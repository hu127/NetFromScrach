import sys
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torchmetrics

from tqdm import tqdm

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

import config
from config import Configs
from biDatasets import BilingualDataset
from models import build_transformer

def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item['translation'][lang]

def get_or_build_tokenizer(tokenizer_path, dataset, lang):
    if Path(tokenizer_path).exists():
        print("Loading tokenizer from", tokenizer_path)
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        print("Building tokenizer from scratch")
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(tokenizer_path)

    return tokenizer

def load_data(configs, verbose=False):
    # load dataset
    dataset_raw = load_dataset(configs['data_source'], f"{configs['lang_src']}-{configs['lang_tgt']}", split='train')

    # get tokenizer
    tockenizer_src_path = configs['tockenizer_file'].format(configs['lang_src'])
    tockenizer_tgt_path = configs['tockenizer_file'].format(configs['lang_tgt'])
    tokenizer_src = get_or_build_tokenizer(tockenizer_src_path, dataset_raw, configs['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(tockenizer_tgt_path, dataset_raw, configs['lang_tgt'])

    train_size = int(len(dataset_raw) * configs['train_ratio'])
    val_size = len(dataset_raw) - train_size
    train_raw, val_raw = random_split(dataset_raw, [train_size, val_size])

    train_dataset = BilingualDataset(train_raw, tokenizer_src, tokenizer_tgt, configs['lang_src'], configs['lang_tgt'], seq_len=configs['seq_len'])
    val_dataset = BilingualDataset(val_raw, tokenizer_src, tokenizer_tgt, configs['lang_src'], configs['lang_tgt'], seq_len=configs['seq_len'])
    
    if verbose:
        print("Number of training examples:", len(train_dataset))
        print("Number of validation examples:", len(val_dataset))

        max_len_src = 0
        max_len_tgt = 0
        for item in dataset_raw:
            src_len = len(tokenizer_src.encode(item['translation'][configs['lang_src']]).ids)
            tgt_len = len(tokenizer_tgt.encode(item['translation'][configs['lang_tgt']]).ids)
            max_len_src = max(max_len_src, src_len)
            max_len_tgt = max(max_len_tgt, tgt_len)
        
        print("Max source length:", max_len_src)
        print("Max target length:", max_len_tgt)

    dataloader_train = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=configs['batch_size'], shuffle=False)

    return dataloader_train, dataloader_val, tokenizer_src, tokenizer_tgt


def get_model(configs, tokenizer_src, tokenizer_tgt, verbose=False):
    print("Building model...")
    if verbose:
        print("Source vocab size:", tokenizer_src.get_vocab_size())
        print("Target vocab size:", tokenizer_tgt.get_vocab_size())
    model = build_transformer(
        src_vocab_size=tokenizer_src.get_vocab_size(),
        tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
        src_seq_len=configs['seq_len'],
        tgt_seq_len=configs['seq_len'],
        d_model=configs['d_model'],
        nhead=configs['nhead'],
        num_encoders=configs['num_encoders'],
        num_decoders=configs['num_decoders'],
        ff_dim=configs['ff_dim'],
        dropout=configs['dropout'],
        verbose=verbose
    )

    return model

def greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device, verbose=False):
    model.eval()

    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    with torch.no_grad():
        encoder_output = model.encoder(encoder_input, encoder_mask)
        # initialize decoder input with sos token
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(device)
    
        while True:
            if decoder_input.size(1) >= max_len:
                break
            decoder_mask = BilingualDataset.tri_mask(decoder_input.size(1)).type_as(encoder_mask).to(device) 
            if verbose:
                print("decoder_input:", decoder_input.shape)
                print("decoder_mask:", decoder_mask.shape)
            decoder_output = model.decoder(decoder_input, encoder_output, encoder_mask, decoder_mask)
            logits = model.project(decoder_output)
            _, next_token = torch.max(logits[:, -1, :], dim=-1)
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)
            if verbose:
                print("next_token:", next_token)
                print("decoder_input_size:", decoder_input.size(1))
                print("decoder_input:", decoder_input)
            if next_token.item() == eos_idx:
                break
    return decoder_input.squeeze(0) # remove the batch dimension
        

def run_validation(model, dataloader_val, tokenizer_tgt, device, writer, num_examples=2, verbose=False):

    # get the width of the console
    try:
        with os.popen('stty size', 'r') as console_size:
            console_height, console_width = map(int, console_size.read().split())
            console_width = int(console_width)
    except:
        console_width = 80


    model.eval()
    count = 0

    source_texts = []
    target_texts = []
    predicted_texts = []

    with torch.no_grad():
        for batch in dataloader_val:
            encoder_input = batch['encoder_input'] 
            decoder_input = batch['decoder_input']
            labels = batch['labels']
            encoder_mask = batch['encoder_mask']
            decoder_mask = batch['decoder_mask']
            if verbose:
                print("encoder_input:", encoder_input.shape)
                print("decoder_input:", decoder_input.shape)
                print("labels:", labels.shape)
                print("encoder_mask:", encoder_mask.shape)
                print("decoder_mask:", decoder_mask.shape)
            
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            encoder_mask = encoder_mask.to(device)
            decoder_mask = decoder_mask.to(device)
            labels = labels.to(device)

            source_text = batch['encoder_text']
            target_text = batch['decoder_text']
            predicted_text = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, labels.size(1), device, verbose)
            predicted_text = predicted_text.detach().cpu() 
            predicted_text = tokenizer_tgt.decode(predicted_text.tolist())

            source_texts.extend(source_text)
            target_texts.extend(target_text)
            predicted_texts.extend(predicted_text)

            # print some examples
            if count < num_examples:
                print('-' * console_width)
                print("Source:", source_text)
                print("Target:", target_text)
                print("Predicted:", predicted_text)
                print('-' * console_width)
                

            count += 1
        
        if writer:
            # evaluate the character level accuracy
            metrics = torchmetrics.CharacterErrorRate()
            cer = metrics(predicted_texts, target_texts) # character error rate
            writer.add_scalar("validation character error rate", cer, global_step)
            writer.flush()

            # evaluate the word level accuracy
            metrics = torchmetrics.WordsErrorRate()
            wer = metrics(predicted_texts, target_texts)
            writer.add_scalar("validation word error rate", wer, global_step)
            writer.flush()

            # compute the BLEU score
            metrics = torchmetrics.BLEU()
            bleu = metrics(predicted_texts, target_texts)
            writer.add_scalar("validation BLEU score", bleu, global_step)
            writer.flush()




def print_config(configs):
    for key, value in configs.items():
        print(" -", key, ":", value)

def train_model(configs, verbose=False):
    print("Training model with the following configs:")
    print_config(configs)

    # define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    if device.type == 'cuda':
        print("Device name:", torch.cuda.get_device_name(0))
        print("Memory allocated:", torch.cuda.memory_allocated(0))
        print("Memory cached:", torch.cuda.memory_reserved(0))
    else:
        print("Using CPU")
    
    # make sure the weights folder exists
    weights_folder = f"{configs['data_source']}_{configs['model_folder']}"
    print("Weights folder:", weights_folder)
    Path(weights_folder).mkdir(parents=True, exist_ok=True)

    # get dataloader & tokenizer
    print("Loading data...")
    dataloader_train, dataloader_val, tokenizer_src, tokenizer_tgt = load_data(configs, verbose=verbose)
    
    model = get_model(configs, tokenizer_src, tokenizer_tgt, verbose).to(device)
    
    print("Model summary:")
    print("Total number of parameters:", sum(p.numel() for p in model.parameters()))
    print("Total number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Total amount of memory:", sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024, "MB")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(configs['learning_rate']))

    writer = SummaryWriter(configs['logs_folder'])
    print("Logging folder:", configs['logs_folder'])

    # loading previous weights if exists
    initial_epoch = 0
    global_step = 0

    preload = configs['preload']
    weights_file_path = None
    if preload == 'latest':
        weights_file_path = config.get_latest_weight(configs)
    elif preload is not None:
        weights_file_path = config.get_weight_file_path(configs, preload)
    
    if weights_file_path is not None:
        print("Loading weights from", weights_file_path)
        checkpoint = torch.load(weights_file_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        initial_epoch = checkpoint['epoch']
        initial_epoch += 1
        global_step = checkpoint['global_step']
        print("Loaded epoch:", initial_epoch)
        print("Loaded global step:", global_step)
    else:
        print("No weights loaded")

    # loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=configs['label_smoothing']).to(device)

    # training loop
    for epoch in range(initial_epoch, configs['num_epochs']):
        # empty cache of GPU to avoid memory leak
        torch.cuda.empty_cache()

        model.train()
        batch_iter = tqdm(dataloader_train, desc=f"Processing epoch {epoch:02d}", unit="batch")
        for batch in batch_iter:
            encoder_input = batch['encoder_input'] # (batch_size, seq_len)
            decoder_input = batch['decoder_input'] # (batch_size, seq_len)
            labels = batch['labels'] # (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'] # (batch_size, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'] # (batch_size, 1, seq_len, seq_len)
            if verbose:
                print("encoder_input:", encoder_input.shape)
                print("decoder_input:", decoder_input.shape)
                print("labels:", labels.shape)
                print("encoder_mask:", encoder_mask.shape)
                print("decoder_mask:", decoder_mask.shape)
            
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            encoder_mask = encoder_mask.to(device)
            decoder_mask = decoder_mask.to(device)
            labels = labels.to(device)

            # forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            logits = model.project(decoder_output)

            # calculate loss with labels
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)) # (batch_size * seq_len, vocab_size)
            batch_iter.set_postfix({"loss": f"{loss.item():6.4f}"})

            # logging
            writer.add_scalar("train loss", loss.item(), global_step)
            writer. flush()

            # backward pass
            loss.backward()
            # update weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # run validation
        run_validation(model, dataloader_val, tokenizer_tgt, device, writer, verbose)

        # save weights
        weights_file_path = configs.get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'global_step': global_step
        }, weights_file_path)


            
    


if __name__ == "__main__":
    config_path = sys.argv[1]
    verbose = sys.argv[2] if len(sys.argv) > 2 else False
    configs = Configs(config_path)
    train_model(configs.get_all_configs(), verbose=verbose)

    