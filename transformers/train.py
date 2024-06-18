import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from configs import Configs
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


def get_model(configs, tokenizer_src, tokenizer_tgt):
    print("Building model...")
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
        dropout=configs['dropout']
    )

    return model

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
    
    model = get_model(configs, tokenizer_src, tokenizer_tgt).to(device)
    


if __name__ == "__main__":
    config_path = sys.argv[1]
    verbose = sys.argv[2] if len(sys.argv) > 2 else False
    configs = Configs(config_path)
    train_model(configs.get_all_configs(), verbose=verbose)

    