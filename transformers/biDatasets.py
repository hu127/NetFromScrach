import torch
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt, seq_len=128):
        super().__init__()

        self.seq_len = seq_len
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.dataset = dataset

        self.sos_token = torch.tensor(tokenizer_tgt.encode("[SOS]").ids, dtype=torch.int64)
        self.eos_token = torch.tensor(tokenizer_tgt.encode("[EOS]").ids, dtype=torch.int64)
        self.pad_token = torch.tensor(tokenizer_tgt.encode("[PAD]").ids, dtype=torch.int64)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get the source and target text
        src_tgt_pair = self.dataset[idx]
        src_text = src_tgt_pair['translation'][self.lang_src]
        tgt_text = src_tgt_pair['translation'][self.lang_tgt]

        # encode the source and target text
        src_encoding = self.tokenizer_src.encode(src_text)
        tgt_encoding = self.tokenizer_tgt.encode(tgt_text)

        # truncate or pad the source and target text
        src_num_padding_tokens = self.seq_len - len(src_encoding.ids) - 2 # 2 for sos and eos tokens
        tgt_num_padding_tokens = self.seq_len - len(tgt_encoding.ids) - 1 # 1 for eos token

        if src_num_padding_tokens < 0 or tgt_num_padding_tokens < 0:
            raise ValueError("Sequence length is too short, increase seq_len")
        
        # add sos and eos tokens
        # src_ids is the input to the encoder
        src_ids = torch.cat([
            self.sos_token, 
            torch.tensor(src_encoding.ids, dtype=torch.int64), 
            self.eos_token, 
            torch.tensor([self.pad_token] * src_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0 # concatenate along the first dimension
            )

        # labels is the target for the model
        labels = torch.cat([
            torch.tensor(tgt_encoding.ids, dtype=torch.int64), 
            self.eos_token, 
            torch.tensor([self.pad_token] * tgt_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0 # concatenate along the first dimension
            )
        
        # tgt_ids is the input for the decoder
        tgt_ids = torch.cat([
            self.sos_token,
            torch.tensor(tgt_encoding.ids, dtype=torch.int64),
            torch.tensor([self.pad_token] * tgt_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0 # concatenate along the first dimension
            )
    
        # check that the lengths are correct
        assert len(src_ids) == self.seq_len
        assert len(labels) == self.seq_len
        assert len(tgt_ids) == self.seq_len

        return {
            'encoder_input': src_ids, # seq_len
            'decoder_input': tgt_ids, # seq_len
            'labels': labels, # seq_len
            'encoder_text': src_text,
            'decoder_text': tgt_text,
            'encoder_mask': (src_ids != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # 1, 1, seq_len
            'decoder_mask': (tgt_ids != self.pad_token).unsqueeze(0).int() & self.tri_mask(self.seq_len), # 1, seq_len, seq_len
        }
    
    # define the mask for the decoder
    def tri_mask(self, size):
        mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
        return mask == 0
        
