import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len):
        super(InputEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        # x: (batch_size, seq_len) --> (batch_size, seq_len, d_model)
        x = self.embedding(x)
        x = x * (self.d_model ** 0.5) # scale the embedding
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # create the positional encodings
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # create position index vector (seq_len, 1)
        # create the div_term: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)) # (d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # add batch dimension (1, seq_len, d_model)
        # register the positional encoding as buffer to avoid it being updated during training
        self.register_buffer('positional_encoding', pe)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.shape[1],:] # extract seq_len length of positional encoding
        x.requires_grad = False # TODO: why positional encoding is not trainable?
        return self.dropout(x)

def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model =512, nhead=8, ff_dim=2048, num_encoders=6, num_decoders=6, dropout=0.1):
    # create the input embeddings
    src_embedded = InputEmbedding(src_vocab_size, d_model, src_seq_len)  # TODO: why don't input embedding has dropout?
    tgt_embedded = InputEmbedding(tgt_vocab_size, d_model, tgt_seq_len)
    
    # create the positional encodings
    src_pos_encoder = PositionalEncoding(d_model, src_seq_len, dropout)



from tokenizers import Tokenizer
text = "The quick brown fox jumps over the lazy dog."
tokenizer = Tokenizer.from_file('/workspaces/NetFromScrach/transformers/configs/tockenizer_en.json')
encoding = tokenizer.encode(text)
print(len(encoding.ids),encoding.ids)
input_embedding = InputEmbedding(tokenizer.get_vocab_size(), 512, 128)
output = input_embedding(torch.tensor(encoding.ids))
print(output.shape)
print(output)

