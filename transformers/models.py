import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len):
        super(InputEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    ##  why don't input embedding has dropout?
    # 目的是为了将输入的 token 转换为 d_model 维度的向量表示
    # input embedding 是在训练过程中通过反向传播不断更新的，这意味着模型可以自动调整和优化嵌入表示，以捕捉输入数据的复杂模式
    # input embedding 和 positional encoding 是叠加在一起的，
    # positional encoding 的 dropout 可以间接对 input embedding 起到正则化的效果，
    # 从而不需要再对 input embedding单独应用 dropout
    def forward(self, x):
        # x: (batch_size, seq_len) --> (batch_size, seq_len, d_model)
        x = self.embedding(x)
        x = x * (self.d_model ** 0.5) # scale the embedding, adjust the frequency of the signal
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
    
    ## why positional encoding is not trainable?
    # 在 Transformer 中不需要经过训练是因为它们是固定的、预定义的，
    # 并且它们的设计目的是为了在模型中引入位置信息，而不需要通过训练来学习这些编码
    # 位置编码需要 dropout 的主要原因是为了防止模型对特定位置模式的过拟合，
    # 因为位置编码是固定的，所以模型可能会学习到这些位置编码的特定模式，而不是真正的序列模式
    def forward(self, x, requires_grad=False):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.shape[1],:] # extract seq_len length of positional encoding
        x.requires_grad = requires_grad 
        return self.dropout(x)

def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model =512, nhead=8, ff_dim=2048, num_encoders=6, num_decoders=6, dropout=0.1):
    # create the input embeddings
    src_embedded = InputEmbedding(src_vocab_size, d_model, src_seq_len) 
    tgt_embedded = InputEmbedding(tgt_vocab_size, d_model, tgt_seq_len)
    
    # create the positional encodings
    src_pos_encoder = PositionalEncoding(d_model, src_seq_len, dropout)




