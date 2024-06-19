import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len, verbose=False):
        super(InputEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        if verbose:
            print(f'The input embedding layer has {vocab_size} tokens and {d_model} dimensions')
            total_params = sum(p.numel() for p in self.parameters())
            print(f'The input embedding layer has {total_params:,} parameters')
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f'The input embedding layer has {num_params:,} trainable parameters')
    
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
    def __init__(self, d_model, seq_len, dropout=0.1, verbose=False):
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
        if verbose:
            print(f'The positional encoding has {seq_len} positions and {d_model} dimensions')
            total_params = sum(p.numel() for p in self.parameters())
            print(f'The positional encoding has {total_params:,} parameters')
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f'The positional encoding has {num_params:,} trainable parameters')
    
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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, verbose=False):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        # check if d_model is divisible by nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_k = d_model // nhead
        # Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        if verbose:
            print(f'The multi-head attention has {nhead} heads and {d_model} dimensions')
            total_params = sum(p.numel() for p in self.parameters())
            print(f'The multi-head attention has {total_params:,} parameters')
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f'The multi-head attention has {num_params:,} trainable parameters')
    
    @staticmethod
    def attention(q, k, v, d_k, mask=None, dropout=None):
        # Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
        # q: (batch_size, nhead, seq_len, d_k)
        # k: (batch_size, nhead, seq_len, d_k)
        # v: (batch_size, nhead, seq_len, d_k)
        # mask: (batch_size, seq_len, seq_len)
        # K^T: (batch_size, nhead, d_k, seq_len)
        attention_scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / (d_k ** 0.5)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # V: (batch_size, nhead, seq_len, d_k)
        output = torch.matmul(attention_scores, v)
        return output, attention_scores

    def forward(self, q, k, v, mask=None):
        # q: (batch_size, seq_len, d_model)
        # k: (batch_size, seq_len, d_model)
        # v: (batch_size, seq_len, d_model)
        # mask: (batch_size, seq_len, seq_len)
        q = self.W_q(q) # (batch_size, seq_len, d_model)
        k = self.W_k(k) # (batch_size, seq_len, d_model)
        v = self.W_v(v) # (batch_size, seq_len, d_model)
        # split the d_model into nhead and reshape the tensor as (batch_size, nhead, seq_len, d_k)
        q = q.view(q.shape[0], q.shape[1], self.nhead, self.d_k).permute(0, 2, 1, 3) # (batch_size, nhead, seq_len, d_k)
        k = k.view(k.shape[0], k.shape[1], self.nhead, self.d_k).permute(0, 2, 1, 3) # (batch_size, nhead, seq_len, d_k)
        v = v.view(v.shape[0], v.shape[1], self.nhead, self.d_k).permute(0, 2, 1, 3) # (batch_size, nhead, seq_len, d_k)
        # Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
        output, self.attention_scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        # output: (batch_size, nhead, seq_len, d_k) --> (batch_size, seq_len, nhead, d_k)
        output = output.permute(0, 2, 1, 3).contiguous().view(output.shape[0], output.shape[2], self.d_model)

        return self.W_o(output)

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, ff_dim, dropout=0.1, verbose=False):
        super(FeedForwardNetwork, self).__init__()
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        if verbose:
            print(f'The feed-forward network has {d_model} dimensions and {ff_dim} inner dimensions')
            total_params = sum(p.numel() for p in self.parameters())
            print(f'The feed-forward network has {total_params:,} parameters')
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f'The feed-forward network has {num_params:,} trainable parameters')
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6, verbose=False):
        super(LayerNormalization, self).__init__()
        self.d_model = d_model
        self.eps = eps # epsilon for numerical stability
        # learnable parameters
        # LayerNorm(x) = gamma * (x - mean) / (std + eps) + beta
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        if verbose:
            print(f'The layer normalization has {d_model} dimensions')
            total_params = sum(p.numel() for p in self.parameters())
            print(f'The layer normalization has {total_params:,} parameters')
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f'The layer normalization has {num_params:,} trainable parameters')
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True) # (batch_size, seq_len, 1)
        std = x.std(dim=-1, keepdim=True) # (batch_size, seq_len, 1)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class AddAndNormLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1, verbose=False):
        super(AddAndNormLayer, self).__init__()
        self.d_model = d_model
        self.layer_norm = LayerNormalization(d_model, verbose=verbose)
        self.dropout = nn.Dropout(dropout)
        if verbose:
            print(f'The add-and-norm layer has {d_model} dimensions')
            total_params = sum(p.numel() for p in self.parameters())
            print(f'The add-and-norm layer has {total_params:,} parameters')
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f'The add-and-norm layer has {num_params:,} trainable parameters')
    
    def forward(self, x, sublayer):
        # x: (batch_size, seq_len, d_model)
        # sublayer: sublayer function
        ## the order of layer normalization
        # 选用第二种方式是因为 Normalization 在 sublayer 之前应用，可以使输入 sublayer 的信号具有均匀的分布，从而使得模型训练更加稳定
        # layer normalization 可以缓解梯度消失或梯度爆炸的问题，使得模型训练更加稳定
        # 第二种方式中 sublayer 的输入已经经过 normalization ，使得即使在应用 dropout 后，信号的变化也不会过大
        # 有助于模型在训练时更好地学习有效的表示，而不是过度依赖于 dropout 的随机性
        # return self.layer_norm(x + self.dropout(sublayer(x))) # 第一种方式，和论文中一致
        return x + self.dropout(sublayer(self.layer_norm(x))) # 第二种方式

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block, feed_forward_network, add_and_norm_layer, verbose=False):
        super(EncoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward = feed_forward_network
        self.add_norm1 = add_and_norm_layer
        self.add_norm2 = add_and_norm_layer
        if verbose:
            total_params = sum(p.numel() for p in self.parameters())
            print(f'The encoder block has {total_params:,} parameters')
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f'The encoder block has {num_params:,} trainable parameters')
    
    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, d_model)
        # mask: (batch_size, seq_len, seq_len)
        x = self.add_norm1(x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.add_norm2(x, self.feed_forward)
        return x

class Encoder(nn.Module):
    def __init__(self, encoder_blocks, norm_layer, verbose=False):
        super(Encoder, self).__init__()
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.layer_norm = norm_layer
        if verbose:
            total_params = sum(p.numel() for p in self.parameters())
            print(f'The encoder has {total_params:,} parameters')
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f'The encoder has {num_params:,} trainable parameters')
    
    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, d_model)
        # mask: (batch_size, seq_len, seq_len)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask)
        return self.layer_norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, cross_attention_block, self_attention_block, feed_forward_network, add_and_norm_layer, verbose=False):
        super(DecoderBlock, self).__init__()
        self.cross_attention_block = cross_attention_block
        self.self_attention_block = self_attention_block
        self.feed_forward = feed_forward_network
        self.add_norm1 = add_and_norm_layer
        self.add_norm2 = add_and_norm_layer
        self.add_norm3 = add_and_norm_layer
        if verbose:
            total_params = sum(p.numel() for p in self.parameters())
            print(f'The decoder block has {total_params:,} parameters')
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f'The decoder block has {num_params:,} trainable parameters')
    
    def forward(self, x, enc_output, mask=None, enc_mask=None):
        # x: (batch_size, tgt_seq_len, d_model)
        # enc_output: (batch_size, src_seq_len, d_model)
        # mask: (batch_size, tgt_seq_len, tgt_seq_len)
        # enc_mask: (batch_size, tgt_seq_len, src_seq_len)
        x = self.add_norm1(x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.add_norm2(x, lambda x: self.cross_attention_block(x, enc_output, enc_output, enc_mask))
        x = self.add_norm3(x, self.feed_forward)
        return x

class Decoder(nn.Module):
    def __init__(self, decoder_blocks, norm_layer, verbose=False):
        super(Decoder, self).__init__()
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.layer_norm = norm_layer
        if verbose:
            total_params = sum(p.numel() for p in self.parameters())
            print(f'The decoder has {total_params:,} parameters')
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f'The decoder has {num_params:,} trainable parameters')
    
    def forward(self, x, enc_output, mask=None, enc_mask=None):
        # x: (batch_size, tgt_seq_len, d_model)
        # enc_output: (batch_size, src_seq_len, d_model)
        # mask: (batch_size, tgt_seq_len, tgt_seq_len)
        # enc_mask: (batch_size, tgt_seq_len, src_seq_len)
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, enc_output, mask, enc_mask)
        return self.layer_norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size, verbose=False):
        super(ProjectionLayer, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.projection = nn.Linear(d_model, vocab_size)
        if verbose:
            print(f'The projection layer has {d_model} dimensions and {vocab_size} vocab size')
            total_params = sum(p.numel() for p in self.parameters())
            print(f'The projection layer has {total_params:,} parameters')
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f'The projection layer has {num_params:,} trainable parameters')
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model) --> (batch_size, seq_len, vocab_size)
        return self.projection(x)

class Transformer(nn.Module):
    def __init__(self, src_embedded, tgt_embedded, src_pos_encoder, tgt_pos_encoder, encoder, decoder, projection):
        super(Transformer, self).__init__()
        self.src_embedded = src_embedded
        self.tgt_embedded = tgt_embedded
        self.src_pos_encoder = src_pos_encoder
        self.tgt_pos_encoder = tgt_pos_encoder
        self.encoder = encoder
        self.decoder = decoder
        self.projection = projection
    
    def encode(self, src, src_mask=None):
        # src: (batch_size, src_seq_len) --> (batch_size, src_seq_len, d_model)
        # src_mask: (batch_size, src_seq_len, src_seq_len)
        x = self.src_embedded(src)
        x = self.src_pos_encoder(x)
        return self.encoder(x, src_mask)

    def decode(self, tgt, enc_output, tgt_mask=None, enc_mask=None):
        # tgt: (batch_size, tgt_seq_len) --> (batch_size, tgt_seq_len, d_model)
        # enc_output: (batch_size, src_seq_len, d_model)
        # tgt_mask: (batch_size, tgt_seq_len, tgt_seq_len)
        # enc_mask: (batch_size, tgt_seq_len, src_seq_len)
        x = self.tgt_embedded(tgt)
        x = self.tgt_pos_encoder(x)
        return self.decoder(x, enc_output, tgt_mask, enc_mask)
    
    def project(self, x):
        # x: (batch_size, tgt_seq_len, d_model) --> (batch_size, tgt_seq_len, vocab_size)
        return self.projection(x)

def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model =512, nhead=8, ff_dim=2048, num_encoders=6, num_decoders=6, dropout=0.1, verbose=False):
    # create the input embeddings
    src_embedded = InputEmbedding(src_vocab_size, d_model, src_seq_len, verbose=verbose) 
    tgt_embedded = InputEmbedding(tgt_vocab_size, d_model, tgt_seq_len, verbose=verbose)
    
    # create the positional encodings
    src_pos_encoder = PositionalEncoding(d_model, src_seq_len, dropout, verbose=verbose)
    tgt_pos_encoder = PositionalEncoding(d_model, tgt_seq_len, dropout, verbose=verbose)

    # encoder blocks
    encoders = []
    for _ in range(num_encoders):
        self_attention_block = MultiHeadAttention(d_model, nhead, dropout, verbose=verbose)
        feed_forward = FeedForwardNetwork(d_model, ff_dim, dropout, verbose=verbose)
        add_norm = AddAndNormLayer(d_model, dropout, verbose=verbose)
        encoders.append(EncoderBlock(self_attention_block, feed_forward, add_norm, verbose=verbose))

    encoder = Encoder(encoders, LayerNormalization(d_model, verbose=verbose), verbose=verbose)

    # decoder blocks
    decoders = []
    for _ in range(num_decoders):
        cross_attention_block = MultiHeadAttention(d_model, nhead, dropout, verbose=verbose)
        self_attention_block = MultiHeadAttention(d_model, nhead, dropout, verbose=verbose)
        feed_forward = FeedForwardNetwork(d_model, ff_dim, dropout, verbose=verbose)
        add_norm = AddAndNormLayer(d_model, dropout, verbose=verbose)
        decoders.append(DecoderBlock(cross_attention_block, self_attention_block, feed_forward, add_norm, verbose=verbose))

    decoder = Decoder(decoders, LayerNormalization(d_model, verbose=verbose), verbose=verbose)

    # create the projection layer
    projection = ProjectionLayer(d_model, tgt_vocab_size, verbose=verbose)

    # create the transformer model
    model = Transformer(src_embedded, tgt_embedded, src_pos_encoder, tgt_pos_encoder, encoder, decoder, projection)

    # initialize the model parameters and count the number of trainable parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f'The model has {total_params:,} parameters')
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'The model has {num_params:,} trainable parameters')

    return model








