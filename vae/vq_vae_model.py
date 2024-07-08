import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embedding, embedding_dim, commitement_cost, verbose=False):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embedding = num_embedding
        # commitement_cost is the beta parameter in the loss function
        self.commitement_cost = commitement_cost
        self.verbose = verbose

        # Create embedding table with size num_embedding x embedding_dim
        self.embedding = nn.Embedding(self.num_embedding, self.embedding_dim)
        # Initialize embedding table with uniform distribution between -1/num_embedding and 1/num_embedding
        self.embedding.weight.data.uniform_(-1/self.num_embedding, 1/self.num_embedding)
    
    def forward(self, inputs):
        if self.verbose:
            print('inputs:', inputs.shape)
        # convert inputs from BCHW -> BHWC
        x = inputs.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape
        if self.verbose:
            print('after vq permute x:', x.shape)

        # Flatten input to BHW x C, where C is the embedding dimension
        flat_x = x.view(-1, self.embedding_dim)
        if self.verbose:
            print('after vq flat x:', flat_x.shape)

        # Calculate distances between input and embedding vectors
        # flat_input: BHW x C
        # self.embedding.weight: num_embedding x C
        # distances: BHW x C
        # distance = ||x - e_k||^2 = ||x||^2 + ||e_k||^2 - 2 * x * e_k
        distances = (torch.sum(flat_x**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_x, self.embedding.weight.t()))
        if self.verbose:
            print('vq distances:', distances.shape)

        # Find closest embedding, i.e. the one with the smallest distance
        # one hot encoding: BHW x num_embedding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embedding, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        if self.verbose:
            print('vq encodings:', encodings.shape)

        # Quantize and unflatten
        # quantized: BHW x C
        quantized = torch.matmul(encodings, self.embedding.weight).view(x_shape)
        if self.verbose:
            print('vq quantized:', quantized.shape)

        # Loss = ||sg[e_k] - z_q||^2 + beta * ||z_q - sg[e_k]||^2
        q_latent_loss = F.mse_loss(quantized.detach(), x) # torch.mean((quantized.detach() - x)**2)
        e_latent_loss = F.mse_loss(quantized, x.detach()) # torch.mean((quantized - x.detach())**2)
        loss = q_latent_loss + self.commitement_cost * e_latent_loss
        if self.verbose:
            print('vq loss:', loss.shape)

        # Using straight through estimator for quantized
        # trick: pass gradient of x through quantized, but retain the value of quantized
        quantized = x + (quantized - x).detach()
        # perplexity = exp(-\sum(p(x) * log(p(x)))
        # a measure to make sure that the model is learning a good representation that encodings are evenly distributed
        # using information theory to measure how well the model is learning
        # higher perplexity means better representation, lower means worse representation
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        quantized_convert = quantized.permute(0, 3, 1, 2).contiguous()
        if self.verbose:
            print('vq requantized:', quantized_convert.shape)

        return loss, quantized_convert, perplexity, encodings
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens, verbose=False):
        super(ResidualBlock, self).__init__()
        self.resblock = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=num_residual_hiddens, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_residual_hiddens, out_channels=num_hiddens, kernel_size=1, stride=1, bias=False)
        )
        self.verbose = verbose
    
    def forward(self, x):
        return x + self.resblock(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, verbose=False):
        super(ResidualStack, self).__init__()
        self.resstack = nn.ModuleList([ResidualBlock(in_channels, num_hiddens, num_residual_hiddens) for _ in range(num_residual_layers)])
        self.verbose = verbose
    
    def forward(self, x):
        for block in self.resstack:
            x = block(x)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, verbose=False):
        super(Encoder, self).__init__()
        
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens//2, kernel_size=4, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=num_hiddens//2, out_channels=num_hiddens, kernel_size=4, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1)
        self.residual_stack = ResidualStack(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)
        self.verbose = verbose
    
    def forward(self, x):
        if self.verbose:
            print('input:', x.shape)
        x = F.relu(self.conv_1(x))
        if self.verbose:
            print('after encoder conv_1:', x.shape)
        x = F.relu(self.conv_2(x))
        if self.verbose:
            print('after encoder conv_2:', x.shape)
        x = self.conv_3(x)
        if self.verbose:
            print('after encoder conv_3:', x.shape)
        x = self.residual_stack(x)
        if self.verbose:
            print('after encoder residual stack:', x.shape)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, verbose=False):
        super(Decoder, self).__init__()
        
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1)
        self.residual_stack = ResidualStack(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)
        self.conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, out_channels=num_hiddens//2, kernel_size=4, stride=2, padding=1)
        self.conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, out_channels=3, kernel_size=4, stride=2, padding=1)
        self.verbose = verbose
    
    def forward(self, x):
        if self.verbose:
            print('input:', x.shape)
        x = self.conv_1(x)
        if self.verbose:
            print('after decoder conv_1:', x.shape)
        x = self.residual_stack(x)
        if self.verbose:
            print('after decoder residual stack:', x.shape)
        x = F.relu(self.conv_trans_1(x))
        if self.verbose:
            print('after decoder conv_trans_1:', x.shape)
        x = self.conv_trans_2(x)
        if self.verbose:
            print('after decoder conv_trans_2:', x.shape)
        return x


class VQ_VAE(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, verbose=False, decay=0):
        super(VQ_VAE, self).__init__()
        
        self.encoder = Encoder(in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,verbose=verbose)
        self.pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost, verbose=verbose)
        self.decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens, verbose=verbose)
        self.decay = decay
        self.verbose = verbose
        if self.verbose:
            print('VQ-VAE model parameters:', sum(p.numel() for p in self.parameters()))
    
    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_vq_conv(z)
        loss, quantized, perplexity, encodings = self.vector_quantizer(z)
        x_recon = self.decoder(quantized)
        return loss, x_recon, perplexity
    