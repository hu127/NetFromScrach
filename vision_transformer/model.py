import torch

class PatchEmbedding(torch.nn.Module):
    def __init__(self, image_width, image_height, patch_size, num_channels, d_model, dropout=0.1,verbose=False):
        super(PatchEmbedding, self).__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.patch_size = patch_size
        # num_patches is the number of patches in the image, equal to the sequence length
        self.num_patches = (image_width // patch_size) * (image_height // patch_size)
        # check if the image size is divisible by the patch size
        assert image_width % patch_size == 0 and image_height % patch_size == 0, 'image dimensions must be divisible by the patch size'
        # the linear projection from the flattened patches to the d_model dimensional space
        self.d_model = d_model
        self.projection = torch.nn.Conv2d(
            in_channels=num_channels, 
            out_channels=d_model, 
            kernel_size=patch_size, 
            stride=patch_size
            )
        
    def forward(self, x):
        # x has shape (batch_size, num_channels, image_size, image_size)
        # projection_out has shape (batch_size, d_model, sqrt(num_patches), sqrt(num_patches))
        projection_out = self.projection(x)
        
        # output has shape (batch_size, num_patches, d_model)
        output = projection_out.flatten(2).transpose(1, 2)
        
        return output

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, num_patches):
        super(PositionalEncoding, self).__init__()
        # current patch size is the square root of the number of patches
        # d_model is the number of channels in the input
        self.num_patches = num_patches
        self.d_model = d_model
        # cls_token is the learnable parameter for the class token
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, d_model))
        # positional_encoding is the learnable parameter for the positional encoding, it has shape (1, num_patches + 1, d_model)
        self.positional_encoding = torch.nn.Parameter(torch.zeros(1, num_patches+1, d_model), requires_grad=True)
    
    def forward(self, x):
        batch_size = x.size(0)
        # cls_tokens has shape (batch_size, 1, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # positional_encoding has shape (batch_size, num_patches + 1, d_model)
        positional_encoding = self.positional_encoding.expand(batch_size, -1, -1, -1)
        # output has shape (batch_size, num_patches + 1, d_model)
        output = torch.cat((cls_tokens, x), dim=1)
        # output has shape (batch_size, num_patches + 1, d_model)
        output += positional_encoding
        
        return output

class MlpBlock(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(MlpBlock, self).__init__()
        self.fc1 = torch.nn.Linear(d_model, d_ff)
        self.fc2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x):
        # x has shape (batch_size, num_patches + 1, d_model)
        # fc1_out has shape (batch_size, num_patches + 1, d_ff)
        fc1_out = torch.nn.functional.relu(self.fc1(x))
        # fc1_out has shape (batch_size, num_patches + 1, d_model)
        fc2_out = self.fc2(fc1_out)
        # output has shape (batch_size, num_patches + 1, d_model)
        output = self.dropout(fc2_out)
        
        return output