import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np

from vq_vae_model import VQ_VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

batch_size = 128

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

learning_rate = 1e-3
num_training_updates = 15000
decay = 0.99


img_transformes = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)) # Normalize image to mean 0 and std 1
])

training_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=img_transformes)
validation_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=img_transformes)
print('Number of training examples:', len(training_data))
print('Number of validation examples:', len(validation_data))

data_variance = np.var(training_data.data / 255.0)
print('Data variance:', data_variance.item())

train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)
valid_loader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True, pin_memory=True)

model = VQ_VAE(in_channels=3, num_hiddens=num_hiddens, num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens, num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost, decay=decay).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

model.train()
train_res_recon_error = []
train_res_vq_error = []

for i in range(num_training_updates):
    (data, _) = next(iter(train_loader))
    data = data.to(device)
    
    optimizer.zero_grad()
    
    vq_loss, data_recon, perplexity = model(data)
    # Reconstruction loss, normalized by data variance
    recon_error = F.mse_loss(data_recon, data) / data_variance
    loss = recon_error + vq_loss
    loss.backward()
    
    optimizer.step()
    
    train_res_recon_error.append(recon_error.item())
    train_res_vq_error.append(vq_loss.item())
    
    if (i+1) % 100 == 0:
        print('%d iterations' % (i+1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
        print('vq_loss: %.3f' % np.mean(train_res_vq_error[-100:]))
        print('perplexity: %.3f' % perplexity)
        print()

# Save the model
torch.save(model.state_dict(), 'vq_vae_model.pth')
print('Model saved to vq_vae_model.pth')

# Plot the training curves
import matplotlib.pyplot as plt
train_res_recon_error = np.array(train_res_recon_error)
train_res_vq_error = np.array(train_res_vq_error)

f = plt.figure(figsize=(16, 6))
ax = f.add_subplot(121)
ax.plot(train_res_recon_error, label='recon_error')
ax.set_yscale('log')
ax.set_title('RMSE Error')
ax.set_xlabel('Iteration')

ax = f.add_subplot(122)
ax.plot(train_res_vq_error, label='vq_error')
ax.set_yscale('log')
ax.set_title('Average codebook usage (perplexity)')
ax.set_xlabel('Iteration')
plt.savefig('vq_vae_training_curves.png')


