import torch
import torch.nn as nn

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


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

class VectorQuantizer(nn.Module):
    def __init__(self, num_embedding, embedding_dim, commitement_cost):
        
    
