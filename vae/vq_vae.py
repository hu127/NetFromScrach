import torch
import torch.nn as nn

import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


img_transformes = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)) # Normalize to [-1, 1]
])

training_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=img_transformes)
validation_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=img_transformes)
