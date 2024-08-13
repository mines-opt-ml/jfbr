import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from src.utils.config import default_config

def load_data(dataset_type):
    """
    Get and save data and return train and test data loaders for a synthetic, MNIST, SVHN, or CIFAR-10 dataset.
    """
    
    train_dataset = None
    test_dataset = None
    
    if dataset_type == 'synthetic':
        pass  #TODO: Implement synthetic data generation
    
    elif dataset_type == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)
    
    elif dataset_type == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.SVHN(root='./data/svhn', split='train', download=True, transform=transform)
        test_dataset = datasets.SVHN(root='./data/svhn', split='test', download=True, transform=transform)
    
    elif dataset_type == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        train_dataset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=default_config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=default_config['batch_size'], shuffle=False)

    return train_loader, test_loader

def get_model():
    pass 
