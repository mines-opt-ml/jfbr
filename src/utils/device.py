import torch

from src.utils.config import default_config

def get_device(verbose=False):
    """ Get the device to use for computation. Priority sequence is preferred CUDA, CUDA, MPS, then CPU."""
    
    # CUDA
    if torch.cuda.is_available():
        preferred_cuda = default_config['preferred-cuda']
        if preferred_cuda in [f'cuda:{i}' for i in range(torch.cuda.device_count())]: 
            device = torch.device(preferred_cuda)
        else:
            device = torch.device('cuda')
    # MPS
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    # CPU
    else:
        device = torch.device('cpu')

    if verbose:
        print(f'Using device: {device}')

    return device