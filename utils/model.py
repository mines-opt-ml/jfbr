import random
import numpy as np
import torch

def set_seed(seed):
    """Utility function to set the random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# def init_weights(m):
#     """Utility function to initialize weights for layers, using Kaiming He method to avoid vanishing/exploding gradients."""
#     if isinstance(m, torch.nn.Linear):
#         torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#         if m.bias is not None:
#             torch.nn.init.constant_(m.bias, 0)

def approximate_norm(W, dim, num_iter=1): 
    """Power iteration method to estimate the norm of matrix W with multiplication only,
    and without storing computational graph for backpropagation."""
    v = torch.randn(dim)
    with torch.no_grad():
        for _ in range(num_iter-1): 
            v = W(v)
            v = v / torch.norm(v, p=2)
        v = W(v)
        norm = torch.norm(v, p=2)
    return norm