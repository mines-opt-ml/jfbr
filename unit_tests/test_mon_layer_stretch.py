import numpy as np
import torch
import matplotlib.pyplot as plt
from models.base_mon_net import MonLayer
from utils.model_utils import set_seed

# Set parameters
seeds = [i for i in range(0, 10)]
input_dim = 10
output_dim = 10
m = 1.0
sample_size = 1000

# Test MonLayer Lipschitz constant for different seeds
for seed in seeds:
    # Initialize MonLayer
    set_seed(seed)
    model = MonLayer(input_dim, output_dim, m)
    model.eval()

    # Sample pairs of points and compute the distance stretch factor after iteration
    min_s = np.inf # Min stretch factor
    max_s = 0 # Max stretch factor
    for _ in range(sample_size):
        x = torch.randn(input_dim)
        z1 = torch.randn(output_dim)
        z2 = torch.randn(output_dim)
        d1 = torch.norm(z1-z2, p=2).item()

        Fz1 = model(x, z1).detach()
        Fz2 = model(x, z2).detach()
        d2 = torch.norm(model(x, z1) - model(x, z2), p=2).item()

        s = d2/d1
        if s > max_s:
            max_s = s
        if s < min_s:
            min_s = s

    print (f'Seed: {seed}, Min stretch factor: {min_s:.3f}, Max stretch factor: {max_s:.3f}')