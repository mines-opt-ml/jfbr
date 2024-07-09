import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.fwd_step_net import FwdStepLayer
from src.utils.seed import set_seed
from src.utils.config import default_config

# Set parameters
seeds = [i for i in range(0, 10)]
in_dim = default_config['in_dim']
out_dim = default_config['out_dim']
m = default_config['m']
sample_size = 1000

# Test MonLayer Lipschitz constant for different seeds
for seed in seeds:
    # Initialize MonLayer
    set_seed(seed)
    model = FwdStepLayer()
    model.eval()

    # Sample pairs of points and compute the distance stretch factor after iteration
    stretches = []
    for _ in range(sample_size):
        x = torch.randn(in_dim)
        z1 = torch.randn(out_dim)
        z2 = torch.randn(out_dim)
        d1 = torch.norm(z1-z2, p=2).item()

        Fz1 = model(x, z1).detach()
        Fz2 = model(x, z2).detach()
        d2 = torch.norm(model(x, z1) - model(x, z2), p=2).item()

        stretches.append(d2/d1)

    min_stretch = min(stretches)
    max_stretch = max(stretches)
    print (f'Seed: {seed}, Min stretch factor: {min_stretch:.3f}, Max stretch factor: {max_stretch:.3f}')

# Plot histogram of stretch factors, from 0 to 1 with 100 bins
plt.hist(stretches, bins=100, range=(0, 1), density=True)
plt.savefig('outputs/stretch_hist.png')