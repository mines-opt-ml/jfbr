import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from src.models.fwd_step_net import FwdStepLayer
from src.utils.seed import set_seed
from src.utils.config import default_config

def get_stretch(z1, z2):
    d1 = torch.norm(z1 - z2, p=2).item()
    d2 = torch.norm(layer(x, z1) - layer(x, z2), p=2).item()
    return d2 / d1

# Set parameters
seed = 0
in_dim = default_config['in_dim']
out_dim = default_config['out_dim']
scales = range(-6, 6)
sample_size = 1000

# Initialize layer function
set_seed(seed)
layer = FwdStepLayer()

# Compute fixed point
x = torch.randn(in_dim)
z = torch.randn(out_dim)
# Iterate until convergence
with torch.no_grad():
    for _ in range(1000):
        z = layer(x, z)
z_fixed = z.detach()

# Sample stretch factor at different scales relative to the fixed point
stretches = []
scale_labels = []

with torch.no_grad():
    for scale in scales:
        for _ in range(sample_size):
            z1 = z_fixed + F.normalize(torch.randn(out_dim), p=2, dim=0) * 10**scale
            z2 = z_fixed + F.normalize(torch.randn(out_dim), p=2, dim=0) * 10**scale
            stretches.append(get_stretch(z_fixed, z1))
            stretches.append(get_stretch(z_fixed, z2))
            stretches.append(get_stretch(z1, z2))
            scale_labels.extend([scale] * 3)

        print(f'Scale {scale}: Min stretch factor: {min(stretches[-3*sample_size:]):.3f}, Max stretch factor: {max(stretches[-3*sample_size:]):.3f}')

# Prepare data for the violin plot
data = [stretches[i * 3 * sample_size:(i + 1) * 3 * sample_size] for i in range(len(scales))]

# Create a violin plot using Matplotlib
plt.figure(figsize=(12, 6))
plt.violinplot(data, positions=scales, showmeans=False, showmedians=True)
plt.xlabel('Scale')
plt.ylabel('Stretch Factor')
plt.ylim(0, 1)
plt.title('Stretch Factors at Different Scales')
plt.xticks(ticks=scales, labels=[str(scale) for scale in scales])

# Ensure the directory exists
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f'{output_dir}/layer_stretch.png')
plt.show()
