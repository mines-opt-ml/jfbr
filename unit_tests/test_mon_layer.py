import numpy as np
import torch
import matplotlib.pyplot as plt
from models.base_mon_net import MonLayer
from utils.model_utils import set_seed

# Set parameters
seeds = [3, 4]
input_dim = 5
output_dim = 5
m = 1.0
max_iter = 100

for seed in seeds:
    # Initialize MonLayer
    set_seed(seed)
    model = MonLayer(input_dim, output_dim, m)
    model.eval()

    # Iterate MonLayer and store differences
    x = torch.randn(input_dim)
    z = torch.randn(output_dim)
    normed_differences = []
    for i in range(max_iter):
        z_new = model(x, z).detach()
        normed_differences.append(torch.norm(z_new - z, p=2).item())
        z = z_new
        print(f'Seed: {seed}, Iteration: {i}, z: {z}')

    # Plot convergence
    plt.plot(normed_differences, label=seed)

# Save plot
plt.xlabel('Iteration')
plt.ylabel('Normed Difference')
plt.yscale('log')
plt.legend(title='Seed')
plt.savefig('results/mon_layer_convergence.png', dpi=600)