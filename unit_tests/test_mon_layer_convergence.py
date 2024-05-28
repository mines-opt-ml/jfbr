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
max_iter = 1000

# Test MonLayer for different seeds
plt.figure(figsize=(15, 5))
for seed in seeds:
    # Initialize MonLayer
    set_seed(seed)
    model = MonLayer(input_dim, output_dim, m)
    model.eval()

    # # Print MonLayer parameters
    # print(f'MonLayer parameters for seed {seed}:')
    # for name, param in model.named_parameters():
    #     print(f'{name}: {param}')

    # Iterate MonLayer and store differences
    x = torch.randn(input_dim)
    z = torch.randn(output_dim)
    normed_differences = []
    with torch.no_grad():
        for i in range(max_iter):
            z_new = model(x, z)
            normed_differences.append(torch.norm(z_new - z, p=2).item())
            z = z_new
            #print(f'Seed: {seed}, Iteration: {i}, z: {z}')

    # Plot convergence
    plt.plot(range(max_iter), normed_differences, label=seed)

# Format and save plot
plt.xlabel('Iteration')
plt.xlim(700, 1000)

plt.ylabel('Normed Difference')
plt.yscale('log')
plt.ylim(1e-8, 1e-6)

# Adjust the figure to make space for the legend
plt.subplots_adjust(right=0.75)

# Place the legend outside the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Seed')

plt.savefig('results/mon_layer_convergence.png', dpi=600)