import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.fwd_step_net import FwdStepLayer
from src.utils.seed import set_seed
from src.utils.config import default_config

# Set parameters
in_dim = default_config['in_dim']
out_dim = default_config['out_dim']
m = default_config['m']
max_iter = 200
num_sequences = 10

# Set seed and initialize FwdStepLayer and input x
set_seed(0)
model = FwdStepLayer()
model.eval()
x = torch.randn(in_dim)

plt.figure(figsize=(15, 5))
for sequence in range(num_sequences):
    # Initialize MonLayer


    # # Print MonLayer parameters
    # print(f'MonLayer parameters for seed {seed}:')
    # for name, param in model.named_parameters():
    #     print(f'{name}: {param}')

    # Iterate MonLayer and store differences
    z = torch.randn(out_dim)
    normed_differences = []
    with torch.no_grad():
        for i in range(max_iter):
            z_new = model(x, z)
            normed_differences.append(torch.norm(z_new - z, p=2).item())
            z = z_new
            #print(f'Seed: {seed}, Iteration: {i}, z: {z}')
        print(f'Sequence: {sequence}, Final z: {z}')

    # Plot convergence
    plt.plot(range(max_iter), normed_differences, label=f'Sequence {sequence}')

# Format and save plot
plt.xlabel('Iteration')
plt.xlim(0, max_iter)

plt.ylabel('Normed Difference')
plt.yscale('log')

# Adjust the figure to make space for the legend
plt.subplots_adjust(right=0.75)

# Place the legend outside the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Seed')

plt.savefig(f'outputs/{model.name()}_convergence.png', dpi=600)