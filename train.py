import torch
import torch.utils

import matplotlib.pyplot as plt

from utils import data_utils 
from utils import model_utils

from models.mon_net_AD import MonNetAD
from models.mon_net_JFB import MonNetJFB
from models.mon_net_JFB_R import MonNetJFBR
from models.mon_net_JFB_CSBO import MonNetJFBCSBO


# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Set parameters
input_dim = 6
output_dim = 3
Models = [MonNetAD,
          MonNetJFB,
          MonNetJFBR,
          MonNetJFBCSBO]
loss_function = torch.nn.MSELoss()
dataset_size = 1024
train_size = round(0.8 * dataset_size)
test_size = dataset_size - train_size
max_epochs = 20
batch_size = 32
lr = 0.01
seed = 0

# Set random seed for ground truth model initialization and synthetic data generation
model_utils.set_seed(seed)

# Synthesize and split data, and instantiate data loaders
data_utils.synthesize_data(MonNetAD, input_dim, output_dim, dataset_size, 'data/dataset.pth')
dataset_dict = torch.load('data/dataset.pth')
X = dataset_dict['X']
Y = dataset_dict['Y']
dataset = torch.utils.data.TensorDataset(X, Y)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Train models
plt.figure(figsize=(10, 5))
for Model in Models:
    # Alter random seed for model initialization
    model_utils.set_seed(seed + 1)

    # Instantiate the model and set the optimizer and loss function
    model = Model(input_dim, output_dim)
    model.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.criterion = loss_function

    # Print first input, output, and prediction as numpy arrays
    print(f'BEFORE TRAINING')
    print(f'X[0]: {X[0].detach().numpy()}')
    print(f'Y[0]: {Y[0].detach().numpy()}')
    model.eval()
    print(f'model(X[0]): {model(X[0]).detach().numpy()}')

    # Check for NaNs in parameters before training
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f'NaN detected in parameter {name} before training.')


    # # Determine why the model is outputing nan
    # for param in model.parameters():
    #     print(type(param), param.size())
    #     print(param)

    # Train the model using batched SGD and save training and testing losses 
    # TODO: create option for randomly sampled batches instead of epoch-wise
    epochs, times, test_losses = model.train_model(train_loader, test_loader, max_epochs)

    # Print first input, output, and prediction as numpy arrays
    print(f'AFTER TRAINING')
    print(f'X[0]: {X[0].detach().numpy()}')
    print(f'Y[0]: {Y[0].detach().numpy()}')
    model.eval()
    print(f'model(X[0]): {model(X[0]).detach().numpy()}')

    # Check for NaNs in parameters after training
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f'NaN detected in parameter {name} after training.')

    # Plotting the training and testing losses
    plt.plot(times, test_losses, label=f'{model.name()}')

plt.xlabel('Time (s)')
plt.yscale('log')
plt.ylabel('Test Loss')
plt.legend()
plt.grid(True)
plt.savefig(f'results/loss_plot.png', dpi=600)