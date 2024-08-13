import torch
import torch.utils
import matplotlib.pyplot as plt

from src.models.fwd_step_net import FwdStepNetAD, FwdStepNetJFB, FwdStepNetJFBR, FwdStepNetCSBO
from src.utils.loading import synthesize_data 
from src.utils.seed import set_seed
from src.utils.config import default_config
from src.utils.device import get_device

device = get_device(verbose=True)

# Set parameters
True_Model = {'class':FwdStepNetAD, 'new_config':{}}
Models = [
    {'class':FwdStepNetAD, 'new_config':{}},
    {'class':FwdStepNetJFB, 'new_config':{}},
    {'class':FwdStepNetJFBR, 'new_config':{}},
    {'class':FwdStepNetCSBO, 'new_config':{}}
    ]
loss_function = torch.nn.MSELoss()
dataset_size = 10000
train_size = round(0.8 * dataset_size)
test_size = dataset_size - train_size
max_epochs = 10
batch_size = 32
lr = 10
seed = 0

# Set random seed for ground truth model initialization and synthetic data generation
set_seed(seed)

# Synthesize and split data, and instantiate data loaders
synthesize_data(True_Model, dataset_size, 'data/dataset.pth')
dataset_dict = torch.load('data/dataset.pth')
X = dataset_dict['X']
Y = dataset_dict['Y']
dataset = torch.utils.data.TensorDataset(X, Y)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Train models
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
plt.subplots_adjust(wspace=0.4)  # Adjust the width space between the subplot

for Model_config in Models:
    # Alter random seed for model initialization
    set_seed(seed + 1)

    # Instantiate the model and set the optimizer and loss function
    Model = Model_config['class']
    new_config = Model_config['new_config']
    config = {**default_config, **new_config} # override default config with any new config
    model = Model(config)
    model.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.criterion = loss_function

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

    # # Print first input, output, and prediction as numpy arrays
    # print(f'AFTER TRAINING')
    # print(f'X[0]: {X[0].detach().numpy()}')
    # print(f'Y[0]: {Y[0].detach().numpy()}')
    # model.eval()
    # print(f'model(X[0]): {model(X[0]).detach().numpy()}')

    # Check for NaNs in parameters after training
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f'NaN detected in parameter {name} after training.')

    # Plotting the training and testing losses
    ax1.plot(epochs, test_losses, label=f'{model.name()}')
    ax2.plot(times, test_losses, label=f'{model.name()}')

ax1.set_title('Test Loss vs. Epochs')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Test Loss')
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True)

ax2.set_title('Test Loss vs. Time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Test Loss')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True)

plt.savefig(f'outputs/loss_plot.png', dpi=600)