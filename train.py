import torch
import torch.utils
import matplotlib.pyplot as plt
import utils.data_utils as data_utils
import models.mon_net_AD
import models.mon_net_JFB
import models.mon_net_JFB_R
import models.mon_net_JFB_CSBO

assert torch.cuda.is_available()

# Set parameters
input_dim = 10
output_dim = 20
Models = [models.mon_net_AD.MonNetAD,
          models.mon_net_JFB.MonNetJFB,
          models.mon_net_JFB_R.MonNetJFBR,
          models.mon_net_JFB_CSBO.MonNetJFBCSBO]
loss_function = torch.nn.MSELoss()
dataset_size = 1024
train_size = round(0.8 * dataset_size)
test_size = dataset_size - train_size
max_epochs = 20
batch_size = 32
lr = 0.01
seed = 0

# Synthesize and split data, and instantiate data loaders
data_utils.synthesize_data(Models[0], input_dim, output_dim, dataset_size, 'data/dataset.pth')
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
    # Instantiate the model and set the optimizer and loss function
    model = Model(input_dim, output_dim)
    model.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.criterion = loss_function

    # Train the model using batched SGD and save training and testing losses 
    # TODO: create option for randomly sampled batches instead of epoch-wise
    epochs, times, test_losses = model.train_model(train_loader, test_loader, max_epochs)

    # Plotting the training and testing losses
    plt.plot(times, test_losses, label=f'{model.name()}')

plt.xlabel('Time (s)')
plt.yscale('log')
plt.ylabel('Test Loss')
plt.legend()
plt.grid(True)
plt.savefig(f'results/loss_plot.png', dpi=600)