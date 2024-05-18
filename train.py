import torch
import torch.utils
import matplotlib.pyplot as plt
import utils.data_utils as data_utils
import models.mon_net_AD
import models.mon_net_JFB
import models.mon_net_JFBR

# Set parameters
input_dim = 10
output_dim = 20
Model = models.mon_net_JFB.MonNetJFB
model = Model(input_dim, output_dim)
loss_function = torch.nn.MSELoss()
dataset_size = 1024
train_size = round(0.8 * dataset_size)
test_size = dataset_size - train_size
max_epochs = 100
batch_size = 32
lr = 0.01
seed = 0

# Synthesize and split data, and instantiate data loaders
data_utils.synthesize_data(Model, input_dim, output_dim, dataset_size, 'data/dataset.pth')
dataset_dict = torch.load('data/dataset.pth')
X = dataset_dict['X']
Y = dataset_dict['Y']
dataset = torch.utils.data.TensorDataset(X, Y)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model and set the optimizer and loss function
net = Model(input_dim, output_dim)
net.optimizer = torch.optim.SGD(net.parameters(), lr=lr)
net.criterion = loss_function

# Train the model using batched SGD and save training and testing losses 
# TODO: create option for randomly sampled batches instead of epoch-wise
train_epochs, train_losses = net.train_model(train_loader, max_epochs, verbose=True)

# Plotting the training and testing losses
plt.figure(figsize=(10, 5))
plt.plot(train_epochs, train_losses, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

#TODO: test model on test data