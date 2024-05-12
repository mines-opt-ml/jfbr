import torch.utils
import models.mon_net
import models.simple_net
import torch
import models
import utils.data_utils as data_utils
import matplotlib.pyplot as plt

# Set parameters
input_dim = 10
output_dim = 20
Model = models.mon_net.MonNet
model = Model(input_dim, output_dim)
loss = torch.nn.MSELoss()
dataset_size = 10000
train_size = round(0.8 * dataset_size)
test_size = dataset_size - train_size
max_epochs = 10
batch_size = 100
lr = 0.01
# TODO: fix generator with random seed and add as argument to random functions for reproducibility

# Synthesize data 
data_utils.synthesize_data(Model, input_dim, output_dim, dataset_size, 'data/dataset.pth')
dataset_dict = torch.load('data/dataset.pth')
X = dataset_dict['X']
Y = dataset_dict['Y']
X_train, X_test = torch.utils.data.random_split(X, [train_size, test_size])
Y_train, Y_test = torch.utils.data.random_split(Y, [train_size, test_size])

# Instantiate the model and set the optimizer and loss function
net = Model(input_dim, output_dim)
net.optimizer = torch.optim.SGD(net.parameters(), lr=lr)
net.criterion = loss

# Train the model using batched SGD and save losses 
# TODO: create option for randomly sampled batches instead of epoch-wise
train_epochs = []
train_losses = []
test_epochs = []
test_losses = []
for epoch in range(max_epochs):
    # Train and save training error for each batch
    for i in range(0, dataset_size, batch_size):
        X_batch = X_train[i:i+batch_size]
        Y_batch = Y_train[i:i+batch_size]
        loss = net.train_step(X_batch, Y_batch) 
        train_epochs.append(epoch + i/dataset_size)
        train_losses.append(loss)
    
    # Save test error at the end of every epoch
    train_epochs.append(epoch)
    test_losses.append(net.criterion(net(X_test), Y_test))

# Plot training and testing loss versus epoch
plt.plot(train_epochs, train_losses, label='Train')
plt.plot(test_epochs, test_losses, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

