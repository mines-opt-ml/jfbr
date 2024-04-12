import models.simple_net
import torch
import models
import utils.data_utils as data_utils
import matplotlib.pyplot as plt

# Set parameters
input_dim = 10
output_dim = 20
dataset_size = 1000
model = models.simple_net.SimpleNet(input_dim, output_dim)
epochs = 100
batch_size = 10

# Synthesize data 
data_utils.synthesize_data(models.simple_net.SimpleNet, input_dim, output_dim, dataset_size)

# Instantiate the model
net = models.simple_net.SimpleNet(input_dim, output_dim)

# Load the data
dataset_dict = torch.load('data/dataset.pth')
X = dataset_dict['X']
Y = dataset_dict['Y']

# Train the model using batched SGD and save losses 
# TODO: create option for randomly sampled batches instead of epoch-wise
losses = []
for epoch in range(epochs):
    for i in range(0, dataset_size, batch_size):
        X_batch = X[i:i+batch_size]
        Y_batch = Y[i:i+batch_size]
        loss = net.train_step(X_batch, Y_batch)
        losses.append(loss)

# Plot loss versus epoch
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

