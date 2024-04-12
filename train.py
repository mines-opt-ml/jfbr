import torch
import models.simple_net
from utils.data_utils import synthesize_data
import models
import matplotlib.pyplot as plt

# Set parameters
input_dim = 10
output_dim = 20
dataset_size = 1000
model = models.simple_net.SimpleNet(input_dim, output_dim)

# Synthesize data 
synthesize_data(models.SimpleNet, input_dim, output_dim, dataset_size)

# Instantiate the model
net = models.SimpleNet(input_dim, output_dim)

# Load the data
dataset_dict = torch.load('data/dataset.pth')
X = dataset_dict['X']
Y = dataset_dict['Y']

# Train the model using batched SGD and plot the loss
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
losses = []
batch_size = 10


# # Train the model and plot the loss
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
# losses = []
# for epoch in range(1000):
#     optimizer.zero_grad()
#     Y_pred = net(X)
#     loss = criterion(Y_pred, Y)
#     loss.backward() 
#     optimizer.step() 
#     losses.append(loss.item()) 
# plt.plot(losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()
