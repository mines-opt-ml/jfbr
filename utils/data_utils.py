import torch
from ..models import simple_net

def synthesize_data(Network, input_dim, output_dim, dataset_size):
    net = Network(input_dim, output_dim)
    net.eval() # No need to train

    # Generate random data
    X = torch.randn(dataset_size, input_dim)
    Y = net(X)

    # Save the data
    dataset_dict = {'X': X, 'Y': Y}
    torch.save(dataset_dict, 'data/dataset.pth')

