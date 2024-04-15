import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, X):
        X = self.fc1(X)
        X = self.fc2(X)
        return X
    
    def train_step(self, X, Y):
        self.optimizer.zero_grad() 
        output = self(X)
        loss = self.criterion(output, Y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    