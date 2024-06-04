import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

class BaseLayer(torch.nn.Module, ABC):
    """ Abstract base class for layer function of implicit network. """

    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim)
        self.A = nn.Linear(out_dim, out_dim, bias=False)
        self.B = nn.Linear(out_dim, out_dim, bias=False)
        self.U = nn.Linear(in_dim, out_dim)

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def forward(self, x, z):
        pass

class BaseNet(torch.nn.Module, ABC):
    """ Base class for implicit networks. """

    def __init__(self, in_dim, out_dim, max_iter=100, tol=1e-6):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.max_iter = max_iter
        self.tol = tol
        self.layer = BaseLayer(in_dim, out_dim)
    
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def forward(self, x, z=None):
        pass
    
    def train_step(self, X_batch, Y_batch):
        self.train() 
        z = self.forward(X_batch)
        loss = self.criterion(z, Y_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_model(self, train_loader, test_loader=None, max_epochs=100):
        epochs = []
        times = []
        test_losses = []

        start_time = time.time()

        # If testing, evaluate model on test data before training
        if test_loader is not None:
            test_loss = self.test_model(test_loader)
            
            epochs.append(0)
            times.append(time.time() - start_time)
            test_losses.append(test_loss)

            print(f'Model: {self.name()}, Epoch: {0}/{max_epochs}, Test Loss: {test_loss:.3f}, Time: {time.time() - start_time:.1f} s')

        for epoch in range(max_epochs):
            for i, (X_batch, Y_batch) in enumerate(train_loader):
                self.train() 
                self.train_step(X_batch, Y_batch)

            # If testing, evaluate model on test data once per epoch
            if test_loader is not None:
                test_loss = self.test_model(test_loader)

                epochs.append(epoch + 1)
                times.append(time.time() - start_time)
                test_losses.append(test_loss)

            print(f'Model: {self.name()}, Epoch: {epoch+1}/{max_epochs}, Test Loss: {test_loss:.5f}, Time: {time.time() - start_time:.1f} s')

        if test_loader is not None:
            return epochs, times, test_losses
    
    def test_model(self, test_loader):
        self.eval()
        total_test_loss = 0
        for X_batch, Y_batch in test_loader:
            total_test_loss += self.criterion(self.forward(X_batch), Y_batch).item()
        test_loss = total_test_loss / len(test_loader)
        return test_loss
