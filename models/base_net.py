import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from utils.config import default_config

class BaseLayer(torch.nn.Module, ABC):
    """ Abstract base class for layer function of implicit network. """

    def __init__(self, config=default_config):
        super().__init__()
        self.in_dim = config['in_dim']
        self.out_dim = config['out_dim']
        
        self.A = nn.Linear(self.out_dim, self.out_dim, bias=False)
        self.B = nn.Linear(self.out_dim, self.out_dim, bias=False)
        self.U = nn.Linear(self.in_dim, self.out_dim)

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def forward(self, x, z):
        pass

class BaseNet(torch.nn.Module, ABC):
    """ Base class for implicit networks. """

    def __init__(self, config=default_config):
        super().__init__()
        self.in_dim = config['in_dim']
        self.out_dim = config['out_dim']
        self.max_iter = config['max_iter']
        self.tol = config['tol']
        self.layer = None
    
    @abstractmethod
    def name(self):
        pass

    def forward(self, x, z=None):
        z = torch.zeros(self.out_dim) if z is None else z
        
        # Train
        if self.training:
            self.forward_train(x, z)

        # Evaluate
        else:
            self.forward_eval(x, z)
    
    @abstractmethod
    def forward_train(self, x, z):
        pass
    
    def forward_eval(self, x, z):
        for _ in range(self.max_iter):
            z_new = self.layer(x, z)
            if torch.norm(z_new - z, p=2) < self.tol:
                z = z_new
                break
            z = z_new
        return z

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
