import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

# Based on https://github.com/locuslab/monotone_op_net/blob/master/mon.py
class MonLayer(torch.nn.Module):
    """ Single monotone layer that computes
            z_(k+1) = ReLU(W z_k + U x + b)
        where 
            W = (1-m)I - A^T A + B - B^T.
    """
    def __init__(self, in_dim, out_dim, m=1.0):
        super().__init__()
        self.U = nn.Linear(in_dim, out_dim)
        self.A = nn.Linear(out_dim, out_dim, bias=False)
        self.B = nn.Linear(out_dim, out_dim, bias=False)
        self.m = m
    
    def name(self):
        return 'MonLayer'

    def forward(self, x, z):
        return F.relu(self.multiply(z) + self.U(x))

    def multiply(self, z):
        ATAz = self.A(z) @ self.A.weight 
        return (1 - self.m) * z - ATAz + self.B(z) - z @ self.B.weight

class BaseMonNet(torch.nn.Module, ABC):

    def __init__(self, in_dim, out_dim, m=1.0, max_iter=100, tol=1e-6, seed=0):
        #TODO: find theoretically motivated choice for tolerance given guaranteed convergence
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.max_iter = max_iter
        self.tol = tol
        self.mon_layer = MonLayer(in_dim, out_dim, m)
    
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def forward(self, x, z=None):
        pass
    
    @abstractmethod
    def train_step(self, X_batch, Y_batch):
        pass
