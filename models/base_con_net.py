import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from models.base_net import BaseLayer, BaseNet
from utils.model import approximate_norm

class ConLayer(BaseLayer):
    """ Single contractive layer that computes
            z_(k+1) = ReLU( (L/||W||) W z_k + U x + b)
        where 
            W = (1-m)I - A^T A + B - B^T
        and L<1 is the Lipschitz constant of the layer with respect to the z.
    """
    def __init__(self, in_dim, out_dim, m=1.0, L=0.9):
        super().__init__(in_dim, out_dim)
        self.m = m
        self.L = L

    def name(self):
        return 'ConLayer'

    def forward(self, x, z):
        W_norm = approximate_norm(self.W, self.A.out_features)
        return F.relu((self.L / W_norm) * self.W(z) + self.U(x)) 

class BaseConNet(BaseNet, ABC):
    """ Base class for contractive networks. """
    def __init__(self, in_dim, out_dim, m=1.0, L=0.9, max_iter=30, tol=1e-6):
        super().__init__(in_dim, out_dim, m, max_iter, tol)
        self.layer = ConLayer(in_dim, out_dim, m, L)
