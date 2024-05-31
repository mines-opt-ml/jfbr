import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from models.base_mon_net import MonLayer, BaseMonNet
from utils.model_utils import get_norm

class ConLayer(MonLayer):
    """ Single contractive layer that computes
            z_(k+1) = ReLU( (L/||W||) W z_k + U x + b)
        where 
            W = (1-m)I - A^T A + B - B^T
        and L<1 is the Lipschitz constant of the layer with respect to the z.
    """
    def __init__(self, in_dim, out_dim, m=1.0, L=0.9):
        super().__init__()
        self.L = L

    def name(self):
        return 'ConLayer'

    def forward(self, x, z):
        W_norm = get_norm(self.W)
        return F.relu((self.L / W_norm) * self.W(z) + self.U(x)) 

    def W(self, z):
        ATAz = self.A(z) @ self.A.weight 
        return (1 - self.m) * z - ATAz + self.B(z) - z @ self.B.weight

class BaseConNet(BaseMonNet, ABC):
    """ Base class for contractive networks. """

    def __init__(self, in_dim, out_dim, m=1.0, L=0.9, max_iter=100, tol=1e-6, seed=0):
        super().__init__()
        self.layer = ConLayer(in_dim, out_dim, m, L)
