import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from models.base_net import BaseLayer, BaseNet
from utils.model import approximate_norm

class MonLipLayer(BaseLayer):
    """ Layer function for BaseFwdStepNet 
            F(z) = (L/||C||) * C z + Ux + b,
        that is m-strongly monotone and L-Lipschitz, where
            C = mI + A^T A + B - B^T
        and
            m = m0 L / ||C||.
    """
    def __init__(self, in_dim, out_dim, m0=0.5, L=1.0):
        super().__init__(in_dim, out_dim)
        self.m0 = m0
        self.m = None
        self.L = L

    def name(self):
        return 'MonLipLayer'

    def forward(self, x, z):
        C_norm = approximate_norm(self.C) 
        self.m = (self.m0 * self.L) / C_norm
        return (self.L / C_norm) * self.C(z) + self.U(x)

    def C(self, z):
        ATAz = self.A(z) @ self.A.weight 
        Cz = self.m0 * z + ATAz + self.B(z) - z @ self.B.weight
        return Cz

class BaseFwdStepNet(BaseNet, ABC):
    """ Base class for forward step networks. """

    def __init__(self, in_dim, out_dim, max_iter=100, tol=1e-6, m0=0.5, L=1.0):  
        super().__init__(in_dim, out_dim, max_iter, tol)
        self.alpha = None
        self.layer = MonLipLayer(in_dim, out_dim, m0, L)
    
    def name(self):
        return 'BaseFwdStepNet'

    def forward(self, x, z=None):
        pass
    