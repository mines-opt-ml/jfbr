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
        self.L = L
        self.C_norm_approx = None
        self.m = None

    def name(self):
        return 'MonLipLayer'

    def forward(self, x, z):
        self.C_norm_approx = approximate_norm(self.C, self.out_dim) 
        self.m = (self.m0 * self.L) / self.C_norm_approx
        return (self.L / self.C_norm_approx) * self.C(z) + self.U(x)

    def C(self, z):
        ATAz = self.A(z) @ self.A.weight 
        Cz = self.m0 * z + ATAz + self.B(z) - z @ self.B.weight
        return Cz
    
    def get_alpha(self):
        if self.C_norm_approx is None :
            self.C_norm_approx = approximate_norm(self.C, self.out_dim) 
        alpha = self.m0 / (self.L * self.C_norm_approx)
        return alpha

class BaseFwdStepNet(BaseNet, ABC):
    """ Base class for forward step networks. """

    def __init__(self, in_dim, out_dim, max_iter=100, tol=1e-6, m0=0.5, L=1.0):  
        super().__init__(in_dim, out_dim, max_iter, tol)
        self.layer = MonLipLayer(in_dim, out_dim, m0, L)
        