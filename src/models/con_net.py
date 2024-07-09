import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

from src.models.base_net import BaseLayer, BaseNet
from src.utils.model import approximate_norm
from src.utils.config import default_config

class ConLayer(BaseLayer):
    """ Single contractive layer that computes
            z_(k+1) = ReLU( (L/||W||) W z_k + U x + b)
        where 
            W = (1-m)I - A^T A + B - B^T
        and L<1 is the Lipschitz constant of the layer with respect to the z.
    """
    def __init__(self, config=default_config):
        super().__init__(config)
        self.m = config['m']
        self.L = config['L']

    def name(self):
        return 'ConLayer'

    def forward(self, x, z):
        W_norm = approximate_norm(self.W, self.A.out_features)
        return F.relu((self.L / W_norm) * self.W(z) + self.U(x)) 
    
    def W(self, z):
        ATAz = self.A(z) @ self.A.weight 
        Wz = (1 - self.m) * z - ATAz + self.B(z) - z @ self.B.weight
        return Wz

class BaseConNet(BaseNet, ABC):
    """ Base class for contractive networks. """
    def __init__(self, config=default_config):
        super().__init__(config)
        self.layer = ConLayer(config)

class ConNetAD(BaseConNet):
    """ Monotone network trained using standard automatic differentiation (AD). """

    def __init__(self, config):
        super().__init__(config)
    
    def name(self):
        return 'ConNetAD'

    def forward_train(self, x, z):
        for _ in range(self.max_iter):
            z = self.layer(x, z)
        return z
        
class ConNetJFB(BaseConNet):
    """ Contractive network trained using Jacobian free backpropagation (JFB). """

    def __init__(self, config=default_config):
        super().__init__(config)
    
    def name(self):
        return 'ConNetJFB'

    def forward_train(self, x, z):
        with torch.no_grad():
            for _ in range(self.max_iter - 1):
                z = self.layer(x, z)
        z = self.layer(x, z)
        return z