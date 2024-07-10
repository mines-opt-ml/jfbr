import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

from src.models.base_net import BaseLayer, BaseNet
from src.models.training_methods import TrainAD, TrainJFB, TrainJFBR, TrainCSBO
from src.utils.config import default_config

# Based on https://github.com/locuslab/monotone_op_net/blob/master/mon.py
class MonLayer(BaseLayer):
    """ Single layer that computes
            z_(k+1) = ReLU(W z_k + U x + b)
        where 
            W = (1-m)I - A^T A + B - B^T.
        The layer is a monotone function composed with ReLU.
        Finding the fixed point of this layer can be done
        with a splitting method.
    """
    def __init__(self, config=default_config):
        super().__init__(config)
        self.m = config['m']

    def name(self):
        return 'MonLayer'

    def forward(self, x, z):
        return F.relu(self.W(z) + self.U(x))

    def W(self, z):
        ATAz = self.A(z) @ self.A.weight 
        Wz = (1 - self.m) * z - ATAz + self.B(z) - z @ self.B.weight
        return Wz

class BaseMonNet(BaseNet, ABC):
    """ Base class for monotone networks, which apply FPI
        to find the fixed point of the monotone layer, unlike
        the paper, which uses splitting methods.
    """

    def __init__(self, config=default_config):
        super().__init__(config)
        self.layer = MonLayer(config)

class MonNetAD(TrainAD, BaseMonNet):
    """ Monotone network trained using automatic differentiation (AD). """

    def __init__(self, config=default_config):
        super().__init__(config)
    
    def name(self):
        return 'MonNetAD'
    
class MonNetJFB(TrainJFB, BaseMonNet):
    """ Monotone network trained using Jacobian free backpropagation (JFB). """

    def __init__(self, config=default_config):
        super().__init__(config)
    
    def name(self):
        return 'MonNetJFB'
    
class MonNetJFBR(TrainJFBR, BaseMonNet):
    """ Monotone network trained using JFB and random selection of number of iterations. """

    def __init__(self, config=default_config):
        super().__init__(config)

        # Initialize probability vector for k, the number of iterations during training
        self.p = torch.tensor([config['decay']**k for k in range(config['max_iter'] + 1)]) # geometric decay of probability
        self.p[0] = 0 # k=0 is not allowed
        self.p = self.p / torch.sum(self.p) # normalize
    
    def name(self):
        return 'MonNetJFBR'
    
class MonNetJFBCSBO(TrainCSBO, BaseMonNet):
    """ Monotone network trained using JFB and gradient update formula from CSBO. """

    def __init__(self, config=default_config):
        super().__init__(config)

        # Initialize propability vector for k, the number of iterations during training
        self.p = torch.tensor([config['decay']**k for k in range(config['max_iter'] + 1)]) # geometric decay of probability
        self.p[0] = 0 # k=0 is not allowed
        self.p = self.p / torch.sum(self.p) # normalize
    
    def name(self):
        return 'MonNetJFBCSBO'