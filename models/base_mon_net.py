import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from models.base_net import BaseLayer, BaseNet
from utils.config import default_config

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
    