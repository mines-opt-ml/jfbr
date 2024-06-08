import torch
import time
from models.base_mon_net import MonLayer, BaseMonNet

class MonNetAD(BaseMonNet):
    """ Monotone network trained using automatic differentiation (AD). """

    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim, m=1.0)
    
    def name(self):
        return 'MonNetAD'

    def forward_train(self, x, z=None):
        for _ in range(self.max_iter):
            z = self.layer(x, z)
        return z