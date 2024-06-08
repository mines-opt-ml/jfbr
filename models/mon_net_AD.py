import torch
import time
from models.base_mon_net import MonLayer, BaseMonNet
from utils.config import default_config

class MonNetAD(BaseMonNet):
    """ Monotone network trained using automatic differentiation (AD). """

    def __init__(self, config=default_config):
        super().__init__(config)
    
    def name(self):
        return 'MonNetAD'

    def forward_train(self, x, z=None):
        for _ in range(self.max_iter):
            z = self.layer(x, z)
        return z