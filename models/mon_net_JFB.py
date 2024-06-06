import torch
import time
from models.base_mon_net import MonLayer, BaseMonNet

class MonNetJFB(BaseMonNet):
    """ Monotone network trained using Jacobian free backpropagation (JFB). """

    def __init__(self, in_dim, out_dim, max_iter, tol, m):
        super().__init__(in_dim, out_dim, max_iter, tol, m)
    
    def name(self):
        return 'MonNetJFB'

    def forward_train(self, x, z=None):
        with torch.no_grad():
            for _ in range(self.max_iter - 1):
                z = self.layer(x, z)
            
        z = self.layer(x, z)
        return z