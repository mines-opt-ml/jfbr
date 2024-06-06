import torch
import time
from models.base_con_net import ConLayer, BaseConNet

class ConNetAD(BaseConNet):
    """ Monotone network trained using standard automatic differentiation (AD). """

    def __init__(self, in_dim, out_dim, m=1.0, L=0.9, max_iter=30, tol=1e-6):
        super().__init__(in_dim, out_dim, m, L, max_iter, tol)
    
    def name(self):
        return 'ConNetAD'

    def forward(self, x, z=None):
        z = torch.zeros(self.out_dim) if z is None else z
        
        # Train
        if self.training:
            for _ in range(self.max_iter):
                z = self.layer(x, z)
            return z

        # Evaluate
        else:
            for _ in range(self.max_iter):
                z_new = self.layer(x, z)
                if torch.norm(z_new - z, p=2) < self.tol:
                    z = z_new
                    break
                z = z_new
            return z