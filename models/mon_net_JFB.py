import torch
import time
from models.base_mon_net import MonLayer, BaseMonNet

class MonNetJFB(BaseMonNet):
    """ Monotone network trained using Jacobian free backpropagation (JFB). """

    def __init__(self, in_dim, out_dim, m=1.0, max_iter=100, tol=1e-6):
        super().__init__(in_dim, out_dim, m, max_iter, tol)
    
    def name(self):
        return 'MonNetJFB'

    def forward(self, x, z=None):
        z = torch.zeros(self.out_dim) if z is None else z
        
        # Training
        if self.training:
            with torch.no_grad():
                for _ in range(self.max_iter - 1):
                    z = self.layer(x, z)
                
            z = self.layer(x, z)
            return z

        # Evaluation
        else:
            for _ in range(self.max_iter):
                z_new = self.layer(x, z)
                if torch.norm(z_new - z, p=2) < self.tol:
                    z = z_new
                    break
                z = z_new
            return z