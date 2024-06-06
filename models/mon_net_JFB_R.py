import torch
import time
from models.base_mon_net import MonLayer, BaseMonNet

class MonNetJFBR(BaseMonNet):
    """ Monotone network trained using JFB and random selection of number of iterations. """

    def __init__(self, in_dim, out_dim, max_iter, tol, m, decay=0.5):
        super().__init__(in_dim, out_dim, max_iter, tol, m)

        # Initialize probability vector for k, the number of iterations during training
        self.p = torch.tensor([decay**k for k in range(max_iter + 1)]) # geometric decay of probability
        self.p[0] = 0 # k=0 is not allowed
        self.p = self.p / torch.sum(self.p) # normalize
    
    def name(self):
        return 'MonNetJFBR'

    def forward_train(self, x, z=None):
        k = torch.multinomial(self.p, 1).item()
        
        with torch.no_grad():
            for _ in range(k - 1):
                z = self.layer(x, z)
            
        z = self.layer(x, z)
        return z