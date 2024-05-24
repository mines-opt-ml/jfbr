import torch
import time
from models.base_mon_net import MonLayer, BaseMonNet

# training approach inspired by CSBO paper https://arxiv.org/abs/2310.18535
class MonNetJFBR(BaseMonNet):
    """ Monotone network trained using JFB and random selection of number of iterations. """

    def __init__(self, in_dim, out_dim, m=1.0, max_iter=100, tol=1e-6, decay=0.5):
        super().__init__(in_dim, out_dim, m, max_iter, tol)

        # Initialize probability vector for k, the number of iterations during training
        self.p = torch.tensor([decay**k for k in range(max_iter + 1)]) # geometric decay of probability
        self.p[0] = 0 # k=0 is not allowed
        self.p = self.p / torch.sum(self.p) # normalize
    
    def name(self):
        return 'MonNetJFBR'

    def forward(self, x, z=None):
        z = torch.zeros(self.out_dim) if z is None else z
        
        # Training
        if self.training:
            k = torch.multinomial(self.p, 1).item()
            
            with torch.no_grad():
                for _ in range(k - 1):
                    z = self.mon_layer(x, z)
                
            z = self.mon_layer(x, z)
            return z

        # Evaluation
        else:
            for _ in range(self.max_iter):
                z_new = self.mon_layer(x, z)
                if torch.norm(z_new - z, p=2) < self.tol:
                    z = z_new
                    break
                z = z_new
            return z
    
    def train_step(self, X_batch, Y_batch):
        z = self.forward(X_batch)
        
        loss = self.criterion(z, Y_batch)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()