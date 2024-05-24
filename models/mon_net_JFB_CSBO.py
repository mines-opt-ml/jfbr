import torch
import time
from models.base_mon_net import MonLayer, BaseMonNet

# training approach inspired by CSBO paper https://arxiv.org/abs/2310.18535
class MonNetJFBCSBO(BaseMonNet):
    """ Monotone network trained using JFB and gradient update formula from CSBO. """

    def __init__(self, in_dim, out_dim, m=1.0, max_iter=100, tol=1e-6, decay=0.5):
        super().__init__(in_dim, out_dim, m, max_iter, tol)

        self.p = torch.tensor([decay**k for k in range(max_iter)])
        self.p = self.p / torch.sum(self.p)
    
    def name(self):
        return 'MonNetJFBCSBO'

    def forward(self, x, z=None):
        z = torch.zeros(self.out_dim) if z is None else z
        
        # Training
        if self.training:
            sampled_k = torch.multinomial(self.p, 1).item()
            
            # Compute z_1 
            z = self.mon_layer(x, z)
            z_1 = z.clone()

            # Compute z_k
            with torch.no_grad():
                for _ in range(sampled_k - 2):
                    z = self.mon_layer(x, z).detach() # detach is likely redundant
            z = self.mon_layer(x, z)
            z_k = z.clone()

            # Compute z_{k+1}
            z.detach()
            z = self.mon_layer(x, z)
            z_k_1 = z.clone()

            return z_1, z_k, z_k_1, self.p[sampled_k]

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
        z_1, z_k, z_k_1, p_k = self.forward(X_batch)
        
        loss_1 = self.criterion(z_1, Y_batch)
        loss_k = self.criterion(z_k, Y_batch)
        loss_k_1 = self.criterion(z_k_1, Y_batch)
        loss = loss_1 + 1/p_k * (loss_k_1 - loss_k)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()