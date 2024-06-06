import torch
import time
from models.base_mon_net import MonLayer, BaseMonNet

# training approach inspired by CSBO paper https://arxiv.org/abs/2310.18535
class MonNetJFBCSBO(BaseMonNet):
    """ Monotone network trained using JFB and gradient update formula from CSBO. """

    def __init__(self, in_dim, out_dim, max_iter, tol, m, decay=0.5):
        super().__init__(in_dim, out_dim, max_iter, tol, m)

        # Initialize propability vector for k, the number of iterations during training
        self.p = torch.tensor([decay**k for k in range(max_iter + 1)]) # geometric decay of probability
        self.p[0] = 0 # k=0 is not allowed
        self.p = self.p / torch.sum(self.p) # normalize
    
    def name(self):
        return 'MonNetJFBCSBO'

    def forward(self, x, z=None):
        k = torch.multinomial(self.p, 1).item()
        
        # Compute z_1 
        z = self.layer(x, z)
        z_1 = z.clone()

        # Compute z_k
        if k > 1:
            with torch.no_grad():
                for _ in range(k - 2):
                    z = self.layer(x, z)
            z = self.layer(x, z)
        z_k = z.clone()

        # Compute z_{k+1}
        z.detach()
        z = self.layer(x, z)
        z_k_1 = z.clone()

        return z_1, z_k, z_k_1, k
    
    def train_step(self, X_batch, Y_batch):
        z_1, z_k, z_k_1, k = self.forward(X_batch)
        
        loss_1 = self.criterion(z_1, Y_batch)
        loss_k = self.criterion(z_k, Y_batch)
        loss_k_1 = self.criterion(z_k_1, Y_batch)
        loss = loss_1 + 1/self.p[k] * (loss_k_1 - loss_k)
        #print(f'k: {k}, loss: {loss.item()}')
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()