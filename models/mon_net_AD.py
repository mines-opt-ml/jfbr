import torch
from models.base_mon_net import MonLayer, BaseMonNet

class MonNetAD(BaseMonNet):
    """ Monotone network trained using standard automatic differentiation (AD). """

    def __init__(self, in_dim, out_dim, m=1.0, max_iter=100, tol=1e-6):
        #TODO: find theoretically motivated choice for tolerance given guaranteed convergence
        super().__init__(in_dim, out_dim, m, max_iter, tol)
    
    def name(self):
        return 'MonNetAD'

    def forward(self, x, z=None):
        iters = 0
        if z is None:
            z = torch.zeros(self.out_dim) # TODO: add customizable (random) latent initialization

        # Apply monotone layer until maximum iterations or convergence
        while iters < self.max_iter:
            z_new = self.mon_layer(x, z)
            if torch.norm(z_new - z, p=2) < self.tol:
                z = z_new
                break

            z = z_new
            iters += 1
        return z
    
    def train_step(self, X_batch, Y_batch):
        self.optimizer.zero_grad()
        Y_hat = self.forward(X_batch)
        loss = self.criterion(Y_hat, Y_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()