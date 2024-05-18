import torch
import time
from models.base_mon_net import MonLayer, BaseMonNet

class MonNetJFB(BaseMonNet):
    """ Monotone network trained using standard automatic differentiation (AD). """

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
    
    def train_model(self, train_loader, max_epochs, verbose=False):
        train_epochs = []
        train_losses = []
        start_time = time.time()
        for epoch in range(max_epochs):
            for i, (X_batch, Y_batch) in enumerate(train_loader):
                self.train() 
                z = self.forward(X_batch)
                loss = self.criterion(z, Y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if verbose:
                    self.eval()
                    train_loss = self.criterion(self.forward(X_batch), Y_batch).item()
                    train_epochs.append(epoch + i / len(train_loader))
                    train_losses.append(train_loss)

            print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Time: {time.time() - start_time:.2f} s')

        if verbose:
            return train_epochs, train_losses