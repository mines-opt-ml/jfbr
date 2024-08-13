import torch

class TrainAD:    
    def forward_train(self, x, z):
        for _ in range(self.max_iter):
            z = self.layer(x, z)
        return z
    
class TrainJFB:    
    def forward_train(self, x, z):
        for _ in range(self.max_iter-1):
            with torch.no_grad():
                z = self.layer(x, z)
        z = self.layer(x, z)
        return z
    
class TrainJFBR:
    def forward_train(self, x, z):
        k = torch.multinomial(self.p, 1).item()
        with torch.no_grad():
            for _ in range(k - 1):
                z = self.layer(x, z)
        z = self.layer(x, z)
        return z

# training approach inspired by CSBO paper https://arxiv.org/abs/2310.18535   
class TrainCSBO:
    def forward_train(self, x, z):
        k = torch.multinomial(self.p, 1).item()
        for i in range(1,k+2):
            for _ in range(1,2**i):
                with torch.no_grad():
                    z = self.layer(x, z)
            z = self.layer(x, z)
            if i == 1:
                z_1 = z.clone()
            if i == k:
                z_k = z.clone()
            elif i == k+1:
                z_k_1 = z.clone()

        return z_1, z_k, z_k_1, k
    
    def train_step(self, X_batch, Y_batch):
        z_1, z_k, z_k_1, k = self.forward(X_batch)
        
        loss_1 = self.criterion(z_1, Y_batch)
        loss_k = self.criterion(z_k, Y_batch)
        loss_k_1 = self.criterion(z_k_1, Y_batch)
        loss = loss_1 + 1/self.p[k] * (loss_k_1 - loss_k)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()