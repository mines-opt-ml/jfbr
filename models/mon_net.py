import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from models.base_net import BaseLayer, BaseNet
from utils.config import default_config

# Based on https://github.com/locuslab/monotone_op_net/blob/master/mon.py
class MonLayer(BaseLayer):
    """ Single layer that computes
            z_(k+1) = ReLU(W z_k + U x + b)
        where 
            W = (1-m)I - A^T A + B - B^T.
        The layer is a monotone function composed with ReLU.
        Finding the fixed point of this layer can be done
        with a splitting method.
    """
    def __init__(self, config=default_config):
        super().__init__(config)
        self.m = config['m']

    def name(self):
        return 'MonLayer'

    def forward(self, x, z):
        return F.relu(self.W(z) + self.U(x))

    def W(self, z):
        ATAz = self.A(z) @ self.A.weight 
        Wz = (1 - self.m) * z - ATAz + self.B(z) - z @ self.B.weight
        return Wz

class BaseMonNet(BaseNet, ABC):
    """ Base class for monotone networks, which apply FPI
        to find the fixed point of the monotone layer, unlike
        the paper, which uses splitting methods.
    """

    def __init__(self, config=default_config):
        super().__init__(config)
        self.layer = MonLayer(config)

class MonNetAD(BaseMonNet):
    """ Monotone network trained using automatic differentiation (AD). """

    def __init__(self, config=default_config):
        super().__init__(config)
    
    def name(self):
        return 'MonNetAD'

    def forward_train(self, x, z):
        for _ in range(self.max_iter):
            z = self.layer(x, z)
        return z
    
class MonNetJFB(BaseMonNet):
    """ Monotone network trained using Jacobian free backpropagation (JFB). """

    def __init__(self, config=default_config):
        super().__init__(config)
    
    def name(self):
        return 'MonNetJFB'

    def forward_train(self, x, z):
        with torch.no_grad():
            for _ in range(self.max_iter - 1):
                z = self.layer(x, z)
            
        z = self.layer(x, z)
        return z
    
class MonNetJFBR(BaseMonNet):
    """ Monotone network trained using JFB and random selection of number of iterations. """

    def __init__(self, config=default_config):
        super().__init__(config)

        # Initialize probability vector for k, the number of iterations during training
        self.p = torch.tensor([config['decay']**k for k in range(config['max_iter'] + 1)]) # geometric decay of probability
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
    
# training approach inspired by CSBO paper https://arxiv.org/abs/2310.18535
class MonNetJFBCSBO(BaseMonNet):
    """ Monotone network trained using JFB and gradient update formula from CSBO. """

    def __init__(self, config=default_config):
        super().__init__(config)

        # Initialize propability vector for k, the number of iterations during training
        self.p = torch.tensor([config['decay']**k for k in range(config['max_iter'] + 1)]) # geometric decay of probability
        self.p[0] = 0 # k=0 is not allowed
        self.p = self.p / torch.sum(self.p) # normalize
    
    def name(self):
        return 'MonNetJFBCSBO'

    def forward_train(self, x, z):
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
    