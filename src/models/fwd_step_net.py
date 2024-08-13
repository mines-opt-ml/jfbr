import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from abc import ABC, abstractmethod

from src.models.training_methods import TrainAD, TrainJFB, TrainJFBR, TrainCSBO
from src.models.base_net import BaseLayer, BaseNet
from src.utils.config import default_config

class FwdStepLayer(BaseLayer):
    """ Layer function (I - alpha F) where 
            F(z) = C z + Ux + b,
        and
            C = mI + A^T A + B - B^T.
        With this parametrization, F is guaranteed to be m-strongly monotone.
        Also, using spectral normalization we approximately 
        force |A| = |B| = 1 so that F is (m+3)-Lipschitz.
        Hence (I - alpha F) is contractive if alpha < 2m/L^2 = 2m/(m+3)^2.
    """
    
    def __init__(self, config=default_config):
        super().__init__(config)
        self.A = spectral_norm(self.A)
        self.B = spectral_norm(self.B)
        self.m = config['m']
        self.L = config['m'] + 3
        self.alpha = self.m / self.L**2

    def name(self):
        return 'FwdStepLayer'

    def forward(self, x, z):
        """ Iteration of contractive layer function (I - alpha F)."""
        return z - self.alpha * (self.C(z) + self.U(x)) 
    
    def C(self, z):
        ATAz = self.A(z) @ self.A.weight 
        Cz = self.m * z + ATAz + self.B(z) - z @ self.B.weight
        return Cz

class BaseFwdStepNet(BaseNet):
    """ Base class for forward step networks. """

    def __init__(self, config=default_config):  
        super().__init__(config)
        self.layer = FwdStepLayer(config)
        self.alpha = self.layer.m / (self.layer.m + 3)**2

class FwdStepNetAD(TrainAD, BaseFwdStepNet):
    """ Forward step network trained via automatic differentation (AD). """

    def __init__(self, config=default_config):
        super().__init__(config)
    
    def name(self):
        return 'FwdStepNetAD'

class FwdStepNetJFB(TrainJFB, BaseFwdStepNet):
    """ Forward step network trained via Jacobian-free backpropagation (JFB). """

    def __init__(self, config=default_config):
        super().__init__(config)
    
    def name(self):
        return 'FwdStepNetJFB'

class FwdStepNetJFBR(TrainJFBR, BaseFwdStepNet):
    """ Forward step network trained via JFB with random number of iterations.  """

    def __init__(self, config=default_config):
        super().__init__(config)

        # Initialize probability vector for k, the number of iterations during training
        self.p = torch.tensor([config['decay']**k for k in range(config['max_iter'] + 1)]) # geometric decay of probability
        self.p[0] = 0 # k=0 is not allowed
        self.p = self.p / torch.sum(self.p) # normalize
    
    def name(self):
        return 'FwdStepNetJFBR'
    
class FwdStepNetCSBO(TrainCSBO, BaseFwdStepNet):
    """ Forward step network trained via method inspired by CSBO paper.  """

    def __init__(self, config=default_config):
        super().__init__(config)

        # Initialize probability vector for k, the number of iterations during training
        self.p = torch.tensor([config['decay']**k for k in range(config['max_iter'] + 1)]) # geometric decay of probability
        self.p[0] = 0 # k=0 is not allowed
        self.p = self.p / torch.sum(self.p) # normalize
    
    def name(self):
        return 'FwdStepNetCSBO'