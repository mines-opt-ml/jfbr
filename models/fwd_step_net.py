import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from abc import ABC, abstractmethod
from models.base_net import BaseLayer, BaseNet
from utils.config import default_config

class MonLipLayer(BaseLayer):
    """ Layer function 
            F(z) = C z + Ux + b,
        where
            C = mI + A^T A + B - B^T.
        With this parametrization, F is guaranteed to be m-strongly monotone.
        Also, using using spectral normalization we approximately 
        force |A| = |B| = 1 so that F is (m+3)-Lipschitz.
        Hence (I-alpha F) is contractive if alpha < 2m/L^2 = 2m/(m+3)^2.
    """
    
    def __init__(self, config=default_config):
        super().__init__(config)
        self.A = spectral_norm(self.A)
        self.B = spectral_norm(self.B)
        self.m = config['m']
        self.L = config['m'] + 3
        self.alpha = self.m / self.L**2

    def name(self):
        return 'MonLipLayer'

    def forward(self, x, z): 
        return z - self.alpha * (self.C(z) + self.U(x))

    def C(self, z):
        ATAz = self.A(z) @ self.A.weight 
        Cz = self.m * z + ATAz + self.B(z) - z @ self.B.weight
        return Cz

class BaseFwdStepNet(BaseNet, ABC):
    """ Base class for forward step networks. """

    def __init__(self, config=default_config):  
        super().__init__(config)
        self.layer = MonLipLayer(config)
        self.alpha = self.layer.m / (self.layer.m + 3)**2

class FwdStepNetAD(BaseFwdStepNet):
    """ Forward step network trained via automatic differentation (AD). """

    def __init__(self, config=default_config):
        super().__init__(config)
    
    def name(self):
        return 'FwdStepNetAD'

    def forward_train(self, x, z):
        for _ in range(self.max_iter):
            z = self.layer(x, z)
        return z

class FwdStepNetJFB(BaseFwdStepNet):
    """ Forward step network trained via Jacobian-free backpropagation (JFB). """

    def __init__(self, config=default_config):
        super().__init__(config)
    
    def name(self):
        return 'FwdStepNetJFB'

    def forward_train(self, x, z):
        for _ in range(self.max_iter-1):
            with torch.no_grad():
                z = self.layer(x, z)
        z = self.layer(x, z)
        return z
      