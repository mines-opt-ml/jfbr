import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from models.base_net import BaseLayer, BaseNet
from utils.model import approximate_norm
from utils.config import default_config

class MonLipLayer(BaseLayer):
    """ Layer function for BaseFwdStepNet 
            F(z) = (L/||C||) * C z + Ux + b,
        that is m-strongly monotone and L-Lipschitz, where
            C = mI + A^T A + B - B^T
        and
            m = m0 L / ||C||.
    """
    
    def __init__(self, config=default_config):
        super().__init__(config)
        self.m0 = config['m0']
        self.L = config['L']
        self.C_norm_approx = None
        self.m = None

    def name(self):
        return 'MonLipLayer'

    def forward(self, x, z):
        self.C_norm_approx = approximate_norm(self.C, self.out_dim) 
        self.m = (self.m0 * self.L) / self.C_norm_approx
        return (self.L / self.C_norm_approx) * self.C(z) + self.U(x)

    def C(self, z):
        ATAz = self.A(z) @ self.A.weight 
        Cz = self.m0 * z + ATAz + self.B(z) - z @ self.B.weight
        return Cz
    
    def get_alpha(self):
        if self.C_norm_approx is None :
            self.C_norm_approx = approximate_norm(self.C, self.out_dim) 
        alpha = self.m0 / (self.L * self.C_norm_approx)
        return alpha

class BaseFwdStepNet(BaseNet, ABC):
    """ Base class for forward step networks. """

    def __init__(self, config=default_config):  
        super().__init__(config)
        self.layer = MonLipLayer(config)

class FwdStepNetAD(BaseFwdStepNet):
    """ Forward step network trained via automatic differentation (AD). """

    def __init__(self, config=default_config):
        super().__init__(config)
    
    def name(self):
        return 'FwdStepNetAD'

    def forward_train(self, x, z):
        alpha = self.layer.m0 * self.layer.L / self.layer.C_norm_approx
        for _ in range(self.max_iter):
            z = z - alpha * self.layer(x, z)
        return z

class FwdStepNetJFB(BaseFwdStepNet):
    """ Forward step network trained via automatic differentation (AD). """

    def __init__(self, config=default_config):
        super().__init__(config)
    
    def name(self):
        return 'FwdStepNetJFB'

    def forward_train(self, x, z):
        with torch.no_grad():
            for _ in range(self.max_iter - 1):
                z = self.layer(x, z)
        z = self.layer(x, z)
        return z



        