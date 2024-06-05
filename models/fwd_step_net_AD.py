import torch
import time
from models.base_fwd_step_net import MonLipLayer, BaseFwdStepNet

class FwdStepNetAD(BaseFwdStepNet):
    """ Forward step network trained via automatic differentation (AD). """

    def __init__(self, in_dim, out_dim, m=1.0, max_iter=100, tol=1e-6):
        super().__init__(in_dim, out_dim, max_iter=100, tol=1e-6, m0=0.5, L=1.0)
    
    def name(self):
        return 'FwdStepNetAD'

    def forward_train(self, x, z):
        alpha = self.m0 * self.L / self.layer.C_norm_approx
        for _ in range(self.max_iter):
            z = z - alpha * self.layer(x, z)
        return z

