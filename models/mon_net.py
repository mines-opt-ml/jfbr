import torch
import torch.nn as nn
import torch.nn.functional as F

# Based on https://github.com/locuslab/monotone_op_net/blob/master/mon.py

class MonLayer(torch.nn.Module):
    """ Single monotone layer that computes
            z_(k+1) = ReLU(W z_k + U x + b)
        where 
            W = (1-m)I - A^T A + B - B^T.
    """
    def __init__(self, in_dim, out_dim, m=1.0):
        super().__init__()
        self.U = nn.Linear(in_dim, out_dim)
        self.A = nn.Linear(out_dim, out_dim, bias=False)
        self.B = nn.Linear(out_dim, out_dim, bias=False)
        self.m = m
    
    def name(self):
        return 'MonNet'

    def forward(self, x, z):
        return F.relu(self.multiply(z) + self.U(x))

    def multiply(self, z):
        ATAz = self.A(z) @ self.A.weight 
        return (1 - self.m) * z - ATAz + self.B(z) - z @ self.B.weight
    
class MonNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, m=1.0):
        super().__init__()
        self.layer = MonLayer(in_dim, out_dim, m)
    
    def name(self):
        return 'MonNet'

    def forward(self, x, z):
        return self.layer(x, z)