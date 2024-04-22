import torch
import torch.nn as nn
import torch.nn.functional as F

# Based on https://github.com/locuslab/monotone_op_net/blob/master/mon.py

class MonLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, m=1.0):
        super().__init__()
        self.U = nn.Linear(in_dim, out_dim)
        self.A = nn.Linear(out_dim, out_dim, bias=False)
        self.B = nn.Linear(out_dim, out_dim, bias=False)
        self.m = m
    
    def name(self):
        return 'MonNet'

    def forward(self, x, z):
        return self.multiply(z) + self.U(x)

    def multiply(self, z):
        ATAz = self.A(z) @ self.A.weight 
        return (1 - self.m) * z - ATAz + self.B(z) - z @ self.B.weight
    
    