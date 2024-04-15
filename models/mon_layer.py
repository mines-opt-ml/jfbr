import torch
import torch.nn as nn
import torch.nn.functional as F

# Based on https://github.com/locuslab/monotone_op_net/blob/master/mon.py
class MonNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, m=1.0):
        super().__init__()
        self.U = nn.Linear(in_dim, out_dim)
        self.A = nn.Linear(out_dim, out_dim, bias=False)
        self.B = nn.Linear(out_dim, out_dim, bias=False)
        self.m = m

    def forward(self, x, *z):
        return (self.U(x) + self.multiply(*z)[0],)

    def multiply(self, *z):
        ATAz = self.A(z[0]) @ self.A.weight 
        z_out = (1 - self.m) * z[0] - ATAz + self.B(z[0]) - z[0] @ self.B.weight
        return (z_out,)
    