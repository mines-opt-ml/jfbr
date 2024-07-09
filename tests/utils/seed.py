import torch
from models.mon_net import MonNetAD
from utils.model import set_seed

set_seed(1)
model_1 = MonNetAD()
for name, param in model_1.named_parameters():
    print(f'Parameter: {name}, Initial value: {param.detach().cpu().numpy()}')

set_seed(2)
model_2 = MonNetAD()
for name, param in model_2.named_parameters():
    print(f'Parameter: {name}, Initial value: {param.detach().cpu().numpy()}')