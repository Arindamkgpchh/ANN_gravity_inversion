import torch
import torch.nn as nn


class InvNet(nn.Module):
def __init__(self, n_in, n_out, width, depth, dropout, rho_min, rho_max):
super().__init__()
layers = []
last = n_in
for _ in range(depth):
layers += [nn.Linear(last, width), nn.ReLU(), nn.Dropout(dropout)]
last = width
self.body = nn.Sequential(*layers)
self.head = nn.Linear(last, n_out)
self.rho_min = rho_min
self.rho_max = rho_max


def forward(self, g):
z = self.body(g)
z = torch.tanh(self.head(z))
r = 0.5*(z+1)*(self.rho_max-self.rho_min)+self.rho_min
return r