import torch
import torch.nn as nn
import torch.nn.functional as F
class Gem_heat(nn.Module):
    def __init__(self, dim = 768, p=3, eps=1e-6):
        super(Gem_heat, self).__init__()
        self.p = nn.Parameter(torch.ones(dim) * p)  # initial p
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3):
        p = F.softmax(p).unsqueeze(-1)
        x = torch.matmul(x,p)
        x = x.view(x.size(0), x.size(1))
        return x
