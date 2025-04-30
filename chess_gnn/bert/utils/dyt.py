import torch
import torch.nn as nn


class DyTNorm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.a = torch.ones(size)
        self.w = torch.ones(size)
        self.b = torch.zeros(size)

    def forward(self, x):
        x = self.w * torch.nn.functional.tanh(self.a * x) + self.b
        return x
