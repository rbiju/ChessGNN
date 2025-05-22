import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        p_attn = F.softmax(scores, dim=-1)

        return torch.matmul(p_attn, value), p_attn
