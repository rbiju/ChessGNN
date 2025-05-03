import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.ffn(x)


class Mlp(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.GELU(),
                                 nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.mlp(x)
