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
