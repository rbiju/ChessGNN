import torch.nn as nn


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, dropout, norm: nn.Module):
        super(SublayerConnection, self).__init__()
        self.norm = norm
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, get_attn: bool = False):
        if get_attn:
            x_, attn = sublayer(self.norm(x))
            x = x + self.dropout(x_)
            return x, attn
        else:
            return x + self.dropout(sublayer(self.norm(x)))
