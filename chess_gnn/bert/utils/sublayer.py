import torch.nn as nn

from .norm_factory import NormFactory


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, dropout, norm_factory: NormFactory):
        super(SublayerConnection, self).__init__()
        self.norm = norm_factory.norm()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
