import copy

import torch.nn as nn

from .attention import MultiHeadedAttention, MultiHeadedAttentionRoPE, RelativeMultiHeadAttention
from .utils import SublayerConnection, PositionwiseFeedForward, NormFactory


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, norm_factory: NormFactory, pos_emb_mode: str = "relative", dropout=0.):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.dim = hidden
        self.pos_emb_mode = pos_emb_mode
        if pos_emb_mode == "relative":
            self.attention = RelativeMultiHeadAttention(hidden, attn_heads)
        elif pos_emb_mode == "rope":
            self.attention = MultiHeadedAttentionRoPE(h=attn_heads, d_model=hidden)
        elif pos_emb_mode == "learned":
            self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        else:
            raise NotImplementedError("Only 'relative', 'rope' or 'learned' are supported positional embedding modes")

        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        norm = norm_factory.norm(hidden)
        self.input_sublayer = SublayerConnection(dropout=dropout, norm=copy.deepcopy(norm))
        self.output_sublayer = SublayerConnection(dropout=dropout, norm=copy.deepcopy(norm))

    def forward(self, x, get_attn: bool = False):
        if get_attn:
            x, attn = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, get_attn=get_attn), get_attn=get_attn)
            x = self.output_sublayer(x, self.feed_forward)
            return x, attn
        else:
            x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, get_attn=get_attn))
            x = self.output_sublayer(x, self.feed_forward)
            return x
