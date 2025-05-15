import torch.nn as nn
import einops

from .single import Attention
from .rope import RotaryEmbedding


class MultiHeadedAttentionRoPE(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        if d_model % h != 0:
            raise ValueError("model size should be divisible by num heads")

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        if self.d_k % 2 != 0:
            raise ValueError("Head dim should be divisible by 2")

        self.rope = RotaryEmbedding(dim=self.d_k // 2)

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, get_attn: bool = False):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        q, k, v = [einops.rearrange(layer(x), 'b n (h d) -> b h n d', h=self.h) for layer, x in zip(self.linear_layers, (query, key, value))]
        q = self.rope.rotate_queries_or_keys(q)
        k = self.rope.rotate_queries_or_keys(k)

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(q, k, v, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        if get_attn:
            return x, attn

        return self.output_linear(x)


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        if d_model % h != 0:
            raise ValueError("model size should be divisible by num heads")

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, get_attn: bool = False):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        q, k, v = [einops.rearrange(layer(x), 'b n (h d) -> b h n d', h=self.h) for layer, x in zip(self.linear_layers, (query, key, value))]

        if get_attn:
            return self.attention(query, key, value)

        # 2) Apply attention on all the projected vectors in batch.
        x = nn.functional.scaled_dot_product_attention(q, k, v)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
