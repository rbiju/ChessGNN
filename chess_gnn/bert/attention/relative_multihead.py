import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_relative_position = 64

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Relative positional embeddings for keys and values
        vocab_size = 2 * self.max_relative_position + 1
        self.rel_k_emb = nn.Parameter(torch.rand(vocab_size, self.head_dim))
        self.rel_v_emb = nn.Parameter(torch.rand(vocab_size, self.head_dim))
        nn.init.trunc_normal_(self.rel_k_emb, std=0.02)
        nn.init.trunc_normal_(self.rel_v_emb, std=0.02)

        self.distance_mat = self._relative_positions().to(self.rel_k_emb.device)

    def _relative_positions(self):
        # Create matrix of relative positions [L, L]
        range_vec = torch.arange(self.max_relative_position + 1)
        dist_mat = range_vec[None, :] - range_vec[:, None]  # shape: [L, L]
        final_mat = dist_mat + self.max_relative_position  # make all positive indices
        return final_mat

    def forward(self, q, k, v):
        B, L, _ = q.size()

        # Project input
        Q = self.q_proj(q).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, L, d]
        K = self.k_proj(k).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, L, d]
        V = self.v_proj(v).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, L, d]

        # Content-based attention scores
        scores_content = torch.matmul(Q, K.transpose(-2, -1))  # [B, h, L, L]

        # Relative position embeddings
        # rel_positions = self._relative_positions(L).to(q.device)  # [L, L]
        rel_K = self.rel_k_emb[self.distance_mat]  # [L, L, d]

        # Expand Q for relative position scores
        Q_ = Q.permute(2, 0, 1, 3)  # [L, B, h, d]
        rel_scores = torch.einsum("lbhd,lrd->bhlr", Q_, rel_K)  # [B, h, L, L]

        scores = scores_content + rel_scores  # combine content + relative scores
        scores = scores / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)

        # Compute attention output
        out_content = torch.matmul(attn, V)  # [B, h, L, d]

        # Add relative positional values
        rel_V = self.rel_v_emb[self.distance_mat]  # [L, L, d]
        rel_out = torch.einsum("bhlr,lrd->lbhd", attn, rel_V).permute(1, 2, 0, 3)  # [B, h, L, d]

        out = out_content + rel_out
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)  # [B, L, d_model]
        return self.out_proj(out)
