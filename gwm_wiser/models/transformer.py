from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        n_dims: int = 2,
        bases: Tuple[float, float] = (10.0, 10000.0),
    ):
        super().__init__()
        assert head_dim % (2 * n_dims) == 0, (
            f"head_dim {head_dim} must be divisible by 2 * n_dims {2 * n_dims}"
        )
        self.head_dim = head_dim
        self.n_dims = n_dims
        self.bases = bases
        self.dim_per_pos = head_dim // n_dims

        # Calculate and register separate inverse frequencies for each dimension
        for i, base in enumerate(self.bases):
            inv_freq = 1.0 / (
                base
                ** (torch.arange(0, self.dim_per_pos, 2).float() / self.dim_per_pos)
            )
            self.register_buffer(f"inv_freq_{i}", inv_freq)

    def forward(self, positions: torch.Tensor):
        # positions: (B, S, n_dims)
        if self.n_dims == 1 and positions.ndim == 2:
            positions = positions.unsqueeze(-1)

        # Compute freqs for each dimension and concatenate
        # positions[:, :, i] is (B, S), inv_freq is (D_per_pos / 2)
        all_freqs = []
        for i in range(self.n_dims):
            # Fetch the specific inv_freq for this dimension
            inv_freq = getattr(self, f"inv_freq_{i}")
            freqs = torch.einsum("bi,d->bid", positions[:, :, i].float(), inv_freq)
            all_freqs.append(freqs)

        freqs = torch.cat(all_freqs, dim=-1)  # (B, S, head_dim / 2)
        return freqs.cos(), freqs.sin()


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)
    q_rotated = torch.cat((q1 * cos - q2 * sin, q1 * sin + q2 * cos), dim=-1)
    k_rotated = torch.cat((k1 * cos - k2 * sin, k1 * sin + k2 * cos), dim=-1)
    return q_rotated.to(q.dtype), k_rotated.to(k.dtype)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.ffn_dim
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.head_dim
        self.n_head = config.n_head
        self.n_kv_head = (
            config.n_kv_head if config.n_kv_head is not None else config.n_head
        )
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.head_dim * self.n_head, config.dim, bias=False)

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split(
            [self.head_dim * self.n_head, kv_size, kv_size], dim=-1
        )

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)

        # Apply rotary embeddings before transposing and caching
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        keys = xk.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = xv.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        if kv_cache is not None:
            # You only need to rotate the new keys and values. cached keys and values are already rotated.
            cached_keys, cached_values = kv_cache
            keys = torch.cat([cached_keys, keys], dim=2)
            values = torch.cat([cached_values, values], dim=2)

        output = F.scaled_dot_product_attention(
            xq,
            keys,
            values,
            attn_mask=mask,
            is_causal=False,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )

        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(bsz, seqlen, self.head_dim * self.n_head)
        )
        output = self.resid_dropout(self.wo(output))
        return output, (keys, values)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        attn_out, updated_kv_cache = self.attention(
            self.attention_norm(x), cos, sin, kv_cache, mask
        )
        h = x + attn_out
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, updated_kv_cache


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_layer = config.n_layer
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        assert config.head_dim % 2 == 0, "head_dim must be even for RoPE"
        # Default to 1D, but GroundedWorldModel will likely want 2D
        self.rotary_emb = RotaryEmbedding(config.head_dim, n_dims=2)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config))

        # output layer
        self.output_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:

        cos, sin = self.rotary_emb(positions)
        h = self.tok_dropout(token_embeddings)

        # transformer blocks
        updated_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for idx, layer in enumerate(self.layers):
            h, this_layer_kv_cache = layer(
                h, cos, sin, None if kv_cache is None else kv_cache[idx], mask
            )
            updated_kv_caches.append(this_layer_kv_cache)

        # output layers
        output = self.output_norm(h)
        return output, updated_kv_caches


@dataclass
class TransformerConfig:
    """Configuration for the transformer backbone."""

    dim: int = 256
    ffn_dim: int = 1024
    head_dim: int = 32
    n_layer: int = 5
    n_head: int = 8
    n_kv_head: int = 8
    token_dropout_p: float = 0.0
    attn_dropout_p: float = 0.0
    ffn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.0
    norm_eps: float = 1e-5
