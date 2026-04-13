"""
Grounded World Model (GWM) - Flow Matching based trajectory embedding prediction.

Architecture:
    Condition: [ImageTokens(540, dim), ActionTokens(60, dim)] → KV cache
    Flow Matching: Denoise x0 ~ N(0,1) → x1 (1, 4096) trajectory embedding
    Inference: Euler ODE solver over cached condition
"""

import torch
import torch.nn as nn

from gwm_wiser.models.transformer import Transformer, TransformerConfig


class GroundedWorldModel(nn.Module):
    """
    Transformer-based Grounded World Model that predicts future trajectory embeddings.

    Architecture:
        Input: (B, 1620, 4096)
        Process: Input -> Projection (dim) -> Transformer -> Projection (4096)
        Output: (B, 1620, 4096)
    """

    def __init__(
        self,
        config: TransformerConfig,
        image_embed_dim: int = 4096,
        image_seq_len: int = 1620,
        output_dim: int = 4096,
    ):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.output_dim = output_dim
        self.image_seq_len = image_seq_len

        # ---- Input projection ----
        # Project input from 4096 → dim
        self.input_proj = nn.Linear(image_embed_dim, config.dim)

        # ---- Transformer backbone ----
        self.backbone = Transformer(config)

        # ---- Output projection ----
        # Project output from dim → 4096
        self.output_proj = nn.Linear(config.dim, output_dim)

        # ---- Pre-create position buffers ----
        # 2D positions: (4, 405) flattened to 1620
        # Dim 1: length 4, Dim 2: length 405
        d1 = torch.arange(4)
        d2 = torch.arange(405)
        # indexing='ij' means grid[0] changes slowly, grid[1] changes fast
        # Resulting grid of shape (4, 405, 2)
        grid = torch.stack(torch.meshgrid(d1, d2, indexing="ij"), dim=-1)
        self.register_buffer("positions", grid.reshape(-1, 2))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in [self.input_proj, self.output_proj]:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, 1620, 4096)

        Returns:
            (B, 1620, 4096)
        """
        bs, seq_len, _ = x.shape

        # Project input: (B, 1620, 4096) → (B, 1620, dim)
        tokens = self.input_proj(x)

        # Position IDs
        # If input sequence length is different from initialized, generate on the fly or slice
        # Assuming fixed size for now based on init, but handle flexibility if needed
        assert seq_len == self.image_seq_len
        positions = self.positions.unsqueeze(0).expand(bs, -1, -1)

        # Forward through transformer
        # Transformer forward signature: token_embeddings, positions, kv_cache=None, mask=None
        output, _ = self.backbone(tokens, positions)

        # Project output: (B, 1620, dim) → (B, 1620, 4096)
        output = self.output_proj(output)

        return output


class ActionConditionedGWM(nn.Module):
    """
    Action-Conditioned Grounded World Model.

    Takes a single-frame embedding from Qwen encoder and an action sequence,
    tokenizes the actions, pads the sequence with learnable tokens to a fixed
    length, and predicts the future trajectory embedding.

    Architecture:
        Input: image_emb (B, N_img, 4096) + actions (B, 60, 8)
        Process:
            1. Project image: (B, N_img, 4096) → (B, N_img, dim)
            2. Tokenize actions: (B, 60, 8) → (B, 60, dim)
            3. Learnable padding: (B, pad_len, dim) where pad_len = target_seq_len - N_img - 60
            4. Concatenate → (B, target_seq_len, dim)
            5. Transformer → (B, target_seq_len, dim)
            6. Project output → (B, target_seq_len, 4096)
        Output: (B, 1620, 4096)
    """

    def __init__(
        self,
        config: TransformerConfig,
        image_embed_dim: int = 4096,
        action_dim: int = 8,
        action_seq_len: int = 60,
        target_seq_len: int = 1620,
        output_dim: int = 4096,
    ):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.output_dim = output_dim
        self.action_dim = action_dim
        self.action_seq_len = action_seq_len
        self.target_seq_len = target_seq_len

        # ---- Input projection (image embedding) ----
        self.input_proj = nn.Linear(image_embed_dim, config.dim)

        # ---- Action tokenizer ----
        self.action_proj = nn.Linear(action_dim, config.dim)

        # ---- Learnable padding tokens ----
        # Max possible padding: target_seq_len - action_seq_len (when N_img = 0)
        max_pad_len = target_seq_len - action_seq_len
        self.pad_tokens = nn.Parameter(torch.randn(max_pad_len, config.dim) * 0.02)

        # ---- Transformer backbone ----
        self.backbone = Transformer(config)

        # ---- Output projection ----
        self.output_proj = nn.Linear(config.dim, output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in [self.input_proj, self.action_proj, self.output_proj]:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        image_emb: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            image_emb: (B, N_img, 4096) — Variable-length image embedding from Qwen encoder
            actions: (B, 60, 8) — Action sequence

        Returns:
            (B, 1620, 4096) — Predicted trajectory embedding
        """
        bs, n_img, _ = image_emb.shape

        # 1. Project image embedding: (B, N_img, 4096) → (B, N_img, dim)
        img_tokens = self.input_proj(image_emb)

        # 2. Tokenize actions: (B, 60, 8) → (B, 60, dim)
        act_tokens = self.action_proj(actions)

        # 3. Compute padding length and expand learnable padding
        pad_len = self.target_seq_len - n_img - self.action_seq_len
        assert pad_len >= 0, (
            f"Image tokens ({n_img}) + action tokens ({self.action_seq_len}) "
            f"exceed target_seq_len ({self.target_seq_len})"
        )
        pad = self.pad_tokens[:pad_len].unsqueeze(0).expand(bs, -1, -1)

        # 4. Concatenate: [image_tokens, action_tokens, pad_tokens] → (B, 1620, dim)
        tokens = torch.cat([img_tokens, act_tokens, pad], dim=1)
        assert tokens.shape[1] == self.target_seq_len

        # 5. Generate 1D position IDs for the concatenated sequence
        # Use 2D format required by Transformer's RotaryEmbedding: (B, seq, 2)
        pos_1d = torch.arange(self.target_seq_len, device=tokens.device)
        positions = (
            torch.stack(
                [
                    torch.zeros_like(pos_1d),  # dim 0: all zeros (flat sequence)
                    pos_1d,  # dim 1: sequential position
                ],
                dim=-1,
            )
            .unsqueeze(0)
            .expand(bs, -1, -1)
        )

        # 6. Forward through transformer
        output, _ = self.backbone(tokens, positions)

        # 7. Project output: (B, 1620, dim) → (B, 1620, 4096)
        output = self.output_proj(output)

        return output
