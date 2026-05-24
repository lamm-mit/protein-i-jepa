from __future__ import annotations

import copy

import torch
from torch import nn

from protein_jepa.masking import make_context_inputs


class ProteinTransformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        max_length: int,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, length = input_ids.shape
        if length > self.max_length:
            raise ValueError(f"Sequence length {length} exceeds max_length={self.max_length}.")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch_size, length)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        padding_mask = ~attention_mask.bool()
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        return self.norm(hidden)


class LatentPredictor(nn.Module):
    def __init__(self, *, embed_dim: int, max_length: int, hidden_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 2
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, context_targets: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
        position_features = self.position_embedding(target_positions)
        return self.net(torch.cat([context_targets, position_features], dim=-1))


class ProteinJEPA(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        max_length: int,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pad_id: int = 0,
        mask_id: int = 1,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.context_encoder = ProteinTransformerEncoder(
            vocab_size=vocab_size,
            max_length=max_length,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            pad_id=pad_id,
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.predictor = LatentPredictor(embed_dim=embed_dim, max_length=max_length, dropout=dropout)
        self.set_target_requires_grad(False)

    def set_target_requires_grad(self, requires_grad: bool) -> None:
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = requires_grad

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.shape != target_mask.shape:
            raise ValueError("input_ids and target_mask must have the same shape.")

        context_input_ids = make_context_inputs(input_ids, target_mask, mask_id=self.mask_id)
        context_states = self.context_encoder(context_input_ids, attention_mask)

        with torch.no_grad():
            target_states = self.target_encoder(input_ids, attention_mask)

        batch_size, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch_size, length)
        context_targets = context_states[target_mask]
        target_positions = positions[target_mask]
        target_latents = target_states[target_mask]
        predicted_latents = self.predictor(context_targets, target_positions)
        return predicted_latents, target_latents

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("EMA momentum must be between 0 and 1.")
        for target_parameter, context_parameter in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
            strict=True,
        ):
            target_parameter.data.mul_(momentum).add_(context_parameter.data, alpha=1.0 - momentum)
