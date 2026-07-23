from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from .ttt_layer import TTTMLP, TTTMultiscaleMoE


class SinusoidalPositionEncoding(nn.Module):
    """Sinusoidal positions that correctly restart after left padding."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.d_model = d_model

    def forward(self, valid_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        positions = valid_mask.long().cumsum(dim=1).sub(1).clamp_min(0)
        angles = positions.to(self.frequencies.dtype).unsqueeze(-1) * self.frequencies
        encoding = torch.zeros(
            *positions.shape,
            self.d_model,
            device=positions.device,
            dtype=self.frequencies.dtype,
        )
        encoding[..., 0::2] = torch.sin(angles)
        odd_width = encoding[..., 1::2].shape[-1]
        encoding[..., 1::2] = torch.cos(angles[..., :odd_width])
        return encoding.to(dtype=dtype) * valid_mask.unsqueeze(-1)


class SelfAttentionMixer(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=int(config["d_model"]),
            num_heads=int(config["nhead"]),
            dropout=float(config["dropout"]),
            batch_first=True,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        cycle_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del cycle_ids
        output, _ = self.attention(
            inputs,
            inputs,
            inputs,
            key_padding_mask=~mask,
            need_weights=False,
        )
        return output


def _build_attention(config: dict[str, Any]) -> nn.Module:
    return SelfAttentionMixer(config)


def _build_ttt_mlp(config: dict[str, Any]) -> nn.Module:
    return TTTMLP(int(config["d_model"]), int(config["nhead"]), config["ttt"])


def _build_ttt_multiscale_moe(config: dict[str, Any]) -> nn.Module:
    return TTTMultiscaleMoE(int(config["d_model"]), int(config["nhead"]), config["ttt"])


MIXER_BUILDERS: dict[str, Callable[[dict[str, Any]], nn.Module]] = {
    "attention": _build_attention,
    "ttt_mlp": _build_ttt_mlp,
    "ttt_multiscale_moe": _build_ttt_multiscale_moe,
}


def build_sequence_mixer(config: dict[str, Any]) -> nn.Module:
    name = str(config["sequence_mixer"])
    try:
        builder = MIXER_BUILDERS[name]
    except KeyError as error:
        raise ValueError(f"Unknown sequence mixer: {name}") from error
    return builder(config)


class TransformerBlock(nn.Module):
    """Shared Transformer block whose sequence mixer is configuration-driven."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(config["d_model"])
        hidden_dim = int(config["dim_feedforward"])
        dropout = float(config["dropout"])
        self.norm_first = bool(config["norm_first"])
        self.sequence_mixer = build_sequence_mixer(config)
        self.mixer_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.mixer_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        activation: nn.Module
        if config["activation"] == "gelu":
            activation = nn.GELU()
        else:
            activation = nn.ReLU()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        cycle_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mixer_inputs = self.mixer_norm(inputs) if self.norm_first else inputs
        mixed = self.sequence_mixer(mixer_inputs, mask, cycle_ids)
        if self.norm_first:
            hidden = inputs + self.mixer_dropout(mixed)
            hidden = hidden + self.ffn_dropout(self.feed_forward(self.ffn_norm(hidden)))
        else:
            hidden = self.mixer_norm(inputs + self.mixer_dropout(mixed))
            hidden = self.ffn_norm(hidden + self.ffn_dropout(self.feed_forward(hidden)))
        hidden = hidden * mask.unsqueeze(-1)
        return hidden


class RULTransformer(nn.Module):
    """Unified Transformer RUL backbone with a replaceable sequence mixer."""

    def __init__(self, input_dim: int, config: dict[str, Any]) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = int(config["d_model"])
        self.sequence_mixer = str(config["sequence_mixer"])
        self.input_projection = nn.Linear(input_dim, self.d_model)
        self.position = SinusoidalPositionEncoding(self.d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(int(config["num_layers"]))
        )
        self.final_norm = nn.LayerNorm(self.d_model)
        self.regression_head = nn.Linear(self.d_model, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.regression_head.weight)
        nn.init.zeros_(self.regression_head.bias)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        cycle_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 3 or mask.shape != features.shape[:2]:
            raise ValueError("Expected features [B,L,F] and mask [B,L]")
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, received {features.shape[-1]}"
            )
        if cycle_ids is not None and cycle_ids.shape != features.shape[:2]:
            raise ValueError("Expected cycle_ids [B,L]")
        if cycle_ids is not None and cycle_ids.dtype != torch.int64:
            raise ValueError("cycle_ids must use torch.int64")
        mask = mask.bool()
        if not torch.all(mask.any(dim=1)):
            raise ValueError("Every sequence must contain at least one valid timestep")
        hidden = self.input_projection(features)
        hidden = hidden + self.position(mask, hidden.dtype)
        hidden = hidden * mask.unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden, mask, cycle_ids)
        positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        last_indices = positions.masked_fill(~mask, -1).max(dim=1).values
        pooled = hidden[
            torch.arange(hidden.shape[0], device=hidden.device), last_indices
        ]
        predictions = self.regression_head(self.final_norm(pooled)).squeeze(-1)
        return predictions
