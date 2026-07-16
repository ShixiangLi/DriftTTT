"""TTT and standard temporal Transformers for C-MAPSS RUL regression."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from .ttt_layer import TTTLayer


def _validate_mask(x: Tensor, padding_mask: Tensor | None) -> None:
    if padding_mask is None:
        return
    if padding_mask.ndim != 2 or padding_mask.shape != x.shape[:2]:
        raise ValueError(
            "padding_mask must have shape [batch, length] matching x; "
            f"got {tuple(padding_mask.shape)} for x {tuple(x.shape)}"
        )
    if padding_mask.dtype is not torch.bool:
        raise TypeError(
            f"padding_mask must have dtype torch.bool, got {padding_mask.dtype}"
        )
    if padding_mask.device != x.device:
        raise ValueError("padding_mask and x must be on the same device")


def _zero_padding(x: Tensor, padding_mask: Tensor | None) -> Tensor:
    if padding_mask is None:
        return x
    return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class ConvPositionalEncoding(nn.Module):
    """Depth-wise conditional positional encoding along the time axis."""

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}"
            )
        self.dim = dim
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )

    def forward(
        self, x: Tensor, padding_mask: Tensor | None = None
    ) -> Tensor:
        _validate_mask(x, padding_mask)
        x = _zero_padding(x, padding_mask)
        position_features = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return _zero_padding(x + position_features, padding_mask)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class TTTTransformerBlock(nn.Module):
    """CPE followed by pre-norm TTT and feed-forward residual branches."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
        qkv_bias: bool = True,
        inner_lr: float = 1.0,
        inner_scale: float = 1.0 / 3.0,
        cpe_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if ffn_ratio <= 0:
            raise ValueError(f"ffn_ratio must be positive, got {ffn_ratio}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        hidden_dim = max(1, int(dim * ffn_ratio))
        self.dim = dim
        self.cpe = ConvPositionalEncoding(dim, cpe_kernel_size)
        self.norm1 = nn.LayerNorm(dim)
        self.ttt = TTTLayer(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            inner_lr=inner_lr,
            inner_scale=inner_scale,
        )
        self.ttt_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim, dropout)

    def forward(
        self, x: Tensor, padding_mask: Tensor | None = None
    ) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(
                f"expected x with shape [batch, length, {self.dim}], "
                f"got {tuple(x.shape)}"
            )
        _validate_mask(x, padding_mask)
        x = self.cpe(x, padding_mask)
        x = x + self.ttt_dropout(self.ttt(self.norm1(x), padding_mask))
        x = _zero_padding(x, padding_mask)
        x = x + self.ffn(self.norm2(x))
        return _zero_padding(x, padding_mask)

    def reset_ttt_state(self) -> None:
        self.ttt.reset_ttt_state()


class SinusoidalPositionalEncoding(nn.Module):
    """Dynamic sinusoidal positions that ignore left or right padding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        frequencies = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / dim)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(
        self, x: Tensor, padding_mask: Tensor | None = None
    ) -> Tensor:
        _validate_mask(x, padding_mask)
        batch_size, length, _ = x.shape
        if padding_mask is None:
            positions = torch.arange(length, device=x.device).expand(batch_size, -1)
        else:
            positions = (~padding_mask).long().cumsum(dim=1) - 1
            positions = positions.clamp_min(0)

        angles = positions.to(self.frequencies.dtype).unsqueeze(-1) * self.frequencies
        encoding = torch.zeros_like(x, dtype=self.frequencies.dtype)
        encoding[..., 0::2] = torch.sin(angles)
        encoding[..., 1::2] = torch.cos(angles[..., : encoding[..., 1::2].shape[-1]])
        return _zero_padding(x + encoding.to(dtype=x.dtype), padding_mask)


class StandardTransformerBlock(nn.Module):
    """Pre-norm standard multi-head self-attention Transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if ffn_ratio <= 0:
            raise ValueError(f"ffn_ratio must be positive, got {ffn_ratio}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.dim = dim
        self.layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=max(1, int(dim * ffn_ratio)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(
        self, x: Tensor, padding_mask: Tensor | None = None
    ) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(
                f"expected x with shape [batch, length, {self.dim}], "
                f"got {tuple(x.shape)}"
            )
        _validate_mask(x, padding_mask)
        x = self.layer(x, src_key_padding_mask=padding_mask)
        return _zero_padding(x, padding_mask)


def _validate_model_dimensions(
    input_dim: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
) -> None:
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    if d_model <= 0:
        raise ValueError(f"d_model must be positive, got {d_model}")
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if d_model % num_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )


class _RULTransformerBase(nn.Module):
    """Shared projection, encoding loop, pooling, and regression head."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        blocks: list[nn.Module],
        config: dict[str, Any],
        position_encoding: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_encoding = position_encoding
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(d_model)
        self.regression_head = nn.Linear(d_model, 1)
        self._config = config

    def forward(
        self, x: Tensor, padding_mask: Tensor | None = None
    ) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [batch, length, features], got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected {self.input_dim} input features, got {x.shape[-1]}"
            )
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("batch size and sequence length must be greater than zero")
        if not x.is_floating_point():
            raise TypeError(f"x must be floating point, got {x.dtype}")
        _validate_mask(x, padding_mask)
        if padding_mask is not None and padding_mask.all(dim=1).any():
            raise ValueError("each sequence must contain at least one valid token")

        hidden = _zero_padding(self.input_projection(x), padding_mask)
        if self.position_encoding is not None:
            hidden = self.position_encoding(hidden, padding_mask)
        for block in self.blocks:
            hidden = block(hidden, padding_mask)
        hidden = _zero_padding(self.final_norm(hidden), padding_mask)

        if padding_mask is None:
            pooled = hidden[:, -1]
        else:
            positions = torch.arange(
                hidden.shape[1], device=hidden.device
            ).expand(hidden.shape[0], -1)
            last_valid = positions.masked_fill(padding_mask, -1).amax(dim=1)
            pooled = hidden[
                torch.arange(hidden.shape[0], device=hidden.device), last_valid
            ]
        return self.regression_head(pooled).squeeze(-1)

    def reset_ttt_state(self) -> None:
        """Reset stateful sequence blocks; standard attention has no state."""
        for block in self.blocks:
            reset = getattr(block, "reset_ttt_state", None)
            if callable(reset):
                reset()

    def get_config(self) -> dict[str, Any]:
        """Return constructor arguments suitable for a checkpoint."""
        return dict(self._config)


class TTTRULTransformer(_RULTransformerBase):
    """Temporal TTT Transformer that predicts one RUL value per window.

    ``padding_mask`` follows the PyTorch convention: ``True`` means padding.
    The regression head consumes the last non-padding token, which supports
    unpadded, left-padded, and right-padded windows without changing labels.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
        qkv_bias: bool = True,
        inner_lr: float = 1.0,
        inner_scale: float = 1.0 / 3.0,
        cpe_kernel_size: int = 3,
    ) -> None:
        _validate_model_dimensions(input_dim, d_model, num_layers, num_heads)
        config: dict[str, Any] = {
            "input_dim": input_dim,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ffn_ratio": float(ffn_ratio),
            "dropout": float(dropout),
            "qkv_bias": bool(qkv_bias),
            "inner_lr": float(inner_lr),
            "inner_scale": float(inner_scale),
            "cpe_kernel_size": cpe_kernel_size,
        }
        blocks = [
            TTTTransformerBlock(
                dim=d_model,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
                qkv_bias=qkv_bias,
                inner_lr=inner_lr,
                inner_scale=inner_scale,
                cpe_kernel_size=cpe_kernel_size,
            )
            for _ in range(num_layers)
        ]
        super().__init__(input_dim, d_model, blocks, config)


class StandardRULTransformer(_RULTransformerBase):
    """Standard self-attention Transformer for the same RUL pipeline."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        _validate_model_dimensions(input_dim, d_model, num_layers, num_heads)
        config: dict[str, Any] = {
            "input_dim": input_dim,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ffn_ratio": float(ffn_ratio),
            "dropout": float(dropout),
        }
        blocks = [
            StandardTransformerBlock(
                dim=d_model,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ]
        super().__init__(
            input_dim,
            d_model,
            blocks,
            config,
            position_encoding=SinusoidalPositionalEncoding(d_model),
        )


RULTransformer = TTTRULTransformer
TTTBlock = TTTTransformerBlock


__all__ = [
    "ConvPositionalEncoding",
    "RULTransformer",
    "SinusoidalPositionalEncoding",
    "StandardRULTransformer",
    "StandardTransformerBlock",
    "TTTBlock",
    "TTTRULTransformer",
    "TTTTransformerBlock",
]
