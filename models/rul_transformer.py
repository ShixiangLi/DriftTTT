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

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        causal: bool = False,
        continuous_state: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}"
            )
        self.dim = dim
        self.kernel_size = kernel_size
        self.causal = bool(causal)
        self.continuous_state = bool(continuous_state)
        if self.continuous_state and not self.causal:
            raise ValueError("continuous_state requires causal positional encoding")
        self._history: Tensor | None = None
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=0,
            groups=dim,
        )

    def forward(
        self, x: Tensor, padding_mask: Tensor | None = None
    ) -> Tensor:
        _validate_mask(x, padding_mask)
        x = _zero_padding(x, padding_mask)
        temporal = x.transpose(1, 2)
        if self.causal:
            if self.continuous_state:
                if x.shape[0] != 1:
                    raise ValueError(
                        "continuous positional state requires one stream per call"
                    )
                history_size = self.kernel_size - 1
                if self._history is None:
                    history = temporal.new_zeros(
                        1, self.dim, history_size
                    )
                else:
                    history = self._history
                    if history.device != x.device or history.dtype != x.dtype:
                        raise RuntimeError(
                            "positional stream state device/dtype changed; reset "
                            "the model state first"
                        )
                temporal = torch.cat([history, temporal], dim=-1)
                self._history = (
                    temporal[:, :, -history_size:].detach()
                    if history_size
                    else temporal[:, :, :0].detach()
                )
            else:
                temporal = nn.functional.pad(
                    temporal, (self.kernel_size - 1, 0)
                )
        else:
            radius = self.kernel_size // 2
            temporal = nn.functional.pad(temporal, (radius, radius))
        position_features = self.conv(temporal).transpose(1, 2)
        return _zero_padding(x + position_features, padding_mask)

    def reset_state(self) -> None:
        self._history = None


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
        causal: bool = False,
        chunk_size: int = 16,
        continuous_state: bool = False,
    ) -> None:
        super().__init__()
        if ffn_ratio <= 0:
            raise ValueError(f"ffn_ratio must be positive, got {ffn_ratio}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        hidden_dim = max(1, int(dim * ffn_ratio))
        self.dim = dim
        self.causal = bool(causal)
        self.cpe = ConvPositionalEncoding(
            dim,
            cpe_kernel_size,
            causal=self.causal,
            continuous_state=continuous_state,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ttt = TTTLayer(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            inner_lr=inner_lr,
            inner_scale=inner_scale,
            causal=self.causal,
            chunk_size=chunk_size,
            continuous_state=continuous_state,
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
        self.cpe.reset_state()
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
        causal: bool = False,
    ) -> None:
        super().__init__()
        if ffn_ratio <= 0:
            raise ValueError(f"ffn_ratio must be positive, got {ffn_ratio}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.dim = dim
        self.causal = bool(causal)
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
        causal_mask = None
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(
                    x.shape[1], x.shape[1], dtype=torch.bool, device=x.device
                ),
                diagonal=1,
            )
        x = self.layer(
            x,
            src_mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
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


def _validate_architecture(
    architecture: str,
    autoregressive: bool,
    autoregressive_loss: str,
    autoregressive_weight: float,
) -> str:
    normalized = str(architecture).strip().lower()
    if normalized not in {"encoder", "decoder"}:
        raise ValueError("architecture must be 'encoder' or 'decoder'")
    if not isinstance(autoregressive, bool):
        raise TypeError("autoregressive must be bool")
    if autoregressive and normalized != "decoder":
        raise ValueError("autoregressive prediction requires decoder architecture")
    if autoregressive_loss not in {"mse", "smooth_l1"}:
        raise ValueError("autoregressive_loss must be 'mse' or 'smooth_l1'")
    if not math.isfinite(autoregressive_weight) or autoregressive_weight <= 0:
        raise ValueError("autoregressive_weight must be finite and positive")
    return normalized


class _RULTransformerBase(nn.Module):
    """Shared projection, encoding loop, pooling, and regression head."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        blocks: list[nn.Module],
        config: dict[str, Any],
        position_encoding: nn.Module | None = None,
        architecture: str = "encoder",
        autoregressive: bool = False,
        autoregressive_loss: str = "smooth_l1",
        autoregressive_weight: float = 0.2,
        continuous_state: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.architecture = architecture
        self.autoregressive = bool(autoregressive)
        self.autoregressive_loss = autoregressive_loss
        self.autoregressive_weight = float(autoregressive_weight)
        self.continuous_state = bool(continuous_state)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_encoding = position_encoding
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(d_model)
        self.regression_head = nn.Linear(d_model, 1)
        self.feature_head = (
            nn.Linear(d_model, input_dim) if self.autoregressive else None
        )
        self._config = config

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor | None = None,
        *,
        return_sequence_predictions: bool = False,
    ) -> Tensor | dict[str, Tensor]:
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
            last_valid = torch.full(
                (hidden.shape[0],),
                hidden.shape[1] - 1,
                dtype=torch.long,
                device=hidden.device,
            )
        else:
            positions = torch.arange(
                hidden.shape[1], device=hidden.device
            ).expand(hidden.shape[0], -1)
            last_valid = positions.masked_fill(padding_mask, -1).amax(dim=1)
        batch_positions = torch.arange(hidden.shape[0], device=hidden.device)
        pooled = hidden[batch_positions, last_valid]
        sequence_predictions = (
            self.regression_head(hidden).squeeze(-1)
            if return_sequence_predictions
            else None
        )
        prediction = (
            self.regression_head(pooled).squeeze(-1)
            if sequence_predictions is None
            else sequence_predictions[batch_positions, last_valid]
        )
        if self.feature_head is None and sequence_predictions is None:
            return prediction
        output = {
            "prediction": prediction,
        }
        if self.feature_head is not None:
            output["next_features"] = self.feature_head(hidden)
        if sequence_predictions is not None:
            output["sequence_predictions"] = sequence_predictions
        return output

    @torch.no_grad()
    def forecast_features(
        self,
        x: Tensor,
        steps: int,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Autoregressively extend normalized feature sequences.

        Stateless models recompute each prefix. Continuous TTT models consume
        the context once and then feed back one generated token at a time.
        """
        if self.feature_head is None:
            raise RuntimeError("forecast_features requires autoregressive decoder mode")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        if padding_mask is not None and padding_mask[:, -1].any():
            raise ValueError("forecast context cannot end with a padding token")

        if self.continuous_state:
            if x.shape[0] != 1:
                raise ValueError(
                    "continuous-state forecasting supports one engine at a time"
                )
            self.reset_ttt_state()
            output = self(x, padding_mask)
            assert isinstance(output, dict)
            generated: list[Tensor] = []
            for step in range(steps):
                next_value = output["next_features"][:, -1]
                generated.append(next_value)
                if step + 1 < steps:
                    output = self(next_value.unsqueeze(1))
                    assert isinstance(output, dict)
            return torch.stack(generated, dim=1)

        sequence = x
        mask = padding_mask
        generated: list[Tensor] = []
        for _ in range(steps):
            output = self(sequence, mask)
            assert isinstance(output, dict)
            next_value = output["next_features"][:, -1]
            generated.append(next_value)
            sequence = torch.cat([sequence, next_value.unsqueeze(1)], dim=1)
            if mask is not None:
                valid = torch.zeros(
                    mask.shape[0], 1, dtype=torch.bool, device=mask.device
                )
                mask = torch.cat([mask, valid], dim=1)
        return torch.stack(generated, dim=1)

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
        architecture: str = "encoder",
        autoregressive: bool = False,
        autoregressive_loss: str = "smooth_l1",
        autoregressive_weight: float = 0.2,
        chunk_size: int = 16,
        continuous_state: bool = False,
    ) -> None:
        _validate_model_dimensions(input_dim, d_model, num_layers, num_heads)
        architecture = _validate_architecture(
            architecture, autoregressive, autoregressive_loss, autoregressive_weight
        )
        causal = architecture == "decoder"
        if not isinstance(continuous_state, bool):
            raise TypeError("continuous_state must be bool")
        if continuous_state and not causal:
            raise ValueError("continuous_state requires decoder architecture")
        config: dict[str, Any] = {
            "input_dim": input_dim,
            "architecture": architecture,
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
        if autoregressive:
            config.update(
                autoregressive=True,
                autoregressive_loss=autoregressive_loss,
                autoregressive_weight=float(autoregressive_weight),
            )
        if causal:
            config["chunk_size"] = chunk_size
        if continuous_state:
            config["continuous_state"] = True
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
                causal=causal,
                chunk_size=chunk_size,
                continuous_state=continuous_state,
            )
            for _ in range(num_layers)
        ]
        super().__init__(
            input_dim,
            d_model,
            blocks,
            config,
            architecture=architecture,
            autoregressive=autoregressive,
            autoregressive_loss=autoregressive_loss,
            autoregressive_weight=autoregressive_weight,
            continuous_state=continuous_state,
        )


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
        architecture: str = "encoder",
        autoregressive: bool = False,
        autoregressive_loss: str = "smooth_l1",
        autoregressive_weight: float = 0.2,
    ) -> None:
        _validate_model_dimensions(input_dim, d_model, num_layers, num_heads)
        architecture = _validate_architecture(
            architecture, autoregressive, autoregressive_loss, autoregressive_weight
        )
        causal = architecture == "decoder"
        config: dict[str, Any] = {
            "input_dim": input_dim,
            "architecture": architecture,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ffn_ratio": float(ffn_ratio),
            "dropout": float(dropout),
        }
        if autoregressive:
            config.update(
                autoregressive=True,
                autoregressive_loss=autoregressive_loss,
                autoregressive_weight=float(autoregressive_weight),
            )
        blocks = [
            StandardTransformerBlock(
                dim=d_model,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
                causal=causal,
            )
            for _ in range(num_layers)
        ]
        super().__init__(
            input_dim,
            d_model,
            blocks,
            config,
            position_encoding=SinusoidalPositionalEncoding(d_model),
            architecture=architecture,
            autoregressive=autoregressive,
            autoregressive_loss=autoregressive_loss,
            autoregressive_weight=autoregressive_weight,
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
