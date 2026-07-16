"""Temporal adaptation of the ViT^3 test-time-training layer.

Adapted from ``ViTTT/ttt_block.py`` by Dongchen Han. The original file is
distributed under the MIT License:

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The spatial 3x3 depth-wise branch is minimally adapted to a temporal 1D
depth-wise convolution with kernel size 3. Fast weights remain local to one
forward call; module parameters are never mutated by the inner update.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TTTLayer(nn.Module):
    """Test-time-training sequence layer with per-sample local fast weights.

    Args:
        dim: Input and output feature dimension.
        num_heads: Number of heads in the simplified SwiGLU branch.
        qkv_bias: Whether the joint q/k/v projection has a bias.
        inner_lr: Learning rate for the single closed-form inner update.

    Inputs:
        x: Floating point tensor with shape ``[batch, length, dim]``.
        padding_mask: Optional boolean tensor with shape ``[batch, length]``.
            ``True`` marks padding and excludes that position from the inner
            update and the returned features.
    """

    _CONV_KERNEL_SIZE = 3

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        inner_lr: float = 1.0,
        inner_scale: float = 1.0 / 3.0,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        if not math.isfinite(inner_lr) or inner_lr < 0:
            raise ValueError(
                f"inner_lr must be finite and non-negative, got {inner_lr}"
            )
        if not math.isfinite(inner_scale) or inner_scale <= 0:
            raise ValueError(
                f"inner_scale must be finite and positive, got {inner_scale}"
            )

        head_dim = dim // num_heads
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_lr = float(inner_lr)

        # Keep the ViT^3 joint projection and its asymmetric second branch.
        self.qkv = nn.Linear(dim, dim * 3 + head_dim * 3, bias=qkv_bias)
        self.w1 = nn.Parameter(torch.zeros(1, num_heads, head_dim, head_dim))
        self.w2 = nn.Parameter(torch.zeros(1, num_heads, head_dim, head_dim))
        self.w3 = nn.Parameter(
            torch.zeros(head_dim, 1, self._CONV_KERNEL_SIZE)
        )
        nn.init.trunc_normal_(self.w1, std=0.02)
        nn.init.trunc_normal_(self.w2, std=0.02)
        nn.init.trunc_normal_(self.w3, std=0.02)
        self.proj = nn.Linear(dim + head_dim, dim)

        # The default preserves ViT^3's fixed 3x3 equivalent head scale.
        self.scale = float(inner_scale)

    @staticmethod
    def _validate_padding_mask(x: Tensor, padding_mask: Tensor | None) -> None:
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
            raise ValueError(
                "padding_mask and x must be on the same device; "
                f"got {padding_mask.device} and {x.device}"
            )

    def inner_train_simplified_swiglu(
        self,
        k: Tensor,
        v: Tensor,
        w1: Tensor,
        w2: Tensor,
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply the hand-derived single update from the original ViT^3 layer."""
        z1 = k @ w1
        z2 = k @ w2
        sig = torch.sigmoid(z2)
        activation = z2 * sig

        if valid_mask is None:
            error = -v / float(v.shape[2]) * self.scale
        else:
            mask = valid_mask[:, None, :, None].to(dtype=v.dtype)
            valid_count = mask.sum(dim=2, keepdim=True).clamp_min(1.0)
            error = -v * mask / valid_count * self.scale

        g1 = k.transpose(-2, -1) @ (error * activation)
        swish_derivative = sig * (1.0 + z2 * (1.0 - sig))
        g2 = k.transpose(-2, -1) @ (
            error * z1 * swish_derivative
        )

        # Preserve the original per-column stabilizing clip.
        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)
        return w1 - self.inner_lr * g1, w2 - self.inner_lr * g2

    def inner_train_depthwise_conv1d(
        self,
        k: Tensor,
        v: Tensor,
        w: Tensor,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        """Update per-sample temporal depth-wise convolution fast weights."""
        batch_size, channels, length = k.shape
        if valid_mask is None:
            error = -v / float(length) * self.scale
        else:
            mask = valid_mask[:, None, :].to(dtype=v.dtype)
            valid_count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            # Zeroing keys is necessary because a valid convolution output can
            # otherwise include a neighboring padded input position.
            k = k * mask
            error = -v * mask / valid_count * self.scale

        radius = self._CONV_KERNEL_SIZE // 2
        padded_k = F.pad(k, (radius, radius))
        gradient_parts = []
        for offset in range(self._CONV_KERNEL_SIZE):
            shifted_k = padded_k[:, :, offset : offset + length]
            gradient_parts.append((shifted_k * error).sum(dim=-1))
        gradient = torch.stack(gradient_parts, dim=-1).reshape(
            batch_size * channels, 1, self._CONV_KERNEL_SIZE
        )

        gradient = gradient / (
            gradient.norm(dim=-1, keepdim=True) + 1.0
        )
        base_weights = w.repeat(batch_size, 1, 1)
        return base_weights - self.inner_lr * gradient

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [batch, length, dim], got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"expected x feature dimension {self.dim}, got {x.shape[-1]}"
            )
        if x.shape[1] == 0:
            raise ValueError("sequence length must be greater than zero")
        if not x.is_floating_point():
            raise TypeError(f"x must be floating point, got {x.dtype}")
        self._validate_padding_mask(x, padding_mask)

        batch_size, length, channels = x.shape
        head_dim = self.head_dim
        q1, k1, v1, q2, k2, v2 = torch.split(
            self.qkv(x),
            [channels, channels, channels, head_dim, head_dim, head_dim],
            dim=-1,
        )

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.reshape(
                batch_size, length, self.num_heads, head_dim
            ).transpose(1, 2)

        q1, k1, v1 = map(split_heads, (q1, k1, v1))
        q2 = q2.transpose(1, 2)
        k2 = k2.transpose(1, 2)
        v2 = v2.transpose(1, 2)

        valid_mask = None if padding_mask is None else ~padding_mask
        if valid_mask is not None:
            head_mask = valid_mask[:, None, :, None].to(dtype=x.dtype)
            temporal_mask = valid_mask[:, None, :].to(dtype=x.dtype)
            q1 = q1 * head_mask
            k1 = k1 * head_mask
            v1 = v1 * head_mask
            q2 = q2 * temporal_mask
            k2 = k2 * temporal_mask
            v2 = v2 * temporal_mask

        # These fast weights have a batch dimension and live only in this call.
        fast_w1, fast_w2 = self.inner_train_simplified_swiglu(
            k1, v1, self.w1, self.w2, valid_mask
        )
        fast_w3 = self.inner_train_depthwise_conv1d(
            k2, v2, self.w3, valid_mask
        )

        swiglu_output = (q1 @ fast_w1) * F.silu(q1 @ fast_w2)
        swiglu_output = swiglu_output.transpose(1, 2).reshape(
            batch_size, length, channels
        )

        conv_output = F.conv1d(
            q2.reshape(1, batch_size * head_dim, length),
            fast_w3,
            padding=self._CONV_KERNEL_SIZE // 2,
            groups=batch_size * head_dim,
        )
        conv_output = conv_output.reshape(
            batch_size, head_dim, length
        ).transpose(1, 2)

        output = self.proj(torch.cat([swiglu_output, conv_output], dim=-1))
        if padding_mask is not None:
            output = output.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return output

    def reset_ttt_state(self) -> None:
        """Reset sequence state.

        The ViT^3 closed-form update creates fast weights locally in
        :meth:`forward`, so there is no persistent state to clear. This explicit
        no-op lets evaluation code mark engine boundaries without depending on
        that implementation detail.
        """
        return None

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_heads={self.num_heads}, "
            f"inner_lr={self.inner_lr}, inner_scale={self.scale}"
        )


# Compatibility with the source project name while keeping the adapted API
# explicit at import sites.
TTT = TTTLayer


__all__ = ["TTT", "TTTLayer"]
