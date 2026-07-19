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
depth-wise convolution with kernel size 3. Fast statistics can remain local to
one forward call or continue across calls for one causal equipment stream;
module parameters are never mutated by the inner update.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class _CausalTTTState:
    """Constant-size sufficient statistics for one causal token stream."""

    g1_sum: Tensor
    g2_sum: Tensor
    g3_sum: Tensor
    seen: Tensor
    pending_g1: Tensor
    pending_g2: Tensor
    pending_g3: Tensor
    pending_count: int
    q_history: Tensor
    k_history: Tensor

    def detached(self) -> "_CausalTTTState":
        return _CausalTTTState(
            g1_sum=self.g1_sum.detach(),
            g2_sum=self.g2_sum.detach(),
            g3_sum=self.g3_sum.detach(),
            seen=self.seen.detach(),
            pending_g1=self.pending_g1.detach(),
            pending_g2=self.pending_g2.detach(),
            pending_g3=self.pending_g3.detach(),
            pending_count=self.pending_count,
            q_history=self.q_history.detach(),
            k_history=self.k_history.detach(),
        )


class TTTLayer(nn.Module):
    """Test-time-training sequence layer with optional continuous fast state.

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
        causal: bool = False,
        chunk_size: int = 16,
        continuous_state: bool = False,
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
        if not isinstance(causal, bool):
            raise TypeError(f"causal must be bool, got {type(causal).__name__}")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError("chunk_size must be an integer")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if not isinstance(continuous_state, bool):
            raise TypeError(
                "continuous_state must be bool, "
                f"got {type(continuous_state).__name__}"
            )
        if continuous_state and not causal:
            raise ValueError("continuous_state requires causal=True")

        head_dim = dim // num_heads
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_lr = float(inner_lr)
        self.causal = causal
        self.chunk_size = chunk_size
        self.continuous_state = continuous_state
        self._causal_state: _CausalTTTState | None = None

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
        with torch.autocast(device_type=k.device.type, enabled=False):
            k, v, w1, w2 = (value.float() for value in (k, v, w1, w2))
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

            # Preserve the original per-column stabilizing clip in FP32.
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
        with torch.autocast(device_type=k.device.type, enabled=False):
            k, v, w = (value.float() for value in (k, v, w))
            batch_size, channels, length = k.shape
            if valid_mask is None:
                error = -v / float(length) * self.scale
            else:
                mask = valid_mask[:, None, :].to(dtype=v.dtype)
                valid_count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
                # A valid output must not include a neighboring padded key.
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

    def _forward_full(
        self,
        q1: Tensor,
        k1: Tensor,
        v1: Tensor,
        q2: Tensor,
        k2: Tensor,
        v2: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor:
        """Apply the original whole-window, non-causal TTT update."""
        batch_size, _, length, _ = q1.shape
        fast_w1, fast_w2 = self.inner_train_simplified_swiglu(
            k1, v1, self.w1, self.w2, valid_mask
        )
        fast_w3 = self.inner_train_depthwise_conv1d(
            k2, v2, self.w3, valid_mask
        )

        swiglu_output = (q1 @ fast_w1) * F.silu(q1 @ fast_w2)
        swiglu_output = swiglu_output.transpose(1, 2).reshape(
            batch_size, length, self.dim
        )
        conv_output = F.conv1d(
            q2.reshape(1, batch_size * self.head_dim, length),
            fast_w3,
            padding=self._CONV_KERNEL_SIZE // 2,
            groups=batch_size * self.head_dim,
        )
        conv_output = conv_output.reshape(
            batch_size, self.head_dim, length
        ).transpose(1, 2)
        return self.proj(torch.cat([swiglu_output, conv_output], dim=-1))

    def _causal_gradient_sums(
        self,
        k1: Tensor,
        v1: Tensor,
        k2: Tensor,
        v2: Tensor,
        k_history: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return unnormalised inner gradients for one contiguous segment."""
        with torch.autocast(device_type=k1.device.type, enabled=False):
            k1, v1, k2, v2, k_history = (
                value.float() for value in (k1, v1, k2, v2, k_history)
            )
            segment_length = k1.shape[2]
            z1 = k1 @ self.w1.float()
            z2 = k1 @ self.w2.float()
            sig = torch.sigmoid(z2)
            error1 = -v1 * self.scale
            g1 = k1.transpose(-2, -1) @ (error1 * (z2 * sig))
            swish_derivative = sig * (1.0 + z2 * (1.0 - sig))
            g2 = k1.transpose(-2, -1) @ (
                error1 * z1 * swish_derivative
            )

            k_context = torch.cat([k_history, k2], dim=-1)
            error2 = -v2 * self.scale
            g3 = torch.stack(
                [
                    (
                        k_context[:, :, offset : offset + segment_length]
                        * error2
                    ).sum(dim=-1)
                    for offset in range(self._CONV_KERNEL_SIZE)
                ],
                dim=-1,
            )
            history_size = self._CONV_KERNEL_SIZE - 1
            return g1, g2, g3, k_context[:, :, -history_size:]

    def _forward_block_causal(
        self,
        q1: Tensor,
        k1: Tensor,
        v1: Tensor,
        q2: Tensor,
        k2: Tensor,
        v2: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor:
        """Use previous chunks for adaptation and causal convolution within chunks.

        Raw inner-gradient statistics are accumulated across completed chunks.
        Consequently, every query is adapted only by earlier positions. Setting
        ``chunk_size=1`` gives token-wise causal adaptation; larger chunks retain
        efficient parallel matrix operations within each chunk.
        """
        batch_size, heads, length, head_dim = q1.shape
        kernel_size = self._CONV_KERNEL_SIZE
        history_size = kernel_size - 1
        dtype = torch.float32
        device = q1.device

        g1_sum = torch.zeros(
            batch_size, heads, head_dim, head_dim, dtype=dtype, device=device
        )
        g2_sum = torch.zeros_like(g1_sum)
        g3_sum = torch.zeros(
            batch_size, head_dim, kernel_size, dtype=dtype, device=device
        )
        seen = torch.zeros(batch_size, dtype=dtype, device=device)
        q_history = torch.zeros(
            batch_size, head_dim, history_size, dtype=q2.dtype, device=device
        )
        k_history = torch.zeros(
            batch_size, head_dim, history_size, dtype=dtype, device=device
        )
        swiglu_chunks: list[Tensor] = []
        conv_chunks: list[Tensor] = []

        base_w3 = self.w3[:, 0, :].unsqueeze(0)
        for start in range(0, length, self.chunk_size):
            stop = min(length, start + self.chunk_size)
            chunk_length = stop - start
            denominator = seen.clamp_min(1.0)

            mean_g1 = g1_sum / denominator[:, None, None, None]
            mean_g2 = g2_sum / denominator[:, None, None, None]
            mean_g3 = g3_sum / denominator[:, None, None]
            mean_g1 = mean_g1 / (mean_g1.norm(dim=-2, keepdim=True) + 1.0)
            mean_g2 = mean_g2 / (mean_g2.norm(dim=-2, keepdim=True) + 1.0)
            mean_g3 = mean_g3 / (mean_g3.norm(dim=-1, keepdim=True) + 1.0)
            fast_w1 = self.w1 - self.inner_lr * mean_g1
            fast_w2 = self.w2 - self.inner_lr * mean_g2
            fast_w3 = (base_w3 - self.inner_lr * mean_g3).reshape(
                batch_size * head_dim, 1, kernel_size
            )

            q1_chunk = q1[:, :, start:stop]
            swiglu_chunks.append(
                (q1_chunk @ fast_w1) * F.silu(q1_chunk @ fast_w2)
            )

            q2_chunk = q2[:, :, start:stop]
            q_context = torch.cat([q_history, q2_chunk], dim=-1)
            conv_chunk = F.conv1d(
                q_context.reshape(1, batch_size * head_dim, -1),
                fast_w3,
                groups=batch_size * head_dim,
            )
            conv_chunks.append(
                conv_chunk.reshape(batch_size, head_dim, chunk_length)
            )
            q_history = q_context[:, :, -history_size:]

            chunk_gradients = self._causal_gradient_sums(
                k1[:, :, start:stop],
                v1[:, :, start:stop],
                k2[:, :, start:stop],
                v2[:, :, start:stop],
                k_history,
            )
            gradient_g1, gradient_g2, gradient_g3, k_history = chunk_gradients
            g1_sum = g1_sum + gradient_g1
            g2_sum = g2_sum + gradient_g2
            g3_sum = g3_sum + gradient_g3
            if valid_mask is None:
                seen = seen + chunk_length
            else:
                seen = seen + valid_mask[:, start:stop].sum(dim=-1).to(dtype)

        swiglu_output = torch.cat(swiglu_chunks, dim=2).transpose(1, 2).reshape(
            batch_size, length, self.dim
        )
        conv_output = torch.cat(conv_chunks, dim=-1).transpose(1, 2)
        return self.proj(torch.cat([swiglu_output, conv_output], dim=-1))

    def _new_causal_state(self, q1: Tensor) -> _CausalTTTState:
        batch_size, heads, _, head_dim = q1.shape
        if batch_size != 1:
            raise ValueError(
                "continuous TTT state requires one chronological stream per "
                "forward call"
            )
        dtype = torch.float32
        device = q1.device
        gradient = torch.zeros(
            1, heads, head_dim, head_dim, dtype=dtype, device=device
        )
        conv_gradient = torch.zeros(
            1,
            head_dim,
            self._CONV_KERNEL_SIZE,
            dtype=dtype,
            device=device,
        )
        q_history = torch.zeros(
            1,
            head_dim,
            self._CONV_KERNEL_SIZE - 1,
            dtype=q1.dtype,
            device=device,
        )
        return _CausalTTTState(
            g1_sum=gradient,
            g2_sum=torch.zeros_like(gradient),
            g3_sum=conv_gradient,
            seen=torch.zeros(1, dtype=dtype, device=device),
            pending_g1=torch.zeros_like(gradient),
            pending_g2=torch.zeros_like(gradient),
            pending_g3=torch.zeros_like(conv_gradient),
            pending_count=0,
            q_history=q_history,
            k_history=torch.zeros(
                1,
                head_dim,
                self._CONV_KERNEL_SIZE - 1,
                dtype=dtype,
                device=device,
            ),
        )

    def _forward_streaming_causal(
        self,
        q1: Tensor,
        k1: Tensor,
        v1: Tensor,
        q2: Tensor,
        k2: Tensor,
        v2: Tensor,
    ) -> Tensor:
        """Continue a causal TTT stream without committing partial chunks early."""
        state = self._causal_state or self._new_causal_state(q1)
        if (
            state.g1_sum.device != q1.device
            or state.q_history.dtype != q1.dtype
        ):
            raise RuntimeError(
                "TTT stream state device/dtype changed; reset_ttt_state() before "
                "moving the model or changing precision"
            )

        batch_size, _, length, _ = q1.shape
        head_dim = self.head_dim
        kernel_size = self._CONV_KERNEL_SIZE
        g1_sum = state.g1_sum
        g2_sum = state.g2_sum
        g3_sum = state.g3_sum
        seen = state.seen
        pending_g1 = state.pending_g1
        pending_g2 = state.pending_g2
        pending_g3 = state.pending_g3
        pending_count = state.pending_count
        q_history = state.q_history
        k_history = state.k_history
        swiglu_segments: list[Tensor] = []
        conv_segments: list[Tensor] = []
        base_w3 = self.w3[:, 0, :].unsqueeze(0)

        start = 0
        while start < length:
            capacity = self.chunk_size - pending_count
            stop = min(length, start + capacity)
            segment_length = stop - start

            denominator = seen.clamp_min(1.0)
            mean_g1 = g1_sum / denominator[:, None, None, None]
            mean_g2 = g2_sum / denominator[:, None, None, None]
            mean_g3 = g3_sum / denominator[:, None, None]
            mean_g1 = mean_g1 / (mean_g1.norm(dim=-2, keepdim=True) + 1.0)
            mean_g2 = mean_g2 / (mean_g2.norm(dim=-2, keepdim=True) + 1.0)
            mean_g3 = mean_g3 / (mean_g3.norm(dim=-1, keepdim=True) + 1.0)
            fast_w1 = self.w1 - self.inner_lr * mean_g1
            fast_w2 = self.w2 - self.inner_lr * mean_g2
            fast_w3 = (base_w3 - self.inner_lr * mean_g3).reshape(
                batch_size * head_dim, 1, kernel_size
            )

            q1_segment = q1[:, :, start:stop]
            swiglu_segments.append(
                (q1_segment @ fast_w1) * F.silu(q1_segment @ fast_w2)
            )
            q2_segment = q2[:, :, start:stop]
            q_context = torch.cat([q_history, q2_segment], dim=-1)
            conv_segment = F.conv1d(
                q_context.reshape(1, batch_size * head_dim, -1),
                fast_w3,
                groups=batch_size * head_dim,
            )
            conv_segments.append(
                conv_segment.reshape(batch_size, head_dim, segment_length)
            )
            q_history = q_context[:, :, -(kernel_size - 1) :]

            segment_gradients = self._causal_gradient_sums(
                k1[:, :, start:stop],
                v1[:, :, start:stop],
                k2[:, :, start:stop],
                v2[:, :, start:stop],
                k_history,
            )
            gradient_g1, gradient_g2, gradient_g3, k_history = segment_gradients
            pending_g1 = pending_g1 + gradient_g1
            pending_g2 = pending_g2 + gradient_g2
            pending_g3 = pending_g3 + gradient_g3
            pending_count += segment_length
            if pending_count == self.chunk_size:
                g1_sum = g1_sum + pending_g1
                g2_sum = g2_sum + pending_g2
                g3_sum = g3_sum + pending_g3
                seen = seen + float(pending_count)
                pending_g1 = torch.zeros_like(pending_g1)
                pending_g2 = torch.zeros_like(pending_g2)
                pending_g3 = torch.zeros_like(pending_g3)
                pending_count = 0
            start = stop

        self._causal_state = _CausalTTTState(
            g1_sum=g1_sum,
            g2_sum=g2_sum,
            g3_sum=g3_sum,
            seen=seen,
            pending_g1=pending_g1,
            pending_g2=pending_g2,
            pending_g3=pending_g3,
            pending_count=pending_count,
            q_history=q_history,
            k_history=k_history,
        ).detached()
        swiglu_output = torch.cat(swiglu_segments, dim=2).transpose(1, 2).reshape(
            batch_size, length, self.dim
        )
        conv_output = torch.cat(conv_segments, dim=-1).transpose(1, 2)
        return self.proj(torch.cat([swiglu_output, conv_output], dim=-1))

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

        if self.causal:
            if self.continuous_state:
                if batch_size != 1:
                    raise ValueError(
                        "continuous TTT state requires batch_size=1 at the "
                        "layer boundary"
                    )
                if valid_mask is None:
                    output = self._forward_streaming_causal(
                        q1, k1, v1, q2, k2, v2
                    )
                else:
                    valid_indices = torch.nonzero(
                        valid_mask[0], as_tuple=False
                    ).flatten()
                    if valid_indices.numel() == 0:
                        raise ValueError(
                            "continuous TTT input must contain a valid token"
                        )
                    output_valid = self._forward_streaming_causal(
                        q1.index_select(2, valid_indices),
                        k1.index_select(2, valid_indices),
                        v1.index_select(2, valid_indices),
                        q2.index_select(2, valid_indices),
                        k2.index_select(2, valid_indices),
                        v2.index_select(2, valid_indices),
                    )
                    output = output_valid.new_zeros(
                        batch_size, length, self.dim
                    ).index_copy(1, valid_indices, output_valid)
                if padding_mask is not None:
                    output = output.masked_fill(
                        padding_mask.unsqueeze(-1), 0.0
                    )
                return output

            restore_order = None
            if valid_mask is not None:
                positions = torch.arange(length, device=x.device).expand(
                    batch_size, -1
                )
                compact_order = torch.argsort(
                    torch.where(valid_mask, positions, positions + length), dim=1
                )

                def compact_heads(tensor: Tensor) -> Tensor:
                    indices = compact_order[:, None, :, None].expand(
                        -1, tensor.shape[1], -1, tensor.shape[-1]
                    )
                    return torch.gather(tensor, 2, indices)

                def compact_temporal(tensor: Tensor) -> Tensor:
                    indices = compact_order[:, None, :].expand(
                        -1, tensor.shape[1], -1
                    )
                    return torch.gather(tensor, 2, indices)

                q1, k1, v1 = map(compact_heads, (q1, k1, v1))
                q2, k2, v2 = map(compact_temporal, (q2, k2, v2))
                valid_mask = torch.gather(valid_mask, 1, compact_order)
                restore_order = torch.argsort(compact_order, dim=1)
            output = self._forward_block_causal(q1, k1, v1, q2, k2, v2, valid_mask)
            if restore_order is not None:
                output = torch.gather(
                    output,
                    1,
                    restore_order.unsqueeze(-1).expand(-1, -1, output.shape[-1]),
                )
        else:
            output = self._forward_full(q1, k1, v1, q2, k2, v2, valid_mask)
        if padding_mask is not None:
            output = output.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return output

    def reset_ttt_state(self) -> None:
        """Clear all accumulated fast statistics at an entity boundary."""
        self._causal_state = None

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_heads={self.num_heads}, "
            f"inner_lr={self.inner_lr}, inner_scale={self.scale}, "
            f"causal={self.causal}, chunk_size={self.chunk_size}, "
            f"continuous_state={self.continuous_state}"
        )


# Compatibility with the source project name while keeping the adapted API
# explicit at import sites.
TTT = TTTLayer


__all__ = ["TTT", "TTTLayer"]
