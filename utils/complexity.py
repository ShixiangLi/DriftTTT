"""Lightweight analytical complexity estimates for the RUL models."""

from __future__ import annotations

from typing import Any

from torch import nn

from models.rul_transformer import StandardTransformerBlock, TTTTransformerBlock


def estimate_model_complexity(
    model: nn.Module,
    sequence_length: int,
) -> dict[str, Any]:
    """Estimate dominant forward MACs for one full, unpadded input window."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    input_dim = int(getattr(model, "input_dim"))
    dim = int(getattr(model, "d_model"))
    macs = sequence_length * input_dim * dim
    model_type: str | None = None

    for block in getattr(model, "blocks"):
        if isinstance(block, StandardTransformerBlock):
            model_type = "transformer"
            hidden_dim = int(block.layer.linear1.out_features)
            attention_macs = (
                4 * sequence_length * dim * dim
                + 2 * sequence_length * sequence_length * dim
            )
            macs += attention_macs + 2 * sequence_length * dim * hidden_dim
        elif isinstance(block, TTTTransformerBlock):
            model_type = "ttt"
            heads = block.ttt.num_heads
            head_dim = block.ttt.head_dim
            hidden_dim = int(block.ffn.layers[0].out_features)
            cpe_kernel = int(block.cpe.conv.kernel_size[0])
            temporal_kernel = int(block.ttt.w3.shape[-1])
            ttt_macs = (
                sequence_length * dim * (3 * dim + 3 * head_dim)
                + 6 * heads * sequence_length * head_dim * head_dim
                + 2 * temporal_kernel * sequence_length * head_dim
                + sequence_length * (dim + head_dim) * dim
                + cpe_kernel * sequence_length * dim
            )
            macs += ttt_macs + 2 * sequence_length * dim * hidden_dim
        else:
            raise TypeError(f"Unsupported block type: {type(block).__name__}")

    if model_type is None:
        raise ValueError("Model must contain at least one supported encoder block")

    macs += dim  # Final regression head.
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "model_type": model_type,
        "input_shape": [1, sequence_length, input_dim],
        "parameters": int(parameters),
        "trainable_parameters": int(trainable_parameters),
        "forward_macs": int(macs),
        "forward_flops": int(2 * macs),
        "method": "analytical",
        "scope": "dominant Linear/MatMul/Conv ops; 1 MAC = 2 FLOPs",
    }


def format_model_complexity(complexity: dict[str, Any]) -> str:
    """Format a compact one-line summary for CLI output."""

    def compact(value: int) -> str:
        for scale, suffix in ((1_000_000_000, "G"), (1_000_000, "M"), (1_000, "K")):
            if value >= scale:
                return f"{value / scale:.3f}{suffix}"
        return str(value)

    return (
        f"complexity input={complexity['input_shape']} "
        f"params={compact(complexity['parameters'])} "
        f"forward_macs={compact(complexity['forward_macs'])} "
        f"forward_flops={compact(complexity['forward_flops'])}"
    )


__all__ = ["estimate_model_complexity", "format_model_complexity"]
