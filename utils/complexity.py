from __future__ import annotations

from typing import Any

from torch import nn


def model_complexity(
    model: nn.Module,
    input_dim: int,
    sequence_length: int,
    model_config: dict[str, Any],
) -> dict[str, int]:
    """Return parameters and analytical per-sample MAC estimates.

    Cycle-aware slow lengths depend on sample metadata, so the multiscale
    estimate conservatively uses the maximum possible number of cycles.
    """
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    length = int(sequence_length)
    width = int(model_config["d_model"])
    feedforward = int(model_config["dim_feedforward"])
    layers = int(model_config["num_layers"])
    input_macs = length * input_dim * width
    mixer = model_config["sequence_mixer"]
    if mixer == "attention":
        mixer_macs = 4 * length * width * width + 2 * length * length * width
    elif mixer in {"ttt_mlp", "ttt_multiscale_moe"}:
        heads = int(model_config["nhead"])
        head_dim = width // heads
        inner_dim = max(
            1,
            int(round(head_dim * float(model_config["ttt"]["hidden_multiplier"]))),
        )
        projections = 4 * length * width * width
        if mixer == "ttt_mlp":
            fast_mlp_and_update = 7 * length * heads * head_dim * inner_dim
            mixer_macs = projections + fast_mlp_and_update
        else:
            multiscale = model_config["ttt"]["multiscale"]
            short_dim = int(round(inner_dim * float(multiscale["short_rank_ratio"])))
            short_dim = min(max(1, short_dim), inner_dim - 1)
            long_dim = inner_dim - short_dim
            slow_length = length
            fast_mlp_and_update = (
                7 * heads * head_dim * (length * short_dim + slow_length * long_dim)
            )
            # Cycle grouping, smoothing, routing, and fusion remain linear in
            # the original sequence length. Actual N-CMAPSS slow lengths are
            # normally much smaller than this conservative upper bound.
            mixer_macs = projections + fast_mlp_and_update + 9 * length * width
    else:
        raise ValueError(f"Unsupported sequence mixer: {mixer}")
    feedforward_macs = 2 * length * width * feedforward
    head_macs = width
    macs = input_macs + layers * (mixer_macs + feedforward_macs) + head_macs
    return {
        "parameters": int(parameters),
        "trainable_parameters": int(trainable),
        "forward_macs_per_sample": int(macs),
        "forward_flops_per_sample": int(2 * macs),
    }
