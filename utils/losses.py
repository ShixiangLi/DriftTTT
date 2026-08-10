from __future__ import annotations

import torch


def cycle_balanced_dense_trajectory_loss(
    sequence_predictions: torch.Tensor,
    endpoint_targets: torch.Tensor,
    cycle_ids: torch.Tensor,
    mask: torch.Tensor,
    *,
    target_scale: float,
    capped_targets: bool,
) -> torch.Tensor:
    """CB-DTS loss over exact, cycle-balanced window RUL trajectories.

    N-CMAPSS RUL is measured in remaining flight cycles.  The endpoint target
    and the raw cycle distance therefore recover the target at every valid
    position without introducing a pseudo-label model.  Errors are averaged
    inside each observed flight cycle before cycles are averaged, preventing
    long flights from dominating the dense objective.
    """
    if sequence_predictions.ndim != 2:
        raise ValueError("sequence_predictions must have shape [B,L]")
    if mask.shape != sequence_predictions.shape:
        raise ValueError("mask must match sequence_predictions")
    if cycle_ids.shape != sequence_predictions.shape:
        raise ValueError("cycle_ids must match sequence_predictions")
    if endpoint_targets.shape != sequence_predictions.shape[:1]:
        raise ValueError("endpoint_targets must have shape [B]")
    if cycle_ids.dtype != torch.int64:
        raise ValueError("cycle_ids must use torch.int64")
    if target_scale <= 0.0:
        raise ValueError("target_scale must be positive")

    valid_mask = mask.bool()
    if not torch.all(valid_mask.any(dim=1)):
        raise ValueError("Every sample must contain at least one valid position")

    if torch.any(cycle_ids[valid_mask] < 0):
        raise ValueError("Valid positions require non-negative cycle IDs")
    adjacent_valid = valid_mask[:, 1:] & valid_mask[:, :-1]
    if torch.any(
        (cycle_ids[:, 1:] < cycle_ids[:, :-1]) & adjacent_valid
    ):
        raise ValueError("cycle_ids must be non-decreasing within a window")

    positions = torch.arange(
        valid_mask.shape[1], device=valid_mask.device
    ).expand_as(valid_mask)
    endpoint_indices = positions.masked_fill(~valid_mask, -1).max(dim=1).values
    endpoint_cycles = cycle_ids.gather(1, endpoint_indices[:, None]).squeeze(1)
    cycle_distance = endpoint_cycles[:, None] - cycle_ids
    dense_targets = endpoint_targets[:, None] + cycle_distance.to(
        endpoint_targets.dtype
    ) / float(target_scale)
    if capped_targets:
        dense_targets = dense_targets.clamp(max=1.0)
    squared_error = (sequence_predictions.float() - dense_targets.float()).square()

    previous_valid = torch.cat(
        (torch.zeros_like(valid_mask[:, :1]), valid_mask[:, :-1]), dim=1
    )
    previous_cycle = torch.cat((cycle_ids[:, :1], cycle_ids[:, :-1]), dim=1)
    cycle_start = valid_mask & (
        ~previous_valid | (cycle_ids != previous_cycle)
    )
    cycle_index = cycle_start.long().cumsum(dim=1).sub(1).clamp_min(0)
    maximum_cycles = valid_mask.shape[1]
    sample_offset = (
        torch.arange(valid_mask.shape[0], device=valid_mask.device) * maximum_cycles
    )[:, None]
    flat_group = (sample_offset + cycle_index)[valid_mask]

    cycle_error_sum = squared_error.new_zeros(
        valid_mask.shape[0] * maximum_cycles
    )
    cycle_token_count = squared_error.new_zeros(
        valid_mask.shape[0] * maximum_cycles
    )
    cycle_error_sum.scatter_add_(0, flat_group, squared_error[valid_mask])
    cycle_token_count.scatter_add_(
        0, flat_group, torch.ones_like(squared_error[valid_mask])
    )
    cycle_error_sum = cycle_error_sum.view(valid_mask.shape[0], maximum_cycles)
    cycle_token_count = cycle_token_count.view(
        valid_mask.shape[0], maximum_cycles
    )
    active_cycles = cycle_token_count > 0
    cycle_mse = cycle_error_sum / cycle_token_count.clamp_min(1.0)
    sample_loss = (cycle_mse * active_cycles).sum(dim=1)
    sample_loss = sample_loss / active_cycles.sum(dim=1).clamp_min(1)
    return sample_loss.mean()
