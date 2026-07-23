from __future__ import annotations

import math
from typing import Any, NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class _FastMLPState(NamedTuple):
    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor


def _clip_fast_state(
    gradients: _FastMLPState, maximum_norm: float | None
) -> _FastMLPState:
    if maximum_norm is None:
        return gradients
    squared_norm = gradients.w1.square().sum(dim=(-2, -1))
    squared_norm = squared_norm + gradients.b1.square().sum(dim=-1)
    squared_norm = squared_norm + gradients.w2.square().sum(dim=(-2, -1))
    squared_norm = squared_norm + gradients.b2.square().sum(dim=-1)
    norm = squared_norm.clamp_min(1.0e-12).sqrt()
    scale = (float(maximum_norm) / norm).clamp(max=1.0)
    return _FastMLPState(
        gradients.w1 * scale[:, :, None, None],
        gradients.b1 * scale[:, :, None],
        gradients.w2 * scale[:, :, None, None],
        gradients.b2 * scale[:, :, None],
    )


class _TTTMLPBase(nn.Module):
    """Shared projection, fast-MLP update, and padding logic for TTT mixers."""

    def __init__(self, d_model: int, nhead: int, config: dict[str, Any]) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.inner_gradient_clip = config["inner_gradient_clip"]
        self.activation = str(config["activation"])

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bool(config["qkv_bias"]))
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, d_model)

    def _reset_common_parameters(self) -> None:
        nn.init.xavier_uniform_(self.qkv.weight)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _activate_with_derivative(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.activation == "silu":
            sigmoid = torch.sigmoid(values)
            activated = values * sigmoid
            derivative = sigmoid * (1.0 + values * (1.0 - sigmoid))
            return activated, derivative
        if self.activation == "gelu":
            activated = F.gelu(values)
            derivative = 0.5 * (1.0 + torch.erf(values / math.sqrt(2.0)))
            derivative = derivative + values * torch.exp(
                -0.5 * values.square()
            ) / math.sqrt(2.0 * math.pi)
            return activated, derivative
        raise RuntimeError(f"Unsupported TTT activation: {self.activation}")

    def _fast_mlp(self, inputs: torch.Tensor, state: _FastMLPState) -> torch.Tensor:
        preactivation = torch.einsum("bshd,bhdm->bshm", inputs, state.w1)
        preactivation = preactivation + state.b1[:, None]
        if self.activation == "silu":
            hidden = F.silu(preactivation)
        elif self.activation == "gelu":
            hidden = F.gelu(preactivation)
        else:
            raise RuntimeError(f"Unsupported TTT activation: {self.activation}")
        return torch.einsum("bshm,bhmd->bshd", hidden, state.w2) + state.b2[:, None]

    def _clip_inner_gradients(self, gradients: _FastMLPState) -> _FastMLPState:
        return _clip_fast_state(gradients, self.inner_gradient_clip)

    def _adapt_chunk(
        self,
        keys: torch.Tensor,
        targets: torch.Tensor,
        update_weight: torch.Tensor,
        state: _FastMLPState,
        learning_rate: float,
    ) -> _FastMLPState:
        preactivation = torch.einsum("bshd,bhdm->bshm", keys, state.w1)
        preactivation = preactivation + state.b1[:, None]
        hidden, activation_derivative = self._activate_with_derivative(preactivation)
        predictions = torch.einsum("bshm,bhmd->bshd", hidden, state.w2)
        predictions = predictions + state.b2[:, None]

        weight = update_weight[:, :, None, None].to(dtype=predictions.dtype)
        error = (predictions - targets) * weight
        mass = update_weight.sum(dim=1).clamp_min(1).to(predictions.dtype)
        inverse_mass = (mass * self.head_dim).reciprocal()
        gradient_scale = inverse_mass[:, None, None, None]

        grad_w2 = torch.einsum("bshm,bshd->bhmd", hidden, error)
        grad_w2 = grad_w2 * gradient_scale
        grad_b2 = error.sum(dim=1) * inverse_mass[:, None, None]
        grad_hidden = torch.einsum("bshd,bhmd->bshm", error, state.w2)
        grad_hidden = grad_hidden * gradient_scale
        grad_preactivation = grad_hidden * activation_derivative
        grad_w1 = torch.einsum("bshd,bshm->bhdm", keys, grad_preactivation)
        grad_b1 = grad_preactivation.sum(dim=1)
        gradients = self._clip_inner_gradients(
            _FastMLPState(grad_w1, grad_b1, grad_w2, grad_b2)
        )
        return _FastMLPState(
            state.w1 - learning_rate * gradients.w1,
            state.b1 - learning_rate * gradients.b1,
            state.w2 - learning_rate * gradients.w2,
            state.b2 - learning_rate * gradients.b2,
        )

    @staticmethod
    def _expand_state(
        w1: torch.Tensor,
        b1: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor,
        batch_size: int,
    ) -> _FastMLPState:
        return _FastMLPState(
            w1.unsqueeze(0).expand(batch_size, -1, -1, -1),
            b1.unsqueeze(0).expand(batch_size, -1, -1),
            w2.unsqueeze(0).expand(batch_size, -1, -1, -1),
            b2.unsqueeze(0).expand(batch_size, -1, -1),
        )

    def _run_expert(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        targets: torch.Tensor,
        update_weight: torch.Tensor,
        output_mask: torch.Tensor,
        state: _FastMLPState,
        chunk_size: int,
        learning_rate: float,
    ) -> tuple[torch.Tensor, _FastMLPState]:
        outputs = []
        for start in range(0, queries.shape[1], chunk_size):
            stop = min(start + chunk_size, queries.shape[1])
            state = self._adapt_chunk(
                keys[:, start:stop],
                targets[:, start:stop],
                update_weight[:, start:stop],
                state,
                learning_rate,
            )
            residual = self._fast_mlp(queries[:, start:stop], state)
            chunk_output_mask = output_mask[:, start:stop, None, None]
            outputs.append(residual * chunk_output_mask)
        return torch.cat(outputs, dim=1), state

    @staticmethod
    def _reconstruction_objective(
        keys: torch.Tensor,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the label-free, same-coordinate TTT reconstruction task."""
        return keys, values - keys, valid_mask

    def _prepare_inputs(
        self, inputs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        values = inputs.float()
        batch_size, sequence_length, _ = values.shape
        qkv = self.qkv(values).view(
            batch_size, sequence_length, 3, self.nhead, self.head_dim
        )
        queries, keys, projected_values = qkv.unbind(dim=2)
        queries = F.layer_norm(queries, (self.head_dim,))
        keys = F.layer_norm(keys, (self.head_dim,))
        # Left alignment makes TTT updates independent of padding width.
        valid_rank = mask.long().cumsum(dim=1).sub(1).clamp_min(0)
        gather_index = valid_rank[:, :, None, None].expand(
            -1, -1, self.nhead, self.head_dim
        )
        valid_values = mask[:, :, None, None]

        def left_align(tensor: torch.Tensor) -> torch.Tensor:
            aligned = torch.zeros_like(tensor)
            return aligned.scatter_add(1, gather_index, tensor * valid_values)

        queries = left_align(queries)
        keys = left_align(keys)
        projected_values = left_align(projected_values)
        valid_lengths = mask.sum(dim=1)
        aligned_mask = torch.arange(sequence_length, device=mask.device)
        aligned_mask = aligned_mask.unsqueeze(0) < valid_lengths.unsqueeze(1)
        return queries, keys, projected_values, aligned_mask, valid_rank

    def _restore_output(
        self,
        aligned_output: torch.Tensor,
        valid_rank: torch.Tensor,
        original_mask: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size, sequence_length = aligned_output.shape[:2]
        mixed = aligned_output.reshape(batch_size, sequence_length, self.d_model)
        restore_index = valid_rank[:, :, None].expand(-1, -1, self.d_model)
        mixed = mixed.gather(1, restore_index)
        mixed = self.output_projection(self.output_norm(mixed))
        mixed = mixed * original_mask.unsqueeze(-1)
        return mixed.to(dtype=output_dtype)

    def _validate_inputs(self, inputs: torch.Tensor, mask: torch.Tensor) -> None:
        if inputs.ndim != 3 or inputs.shape[-1] != self.d_model:
            raise ValueError(f"Expected TTT inputs [B,L,{self.d_model}]")
        if mask.shape != inputs.shape[:2]:
            raise ValueError("TTT mask must have shape [B,L]")


class TTTMLP(_TTTMLPBase):
    """Sample-local test-time-training mixer with one fast MLP."""

    def __init__(self, d_model: int, nhead: int, config: dict[str, Any]) -> None:
        super().__init__(d_model, nhead, config)
        self.hidden_dim = max(
            1, int(round(self.head_dim * float(config["hidden_multiplier"])))
        )
        self.inner_learning_rate = float(config["inner_learning_rate"])
        self.chunk_size = int(config["chunk_size"])
        self.fast_w1 = nn.Parameter(torch.empty(nhead, self.head_dim, self.hidden_dim))
        self.fast_b1 = nn.Parameter(torch.zeros(nhead, self.hidden_dim))
        self.fast_w2 = nn.Parameter(torch.empty(nhead, self.hidden_dim, self.head_dim))
        self.fast_b2 = nn.Parameter(torch.zeros(nhead, self.head_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self._reset_common_parameters()
        nn.init.xavier_uniform_(self.fast_w1)
        nn.init.xavier_uniform_(self.fast_w2, gain=0.25)
        nn.init.zeros_(self.fast_b1)
        nn.init.zeros_(self.fast_b2)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        cycle_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del cycle_ids
        self._validate_inputs(inputs, mask)
        input_dtype = inputs.dtype
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            queries, keys, values, aligned_mask, valid_rank = self._prepare_inputs(
                inputs, mask
            )
            update_keys, targets, update_mask = self._reconstruction_objective(
                keys, values, aligned_mask
            )
            state = self._expand_state(
                self.fast_w1,
                self.fast_b1,
                self.fast_w2,
                self.fast_b2,
                inputs.shape[0],
            )
            residual, _ = self._run_expert(
                queries,
                update_keys,
                targets,
                update_mask,
                aligned_mask,
                state,
                self.chunk_size,
                self.inner_learning_rate,
            )
            mixed = queries + residual
            return self._restore_output(mixed, valid_rank, mask, input_dtype)


class TTTMultiscaleMoE(_TTTMLPBase):
    """Cycle-aware TTT mixer formed by partitioning one fast MLP rank budget.

    The short expert models base-token high-frequency variation. The long expert
    operates on contiguous cycle summaries. Its gated contribution is centered
    over valid observations so that the query path retains absolute health.
    Both experts partition one fixed hidden rank.
    """

    def __init__(self, d_model: int, nhead: int, config: dict[str, Any]) -> None:
        super().__init__(d_model, nhead, config)
        self.hidden_dim = max(
            1, int(round(self.head_dim * float(config["hidden_multiplier"])))
        )
        if self.hidden_dim < 2:
            raise ValueError(
                "TTTMultiscaleMoE requires at least two hidden channels per head"
            )
        multiscale = config["multiscale"]
        requested_ratio = float(multiscale["short_rank_ratio"])
        self.short_hidden_dim = int(round(self.hidden_dim * requested_ratio))
        self.short_hidden_dim = min(max(1, self.short_hidden_dim), self.hidden_dim - 1)
        self.short_rank_ratio = self.short_hidden_dim / self.hidden_dim
        self.short_chunk_size = int(config["chunk_size"])
        self.long_update_interval = int(multiscale["long_update_interval"])
        self.short_learning_rate = float(config["inner_learning_rate"])
        self.long_learning_rate = float(multiscale["long_inner_learning_rate"])
        self.long_ema_decay = float(multiscale["long_ema_decay"])
        self.center_long_residual = bool(multiscale["center_long_residual"])

        self.fast_w1 = nn.Parameter(torch.empty(nhead, self.head_dim, self.hidden_dim))
        self.fast_b1 = nn.Parameter(torch.zeros(nhead, self.hidden_dim))
        self.fast_w2 = nn.Parameter(torch.empty(nhead, self.hidden_dim, self.head_dim))
        self.fast_b2 = nn.Parameter(torch.zeros(nhead, self.head_dim))
        self.gate_weight = nn.Parameter(torch.zeros(nhead, self.head_dim))
        self.gate_bias = nn.Parameter(torch.empty(nhead))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self._reset_common_parameters()
        nn.init.xavier_uniform_(self.fast_w1)
        nn.init.xavier_uniform_(self.fast_w2, gain=0.25)
        nn.init.zeros_(self.fast_b1)
        nn.init.zeros_(self.fast_b2)
        nn.init.zeros_(self.gate_weight)
        prior_logit = math.log(self.short_rank_ratio / (1.0 - self.short_rank_ratio))
        nn.init.constant_(self.gate_bias, prior_logit)

    @staticmethod
    def _align_cycle_ids(
        cycle_ids: torch.Tensor,
        mask: torch.Tensor,
        valid_rank: torch.Tensor,
        aligned_mask: torch.Tensor,
    ) -> torch.Tensor:
        aligned = torch.zeros_like(cycle_ids)
        aligned.scatter_add_(1, valid_rank, cycle_ids.clamp_min(0) * mask)
        return aligned.masked_fill(~aligned_mask, -1)

    def _cycle_slow_clock(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        cycle_ids: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Aggregate contiguous tokens on the physical engine-cycle clock."""
        previous_valid = torch.cat(
            (torch.zeros_like(valid_mask[:, :1]), valid_mask[:, :-1]), dim=1
        )
        previous_cycle = torch.cat((cycle_ids[:, :1], cycle_ids[:, :-1]), dim=1)
        boundary = valid_mask & (~previous_valid | (cycle_ids != previous_cycle))
        group_index = boundary.long().cumsum(dim=1).sub(1).clamp_min(0)
        group_count = int(boundary.sum(dim=1).max().item())
        if group_count < 1:
            raise ValueError("Every TTT sample must contain at least one cycle")

        index = group_index[:, :, None, None].expand(
            -1, -1, values.shape[2], values.shape[3]
        )
        valid = valid_mask[:, :, None, None].to(values.dtype)
        cycle_sum = values.new_zeros(
            values.shape[0], group_count, values.shape[2], values.shape[3]
        )
        cycle_sum.scatter_add_(1, index, values * valid)
        durations = values.new_zeros(values.shape[0], group_count)
        durations.scatter_add_(1, group_index, valid_mask.to(values.dtype))
        cycle_mask = durations > 0
        cycle_mean = cycle_sum / durations.clamp_min(1)[:, :, None, None]
        if self.long_ema_decay == 0.0:
            return cycle_mean, cycle_mask, group_index

        cycle_numbers = cycle_ids.new_zeros(values.shape[0], group_count)
        cycle_numbers.scatter_add_(1, group_index, cycle_ids.clamp_min(0) * boundary)
        previous_numbers = torch.cat(
            (torch.zeros_like(cycle_numbers[:, :1]), cycle_numbers[:, :-1]), dim=1
        )
        transition_steps = (cycle_numbers - previous_numbers).clamp_min(1)

        state = torch.zeros_like(cycle_mean[:, 0])
        initialized = torch.zeros(
            values.shape[0], device=values.device, dtype=torch.bool
        )
        slow_states = []
        decay = values.new_tensor(self.long_ema_decay)
        for index_value in range(group_count):
            active = cycle_mask[:, index_value]
            gap = transition_steps[:, index_value]
            cycle_decay = decay.pow(gap.to(values.dtype))[:, None, None]
            current_mean = cycle_mean[:, index_value]
            updated = cycle_decay * state + (1.0 - cycle_decay) * current_mean
            updated = torch.where(initialized[:, None, None], updated, current_mean)
            state = torch.where(active[:, None, None], updated, state)
            slow_states.append(state * active[:, None, None])
            initialized = initialized | active
        return (
            torch.stack(slow_states, dim=1),
            cycle_mask,
            group_index,
        )

    @staticmethod
    def _broadcast_slow(
        values: torch.Tensor,
        group_index: torch.Tensor,
        base_mask: torch.Tensor,
    ) -> torch.Tensor:
        index = group_index[:, :, None, None].expand(
            -1, -1, values.shape[2], values.shape[3]
        )
        return values.gather(1, index) * base_mask[:, :, None, None]

    @staticmethod
    def _center_valid_sequence(
        residual: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        weight = valid_mask[:, :, None, None].to(residual.dtype)
        mean = (residual * weight).sum(dim=1)
        mean = mean / valid_mask.sum(dim=1).clamp_min(1)[:, None, None]
        return (residual - mean[:, None]) * weight

    def _expert_state(
        self, hidden_slice: slice, output_bias_scale: float, batch_size: int
    ) -> _FastMLPState:
        return self._expand_state(
            self.fast_w1[:, :, hidden_slice],
            self.fast_b1[:, hidden_slice],
            self.fast_w2[:, hidden_slice, :],
            self.fast_b2 * output_bias_scale,
            batch_size,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        cycle_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_inputs(inputs, mask)
        if cycle_ids is None or cycle_ids.shape != mask.shape:
            raise ValueError("Cycle-aware MoE-TTT requires cycle_ids with shape [B,L]")
        if torch.any(cycle_ids[mask] < 0):
            raise ValueError("Valid TTT tokens require non-negative cycle_ids")
        input_dtype = inputs.dtype
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            queries, keys, values, aligned_mask, valid_rank = self._prepare_inputs(
                inputs, mask
            )
            aligned_cycles = self._align_cycle_ids(
                cycle_ids, mask, valid_rank, aligned_mask
            )
            cycle_delta = aligned_cycles[:, 1:] - aligned_cycles[:, :-1]
            adjacent_valid = aligned_mask[:, 1:] & aligned_mask[:, :-1]
            if torch.any(cycle_delta[adjacent_valid] < 0):
                raise ValueError("cycle_ids must be non-decreasing within a window")
            combined = torch.cat((queries, keys, values), dim=-1)
            (
                slow_components,
                slow_mask,
                group_index,
            ) = self._cycle_slow_clock(combined, aligned_mask, aligned_cycles)
            long_broadcast = self._broadcast_slow(
                slow_components, group_index, aligned_mask
            )
            short_components = combined - long_broadcast
            short_queries, short_keys, short_values = short_components.split(
                self.head_dim, dim=-1
            )
            long_queries, long_keys, long_values = slow_components.split(
                self.head_dim, dim=-1
            )
            short_update_keys, short_targets, short_update_mask = (
                self._reconstruction_objective(short_keys, short_values, aligned_mask)
            )
            long_update_keys, long_targets, long_update_mask = (
                self._reconstruction_objective(long_keys, long_values, slow_mask)
            )

            short_prior = self.short_rank_ratio
            short_state = self._expert_state(
                slice(0, self.short_hidden_dim), short_prior, inputs.shape[0]
            )
            long_state = self._expert_state(
                slice(self.short_hidden_dim, None),
                1.0 - short_prior,
                inputs.shape[0],
            )
            short_residual, _ = self._run_expert(
                short_queries,
                short_update_keys,
                short_targets,
                short_update_mask,
                aligned_mask,
                short_state,
                self.short_chunk_size,
                self.short_learning_rate,
            )
            slow_residual, _ = self._run_expert(
                long_queries,
                long_update_keys,
                long_targets,
                long_update_mask,
                slow_mask,
                long_state,
                self.long_update_interval,
                self.long_learning_rate,
            )
            long_residual = self._broadcast_slow(
                slow_residual, group_index, aligned_mask
            )

            gate_logits = torch.einsum(
                "bshd,hd->bsh", queries, self.gate_weight
            ) / math.sqrt(self.head_dim)
            gate = torch.sigmoid(gate_logits + self.gate_bias)
            has_cycle_transition = slow_mask.sum(dim=1) > 1
            gate = torch.where(
                has_cycle_transition[:, None, None],
                gate,
                gate.new_full((), short_prior),
            )
            short_scale = (gate / short_prior).unsqueeze(-1)
            long_scale = ((1.0 - gate) / (1.0 - short_prior)).unsqueeze(-1)
            long_contribution = long_scale * long_residual
            long_contribution = (
                long_contribution * has_cycle_transition[:, None, None, None]
            )
            if self.center_long_residual:
                long_contribution = self._center_valid_sequence(
                    long_contribution, aligned_mask
                )
            residual = short_scale * short_residual + long_contribution
            mixed = queries + residual
            return self._restore_output(mixed, valid_rank, mask, input_dtype)
