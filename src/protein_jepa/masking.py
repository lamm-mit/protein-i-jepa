from __future__ import annotations

import torch


def sample_span_mask(
    lengths: torch.Tensor,
    max_length: int,
    *,
    mask_fraction: float = 0.25,
    min_span: int = 4,
    max_span: int = 32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not 0.0 < mask_fraction < 1.0:
        raise ValueError("mask_fraction must be between 0 and 1.")
    if min_span < 1 or max_span < min_span:
        raise ValueError("Span bounds must satisfy 1 <= min_span <= max_span.")

    device = lengths.device
    target_mask = torch.zeros((lengths.numel(), max_length), dtype=torch.bool, device=device)
    for batch_index, raw_length in enumerate(lengths.tolist()):
        length = int(raw_length)
        if length < 2:
            continue
        desired = max(1, int(round(length * mask_fraction)))
        attempts = 0
        while int(target_mask[batch_index].sum().item()) < desired and attempts < desired * 8:
            attempts += 1
            span_high = min(max_span, length)
            span_low = min(min_span, span_high)
            span_length = _randint(span_low, span_high + 1, generator=generator, device=device)
            start = _randint(0, length - span_length + 1, generator=generator, device=device)
            target_mask[batch_index, start : start + span_length] = True
        if not bool(target_mask[batch_index, :length].any()):
            start = _randint(0, length, generator=generator, device=device)
            target_mask[batch_index, start] = True
        target_mask[batch_index, length:] = False
    return target_mask


def _randint(
    low: int,
    high: int,
    *,
    generator: torch.Generator | None,
    device: torch.device,
) -> int:
    value = torch.randint(low, high, (1,), generator=generator, device=device)
    return int(value.item())


def make_context_inputs(input_ids: torch.Tensor, target_mask: torch.Tensor, *, mask_id: int) -> torch.Tensor:
    if input_ids.shape != target_mask.shape:
        raise ValueError("input_ids and target_mask must have the same shape.")
    context_input_ids = input_ids.clone()
    context_input_ids[target_mask] = mask_id
    return context_input_ids

