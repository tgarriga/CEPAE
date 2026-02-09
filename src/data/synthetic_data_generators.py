from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _sample_trend(rng: np.random.Generator) -> float:
    # Symmetric range around 0.
    return float(rng.uniform(-0.1, 0.1))


def _sample_secondary_step(rng: np.random.Generator, seq_len: int) -> int:
    """Sample an additional change-point late in the sequence."""
    if seq_len < 3:
        return 1
    low = max(1, int(0.7 * seq_len))
    high = seq_len - 1
    if low >= high:
        low = 1
        high = seq_len - 1
    return int(rng.integers(low, high))


def create_dataset(
    n: int,
    seq_len: int,
    key_step: int,
    uniform_change: float = 0.7,
    scale_param: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a simple synthetic dataset with a binary event.

    The dataset is balanced: half the series receive an additive drop at ``key_step``
    (event=1) and the other half do not (event=0). All series also include a second
    random change-point near the end of the sequence.

    Args:
        n: Number of series.
        seq_len: Length of each series.
        key_step: Index at which the event drop is applied for event=1 series.
        uniform_change: Magnitude bound for the secondary change-point.
        scale_param: Standard deviation of Gaussian noise added to each step.
        seed: Optional RNG seed for reproducibility.

    Returns:
        labels: Array of shape (n, 1) with values in {0, 1}.
        data: Array of shape (n, seq_len, 1).
    """
    if not (0 <= key_step < seq_len):
        raise ValueError(f"key_step must be in [0, {seq_len-1}]")

    rng = _rng(seed)
    level = 0.0
    drop = 0.7

    data = np.empty((n, seq_len), dtype=np.float32)
    labels = np.zeros((n, 1), dtype=np.float32)
    labels[n // 2 :] = 1.0

    for i in range(n):
        trend = _sample_trend(rng)
        change = float(rng.uniform(-uniform_change, uniform_change))
        key_step2 = _sample_secondary_step(rng, seq_len)

        step = level
        ts = np.empty(seq_len, dtype=np.float32)
        for j in range(seq_len):
            ts[j] = step
            step += trend

            # Event drop for the second half of the dataset.
            if (i >= n // 2) and (j == key_step):
                step -= drop

            # Secondary change-point (applies to all series).
            if j == key_step2:
                step -= change

        if scale_param > 0:
            ts += rng.normal(0.0, scale_param, size=seq_len).astype(np.float32)

        data[i] = ts

    return labels, data[..., None]


def create_dataset_counterfactuals(
    n: int,
    seq_len: int,
    key_step: int,
    uniform_change: float = 0.7,
    scale_param: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate paired potential outcomes (y0, y1) for the unconfounded setting.

    For each unit, ``y0`` is generated without the event drop and ``y1`` with the
    event drop at ``key_step``. Both share the same latent trend, secondary change
    point, and noise, so the only systematic difference is the event intervention.

    Returns:
        data_0: Potential outcomes under no event, shape (n, seq_len, 1).
        data_1: Potential outcomes under event, shape (n, seq_len, 1).
    """
    if not (0 <= key_step < seq_len):
        raise ValueError(f"key_step must be in [0, {seq_len-1}]")

    rng = _rng(seed)
    level = 0.0
    drop = 0.7

    data_0 = np.empty((n, seq_len), dtype=np.float32)
    data_1 = np.empty((n, seq_len), dtype=np.float32)

    for i in range(n):
        trend = _sample_trend(rng)
        change = float(rng.uniform(-uniform_change, uniform_change))
        key_step2 = _sample_secondary_step(rng, seq_len)

        step0 = level
        step1 = level
        ts0 = np.empty(seq_len, dtype=np.float32)
        ts1 = np.empty(seq_len, dtype=np.float32)

        for j in range(seq_len):
            ts0[j] = step0
            ts1[j] = step1
            step0 += trend
            step1 += trend

            if j == key_step:
                step1 -= drop
            if j == key_step2:
                step0 -= change
                step1 -= change

        if scale_param > 0:
            noise = rng.normal(0.0, scale_param, size=seq_len).astype(np.float32)
            ts0 += noise
            ts1 += noise

        data_0[i] = ts0
        data_1[i] = ts1

    return data_0[..., None], data_1[..., None]


def create_dataset_confounded(
    n: int,
    seq_len: int,
    key_step: int,
    uniform_change: float = 0.7,
    scale_param: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a confounded dataset where treatment depends on a latent trend.

    The probability of an event at ``key_step`` depends monotonically on the sampled
    trend. This induces confounding between the event indicator and outcomes.

    Returns:
        labels: Event indicator, shape (n, 1).
        data: Observed outcomes, shape (n, seq_len, 1).
    """
    if not (0 <= key_step < seq_len):
        raise ValueError(f"key_step must be in [0, {seq_len-1}]")

    rng = _rng(seed)
    level = 0.0
    drop = 0.7

    data = np.empty((n, seq_len), dtype=np.float32)
    labels = np.empty((n, 1), dtype=np.float32)

    for i in range(n):
        trend = _sample_trend(rng)
        change = float(rng.uniform(-uniform_change, uniform_change))
        key_step2 = _sample_secondary_step(rng, seq_len)

        # Probability of event increases with trend (trend in [-0.1, 0.1]).
        drop_prob = (trend + 0.1) / 0.2
        is_drop = float(rng.binomial(1, drop_prob))

        step = level
        ts = np.empty(seq_len, dtype=np.float32)
        for j in range(seq_len):
            ts[j] = step
            step += trend

            if j == key_step:
                step -= drop * is_drop
            if j == key_step2:
                step -= change

        if scale_param > 0:
            ts += rng.normal(0.0, scale_param, size=seq_len).astype(np.float32)

        data[i] = ts
        labels[i, 0] = is_drop

    return labels, data[..., None]


def create_dataset_counterfactuals_confounded(
    n: int,
    seq_len: int,
    key_step: int,
    uniform_change: float = 0.7,
    scale_param: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate paired potential outcomes (y0, y1) for the confounded setting.

    This matches :func:`create_dataset_counterfactuals`. The confounding mechanism
    is handled by :func:`create_dataset_confounded` when sampling observed labels.

    Returns:
        data_0: Potential outcomes under no event, shape (n, seq_len, 1).
        data_1: Potential outcomes under event, shape (n, seq_len, 1).
    """
    return create_dataset_counterfactuals(
        n=n,
        seq_len=seq_len,
        key_step=key_step,
        uniform_change=uniform_change,
        scale_param=scale_param,
        seed=seed,
    )
