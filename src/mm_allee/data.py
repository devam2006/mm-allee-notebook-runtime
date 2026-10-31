"""Observation splitting, noise generation, and strict leakage auditing for WP05."""

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ObservationSplit:
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]


def validate_splits(train_ids: Sequence, val_ids: Sequence, test_ids: Sequence) -> None:
    """Validate that train, val, and test splits are non-empty, have no internal duplicates,

    and do not overlap.
    """
    groups = [list(ids) for ids in (train_ids, val_ids, test_ids)]
    if any(len(ids) == 0 for ids in groups):
        raise ValueError("Training, validation and test splits must be nonempty")
    if any(len(set(ids)) != len(ids) for ids in groups):
        raise ValueError("Duplicate observation IDs in a split")
    train, val, test = map(set, groups)
    if train & val or train & test or val & test:
        raise ValueError("Observation IDs overlap across splits")


def validate_time_windows(
    train_times: Sequence[float],
    val_times: Sequence[float],
    test_times: Sequence[float],
) -> None:
    """Validate that contiguous time splits are strictly ordered:

    max(train) <= min(val) and max(val) <= min(test).
    """
    for name, times in (("Train", train_times), ("Val", val_times), ("Test", test_times)):
        if len(times) == 0:
            raise ValueError(f"{name} time split must be nonempty")
        if not np.all(np.isfinite(times)):
            raise ValueError(f"{name} times must be finite")
        if np.any(np.diff(times) < 0):
            raise ValueError(f"{name} times must be non-decreasing")

    if np.max(train_times) > np.min(val_times):
        raise ValueError("Train times must precede or meet validation times")
    if np.max(val_times) > np.min(test_times):
        raise ValueError("Validation times must precede or meet test times")


def create_contiguous_splits(
    n_total: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create contiguous indices for 60/20/20 train/val/test splits."""
    if n_total < 3:
        raise ValueError("At least 3 observations required for train/val/test splitting")
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    n_train = max(1, int(round(n_total * train_ratio)))
    n_val = max(1, int(round(n_total * val_ratio)))
    # Ensure all splits are non-empty
    if n_train + n_val >= n_total:
        n_train = max(1, n_total - 2)
        n_val = 1
    n_test = n_total - n_train - n_val
    if n_test <= 0:
        n_test = 1
        n_train -= 1

    idx_train = np.arange(0, n_train)
    idx_val = np.arange(n_train, n_train + n_val)
    idx_test = np.arange(n_train + n_val, n_total)
    return idx_train, idx_val, idx_test


def generate_noisy_observations(
    reference_times: np.ndarray,
    reference_populations: np.ndarray,
    *,
    trajectory_id: str,
    model: str,
    noise_sigma: float,
    noise_seed: int,
    n_ref: float = 100.0,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> dict[str, np.ndarray | list[str]]:
    """Generate noisy fitting observations adhering to the WP05 contract:

    N_obs = N_ref_clean + sigma * N_ref * epsilon, epsilon ~ Normal(0, 1).
    Observations are never silently clipped to preserve measurement noise statistics.
    Returns observation dictionary with unique IDs and frozen train/val/test split labels.
    """
    times = np.asarray(reference_times, dtype=float)
    clean_pop = np.asarray(reference_populations, dtype=float)
    if times.shape != clean_pop.shape:
        raise ValueError("reference_times and reference_populations must have matching shapes")
    if not isfinite(noise_sigma) or noise_sigma < 0:
        raise ValueError("noise_sigma must be finite and nonnegative")
    if not isfinite(n_ref) or n_ref <= 0:
        raise ValueError("n_ref must be finite and positive")

    n_total = len(times)
    rng = np.random.default_rng(noise_seed)
    noise = rng.normal(loc=0.0, scale=noise_sigma * n_ref, size=n_total)
    observed_population = clean_pop + noise

    idx_train, idx_val, idx_test = create_contiguous_splits(
        n_total, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
    )

    observation_ids = [f"{trajectory_id}:{i:05d}" for i in range(n_total)]
    split_labels = np.empty(n_total, dtype=object)
    split_labels[idx_train] = "train"
    split_labels[idx_val] = "val"
    split_labels[idx_test] = "test"

    train_ids = [observation_ids[i] for i in idx_train]
    val_ids = [observation_ids[i] for i in idx_val]
    test_ids = [observation_ids[i] for i in idx_test]
    validate_splits(train_ids, val_ids, test_ids)
    validate_time_windows(times[idx_train], times[idx_val], times[idx_test])

    return {
        "observation_id": observation_ids,
        "trajectory_id": [trajectory_id] * n_total,
        "model": [model] * n_total,
        "time": times,
        "observed_population": observed_population,
        "split": split_labels,
        "noise_sigma": float(noise_sigma),
        "noise_seed": int(noise_seed),
        "is_synthetic": True,
        "train_indices": idx_train,
        "val_indices": idx_val,
        "test_indices": idx_test,
    }
