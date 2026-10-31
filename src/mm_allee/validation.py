"""Validation routines, metric audits, and experiment reporting for WP05."""

import csv
from math import isfinite
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .data import validate_splits, validate_time_windows
from .plotting import save_figure


def audit_splits_and_leakage(observations: dict[str, Any]) -> None:
    """Audit split integrity, strict leakage protection, and temporal ordering."""
    obs_ids = observations["observation_id"]
    idx_train = observations["train_indices"]
    idx_val = observations["val_indices"]
    idx_test = observations["test_indices"]
    times = np.asarray(observations["time"], dtype=float)

    train_ids = [obs_ids[i] for i in idx_train]
    val_ids = [obs_ids[i] for i in idx_val]
    test_ids = [obs_ids[i] for i in idx_test]

    validate_splits(train_ids, val_ids, test_ids)
    validate_time_windows(times[idx_train], times[idx_val], times[idx_test])


def compute_parameter_metrics(
    p_true: dict[str, float],
    p_est: dict[str, float],
    fit_params: Sequence[str],
) -> list[dict[str, Any]]:
    """Compute absolute and relative parameter errors."""
    records = []
    for param in fit_params:
        true_val = float(p_true[param])
        est_val = float(p_est.get(param, float("nan")))

        abs_err = (
            abs(est_val - true_val) if isfinite(est_val) and isfinite(true_val) else float("nan")
        )
        rel_err = (
            abs_err / abs(true_val)
            if isfinite(abs_err) and true_val != 0
            else (None if true_val == 0 else float("nan"))
        )

        records.append(
            {
                "parameter": param,
                "true_value": true_val,
                "estimated_value": est_val,
                "absolute_error": abs_err,
                "relative_error": rel_err,
            }
        )
    return records


def write_parameter_metrics_csv(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Write standard WP05 parameter metrics CSV."""
    columns = [
        "run_id",
        "model",
        "regime",
        "method",
        "noise_sigma",
        "seed",
        "noise_seed",
        "fit_seed",
        "parameter",
        "true_value",
        "estimated_value",
        "absolute_error",
        "relative_error",
        "status",
    ]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in columns})


def write_prediction_metrics_csv(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Write standard WP05 trajectory prediction metrics CSV."""
    columns = [
        "run_id",
        "model",
        "regime",
        "method",
        "noise_sigma",
        "seed",
        "noise_seed",
        "fit_seed",
        "train_rmse",
        "val_rmse",
        "test_rmse",
        "test_mae",
        "test_relative_l2",
        "test_clean_rmse",
        "test_clean_mae",
        "test_clean_relative_l2",
        "wall_seconds",
        "status",
        "lbfgs_outcome",
        "lbfgs_message",
    ]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in columns})


def plot_parameter_recovery_noise(
    records: list[dict[str, Any]],
    output_stem: str | Path,
) -> list[Path]:
    """Generate publication-quality figure: Parameter recovery error vs noise for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    models = ["A", "B"]
    for ax, model_name in zip(axes, models, strict=True):
        m_recs = [r for r in records if r.get("model") == model_name]
        if not m_recs:
            ax.text(0.5, 0.5, f"No data for Model {model_name}", ha="center", va="center")
            continue

        sigmas = sorted(list({r["noise_sigma"] for r in m_recs}))
        methods = sorted(list({r["method"] for r in m_recs}))

        for method in methods:
            for param in sorted(list({r["parameter"] for r in m_recs if r["method"] == method})):
                means = []
                stds = []
                valid_sigmas = []
                for s in sigmas:
                    vals = [
                        r["relative_error"]
                        for r in m_recs
                        if r["noise_sigma"] == s
                        and r["method"] == method
                        and r["parameter"] == param
                        and r["relative_error"] is not None
                        and isfinite(r["relative_error"])
                    ]
                    if vals:
                        valid_sigmas.append(s * 100)  # as percentage
                        means.append(np.mean(vals))
                        stds.append(np.std(vals))

                if valid_sigmas:
                    marker = "o" if method == "ODE_LS" else "s"
                    ls = "-" if method == "ODE_LS" else "--"
                    label = f"{method} ({param})"
                    ax.errorbar(
                        valid_sigmas,
                        means,
                        yerr=stds,
                        marker=marker,
                        linestyle=ls,
                        capsize=4,
                        label=label,
                    )

        ax.set_title(f"Model {model_name} Parameter Recovery vs Noise")
        ax.set_xlabel("Noise level (% of $N_{ref}$)")
        ax.set_ylabel("Mean Relative Parameter Error")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    return save_figure(fig, output_stem)


def plot_heldout_failure(
    trajectory_runs: list[dict[str, Any]],
    output_stem: str | Path,
) -> list[Path]:
    """Generate publication-quality figure: Disclosed train/val/test regions

    and near-fold / extinction failure cases.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for ax, run in zip(axes.flat, trajectory_runs[:4], strict=False):
        times = np.asarray(run["time"])
        obs_pop = np.asarray(run.get("observed_population", run.get("obs_population", [])))
        pred_pop = np.asarray(run.get("predicted_population", run.get("pred_population", [])))

        idx_train = run.get("train_indices")
        idx_val = run.get("val_indices")
        idx_test = run.get("test_indices")
        if idx_train is None and "obs_split" in run:
            splits = np.asarray(run["obs_split"])
            idx_train = np.where(splits == "train")[0]
            idx_val = np.where(splits == "val")[0]
            idx_test = np.where(splits == "test")[0]

        regime = run.get("regime", run.get("case_id", ""))
        title = f"{run['model']} - {regime} ({run['method']})"

        # Background color for train/val/test splits
        if idx_train is not None and len(idx_train) > 0:
            t_train_max = float(times[idx_train[-1]])
            ax.axvspan(0, t_train_max, color="blue", alpha=0.08, label="Train split (60%)")
        if idx_val is not None and len(idx_val) > 0:
            t_train_max = (
                float(times[idx_train[-1]]) if idx_train is not None and len(idx_train) > 0 else 0
            )
            t_val_max = float(times[idx_val[-1]])
            ax.axvspan(t_train_max, t_val_max, color="green", alpha=0.08, label="Val split (20%)")
        if idx_test is not None and len(idx_test) > 0:
            t_val_max = float(times[idx_val[-1]]) if idx_val is not None and len(idx_val) > 0 else 0
            t_max = float(times[-1])
            ax.axvspan(t_val_max, t_max, color="red", alpha=0.08, label="Held-out Test (20%)")

        if len(obs_pop) > 0:
            obs_times = np.asarray(run.get("obs_time", times))
            ax.scatter(
                obs_times, obs_pop, s=15, color="black", alpha=0.6, label="Noisy Observations"
            )
        if len(pred_pop) > 0:
            ax.plot(times, pred_pop, color="crimson", lw=2, label="Fitted Trajectory")

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time")
        ax.set_ylabel("Population")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    plt.tight_layout()
    return save_figure(fig, output_stem)
