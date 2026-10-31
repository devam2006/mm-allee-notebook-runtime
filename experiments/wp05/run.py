"""CLI runner for WP05 staged inverse parameter estimation, ODE baseline, and noise sweeps.

Shared JAX neural engine:
    python experiments/wp05/run.py --config configs/wp05_smoke.json --output results/wp05
"""

import argparse
import csv
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mm_allee.data import generate_noisy_observations  # noqa: E402
from mm_allee.inference import fit_ode_least_squares  # noqa: E402
from mm_allee.metrics import errors  # noqa: E402
from mm_allee.validation import (  # noqa: E402
    audit_splits_and_leakage,
    compute_parameter_metrics,
    plot_heldout_failure,
    plot_parameter_recovery_noise,
    write_parameter_metrics_csv,
    write_prediction_metrics_csv,
)

# Optional JAX import
try:
    import jax  # noqa: F401  (availability check)

    from mm_allee.pinn.inverse import InversePINNTrainer, TrainableParameters
    from mm_allee.pinn.network import NetworkConfig

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load clean reference times and populations from CSV."""
    times = []
    pops = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time"]))
            pops.append(float(row["population"]))
    return np.array(times), np.array(pops)


def run_experiment(config_path: Path, output_dir: Path) -> dict:
    start_wall_time = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load separate evaluation truth metadata (truth.json)
    truth_file = REPO_ROOT / "data" / "fixtures" / "wp03_reference" / "truth.json"
    if not truth_file.exists():
        raise FileNotFoundError(f"Required evaluation truth file not found: {truth_file}")
    truth_data = json.loads(truth_file.read_text(encoding="utf-8")).get("trajectories", {})

    cases = config["cases"]
    sigmas = config["noise_sigmas"]
    seeds = config["seeds"]
    n_ref = float(config.get("fixed_reference_scales", {}).get("population", 100.0))
    fit_seed_offset = int(config.get("fit_seed_offset", 1000))

    param_metric_records = []
    pred_metric_records = []
    trajectory_runs_for_plot = []

    print(f"Starting WP05 experiment: {config.get('scope', 'Inverse inference')}")
    print(f"Cases: {len(cases)}, Noise levels: {sigmas}, Seeds: {seeds}, JAX available: {HAS_JAX}")

    run_counter = 0

    for case in cases:
        case_id = case["case_id"]
        model = case["model"]
        regime = case["regime"]
        ref_csv = REPO_ROOT / case["reference_csv"]
        fit_params = case["fit_params"]
        known_cond = case["known_conditions"]
        init_guess = case.get("initial_guess", {})

        ref_times, ref_pops = load_reference_csv(ref_csv)

        # Ground truth parameters for evaluation ONLY (never passed to trainer)
        if case_id not in truth_data:
            raise KeyError(
                f"Evaluation ground truth metadata missing for case '{case_id}' in {truth_file}"
            )
        p_true = truth_data[case_id]["parameters"]

        for sigma in sigmas:
            for seed in seeds:
                run_counter += 1
                noise_seed = seed
                fit_seed = fit_seed_offset + seed
                run_id = f"wp05_{case_id}_s{int(sigma * 100):02d}_seed{seed}"

                # Generate noisy observations with frozen 60/20/20 train/val/test splits
                obs = generate_noisy_observations(
                    ref_times,
                    ref_pops,
                    trajectory_id=case_id,
                    model=model,
                    noise_sigma=sigma,
                    noise_seed=noise_seed,
                    n_ref=n_ref,
                )
                audit_splits_and_leakage(obs)

                idx_test = obs["test_indices"]

                # ==============================================================
                # 1. Conventional ODE Least-Squares Baseline
                # ==============================================================
                ode_res = fit_ode_least_squares(
                    model,
                    observations=obs,
                    known_conditions=known_cond,
                    fit_params=fit_params,
                    initial_guess=init_guess,
                    n_ref=n_ref,
                    max_nfev=config.get("ode_settings", {}).get("max_nfev", 300),
                )

                # Parameter metrics for ODE
                ode_param_metrics = compute_parameter_metrics(
                    p_true, ode_res.estimated_parameters, fit_params
                )
                for pm in ode_param_metrics:
                    param_metric_records.append(
                        {
                            "run_id": run_id,
                            "model": model,
                            "regime": regime,
                            "method": "ODE_LS",
                            "noise_sigma": sigma,
                            "seed": seed,
                            "noise_seed": noise_seed,
                            "fit_seed": fit_seed,
                            "parameter": pm["parameter"],
                            "true_value": pm["true_value"],
                            "estimated_value": pm["estimated_value"],
                            "absolute_error": pm["absolute_error"],
                            "relative_error": pm["relative_error"],
                            "status": ode_res.status,
                        }
                    )

                # Prediction metrics against both noisy observations and clean reference
                ode_pred_pop = ode_res.predicted_trajectory["population"]
                ode_clean_test_err = (
                    errors(ode_pred_pop[idx_test], ref_pops[idx_test])
                    if len(idx_test) > 0
                    else {"rmse": float("nan"), "mae": float("nan"), "relative_l2": None}
                )

                pred_metric_records.append(
                    {
                        "run_id": run_id,
                        "model": model,
                        "regime": regime,
                        "method": "ODE_LS",
                        "noise_sigma": sigma,
                        "seed": seed,
                        "noise_seed": noise_seed,
                        "fit_seed": fit_seed,
                        "train_rmse": ode_res.train_metrics["rmse"],
                        "val_rmse": ode_res.val_metrics["rmse"],
                        "test_rmse": ode_res.test_metrics["rmse"],
                        "test_mae": ode_res.test_metrics["mae"],
                        "test_relative_l2": ode_res.test_metrics["relative_l2"],
                        "test_clean_rmse": ode_clean_test_err["rmse"],
                        "test_clean_mae": ode_clean_test_err["mae"],
                        "test_clean_relative_l2": ode_clean_test_err["relative_l2"],
                        "wall_seconds": ode_res.wall_seconds,
                        "status": ode_res.status,
                    }
                )

                if seed == seeds[0]:
                    trajectory_runs_for_plot.append(
                        {
                            "case_id": case_id,
                            "model": model,
                            "regime": regime,
                            "method": "ODE_LS",
                            "noise_sigma": sigma,
                            "time": ode_res.predicted_trajectory["time"],
                            "pred_population": ode_res.predicted_trajectory["population"],
                            "predicted_population": ode_res.predicted_trajectory["population"],
                            "ref_population": ref_pops,
                            "obs_time": np.asarray(obs["time"]),
                            "obs_population": np.asarray(obs["observed_population"]),
                            "observed_population": np.asarray(obs["observed_population"]),
                            "obs_split": np.asarray(obs["split"]),
                            "train_indices": obs["train_indices"],
                            "val_indices": obs["val_indices"],
                            "test_indices": obs["test_indices"],
                        }
                    )

                # ==============================================================
                # 2. Physics-Informed Neural Network (JAX Inverse Engine)
                # ==============================================================
                if HAS_JAX:
                    pinn_cfg = config.get("pinn_settings", {})
                    hidden_layers = pinn_cfg.get("hidden_layers", [32, 32])
                    layers = tuple([1] + list(hidden_layers) + [1])
                    act = pinn_cfg.get("activation", "tanh")
                    lbfgs_iters = pinn_cfg.get("lbfgs_iters", 200)

                    # Explicit seeding before network weight construction using fit_seed
                    ncfg = NetworkConfig(
                        layers=layers,
                        t_end=float(ref_times.max()),
                        x0=float(ref_pops[0]) / n_ref,
                        seed=fit_seed,
                        activation=act,
                    )
                    tp = TrainableParameters(
                        model=model,
                        fit_params=fit_params,
                        known_conditions=known_cond,
                        initial_guesses=init_guess,
                    )
                    trainer = InversePINNTrainer(
                        network_config=ncfg,
                        trainable_params=tp,
                        lr_net=pinn_cfg.get("lr_net", 0.001),
                        lr_params=pinn_cfg.get("lr_params", 0.01),
                        lambda_data=pinn_cfg.get("lambda_data", 1.0),
                        lambda_phys=pinn_cfg.get("lambda_phys", 0.1),
                        lbfgs_iters=lbfgs_iters,
                        lbfgs_ftol=pinn_cfg.get("lbfgs_ftol", 1e-12),
                        lbfgs_gtol=pinn_cfg.get("lbfgs_gtol", 1e-8),
                        validation_selection=pinn_cfg.get("validation_selection", True),
                    )

                    try:
                        pinn_res = trainer.fit(
                            obs,
                            known_conditions=known_cond,
                            epochs=pinn_cfg.get("epochs", 1000),
                            patience=pinn_cfg.get("patience", 100),
                            n_collocation=pinn_cfg.get("n_collocation", 100),
                            seed=fit_seed,
                            eval_every=pinn_cfg.get("eval_every", 10),
                            adam_iters=pinn_cfg.get("adam_iters", None),
                            lbfgs_iters=lbfgs_iters,
                        )
                    except Exception as exc:
                        # Record failed run per seed without crashing the whole sweep
                        print(f"Warning: PINN fit failed on {run_id}: {exc}")
                        pinn_res = None
                        fail_status = f"failed_{type(exc).__name__}"

                    if pinn_res is not None:
                        pinn_param_metrics = compute_parameter_metrics(
                            p_true, pinn_res.estimated_parameters, fit_params
                        )
                        for pm in pinn_param_metrics:
                            param_metric_records.append(
                                {
                                    "run_id": run_id,
                                    "model": model,
                                    "regime": regime,
                                    "method": "PINN",
                                    "noise_sigma": sigma,
                                    "seed": seed,
                                    "noise_seed": noise_seed,
                                    "fit_seed": fit_seed,
                                    "parameter": pm["parameter"],
                                    "true_value": pm["true_value"],
                                    "estimated_value": pm["estimated_value"],
                                    "absolute_error": pm["absolute_error"],
                                    "relative_error": pm["relative_error"],
                                    "status": pinn_res.status,
                                }
                            )

                        pinn_pred_pop = pinn_res.predicted_trajectory["population"]
                        pinn_clean_test_err = (
                            errors(pinn_pred_pop[idx_test], ref_pops[idx_test])
                            if len(idx_test) > 0
                            else {"rmse": float("nan"), "mae": float("nan"), "relative_l2": None}
                        )

                        pred_metric_records.append(
                            {
                                "run_id": run_id,
                                "model": model,
                                "regime": regime,
                                "method": "PINN",
                                "noise_sigma": sigma,
                                "seed": seed,
                                "noise_seed": noise_seed,
                                "fit_seed": fit_seed,
                                "train_rmse": pinn_res.train_metrics["rmse"],
                                "val_rmse": pinn_res.val_metrics["rmse"],
                                "test_rmse": pinn_res.test_metrics["rmse"],
                                "test_mae": pinn_res.test_metrics["mae"],
                                "test_relative_l2": pinn_res.test_metrics["relative_l2"],
                                "test_clean_rmse": pinn_clean_test_err["rmse"],
                                "test_clean_mae": pinn_clean_test_err["mae"],
                                "test_clean_relative_l2": pinn_clean_test_err["relative_l2"],
                                "wall_seconds": pinn_res.wall_seconds,
                                "status": pinn_res.status,
                                "lbfgs_outcome": pinn_res.optimizer.get("lbfgs_outcome"),
                                "lbfgs_message": pinn_res.optimizer.get("error")
                                or pinn_res.optimizer.get("lbfgs_message"),
                            }
                        )

                        if seed == seeds[0]:
                            trajectory_runs_for_plot.append(
                                {
                                    "case_id": case_id,
                                    "model": model,
                                    "regime": regime,
                                    "method": "PINN",
                                    "noise_sigma": sigma,
                                    "time": pinn_res.predicted_trajectory["time"],
                                    "pred_population": pinn_res.predicted_trajectory["population"],
                                    "predicted_population": pinn_res.predicted_trajectory[
                                        "population"
                                    ],
                                    "ref_population": ref_pops,
                                    "obs_time": np.asarray(obs["time"]),
                                    "obs_population": np.asarray(obs["observed_population"]),
                                    "observed_population": np.asarray(obs["observed_population"]),
                                    "obs_split": np.asarray(obs["split"]),
                                    "train_indices": obs["train_indices"],
                                    "val_indices": obs["val_indices"],
                                    "test_indices": obs["test_indices"],
                                }
                            )
                    else:
                        for p_name in fit_params:
                            param_metric_records.append(
                                {
                                    "run_id": run_id,
                                    "model": model,
                                    "regime": regime,
                                    "method": "PINN",
                                    "noise_sigma": sigma,
                                    "seed": seed,
                                    "noise_seed": noise_seed,
                                    "fit_seed": fit_seed,
                                    "parameter": p_name,
                                    "true_value": p_true.get(p_name, float("nan")),
                                    "estimated_value": float("nan"),
                                    "absolute_error": float("nan"),
                                    "relative_error": float("nan"),
                                    "status": fail_status,
                                }
                            )
                        pred_metric_records.append(
                            {
                                "run_id": run_id,
                                "model": model,
                                "regime": regime,
                                "method": "PINN",
                                "noise_sigma": sigma,
                                "seed": seed,
                                "noise_seed": noise_seed,
                                "fit_seed": fit_seed,
                                "train_rmse": float("nan"),
                                "val_rmse": float("nan"),
                                "test_rmse": float("nan"),
                                "test_mae": float("nan"),
                                "test_relative_l2": None,
                                "test_clean_rmse": float("nan"),
                                "test_clean_mae": float("nan"),
                                "test_clean_relative_l2": None,
                                "wall_seconds": 0.0,
                                "status": fail_status,
                            }
                        )
                else:
                    for p_name in fit_params:
                        param_metric_records.append(
                            {
                                "run_id": run_id,
                                "model": model,
                                "regime": regime,
                                "method": "PINN",
                                "noise_sigma": sigma,
                                "seed": seed,
                                "noise_seed": noise_seed,
                                "fit_seed": fit_seed,
                                "parameter": p_name,
                                "true_value": p_true.get(p_name, float("nan")),
                                "estimated_value": float("nan"),
                                "absolute_error": float("nan"),
                                "relative_error": float("nan"),
                                "status": "skipped_no_jax_environment",
                            }
                        )
                    pred_metric_records.append(
                        {
                            "run_id": run_id,
                            "model": model,
                            "regime": regime,
                            "method": "PINN",
                            "noise_sigma": sigma,
                            "seed": seed,
                            "noise_seed": noise_seed,
                            "fit_seed": fit_seed,
                            "train_rmse": float("nan"),
                            "val_rmse": float("nan"),
                            "test_rmse": float("nan"),
                            "test_mae": float("nan"),
                            "test_relative_l2": None,
                            "test_clean_rmse": float("nan"),
                            "test_clean_mae": float("nan"),
                            "test_clean_relative_l2": None,
                            "wall_seconds": 0.0,
                            "status": "skipped_no_jax_environment",
                        }
                    )

    # Save output tables
    param_table_path = output_dir / "wp05_parameter_metrics.csv"
    pred_table_path = output_dir / "wp05_prediction_metrics.csv"
    write_parameter_metrics_csv(param_table_path, param_metric_records)
    write_prediction_metrics_csv(pred_table_path, pred_metric_records)

    # Generate paired figures
    fig_stem_recovery = output_dir / "wp05_recovery_noise"
    fig_stem_heldout = output_dir / "wp05_heldout_failure"
    try:
        plot_parameter_recovery_noise(param_metric_records, fig_stem_recovery)
        if trajectory_runs_for_plot:
            plot_heldout_failure(trajectory_runs_for_plot, fig_stem_heldout)
    except Exception as e:
        print(f"Warning: Figure plotting error: {e}")

    total_wall_time = time.perf_counter() - start_wall_time

    # Build manifest
    manifest = {
        "schema_version": 1,
        "run_id": f"wp05_experiment_{int(time.time())}",
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_wall_seconds": total_wall_time,
        "cases_evaluated": [c["case_id"] for c in cases],
        "noise_sigmas": sigmas,
        "seeds": seeds,
        "fit_seed_offset": fit_seed_offset,
        "methods": ["ODE_LS", "PINN" if HAS_JAX else "PINN (code ready, jax pending)"],
        "summary": {
            "total_runs": run_counter,
            "parameter_records": len(param_metric_records),
            "prediction_records": len(pred_metric_records),
        },
        "output_files": {
            "parameter_metrics": str(param_table_path.name),
            "prediction_metrics": str(pred_table_path.name),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Experiment complete in {total_wall_time:.2f}s. Outputs saved to {output_dir}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="WP05 Inverse Parameter Estimation Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    output_dir = Path(args.output).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    run_experiment(config_path, output_dir)


if __name__ == "__main__":
    main()
