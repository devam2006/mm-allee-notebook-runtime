"""WP04 forward-PINN experiment: PINN vs a tight reference solver for Models A and B.

    python experiments/wp04/run.py --config configs/wp04_smoke.json --output results/wp04

Importing this module trains nothing; all work happens inside ``run``.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mm_allee.models import rhs  # noqa: E402
from mm_allee.solver import solve_trajectory  # noqa: E402

METRIC_COLUMNS = [
    "run_id",
    "case",
    "model",
    "regime",
    "method",
    "seed",
    "split",
    "noise_sigma",
    "rmse",
    "mae",
    "relative_l2",
    "max_abs",
    "max_abs_near_zero",
    "boundary_violations",
    "wall_seconds",
    "status",
    "final_loss",
    "iterations",
    "lbfgs_status",
    "lbfgs_message",
    "lbfgs_iterations",
    "note",
    "t_end",
    "extinction_tau",
]


def reference_trajectory(case, ref):
    """Independent DOP853 reference on a dense grid; A stops exactly at its zero crossing.

    Returns ``(tau, x, extinction_tau)``. No sample is produced after the event, so the
    original ODE residual and every error metric stay on the physical interval.
    """
    a, h, x0 = case["a"], case["harvest"], case["x0"]
    model = case["model"]

    def field_(tau, state):
        return [rhs(state[0], model, a=a, harvest=h)]

    constant_yield = model == "A" and h > 0

    def extinction(tau, state):
        return state[0]

    extinction.terminal = True
    extinction.direction = -1
    sol = solve_ivp(
        field_,
        (0.0, case["tau_end"]),
        [x0],
        method=ref["method"],
        rtol=ref["rtol"],
        atol=ref["atol"],
        dense_output=True,
        events=extinction if constant_yield else None,
    )
    if not sol.success:
        raise RuntimeError(f"Reference solve failed: {sol.message}")
    hit = float(sol.t_events[0][0]) if constant_yield and len(sol.t_events[0]) else None
    end = case["tau_end"] if hit is None else hit
    tau = np.linspace(0.0, end, ref["n_points"])
    x = np.maximum(sol.sol(tau)[0], 0.0)
    if hit is not None:
        x[-1] = 0.0
    return tau, x, hit


def run_pinn(case, seed, cfg, extinction_tau):
    from mm_allee.pinn.train_forward import ForwardConfig, train_forward

    allowed = {f.name for f in fields(ForwardConfig)}
    settings = {k: v for k, v in cfg["training"].items() if k in allowed}
    t_end = case["tau_end"] if extinction_tau is None else extinction_tau
    config = ForwardConfig(
        model=case["model"],
        a=case["a"],
        harvest=case["harvest"],
        x0=case["x0"],
        t_end=t_end,
        seed=seed,
        **settings,
    )
    return train_forward(config)


def label_status(train_status, ev, threshold):
    """Outcome label of one scored run.

    A training status other than ``ok`` is kept. A prediction with NaN/inf is
    ``invalid_prediction`` (it has no error score and never counts as accurate). Otherwise the
    run is ``ok`` or ``inaccurate`` against ``threshold`` on the relative L2 error, falling
    back to the maximum absolute error when the relative error is undefined (tiny reference).
    """
    if train_status != "ok":
        return train_status
    if not ev["valid"]:
        return "invalid_prediction"
    score = ev["max_abs"] if ev["relative_l2"] is None else ev["relative_l2"]
    return "ok" if score <= threshold else "inaccurate"


def failed_eval():
    """Scores of a run that produced no prediction (it raised an exception)."""
    return {
        "valid": False, "n_invalid": 0, "rmse": None, "mae": None, "relative_l2": None,
        "max_abs": None, "max_abs_near_zero": None, "boundary_violations": None,
    }  # fmt: skip


def fit_and_score(case, seed, cfg, ref, hit):
    """Train one PINN and score it against the reference; never lets one run end the sweep.

    Returns ``(result, pred, ev, status, note)``. An exception during training or scoring gives
    ``result=None``, ``status="failed"`` and the exception text in ``note``.
    """
    from mm_allee.pinn.train_forward import evaluate_prediction, predict

    tau, ref_x = ref
    try:
        result = run_pinn(case, seed, cfg, hit)
        pred = predict(result.params, result.config, tau)
        ev = evaluate_prediction(pred, ref_x)
    except Exception as exc:  # noqa: BLE001  (a failed run is recorded, not fatal)
        return None, None, failed_eval(), "failed", f"{type(exc).__name__}: {exc}"
    status = label_status(result.status, ev, cfg["inaccuracy_threshold_relative_l2"])
    note = f"{ev['n_invalid']} non-finite predicted samples" if not ev["valid"] else None
    return result, pred, ev, status, note


def run(config_path, output, artifacts=None):
    from mm_allee.pinn.train_forward import evaluate_prediction, save_checkpoint

    config_path = Path(config_path)
    cfg = json.loads(config_path.read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows, curves = [], []
    for case in cfg["cases"]:
        tau, ref_x, hit = reference_trajectory(case, cfg["reference"])
        # Classical baseline (project RK45 wrapper, default tolerances) on the same grid.
        t0 = time.perf_counter()
        rk = solve_trajectory(
            case["model"],
            a=case["a"],
            harvest=case["harvest"],
            x0=case["x0"],
            tau_end=case["tau_end"],
            n_points=cfg["reference"]["n_points"],
        )
        rk_seconds = time.perf_counter() - t0
        rk_x = np.interp(tau, rk.tau, rk.x)
        rows.append(
            metric_row(case, "rk45", None, evaluate_prediction(rk_x, ref_x), rk_seconds, "ok", hit)
        )
        for seed in cfg["seeds"]:
            result, pred, ev, status, note = fit_and_score(case, seed, cfg, (tau, ref_x), hit)
            seconds = 0.0 if result is None else result.wall_seconds
            rows.append(metric_row(case, "pinn", seed, ev, seconds, status, hit, result, note))
            if result is None:
                continue
            curves.append(
                {
                    "case": case["id"],
                    "seed": seed,
                    "tau": tau,
                    "ref": ref_x,
                    "pred": pred,
                    "history": result.history,
                }
            )
            if cfg.get("save_checkpoints", True):
                save_checkpoint(output / "checkpoints" / f"{case['id']}_seed{seed}.npz", result)
    write_metrics(output / "wp04_forward_metrics.csv", rows)
    figures = make_figures(cfg, curves, output / "figures")
    manifest = build_manifest(
        config_path, cfg, rows, time.perf_counter() - started, output, figures
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if artifacts:
        publish(Path(artifacts), output, figures, manifest, cfg)
    return rows, manifest


def metric_row(case, method, seed, ev, seconds, status, hit, result=None, note=None):
    def last(key):
        return result.history[key][-1] if result is not None and result.history[key] else None

    lbfgs = (result.lbfgs if result is not None else None) or {}
    return {
        "run_id": f"{case['id']}-{method}-{seed}",
        "case": case["id"],
        "model": case["model"],
        "regime": case["regime"],
        "method": method,
        "seed": seed,
        "split": "dense_reference",
        "noise_sigma": None,
        "rmse": ev["rmse"],
        "mae": ev["mae"],
        "relative_l2": ev["relative_l2"],
        "max_abs": ev["max_abs"],
        "max_abs_near_zero": ev["max_abs_near_zero"],
        "boundary_violations": ev["boundary_violations"],
        "wall_seconds": round(seconds, 4),
        "status": status,
        "final_loss": last("loss"),
        "iterations": None if result is None else len(result.history["loss"]),
        "lbfgs_status": lbfgs.get("status"),
        "lbfgs_message": lbfgs.get("message"),
        "lbfgs_iterations": lbfgs.get("iterations"),
        "note": note,
        "t_end": hit if hit is not None else case["tau_end"],
        "extinction_tau": hit,
    }


def format_cell(value):
    """Errors, losses and times in scientific notation; None becomes an empty cell."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6e}"
    return value


def write_metrics(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: format_cell(v) for k, v in row.items()})


def save_figure_with_pdf(fig, stem):
    """PNG + EPS from the shared helper, plus a PDF so plain ``latexmk -pdf`` can include it."""
    from mm_allee.plotting import save_figure

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)  # the PDF is written before save_figure runs
    pdf = stem.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white", transparent=False)
    return [*save_figure(fig, stem), pdf]


def make_figures(cfg, curves, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = cfg["cases"]
    colours = {seed: c for seed, c in zip(cfg["seeds"], ["#0072B2", "#D55E00", "#009E73"])}
    styles = {seed: s for seed, s in zip(cfg["seeds"], ["--", "-.", ":"])}
    fig1, axes1 = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
    fig2, axes2 = plt.subplots(2, len(cases), figsize=(3.4 * len(cases), 6.4), squeeze=False)
    for index, case in enumerate(cases):
        ax = axes1[index // 3][index % 3]
        mine = [c for c in curves if c["case"] == case["id"]]
        if not mine:
            continue  # every seed of this case failed: nothing to draw
        ax.plot(mine[0]["tau"], mine[0]["ref"], "k-", lw=2.4, label="DOP853 reference")
        err_ax, loss_ax = axes2[0][index], axes2[1][index]
        for c in mine:
            ax.plot(
                c["tau"],
                c["pred"],
                styles[c["seed"]],
                color=colours[c["seed"]],
                lw=1.8,
                label=f"PINN seed {c['seed']}",
            )
            err_ax.semilogy(
                c["tau"],
                np.abs(c["pred"] - c["ref"]) + 1e-16,
                styles[c["seed"]],
                color=colours[c["seed"]],
                label=f"seed {c['seed']}",
            )
            loss_ax.semilogy(
                c["history"]["loss"],
                color=colours[c["seed"]],
                ls=styles[c["seed"]],
                label=f"seed {c['seed']}",
            )
        loss_ax.axvline(cfg["training"]["adam_iters"], color="0.5", lw=0.8)  # Adam -> L-BFGS
        title = f"Model {case['model']}, {case['regime']}\n a={case['a']}, "
        title += f"{'q' if case['model'] == 'A' else 'e'}={case['harvest']}, x0={case['x0']}"
        ax.set(title=title, xlabel=r"time $\tau=rt$", ylabel=r"population $x=N/K$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        err_ax.set(
            title=f"{case['model']}, {case['regime']}",
            xlabel=r"$\tau$",
            ylabel=r"$|\hat x-x_{ref}|$",
        )
        loss_ax.set(xlabel="iteration (Adam, then L-BFGS)", ylabel="physics loss")
        for extra in (err_ax, loss_ax):
            extra.grid(alpha=0.3)
        err_ax.legend(fontsize=7)
    fig1.suptitle("WP04: forward PINN vs reference (synthetic, dimensionless)")
    fig2.suptitle("WP04: absolute error (top) and optimisation history (bottom)")
    fig1.tight_layout()
    fig2.tight_layout()
    paths = save_figure_with_pdf(fig1, directory / "wp04_forward") + save_figure_with_pdf(
        fig2, directory / "wp04_error_loss"
    )
    plt.close(fig1)
    plt.close(fig2)
    return paths


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state():
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()

    try:
        return {
            "sha": git("rev-parse", "HEAD") or None,
            "dirty": bool(git("status", "--porcelain")),
        }
    except OSError:
        return {"sha": None, "dirty": None}


def environment_pins():
    """Exact versions of the packages that determine the numbers (for reproduction)."""
    from importlib import metadata

    pins = {}
    for name in ("jax", "jaxlib", "numpy", "scipy", "matplotlib"):
        pins[name] = metadata.version(name)
    return pins


def build_manifest(config_path, cfg, rows, elapsed, output, figures):
    pins = environment_pins()

    pinn_rows = [r for r in rows if r["method"] == "pinn"]
    counts, lbfgs_exits = {}, {}
    for row in pinn_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        if row["lbfgs_message"]:
            lbfgs_exits[row["lbfgs_message"]] = lbfgs_exits.get(row["lbfgs_message"], 0) + 1
    return {
        "schema_version": 1,
        "run_id": f"wp04-{datetime.now(UTC):%Y%m%dT%H%M%SZ}",
        "created_utc": datetime.now(UTC).isoformat(),
        "git": git_state(),
        "command": f"python experiments/wp04/run.py --config {config_path.as_posix()} --output {Path(output).as_posix()}",
        "config": cfg,
        "seeds": cfg["seeds"],
        "model_and_units": "dimensionless x=N/K, tau=r*t; A: q=H/(rK), B: e=h/r; synthetic only",
        "reference_solver": cfg["reference"],
        "baseline_solver": "mm_allee.solver.solve_trajectory (RK45, rtol=1e-9, atol=1e-11)",
        "optimiser": {
            "scheme": "Adam then L-BFGS-B (SciPy) on the JAX gradient",
            **{k: cfg["training"].get(k) for k in ("adam_iters", "lr", "lbfgs_iters")},
        },
        "loss": "physics residual MSE only; initial condition enforced exactly (hard constraint)",
        "runtime_budget_s_per_run": cfg["training"].get("time_budget_s"),
        "supervised_labels_used": False,
        "pinn_status_counts": counts,
        "lbfgs_exit_counts": lbfgs_exits,
        "elapsed_seconds": round(elapsed, 2),
        "dependencies": {
            "python": platform.python_version(),
            **pins,
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "device": "cpu",
            "dtype": "float64",
        },
        "outputs": {
            "wp04_forward_metrics.csv": sha256(Path(output) / "wp04_forward_metrics.csv"),
            **{p.name: sha256(p) for p in figures},
        },
        "status": "failed_runs_present"
        if any(counts.get(k) for k in ("nonfinite_loss", "failed", "invalid_prediction"))
        else "completed",
    }


def publish(artifacts, output, figures, manifest, cfg):
    """Copy curated small outputs (figures, table, manifest) into ``artifacts/``."""
    import shutil

    (artifacts / "figures").mkdir(parents=True, exist_ok=True)
    (artifacts / "tables").mkdir(parents=True, exist_ok=True)
    (artifacts / "manifests").mkdir(parents=True, exist_ok=True)
    for path in figures:
        shutil.copy2(path, artifacts / "figures" / path.name)
    shutil.copy2(
        output / "wp04_forward_metrics.csv", artifacts / "tables" / "wp04_forward_metrics.csv"
    )
    name = cfg.get("manifest_name", "wp04_forward.json")
    (artifacts / "manifests" / name).write_text(json.dumps(manifest, indent=2))
    pins = manifest["dependencies"]
    lines = [f"{k}=={pins[k]}" for k in ("jax", "jaxlib", "numpy", "scipy", "matplotlib")]
    header = f"# exact versions used for {name}; Python {pins['python']}, {manifest['hardware']['platform']}\n"
    (artifacts / "manifests" / f"{Path(name).stem}_requirements.txt").write_text(
        header + "\n".join(lines) + "\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifacts", help="also copy curated figures/table/manifest here")
    args = parser.parse_args(argv)
    rows, manifest = run(args.config, args.output, args.artifacts)
    print(
        json.dumps(
            {
                "pinn_status_counts": manifest["pinn_status_counts"],
                "elapsed_seconds": manifest["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
