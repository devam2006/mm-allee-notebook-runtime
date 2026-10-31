"""WP04 diagnostic (exploratory, NOT the frozen sweep): why do the *regular* A/B runs fail?

    python experiments/wp04/diagnose.py --config configs/wp04_diagnose.json --output results/wp04_diagnose

Hypotheses and predictions are written down in experiments/wp04/README.md before running.
For every (case, collocation variant, seed) it trains the forward PINN, scores it against the
DOP853 reference, and also measures the ODE residual *between* the training points.
Importing this module trains nothing.
"""

import argparse
import csv
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from run import format_cell, git_state, reference_trajectory, sha256  # noqa: E402

RUN_COLUMNS = [
    "case", "variant", "seed", "relative_l2", "class", "final_loss", "max_R_collocation",
    "max_R_dense", "x_mid", "boundary_violations", "wall_seconds", "status",
]  # fmt: skip
SUMMARY_COLUMNS = [
    "case", "variant", "n_runs", "n_accurate", "n_unstable_root", "n_zero", "n_negative_state",
    "n_not_converged",
    "n_other", "n_invalid", "median_relative_l2", "median_max_R_dense_accurate", "median_max_R_dense_failed",
    "median_wall_seconds",
]  # fmt: skip
CLASSES = [
    "accurate", "unstable_root", "zero", "negative_state", "not_converged", "other", "invalid",
]  # fmt: skip


def make_grid(kind, n, t_end):
    """Collocation times in [0, t_end], both ends included."""
    if kind == "uniform":  # identical (bit for bit) to the engine's default grid
        import jax.numpy as jnp

        return np.asarray(jnp.linspace(0.0, t_end, n))
    u = np.linspace(0.0, 1.0, n)
    if kind == "power2":  # clustered near tau=0: spacing grows like 2u
        return t_end * u**2
    raise ValueError(f"unknown grid kind {kind!r}")


def equilibrium_roots(case):
    """Positive roots of f(x)=0 by an independent polynomial solve (not the engine's RHS)."""
    a, h = case["a"], case["harvest"]
    # G(x) = -x^3/a + (1+a)x^2/a - x ;  A: G-q=0 ;  B: G-e x = 0
    coeffs = (
        [-1 / a, (1 + a) / a, -1.0, -h]
        if case["model"] == "A"
        else [-1 / a, (1 + a) / a, -1.0 - h, 0.0]
    )
    roots = np.roots(coeffs)
    return np.sort(roots[np.isreal(roots)].real)


def classify(rel_l2, final_loss, x_mid, x_unstable, cfg):
    if rel_l2 is None:  # NaN/inf prediction (or undefined error): never scored as accurate
        return "invalid"
    if rel_l2 <= cfg["inaccuracy_threshold_relative_l2"]:
        return "accurate"
    if final_loss > cfg["not_converged_loss"]:
        return "not_converged"
    if abs(x_mid - x_unstable) < cfg["state_tolerance"]:
        return "unstable_root"
    if abs(x_mid) < cfg["state_tolerance"]:
        return "zero"
    if x_mid < -cfg["state_tolerance"]:  # added after the first pass; see README
        return "negative_state"
    return "other"


def run_one(case, variant, seed, cfg, ref):
    import jax.numpy as jnp

    from mm_allee.pinn.network import x_hat
    from mm_allee.pinn.residual import collocation_residual
    from mm_allee.pinn.train_forward import (
        ForwardConfig,
        evaluate_prediction,
        predict,
        train_forward,
    )

    tau_ref, x_ref, hit = ref
    t_end = case["tau_end"] if hit is None else hit
    config = ForwardConfig(
        model=case["model"], a=case["a"], harvest=case["harvest"], x0=case["x0"], t_end=t_end,
        seed=seed, n_collocation=variant["n"], adam_iters=variant["adam_iters"],
        lbfgs_iters=variant["lbfgs_iters"], lr=cfg["lr"], time_budget_s=cfg["time_budget_s"],
    )  # fmt: skip
    grid = make_grid(variant["grid"], variant["n"], t_end)
    result = train_forward(config, tau=grid)
    pred = predict(result.params, config, tau_ref)
    ev = evaluate_prediction(pred, x_ref)

    def fn(t):
        return x_hat(result.params, t, config.network_config())

    kw = dict(a=case["a"], harvest=case["harvest"])
    _, r_col = collocation_residual(fn, grid, case["model"], **kw)
    dense = np.linspace(0.0, t_end, cfg["dense_points"])
    _, r_den = collocation_residual(fn, dense, case["model"], **kw)
    x_mid = float(predict(result.params, config, np.array([t_end / 2]))[0])
    roots = equilibrium_roots(case)
    x_unstable = float(roots[roots > 0][0])
    final_loss = result.history["loss"][-1]
    rel = ev["relative_l2"]
    return {
        "case": case["id"], "variant": variant["id"], "seed": seed, "relative_l2": rel,
        "class": classify(rel, final_loss, x_mid, x_unstable, cfg), "final_loss": final_loss,
        "max_R_collocation": float(jnp.max(jnp.abs(r_col))), "max_R_dense": float(jnp.max(jnp.abs(r_den))),
        "x_mid": x_mid, "boundary_violations": ev["boundary_violations"],
        "wall_seconds": round(result.wall_seconds, 3), "status": result.status,
    }  # fmt: skip


def summarise(rows):
    out = []
    for case in sorted({r["case"] for r in rows}):
        for variant in dict.fromkeys(r["variant"] for r in rows if r["case"] == case):
            mine = [r for r in rows if r["case"] == case and r["variant"] == variant]
            ok = [r["max_R_dense"] for r in mine if r["class"] == "accurate"]
            bad = [r["max_R_dense"] for r in mine if r["class"] != "accurate"]
            count = lambda c: sum(r["class"] == c for r in mine)  # noqa: E731
            out.append({
                "case": case, "variant": variant, "n_runs": len(mine),
                "n_accurate": count("accurate"), "n_unstable_root": count("unstable_root"),
                "n_zero": count("zero"), "n_negative_state": count("negative_state"),
                "n_not_converged": count("not_converged"),
                "n_other": count("other"),
                "n_invalid": count("invalid"),
                "median_relative_l2": (
                    statistics.median(r["relative_l2"] for r in mine if r["relative_l2"] is not None)
                    if any(r["relative_l2"] is not None for r in mine)
                    else None
                ),
                "median_max_R_dense_accurate": statistics.median(ok) if ok else None,
                "median_max_R_dense_failed": statistics.median(bad) if bad else None,
                "median_wall_seconds": statistics.median(r["wall_seconds"] for r in mine),
            })  # fmt: skip
    return out


def write_csv(path, columns, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: format_cell(row[k]) for k in columns})


def make_figure(cfg, summary, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mm_allee.plotting import save_figure

    colours = {"accurate": "#009E73", "unstable_root": "#D55E00", "zero": "#0072B2",
               "negative_state": "#E69F00", "not_converged": "#CC79A7", "other": "#999999", "invalid": "#000000"}  # fmt: skip
    cases = [c["id"] for c in cfg["cases"]]
    variants = [v["id"] for v in cfg["variants"]]
    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 4.2), squeeze=False)
    for ax, case in zip(axes[0], cases):
        bottom = np.zeros(len(variants))
        for cls in CLASSES:
            counts = np.array([
                next(s[f"n_{cls}"] for s in summary if s["case"] == case and s["variant"] == v)
                for v in variants
            ])  # fmt: skip
            ax.bar(variants, counts, bottom=bottom, color=colours[cls], label=cls.replace("_", " "),
                   edgecolor="white")  # fmt: skip
            bottom += counts
        ax.set(title=f"{case}: outcome of {len(cfg['seeds'])} seeds", ylabel="number of seeds",
               xlabel="collocation / budget variant")  # fmt: skip
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.suptitle("WP04 diagnostic (exploratory): outcomes by collocation design")
    fig.tight_layout()
    paths = save_figure(fig, directory / "wp04_diagnose")
    plt.close(fig)
    return paths


def run(config_path, output, artifacts=None):
    import jax

    config_path = Path(config_path)
    cfg = json.loads(config_path.read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = []
    for case in cfg["cases"]:
        ref = reference_trajectory(case, cfg["reference"])
        for variant in cfg["variants"]:
            for seed in cfg["seeds"]:
                rows.append(run_one(case, variant, seed, cfg, ref))
    summary = summarise(rows)
    write_csv(output / "wp04_diagnose_runs.csv", RUN_COLUMNS, rows)
    write_csv(output / "wp04_diagnose_summary.csv", SUMMARY_COLUMNS, summary)
    figures = make_figure(cfg, summary, output / "figures")
    manifest = {
        "schema_version": 1,
        "scope": "exploratory diagnostic; not part of the frozen WP04 sweep",
        "run_id": f"wp04-diagnose-{datetime.now(UTC):%Y%m%dT%H%M%SZ}",
        "git": git_state(),
        "command": f"python experiments/wp04/diagnose.py --config {config_path.as_posix()} --output {output.as_posix()}",
        "config": cfg,
        "jax": jax.__version__,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "outputs": {
            p.name: sha256(p) for p in [output / "wp04_diagnose_runs.csv", output / "wp04_diagnose_summary.csv", *figures]
        },
    }  # fmt: skip
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if artifacts:
        import shutil

        artifacts = Path(artifacts)
        for sub in ("figures", "tables", "manifests"):
            (artifacts / sub).mkdir(parents=True, exist_ok=True)
        for p in figures:
            shutil.copy2(p, artifacts / "figures" / p.name)
        for name in ("wp04_diagnose_runs.csv", "wp04_diagnose_summary.csv"):
            shutil.copy2(output / name, artifacts / "tables" / name)
        (artifacts / "manifests" / "wp04_diagnose.json").write_text(json.dumps(manifest, indent=2))
    return rows, summary, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifacts", help="also copy curated figure/tables/manifest here")
    args = parser.parse_args(argv)
    _, summary, manifest = run(args.config, args.output, args.artifacts)
    print(json.dumps({"elapsed_seconds": manifest["elapsed_seconds"], "groups": len(summary)}))


if __name__ == "__main__":
    main()
