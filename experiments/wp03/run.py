"""Reproduce WP03 collapse, basin, bottleneck and optional sensitivity/MSY evidence."""

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import scipy

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from mm_allee.collapse import positive_branches
from mm_allee.collapse_study import build_study
from mm_allee.datasets import sha256, write_json
from mm_allee.plotting import save_figure
from mm_allee.thresholds import critical_a, critical_b

ROOT = Path(__file__).resolve().parents[2]


def plot_basin_grid(axis, controls, initials, grid, colours):
    """Draw sampled basins with an exact cell edge at the critical control.

    Fold samples belong to the boundary line, never a finite-width cell.
    Draw each side separately so persistence colours cannot spill above 1,
    including when the control grid does not sample the fold.
    """

    def edges(values):
        return np.r_[
            values[0] - (values[1] - values[0]) / 2,
            (values[:-1] + values[1:]) / 2,
            values[-1] + (values[-1] - values[-2]) / 2,
        ]

    control_edges, initial_edges = edges(controls), edges(initials)
    meshes = []
    for mask in (controls < 1, controls > 1):
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        left, right = indices[0], indices[-1] + 1
        region_edges = control_edges[left : right + 1].copy()
        if controls[left] > 1 and controls[0] <= 1:
            region_edges[0] = 1
        if controls[right - 1] < 1 and controls[-1] >= 1:
            region_edges[-1] = 1
        meshes.append(
            axis.pcolormesh(
                region_edges,
                initial_edges,
                grid[:, left:right],
                shading="flat",
                cmap=ListedColormap(colours),
                vmin=0,
                vmax=4,
                # Matching strokes hide PDF viewers' antialiasing seams
                # between adjacent filled vector polygons.
                edgecolors="face",
                linewidth=0.5,
            )
        )
    return meshes


def plot_results(study, config, output):
    p = study["parameters"]
    a = p.A / p.K
    star, qc = critical_a(a)
    subtitle = f"Synthetic model: r={p.r:g}, K={p.K:g}, A={p.A:g}; declared model units"
    paths = []

    def finish(fig, name, *, legend_space=False):
        fig.suptitle(subtitle, fontsize=10)
        fig.tight_layout(rect=(0, 0.14 if legend_space else 0, 1, 0.94))
        paths.extend(save_figure(fig, output / name))
        # Keep the shared PNG/EPS helper's API; WP03 also supplies vector
        # PDFs for plain pdflatex/latexmk without EPS shell conversion.
        pdf = output / f"{name}.pdf"
        fig.savefig(pdf, bbox_inches="tight", facecolor="white", transparent=False)
        paths.append(pdf)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for fraction in config["initial_curve_harvest_fractions"]:
        rows = [
            r
            for r in study["tables"]["extinction"]
            if r["curve"] == "vary_initial" and r["harvest_fraction"] == fraction
        ]
        values = [r["t_ext"] if r["t_ext"] is not None and r["t_ext"] > 0 else np.nan for r in rows]
        axes[0].plot(
            [r["N0"] for r in rows],
            values,
            label=f"H={fraction * qc * p.r * p.K:.3g} ({fraction:g} Hc)",
        )
    for initial in config["harvest_curve_initial_x"]:
        rows = [
            r
            for r in study["tables"]["extinction"]
            if r["curve"] == "vary_harvest" and r["x0"] == initial
        ]
        axes[1].plot(
            [r["H"] for r in rows],
            [r["t_ext"] if r["t_ext"] is not None and r["t_ext"] > 0 else np.nan for r in rows],
            label=f"N0={initial * p.K:g}",
        )
    for axis in axes:
        axis.set_yscale("log")
        axis.set_ylabel(f"Extinction time [{config['units']['time']}] (log scale)")
        axis.grid(color="0.9")
        axis.legend(fontsize=8, framealpha=1)
    axes[0].set(
        title="Model A: vary initial population",
        xlabel=f"Initial population [{config['units']['population']}]",
    )
    axes[1].set(
        title="Model A: vary constant catch", xlabel="H [population_unit / model_time_unit]"
    )
    axes[1].axvline(qc * p.r * p.K, color="black", linestyle="--", linewidth=1)
    finish(fig, "wp03_extinction")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    colours = ["#D55E00", "#E6AB02", "#0072B2", "#FFFFFF", "#999999"]
    for model, axis in zip(("A", "B"), axes, strict=True):
        plot_basin_grid(
            axis, study["controls"], study["initials"], study["basin_grids"][model], colours
        )
        fractions = np.linspace(0, 1, 101)
        critical = qc if model == "A" else critical_b(a)
        boundaries = [positive_branches(model, a=a, harvest=float(f * critical)) for f in fractions]
        axis.plot(fractions, [x[0] for x in boundaries], color="black", linewidth=1.5)
        axis.plot(fractions, [x[1] for x in boundaries], color="black", linestyle="--", linewidth=1)
        axis.axvline(1, color="black", linestyle=":", linewidth=1)
        axis.set(
            title=f"Model {model}: analytical basins",
            xlabel="H / Hc" if model == "A" else "h / hc",
            ylabel="Initial x0 = N0 / K",
            xlim=(0, config["max_harvest_fraction"]),
            ylim=(0, config["max_initial_x"]),
        )
    labels = [
        "Finite extinction (A)",
        "Asymptotic decline",
        "Positive equilibrium / basin",
        "Zero equilibrium",
        "Boundary unresolved",
    ]
    fig.legend(
        handles=[
            Patch(facecolor=c, edgecolor="0.3", label=label)
            for c, label in zip(colours, labels, strict=True)
        ],
        loc="lower center",
        ncol=3,
        fontsize=8,
        framealpha=1,
    )
    finish(fig, "wp03_basins", legend_space=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for fraction, solution in study["trajectories"]:
        axes[0].plot(solution.tau, solution.x, label=f"H/Hc={1 + fraction:g}")
    axes[0].axhline(star, color="black", linestyle="--", linewidth=1, label="x_star (fold state)")
    axes[0].set(
        title="Post-collapse bottleneck (A)",
        xlabel="Dimensionless time tau = r t",
        ylabel="Population x = N / K",
    )
    axes[0].legend(fontsize=8, framealpha=1)
    rows = study["tables"]["bottleneck"]
    delta = np.array([r["delta_q"] for r in rows])
    axes[1].loglog(
        delta, [r["tau_ext"] for r in rows], "o-", label="Quadrature, full extinction time"
    )
    axes[1].loglog(
        delta,
        [r["asymptotic_scaled_time"] / np.sqrt(r["delta_q"]) for r in rows],
        "k--",
        label="Leading inverse-square-root term",
    )
    axes[1].set(
        title="Approach to the collapse cliff",
        xlabel="delta = q - qc > 0",
        ylabel="Dimensionless extinction time tau_ext",
    )
    axes[1].legend(fontsize=8, framealpha=1)
    for axis in axes:
        axis.grid(color="0.9")
    finish(fig, "wp03_bottleneck")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for i, name in enumerate(("H_c", "h_c")):
        rows = [r for r in study["tables"]["sensitivity"] if r["threshold"] == name]
        axes[0].bar(
            np.arange(3) + (i - 0.5) * 0.32, [r["elasticity"] for r in rows], width=0.32, label=name
        )
    axes[0].set(
        xticks=np.arange(3),
        xticklabels=["r", "K", "A"],
        title="Optional: dimensional elasticities",
        ylabel="(p / threshold) d(threshold) / dp",
    )
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].legend(framealpha=1)
    rows = study["tables"]["yield"]
    axes[1].plot(
        [r["effort_fraction"] for r in rows],
        [r["realised_yield"] for r in rows],
        label="B upper equilibrium branch",
    )
    msy = study["msy"]
    axes[1].plot(msy["h_MSY"] / msy["h_c"], msy["Y_MSY_B"], "ko", label="Interior MSY")
    axes[1].plot(
        1,
        rows[-1]["realised_yield"],
        marker="s",
        color="#D55E00",
        linestyle="none",
        label="Effort cliff (semistable)",
    )
    axes[1].set(
        title="Optional: yield is not effort",
        xlabel="Effort h / hc",
        ylabel="Realised yield [population_unit / model_time_unit]",
    )
    axes[1].legend(fontsize=8, framealpha=1)
    finish(fig, "wp03_sensitivity_yield")
    return paths


def git_value(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(config, output):
    started = time.perf_counter()
    study = build_study(config)  # Validate and compute before creating output files.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, rows in study["tables"].items():
        path = output / f"wp03_{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        paths.append(path)
    paths.extend(plot_results(study, config, output))
    summary_path = output / "wp03_summary.json"
    write_json(
        summary_path,
        {"thresholds_and_msy": study["msy"], "example_policy_margins": study["policies"]},
    )
    paths.append(summary_path)
    source = [
        Path(__file__),
        *[
            ROOT / "src/mm_allee" / name
            for name in (
                "models.py",
                "datasets.py",
                "solver.py",
                "thresholds.py",
                "collapse.py",
                "collapse_study.py",
                "plotting.py",
            )
        ],
    ]
    git_status = git_value("status", "--porcelain")
    manifest = {
        "schema_version": 1,
        "run_id": "wp03-collapse-analysis-v1",
        "status": "passed",
        "is_synthetic": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_value("rev-parse", "HEAD"),
        "git_dirty": None if git_status is None else bool(git_status),
        "configuration": config,
        "configuration_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "source_files_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in source},
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "hardware": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "wall_seconds": time.perf_counter() - started,
        "table_rows": {name: len(rows) for name, rows in study["tables"].items()},
        "validation": {
            "threshold_maxima": "passed",
            "event_integrations": "passed",
            "sensitivity_finite_differences": "passed",
            "yield_numerical_maximum": "passed",
        },
        "seeds": None,
        "noise": None,
        "splits": None,
        "scope": "Deterministic WP03 analysis; no noise/splits/PINN, no certified final group submission",
        "paths_relative_to": "output_directory",
        "files_sha256": {p.name: sha256(p) for p in paths},
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def curate(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    hashes = {}
    for name in manifest["files_sha256"]:
        folder = (
            "figures"
            if name.endswith((".png", ".eps", ".pdf"))
            else "manifests"
            if name.endswith(".json")
            else "tables"
        )
        destination = ROOT / "artifacts" / folder / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (output / name).resolve() != destination.resolve():
            shutil.copyfile(output / name, destination)
        if destination.suffix in {".json", ".csv", ".eps"}:
            # Hash the canonical LF bytes that Git stores on every platform.
            destination.write_bytes(destination.read_bytes().replace(b"\r\n", b"\n"))
        hashes[destination.relative_to(ROOT).as_posix()] = sha256(destination)
    manifest["files_sha256"], manifest["paths_relative_to"] = hashes, "repository_root"
    write_json(ROOT / "artifacts/manifests/wp03_analysis.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--curate", action="store_true", help="Refresh the versioned WP03 figures/tables/manifest"
    )
    args = parser.parse_args()
    try:
        result = run(json.loads(args.config.read_text()), args.output)
        if args.curate:
            curate(args.output)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, ArithmeticError) as exc:
        parser.exit(1, f"WP03 study failed: {exc}\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "table_rows": result["table_rows"],
                "wall_seconds": result["wall_seconds"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
