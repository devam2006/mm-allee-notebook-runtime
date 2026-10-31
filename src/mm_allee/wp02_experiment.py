"""Reproducible WP02 experiments and opaque publication figures.

Runtime budget is checked between bounded solves (not a hard OS timeout).
All data are synthetic. Finite-time observations and theoretical outcomes are
separate; a numerical smallness cutoff never becomes B's extinction event.
"""

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .comparison import compare_policies
from .equilibria import fold_a
from .history import ramp_schedule, run_history
from .model_b import critical_effort, equilibria_b
from .models import Parameters
from .plotting import save_figure
from .solver import MIN_RK45_RTOL


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        # Canonical LF keeps manifest hashes unchanged by Git newline normalisation.
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def branch_figure(a):
    ec = critical_effort(a)
    grid = np.linspace(0, ec, 301)
    records = [equilibria_b(a, float(e)) for e in grid]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid / ec, [r[1].x for r in records], "--", color="#bd4b32", label="unstable threshold")
    ax.plot(grid / ec, [r[-1].x for r in records], color="#176b87", label="stable population")
    ax.plot([0, 1.3], [0, 0], color="#222222", label="stable zero")
    ax.scatter([1], [(1 + a) / 2], color="#7b3e9b", marker="D", label="semistable fold", zorder=4)
    ax.set(
        xlabel=r"Effort / critical effort $e/e_c$",
        ylabel=r"Population $x=N/K$",
        title=f"Model B equilibria (a={a:g}); zero remains stable",
        ylim=(-0.04, 1.06),
    )
    ax.legend(framealpha=1)
    fig.tight_layout()
    return fig


def history_figure(runs, config):
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex="col")
    colors = ["#176b87", "#bd4b32"]
    for col, model in enumerate(["A", "B"]):
        for j, dwell in enumerate(config["history"]["dwells"]):
            plain, stocked, critical = runs[(model, dwell)]
            color = colors[j % len(colors)]
            for h, style, name in [(plain, "-", "unassisted"), (stocked, "--", "restocked")]:
                axes[1, col].plot(
                    [r["tau"] for r in h.rows],
                    [r["x"] for r in h.rows],
                    style,
                    color=color,
                    label=f"dwell={dwell:g}, {name}",
                )
            axes[0, col].plot(
                [r["tau"] for r in plain.rows],
                [r["control"] / critical for r in plain.rows],
                color=color,
                label=f"dwell={dwell:g}",
            )
            # Prescribed control is distinct from shutdown's effective control for A.
            axes[2, col].plot(
                [r["tau"] for r in plain.rows],
                [r["yield_dimensionless"] for r in plain.rows],
                color=color,
                label=f"dwell={dwell:g}",
            )
            jump = next(r for r in stocked.rows if r["label"] == "restock_after")
            axes[1, col].scatter([jump["tau"]], [jump["x"]], marker="^", color=color, zorder=5)
        axes[0, col].axhline(1, color="black", linestyle=":")
        axes[1, col].axhline(
            config["parameters"]["A"] / config["parameters"]["K"], color="black", linestyle=":"
        )
        axes[0, col].set_title(f"Model {model}: prescribed harvest")
        axes[0, col].set_ylabel("Control / critical control")
        axes[1, col].set_ylabel("Population x=N/K")
        axes[2, col].set(xlabel=r"Time $\tau=rt$", ylabel="Unassisted catch / (rK)")
        axes[0, col].legend(framealpha=1, fontsize=8)
        axes[1, col].legend(framealpha=1, fontsize=7)
    fig.suptitle(
        f"Synthetic histories: triangles add {config['history']['restock_addition']:g}K; dotted population line is A/K\n"
        "A shuts down at extinction; B smallness is not finite extinction",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def run_experiment(config, output, root):
    """Write results/figures/manifest and return summary, recording failures too."""
    output, root = Path(output), Path(root)
    requested_rtol = config.get("solver", {}).get("rtol", 1e-9)
    if not np.isfinite(requested_rtol) or requested_rtol < MIN_RK45_RTOL:
        raise ValueError("RK45 rtol is below its supported minimum or nonfinite")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    manifest = dict(
        schema_version=1,
        run_id=datetime.now(timezone.utc).strftime("wp02-%Y%m%dT%H%M%SZ"),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        config=config,
        **source_state(root),
        hardware=platform.platform() + " / " + platform.machine(),
        python=platform.python_version(),
        versions={
            name: importlib.metadata.version(name) for name in ["numpy", "scipy", "matplotlib"]
        },
        seeds=[],
        failed_seeds=[],
        status="running",
        solver_method="RK45",
        boundary_policy="A first-zero event then shutdown; B no finite extinction or cutoff reset",
        noise="none",
        split_ids=None,
        units=config["units"],
        model=["A", "B", "logistic"],
    )
    source_paths = [
        root / "src/mm_allee" / name
        for name in [
            "models.py",
            "solver.py",
            "equilibria.py",
            "thresholds.py",
            "collapse.py",
            "model_b.py",
            "history.py",
            "comparison.py",
            "wp02_experiment.py",
        ]
    ]
    manifest["source_sha256"] = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_paths
    }

    def budget():
        if time.monotonic() - started > config["budget_seconds"]:
            raise TimeoutError("Recorded WP02 runtime budget exceeded")

    try:
        p = Parameters(**config["parameters"])
        a = p.A / p.K
        rows = []
        c = config["comparison"]
        for protocol in ["initial_catch", "threshold_fraction"]:
            for x0 in c["x0_values"]:
                for fraction in c["fractions"]:
                    budget()
                    rows.extend(
                        compare_policies(
                            p,
                            x0=x0,
                            fraction=fraction,
                            protocol=protocol,
                            tau_end=c["tau_end"],
                            n_points=c["n_points"],
                            **config["solver"],
                        )
                    )
        write_csv(output / "wp02_comparison.csv", rows)
        equilibrium_rows = [
            root.as_row()
            for fraction in [0, 0.5, 0.999999, 1, 1.2]
            for root in equilibria_b(a, fraction * critical_effort(a))
        ]
        write_csv(output / "wp02_b_equilibria.csv", equilibrium_rows)
        h = config["history"]
        runs, summaries, all_rows = {}, [], []
        convergence = config["convergence"]
        tight_settings = {key: convergence[key] for key in ["rtol", "atol", "max_step"]}
        for model, critical in [("A", fold_a(a)[1]), ("B", critical_effort(a))]:
            for dwell in h["dwells"]:
                budget()
                times, controls, last = ramp_schedule(
                    critical,
                    steps=h["steps"],
                    dwell=dwell,
                    peak_fraction=h["peak_fraction"],
                    peak_hold=h["peak_hold"],
                    recovery_hold=h["recovery_hold"],
                    restock_hold=h["restock_hold"],
                )
                args = dict(a=a, x0=h["x0"], times=times, controls=controls, n_points=h["n_points"])
                plain = run_history(model, **args, **config["solver"])
                stocked = run_history(
                    model, **args, restocking={last: h["restock_addition"]}, **config["solver"]
                )
                tight = run_history(
                    model, **args, restocking={last: h["restock_addition"]}, **tight_settings
                )
                if len(stocked.rows) != len(tight.rows) or len(stocked.extinction_times) != len(
                    tight.extinction_times
                ):
                    raise AssertionError("Event topology changed on refinement")
                error = max(
                    abs(left["x"] - r["x"])
                    for left, r in zip(stocked.rows, tight.rows, strict=True)
                )
                event_error = max(
                    [
                        abs(left - r)
                        for left, r in zip(
                            stocked.extinction_times, tight.extinction_times, strict=True
                        )
                    ]
                    or [0]
                )
                if (
                    error > convergence["max_population_error"]
                    or event_error > convergence["max_event_time_error"]
                ):
                    raise AssertionError("History tolerance convergence failed")
                small = next((r["tau"] for r in plain.rows if r["x"] <= h["epsilon"]), None)
                before = next(r for r in stocked.rows if r["label"] == "restock_before")
                summaries.append(
                    dict(
                        model=model,
                        dwell=dwell,
                        critical_control=critical,
                        extinction_tau=plain.extinction_times[0]
                        if plain.extinction_times
                        else None,
                        first_sample_below_epsilon_tau=small,
                        epsilon=h["epsilon"],
                        restock_tau=float(times[last]),
                        population_before_restock=before["x"],
                        restock_addition=h["restock_addition"],
                        final_unassisted=plain.rows[-1]["x"],
                        final_restocked=stocked.rows[-1]["x"],
                        max_refinement_x_error=error,
                        max_refinement_event_tau_error=event_error,
                    )
                )
                runs[(model, dwell)] = plain, stocked, critical
                for label, trajectory in [("unassisted", plain), ("restocked", stocked)]:
                    all_rows.extend(
                        dict(model=model, dwell=dwell, run=label, **r) for r in trajectory.rows
                    )
        write_csv(output / "wp02_history_summary.csv", summaries)
        write_csv(output / "wp02_history_samples.csv", all_rows)
        for name, fig in [
            ("wp02_b_branches", branch_figure(a)),
            ("wp02_history", history_figure(runs, config)),
        ]:
            save_figure(fig, output / name)
            fig.savefig(output / f"{name}.pdf", bbox_inches="tight", facecolor="white")
            plt.close(fig)
        budget()
        manifest["status"] = "success"
        manifest["summary"] = summaries
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["output_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.glob("wp02_*"))
            if p.is_file()
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
    return manifest


def source_state(root):
    """Preserve provenance when running an exported notebook without Git."""
    snapshot = root / "SOURCE_SNAPSHOT.json"
    if snapshot.is_file():
        return dict(git_sha=None, git_dirty=None, source_snapshot=json.loads(snapshot.read_text()))
    try:
        return dict(
            git_sha=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip(),
            git_dirty=bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=root,
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            ),
        )
    except (OSError, subprocess.CalledProcessError):
        return dict(git_sha=None, git_dirty=None, source_snapshot="Git metadata unavailable")
