"""Reproduce every WP01 figure and table.

Usage (from the repository root):

    python experiments/wp01/run.py --config configs/wp01.json --output results/wp01

Add ``--curate`` to also copy the figures, table and manifest into ``artifacts/``.
Everything is deterministic (no random numbers); the run takes a few seconds on a CPU.
"""

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402

from mm_allee import __version__  # noqa: E402
from mm_allee.equilibria import (  # noqa: E402
    BRENT_RTOL,
    FOLD_RTOL,
    PROBE_MAX,
    equilibria_a,
    equilibria_logistic,
    fold_a,
)
from mm_allee.equilibria_plots import (  # noqa: E402
    branch_figure,
    perturbation_figure,
    production_figure,
)
from mm_allee.plotting import save_figure  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLE_COLUMNS = [
    "model",
    "a",
    "q",
    "q_over_qc",
    "regime",
    "x",
    "domain_status",
    "physical",
    "multiplicity",
    "derivative",
    "left_sign",
    "right_sign",
    "stability",
    "attracts_from",
    "residual",
    "note",
]


def save_all(fig, stem):
    """PNG + EPS (course originals) via the shared helper, plus a vector PDF for pdflatex.

    pdflatex in its default restricted mode cannot convert EPS stored outside the report
    directory, so the report includes the PDF variant; the EPS files remain the submitted
    originals and are produced from the same figure object.
    """
    paths = save_figure(fig, stem)
    pdf = Path(stem).with_suffix(".pdf")
    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
        metadata={"CreationDate": None, "ModDate": None},
    )
    return paths + [pdf]


SUPPORTED_SOLVER_KEYS = {"method", "rtol", "atol", "max_step"}


def solver_settings(config):
    """Validate the config's solver block and return the keyword arguments actually used.

    The shared solver integrates with RK45 only, so any other method is rejected rather
    than silently recorded. Tolerances must be positive finite numbers.
    """
    block = dict(config.get("solver", {}))
    unknown = set(block) - SUPPORTED_SOLVER_KEYS
    if unknown:
        raise ValueError(f"Unsupported solver settings: {sorted(unknown)}")
    if block.get("method", "RK45") != "RK45":
        raise ValueError("WP01 uses the shared RK45 solver; method must be 'RK45'")
    settings = {
        k: float(block.get(k, d)) for k, d in (("rtol", 1e-9), ("atol", 1e-11), ("max_step", 0.1))
    }
    if not all(np.isfinite(v) and v > 0 for v in settings.values()):
        raise ValueError("Solver tolerances and max_step must be positive and finite")
    minimum_rtol = 100 * np.finfo(float).eps
    if settings["rtol"] < minimum_rtol:
        raise ValueError(f"RK45 rtol must be at least {minimum_rtol:.17g}")
    return settings


def regime(q, q_c):
    if q == 0:
        return "zero_harvest"
    if q == q_c:
        return "critical"
    return "subcritical" if q < q_c else "supercritical"


def table_rows(config):
    rows = []
    for q in config["table_cases"]["logistic"]:
        records = equilibria_logistic(q)
        rows += _rows("logistic", None, q, 0.25, records)
    for a in config["table_cases"]["A"]["a_values"]:
        _, q_c = fold_a(a)
        for frac in config["table_cases"]["A"]["q_fraction_of_qc"]:
            q = q_c if frac == 1.0 else frac * q_c
            rows += _rows("A", a, q, q_c, equilibria_a(a, q, include_excluded=True))
    return rows


def _rows(model, a, q, q_c, records):
    if not records:
        return [
            {
                "model": model,
                "a": a,
                "q": q,
                "q_over_qc": q / q_c,
                "regime": regime(q, q_c),
                "x": None,
                "domain_status": "none",
                "physical": None,
                "multiplicity": 0,
                "derivative": None,
                "left_sign": None,
                "right_sign": None,
                "stability": "no_equilibrium",
                "attracts_from": None,
                "residual": None,
                "note": "no root of the cubic/quadratic; collapse from every start",
            }
        ]
    out = []
    for r in records:
        row = r.as_row()
        row.update(
            q_over_qc=q / q_c,
            regime=regime(q, q_c),
            domain_status="physical_population" if r.physical else "excluded_negative_root",
        )
        out.append({k: row[k] for k in TABLE_COLUMNS})
    return out


def write_csv(path, rows):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TABLE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(v) for k, v in row.items()})


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.15g}"
    return value


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def thresholds_summary(config):
    a = config["a"]
    x_star, q_c = fold_a(a)
    ex = config["dimensional_example"]
    r, K, A = ex["r_per_year"], ex["K_tonnes"], ex["A_tonnes"]
    assert abs(A / K - a) < 1e-12, "dimensional example must share the configured a"
    return {
        "a": a,
        "model_A": {
            "x_star": x_star,
            "q_c": q_c,
            "N_star_tonnes": K * x_star,
            "H_c_tonnes_per_year": r * K * q_c,
        },
        "logistic": {
            "x_star": 0.5,
            "q_c": 0.25,
            "N_star_tonnes": K / 2,
            "H_c_tonnes_per_year": r * K / 4,
        },
        "dimensional_example": ex,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--curate", action="store_true", help="copy figures/table/manifest into artifacts/"
    )
    args = parser.parse_args(argv)
    start = time.perf_counter()
    config = json.loads(args.config.read_text())
    solver = solver_settings(config)  # fail before writing anything
    out = args.output
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    a = config["a"]
    _, q_c = fold_a(a)
    files = []
    fig = production_figure(
        a,
        tuple(config["figure_logistic_q"]),
        tuple(q_c if f == 1.0 else f * q_c for f in config["figure_allee_q_fraction_of_qc"]),
    )
    files += save_all(fig, out / "figures" / "wp01_logistic_vs_allee")
    plt.close(fig)
    fig = branch_figure(a, config["branch_n_q"])
    files += save_all(fig, out / "figures" / "wp01_model_a_branches")
    plt.close(fig)
    pert = config["perturbation"]
    fig = perturbation_figure(a, pert["q"], pert["delta"], pert["tau_end"], **solver)
    files += save_all(fig, out / "figures" / "wp01_model_a_perturbations")
    plt.close(fig)

    table = out / "tables" / "wp01_equilibria.csv"
    rows = table_rows(config)
    write_csv(table, rows)
    files.append(table)
    summary = out / "tables" / "wp01_thresholds.json"
    summary.write_text(json.dumps(thresholds_summary(config), indent=2) + "\n")
    files.append(summary)

    sha, dirty = git_state()
    manifest = {
        "schema_version": 1,
        "run_id": str(uuid.uuid4()),
        "task": "WP01",
        "utc_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": sha,
        "git_dirty": dirty,
        "command": "python experiments/wp01/run.py --config "
        f"{args.config.as_posix()} --output {args.output.as_posix()}",
        "config": config,
        "seeds": [],
        "models": ["logistic", "A"],
        "units": config["units"],
        "data_type": "analytical/numerical root finding and RK45 simulation; no observations",
        "root_finding": {
            "method": "scipy.optimize.brentq on calculus-derived brackets",
            "brent_relative_tolerance": BRENT_RTOL,
            "fold_relative_tolerance_q": 0.0,
            "uncertainty_scale_relative_q": FOLD_RTOL,
            "fold_policy": "exact equality to shared computed critical control",
            "sign_probe_max_offset": PROBE_MAX,
        },
        "solver_effective": {
            "method": "RK45",
            **solver,
            "event_policy": "stop at first downward zero for q>0",
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "mm_allee": __version__,
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
        },
        "table_rows": len(rows),
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "status": "success",
        "outputs": {p.relative_to(out).as_posix(): sha256(p) for p in files},
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    if args.curate:
        for p in files:
            kind = "figures" if p.parent.name == "figures" else "tables"
            shutil.copy2(p, ROOT / "artifacts" / kind / p.name)
        shutil.copy2(manifest_path, ROOT / "artifacts" / "manifests" / "wp01_run.json")
    print(f"WP01: wrote {len(files)} files and manifest to {out}")
    return manifest


if __name__ == "__main__":
    main()
