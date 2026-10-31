"""Generate and independently verify small clean A/B reference fixtures."""

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

from mm_allee.datasets import (
    generate_reference,
    sha256,
    validate_config,
    write_json,
    write_reference_files,
)
from mm_allee.plotting import save_figure
from mm_allee.reference_validation import validate_reference

ROOT = Path(__file__).resolve().parents[2]


def git_value(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(config, output):
    start = time.perf_counter()
    validate_config(config)
    references = [generate_reference(case, config.get("solver")) for case in config["cases"]]
    checks = [
        validate_reference(case, ref, config.get("solver"), **config.get("validation", {}))
        for case, ref in zip(config["cases"], references, strict=True)
    ]
    if any(check["status"] != "passed" for check in checks):
        raise RuntimeError("Numerical reference validation failed: " + json.dumps(checks))
    output = Path(output)
    paths = write_reference_files(output, references, config)
    validation_path = output / "validation.csv"
    with validation_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checks[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(checks)
    paths.append(validation_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ref in references:
        axis = axes[0 if ref.model == "A" else 1]
        axis.plot(ref.time, ref.population, label=ref.trajectory_id)
        if ref.extinction_time is not None:
            axis.plot(ref.extinction_time, 0, "ko", markersize=4)
    for model, axis in zip(("A", "B"), axes, strict=True):
        axis.set(
            title=f"Model {model}: clean synthetic RK45 references",
            xlabel=f"Time [{config['units']['time']}]",
            ylabel=f"Population [{config['units']['population']}]",
        )
        axis.legend(fontsize=7, framealpha=1)
        axis.grid(alpha=1, color="0.9")
        axis.set_ylim(bottom=0)
    fig.tight_layout()
    paths.extend(save_figure(fig, output / "wp03_rk45_reference"))
    plt.close(fig)
    source_paths = [
        Path(__file__),
        ROOT / "src/mm_allee/models.py",
        ROOT / "src/mm_allee/solver.py",
        ROOT / "src/mm_allee/datasets.py",
        ROOT / "src/mm_allee/reference_validation.py",
        ROOT / "src/mm_allee/plotting.py",
    ]
    git_status = git_value("status", "--porcelain")
    manifest = {
        "schema_version": 1,
        "run_id": "wp03-clean-reference-v1",
        "status": "passed",
        "is_synthetic": True,
        "evaluation_metadata_only": True,
        "scope": "RK45 reference milestone only; no noisy observations, splits or inference",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_value("rev-parse", "HEAD"),
        "git_dirty": None if git_status is None else bool(git_status),
        "source_files_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
        "configuration": config,
        "configuration_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "solver": "SciPy RK45 (Dormand-Prince 5(4)); dense output; terminal A zero event",
        "independent_checks": "Dimensional polynomial DOP853, tighter RK45, A quadrature where safe",
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
        "wall_seconds": time.perf_counter() - start,
        "noise": None,
        "splits": None,
        "seeds": None,
        "noise_and_splits_owner": "WP05; make and freeze observation files before fitting",
        "validation": checks,
        "files_sha256": {p.name: sha256(p) for p in paths},
        "paths_relative_to": "output_directory",
        "reproducibility": "CSV/conditions/truth repeat exactly in the same environment; manifest timestamps/runtime and figure metadata vary",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def curate_output(output):
    """Publish the verified small handoff into versioned repository locations."""
    output = Path(output)
    fixture = ROOT / "data/fixtures/wp03_reference"
    fixture.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((output / "manifest.json").read_text())
    curated_hashes = {}
    for name in manifest["files_sha256"]:
        destination = (
            ROOT / "artifacts/figures" / name if name.endswith((".png", ".eps")) else fixture / name
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (output / name).resolve() != destination.resolve():
            shutil.copyfile(output / name, destination)
        curated_hashes[str(destination.relative_to(ROOT))] = sha256(destination)
    table = ROOT / "artifacts/tables/wp03_reference_validation.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output / "validation.csv", table)
    curated_hashes[str(table.relative_to(ROOT))] = sha256(table)
    manifest["files_sha256"] = curated_hashes
    manifest["paths_relative_to"] = "repository_root"
    manifest["curated_locations"] = {
        "reference_files": "data/fixtures/wp03_reference/",
        "validation_table": "artifacts/tables/wp03_reference_validation.csv",
        "figures": "artifacts/figures/wp03_rk45_reference.png/.eps",
    }
    write_json(fixture / "manifest.json", manifest)
    target = ROOT / "artifacts/manifests/wp03_reference.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--curate",
        action="store_true",
        help="Also refresh the versioned small fixtures and evidence",
    )
    args = parser.parse_args()
    try:
        manifest = run(json.loads(args.config.read_text()), args.output)
        if args.curate:
            curate_output(args.output)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Reference generation failed: {exc}\n")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "trajectories": len(manifest["validation"]),
                "output": str(args.output),
                "wall_seconds": manifest["wall_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
