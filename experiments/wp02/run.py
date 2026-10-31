"""Run WP02's bounded deterministic experiment; --curate copies small evidence."""

import argparse
import json
import shutil
from pathlib import Path

from mm_allee.wp02_experiment import run_experiment

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--curate", action="store_true")
    args = parser.parse_args()
    manifest = run_experiment(json.loads(args.config.read_text()), args.output, ROOT)
    if args.curate:
        for path in args.output.glob("wp02_*"):
            if path.name == "wp02_history_samples.csv":
                continue  # bulk trajectories remain local
            folder = "tables" if path.suffix == ".csv" else "figures"
            target = ROOT / "artifacts" / folder / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        shutil.copyfile(args.output / "manifest.json", ROOT / "artifacts/manifests/wp02_run.json")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "elapsed_seconds": manifest["elapsed_seconds"],
                "history": manifest["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
