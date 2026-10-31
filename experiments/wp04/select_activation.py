"""Compare activation sweeps and apply the pre-declared selection rule (see README).

    python experiments/wp04/select_activation.py --inputs results/wp04_act_tanh \
        results/wp04_act_swish results/wp04_act_sigmoid --output artifacts/tables/wp04_activation_selection.csv

Rule: most accurate PINN runs (relative L2 <= threshold); ties -> smaller geometric mean of
the relative L2 over all PINN runs. Importing this module does nothing.
"""

import argparse
import csv
import math
import statistics
from pathlib import Path

THRESHOLD = 5e-2
COLUMNS = [
    "activation", "n_runs", "n_accurate", "n_invalid", "geomean_relative_l2", "median_relative_l2",
    "boundary_violations_total", "median_wall_seconds", "selected",
]  # fmt: skip


def load_pinn_rows(directory):
    path = Path(directory) / "wp04_forward_metrics.csv"
    with open(path, newline="") as handle:
        return [r for r in csv.DictReader(handle) if r["method"] == "pinn"]


def _errors(rows):
    """Relative L2 per run; None for a run with no valid score (failed / invalid prediction)."""
    return [float(r["relative_l2"]) if r["relative_l2"] not in ("", None) else None for r in rows]


def summarise(name, rows, threshold=THRESHOLD):
    """A run without a valid error counts as *not accurate* and is excluded from the error
    averages (``n_invalid`` reports how many there were)."""
    errs = _errors(rows)
    valid = [e for e in errs if e is not None]
    return {
        "activation": name,
        "n_runs": len(rows),
        "n_accurate": sum(e <= threshold for e in valid),
        "n_invalid": len(errs) - len(valid),
        "geomean_relative_l2": math.exp(sum(math.log(e) for e in valid) / len(valid))
        if valid
        else None,
        "median_relative_l2": statistics.median(valid) if valid else None,
        "boundary_violations_total": sum(int(r["boundary_violations"] or 0) for r in rows),
        "median_wall_seconds": statistics.median(float(r["wall_seconds"]) for r in rows),
    }


def select(summaries):
    """Most accurate runs wins; ties go to the smaller geometric-mean error (None counts as worst)."""
    return min(
        summaries,
        key=lambda s: (
            -s["n_accurate"],
            math.inf if s["geomean_relative_l2"] is None else s["geomean_relative_l2"],
        ),
    )["activation"]


def per_case(rows, threshold=THRESHOLD):
    out = {}
    for case in dict.fromkeys(r["case"] for r in rows):
        mine = _errors([r for r in rows if r["case"] == case])
        valid = [e for e in mine if e is not None]
        out[case] = (
            sum(e <= threshold for e in valid),
            len(mine),
            statistics.median(valid) if valid else None,
        )
    return out


def _fmt(value):
    return "n/a" if value is None else f"{value:.2e}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    summaries, cases = [], {}
    for directory in args.inputs:
        name = Path(directory).name.removeprefix("wp04_act_")
        rows = load_pinn_rows(directory)
        summaries.append(summarise(name, rows))
        cases[name] = per_case(rows)
    winner = select(summaries)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for s in summaries:
            row = {**s, "selected": s["activation"] == winner}
            for key in ("geomean_relative_l2", "median_relative_l2"):
                row[key] = "" if row[key] is None else f"{row[key]:.6e}"
            writer.writerow(row)
    for s in summaries:
        print(
            f"{s['activation']:8s} accurate {s['n_accurate']:>2d}/{s['n_runs']} (invalid {s['n_invalid']})  "
            f"geomean relL2 {_fmt(s['geomean_relative_l2'])}  median {_fmt(s['median_relative_l2'])}  "
            f"neg-samples {s['boundary_violations_total']}  median {s['median_wall_seconds']:.1f}s"
        )
        for case, (ok, n, med) in cases[s["activation"]].items():
            print(f"    {case:13s} {ok}/{n} accurate, median relL2 {_fmt(med)}")
    print(f"SELECTED: {winner}")


if __name__ == "__main__":
    main()
