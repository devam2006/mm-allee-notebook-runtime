"""Analyse the noise study: optimiser convergence and parameter recovery versus noise level.

    python experiments/integration/noise_convergence.py --runs results/shared_noise/inverse_runs.csv

Reads the per-fit table written by ``shared_settings.py``. Two notions are kept apart:
optimiser convergence (training RMSE relative to the noise sd) and parameter recovery
(error relative to the Cramer-Rao bound, only where the bound is informative).
"""

import argparse
import csv
import json
import statistics

INFORMATIVE_BOUND = 0.25  # every parameter's Cramer-Rao relative sd must be <= this


def load(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for r in rows:
        r["sigma"] = float(r["sigma"])
        r["max_rel_error"] = float(r["max_rel_error"]) if r["max_rel_error"] else None
        r["noise_floor_ratio"] = float(r["noise_floor_ratio"]) if r["noise_floor_ratio"] else None
        r["err_over_crb"] = float(r["err_over_crb"]) if r["err_over_crb"] else None
        r["min_r_A"] = float(r["min_r_A_estimate"]) if r["min_r_A_estimate"] else None
        r["crb"] = json.loads(r["crb_rel"]) if r["crb_rel"] else None
        r["errs"] = json.loads(r["param_errors"]) if r["param_errors"] else None
    return rows


def informative(row):
    return row["crb"] is not None and max(row["crb"].values()) <= INFORMATIVE_BOUND


def summarise(rows, candidate, sigma):
    mine = [r for r in rows if r["candidate"] == candidate and r["sigma"] == sigma]
    out = {
        "fits": len(mine),
        "collapsed": sum(1 for r in mine if r["min_r_A"] is not None and r["min_r_A"] < 0.01),
    }
    errs = [r["max_rel_error"] for r in mine if r["max_rel_error"] is not None]
    out["within_5pct"] = sum(e <= 0.05 for e in errs)
    out["median_max_error"] = statistics.median(errs) if errs else None
    ratios = [r["noise_floor_ratio"] for r in mine if r["noise_floor_ratio"] is not None]
    if ratios:
        out["median_floor_ratio"] = statistics.median(ratios)
        out["at_floor"] = sum(0.5 <= x <= 2 for x in ratios)
        out["overfit"] = sum(x < 0.5 for x in ratios)
        out["underfit"] = sum(x > 2 for x in ratios)
    info = [r for r in mine if informative(r)]
    out["informative_fits"] = len(info)
    out["informative_cases"] = sorted({r["case"] for r in info})
    out["within_2crb"] = sum(
        1 for r in info if r["errs"] and all(r["errs"][k] <= 2 * r["crb"][k] for k in r["errs"])
    )
    ec = [r["err_over_crb"] for r in info if r["err_over_crb"] is not None]
    out["median_err_over_crb"] = statistics.median(ec) if ec else None
    return out


def fmt(x, spec=".2g"):
    return "n/a" if x is None else format(x, spec)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    args = parser.parse_args(argv)
    rows = load(args.runs)
    candidates = list(dict.fromkeys(r["candidate"] for r in rows))
    sigmas = sorted({r["sigma"] for r in rows})
    cases = list(dict.fromkeys(r["case"] for r in rows))
    for c in candidates:
        print(f"\n=== {c}")
        print(
            f"{'sigma':>7s} {'fits':>4s} {'<=5% err':>8s} {'collapsed':>9s} | {'floor ratio med':>15s} {'at/over/under':>14s} | {'informative fits':>16s} {'within 2xCRB':>12s} {'med err/CRB':>11s}"
        )
        for s in sigmas:
            o = summarise(rows, c, s)
            floor = f"{fmt(o.get('median_floor_ratio'))}" if "median_floor_ratio" in o else "n/a"
            split = (
                f"{o.get('at_floor', '-')}/{o.get('overfit', '-')}/{o.get('underfit', '-')}"
                if "at_floor" in o
                else "n/a"
            )
            print(
                f"{s:>7g} {o['fits']:>4d} {o['within_5pct']:>8d} {o['collapsed']:>9d} | {floor:>15s} {split:>14s} | {o['informative_fits']:>16d} {o['within_2crb']:>12d} {fmt(o['median_err_over_crb']):>11s}"
            )
        print(
            "  informative cases:",
            {f"{s:g}": summarise(rows, c, s)["informative_cases"] for s in sigmas if s > 0},
        )
    print("\n=== median of the worst-parameter relative error per case (3 seeds)")
    print(f"{'candidate':12s}{'sigma':>7s}" + "".join(f"{k:>14s}" for k in cases))
    for c in candidates:
        for s in sigmas:
            vals = []
            for k in cases:
                e = [
                    r["max_rel_error"]
                    for r in rows
                    if r["candidate"] == c
                    and r["sigma"] == s
                    and r["case"] == k
                    and r["max_rel_error"] is not None
                ]
                vals.append(f"{statistics.median(e):.1e}" if e else "n/a")
            print(f"{c:12s}{s:>7g}" + "".join(f"{v:>14s}" for v in vals))
    bounds = {}
    for r in rows:
        if r["crb"] and r["candidate"] == candidates[0]:
            bounds.setdefault(r["sigma"], {})[r["case"]] = max(r["crb"].values())
    print("\n=== Cramer-Rao bound (worst parameter, relative sd) per case")
    print(f"{'sigma':>7s}" + "".join(f"{k:>14s}" for k in cases))
    for s in sorted(bounds):
        print(f"{s:>7g}" + "".join(f"{bounds[s].get(k, float('nan')):>14.2g}" for k in cases))


if __name__ == "__main__":
    main()
