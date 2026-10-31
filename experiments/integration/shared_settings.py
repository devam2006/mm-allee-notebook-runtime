"""Shared-settings study: one PINN setting that works for BOTH the forward (WP04) and the
inverse (WP05) problem.

    python experiments/integration/shared_settings.py --config configs/shared_stage1.json \
        --output results/shared_stage1

Forward scoring reuses ``experiments/wp04/run.py`` (same reference, labels and invalid-prediction
rule). The inverse side runs Rohan's ``InversePINNTrainer`` unchanged on the WP03 fixtures.
The selection rule is fixed in ``experiments/integration/README.md`` before any run:
score = (forward runs accurate, of 18) + (inverse fits recovered, of 18); ties -> lower median
parameter error over the recovered fits. Importing this module trains nothing.
"""

import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

FORWARD_COLUMNS = [
    "candidate", "case", "seed", "relative_l2", "status", "wall_seconds",
]  # fmt: skip
INVERSE_COLUMNS = [
    "candidate", "case", "seed", "sigma", "max_rel_error", "recovered", "param_errors",
    "min_r_A_estimate", "train_relative_l2", "train_rmse", "train_loss", "noise_floor_ratio",
    "err_over_crb", "crb_rel", "status", "wall_seconds",
]  # fmt: skip
SUMMARY_COLUMNS = [
    "candidate", "forward_accurate", "forward_runs", "inverse_recovered", "inverse_fits", "score",
    "median_recovered_error", "collapsed_fits", "median_forward_seconds", "median_inverse_seconds",
    "inverse_by_sigma", "selected",
]  # fmt: skip


def load_wp04_runner():
    spec = importlib.util.spec_from_file_location(
        "wp04_run_shared", ROOT / "experiments/wp04/run.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["wp04_run_shared"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- pure scoring (unit-tested)


def summarise_candidate(name, forward_rows, inverse_rows, recovery_threshold):
    """Counts for one candidate. A run with no valid score is never accurate / recovered."""
    f_ok = sum(r["status"] == "ok" for r in forward_rows)

    def is_recovered(r):
        if "recovered" in r:  # set by run_inverse (uses the Cramer-Rao criterion under noise)
            return bool(r["recovered"])
        return r["max_rel_error"] is not None and r["max_rel_error"] <= recovery_threshold

    inv_ok = [r for r in inverse_rows if is_recovered(r)]
    return {
        "candidate": name,
        "forward_accurate": f_ok,
        "forward_runs": len(forward_rows),
        "inverse_recovered": len(inv_ok),
        "inverse_fits": len(inverse_rows),
        "score": f_ok + len(inv_ok),
        "median_recovered_error": statistics.median(r["max_rel_error"] for r in inv_ok)
        if inv_ok
        else None,
        "collapsed_fits": sum(r["collapsed"] for r in inverse_rows if "collapsed" in r),
        "median_forward_seconds": statistics.median(r["wall_seconds"] for r in forward_rows)
        if forward_rows
        else None,
        "median_inverse_seconds": statistics.median(r["wall_seconds"] for r in inverse_rows)
        if inverse_rows
        else None,
    }


def choose(summaries):
    """Highest score wins; ties go to the lower median parameter error over recovered fits."""

    def key(s):
        err = s["median_recovered_error"]
        return (-s["score"], math.inf if err is None else err)

    return min(summaries, key=key)["candidate"]


# ---------------------------------------------------------------- forward


def run_forward(cand, cfg, wp04):
    rows = []
    fixed, ref_cfg = cfg["fixed"], cfg["forward"]["reference"]
    thr = cfg["forward"]["inaccuracy_threshold_relative_l2"]
    training = {
        "layers": [1, *cand["hidden"], 1],
        "activation": cand["activation"],
        "lr": cand["lr_net"],
        "n_collocation": cand.get("n_collocation", fixed["n_collocation"]),
        "adam_iters": fixed["forward_adam_iters"],
        "lbfgs_iters": fixed["lbfgs_iters"],
        "time_budget_s": 300,
    }
    for case in cfg["forward"]["cases"]:
        tau, ref_x, hit = wp04.reference_trajectory(case, ref_cfg)
        for seed in cfg["seeds"]:
            run_cfg = {"inaccuracy_threshold_relative_l2": thr, "training": training}
            result, _, ev, status, _ = wp04.fit_and_score(case, seed, run_cfg, (tau, ref_x), hit)
            rows.append(
                {
                    "candidate": cand["id"],
                    "case": case["id"],
                    "seed": seed,
                    "relative_l2": ev["relative_l2"],
                    "status": status,
                    "wall_seconds": 0.0 if result is None else round(result.wall_seconds, 3),
                }
            )
    return rows


# ---------------------------------------------------------------- inverse


def load_fixture(case_id):
    path = ROOT / f"data/fixtures/wp03_reference/{case_id}.csv"
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return np.array([float(r["time"]) for r in rows]), np.array(
        [float(r["population"]) for r in rows]
    )


def crb_relative_sd(model, fit_params, known, p_true, train_times, sigma_abs):
    """Cramer-Rao lower bound on the relative standard deviation of each fitted parameter.

    Fisher information of the ODE model at the true parameters on the training times with
    Gaussian noise of absolute standard deviation ``sigma_abs`` (central finite differences
    of the DOP853 trajectory). No estimator can do better on average; it is the yardstick
    for noisy recovery.
    """
    from scipy.integrate import solve_ivp

    K, N0 = known["known_K"], known["known_initial_population"]
    names = list(fit_params)
    theta0 = np.array([p_true[k] for k in names], dtype=float)
    t = np.asarray(train_times, dtype=float)

    def trajectory(theta):
        p = dict(p_true)
        p.update(dict(zip(names, theta)))
        r, A, H, h = p["r"], p["A"], p.get("H", 0.0), p.get("h", 0.0)
        if model == "A":
            f = lambda tt, N: [r * N[0] * (1 - N[0] / K) * (N[0] / A - 1) - H]  # noqa: E731
        else:
            f = lambda tt, N: [r * N[0] * (1 - N[0] / K) * (N[0] / A - 1) - h * N[0]]  # noqa: E731
        return solve_ivp(
            f, (0.0, t[-1]), [N0], t_eval=t, method="DOP853", rtol=1e-11, atol=1e-11
        ).y[0]

    J = np.zeros((len(t), len(names)))
    for j in range(len(names)):
        step = 1e-6 * abs(theta0[j])
        up, dn = theta0.copy(), theta0.copy()
        up[j] += step
        dn[j] -= step
        J[:, j] = (trajectory(up) - trajectory(dn)) / (2 * step)
    cov = np.linalg.inv(J.T @ J / sigma_abs**2)
    return {k: float(np.sqrt(cov[i, i]) / abs(theta0[i])) for i, k in enumerate(names)}


def run_inverse(cand, cfg, sigmas=None):
    from mm_allee.data import generate_noisy_observations
    from mm_allee.pinn.inverse import InversePINNTrainer, TrainableParameters
    from mm_allee.pinn.network import NetworkConfig

    inv = json.loads((ROOT / cfg["inverse"]["cases_config"]).read_text(encoding="utf-8"))
    truth = json.loads(
        (ROOT / "data/fixtures/wp03_reference/truth.json").read_text(encoding="utf-8")
    )["trajectories"]
    n_ref = inv["fixed_reference_scales"]["population"]
    fixed, thr = cfg["fixed"], cfg["inverse"]["recovery_threshold_relative_error"]
    rows = []
    for case in inv["cases"]:
        cid, model = case["case_id"], case["model"]
        times, pops = load_fixture(cid)
        p_true = truth[cid]["parameters"]
        for sigma in sigmas if sigmas is not None else cfg["inverse"]["sigmas"]:
            for seed in cfg["seeds"]:
                obs = generate_noisy_observations(
                    times,
                    pops,
                    trajectory_id=cid,
                    model=model,
                    noise_sigma=sigma,
                    noise_seed=seed,
                    n_ref=n_ref,
                )
                fit_seed = inv.get("fit_seed_offset", 1000) + seed
                ncfg = NetworkConfig(
                    layers=tuple([1, *cand["hidden"], 1]), t_end=float(times.max()),
                    x0=float(pops[0]) / n_ref, seed=fit_seed, activation=cand["activation"],
                )  # fmt: skip
                tp = TrainableParameters(
                    model=model, fit_params=case["fit_params"], known_conditions=case["known_conditions"],
                    initial_guesses=case["initial_guess"],
                )  # fmt: skip
                trainer = InversePINNTrainer(
                    network_config=ncfg, trainable_params=tp, lr_net=cand["lr_net"],
                    lr_params=cand["lr_params"], lambda_data=1.0, lambda_phys=cand["lambda_phys"],
                    lbfgs_iters=cand.get("lbfgs_iters", fixed["lbfgs_iters"]), lbfgs_ftol=cand.get("lbfgs_ftol", 0.0),
                    lbfgs_gtol=cand.get("lbfgs_gtol", 0.0), time_budget_s=300,
                    validation_selection=cand.get("validation_selection", True),
                )  # fmt: skip
                crb = None
                if sigma > 0:
                    train_t = np.asarray(obs["time"], dtype=float)[
                        np.asarray(obs["split"]) == "train"
                    ]
                    crb = crb_relative_sd(
                        model,
                        case["fit_params"],
                        case["known_conditions"],
                        p_true,
                        train_t,
                        sigma * n_ref,
                    )
                row = {"candidate": cand["id"], "case": cid, "seed": seed, "sigma": sigma}
                t0 = time.perf_counter()
                try:
                    res = trainer.fit(
                        obs, known_conditions=case["known_conditions"], epochs=cand.get("inverse_adam_epochs", fixed["inverse_adam_epochs"]),
                        patience=cand.get("patience", 150), n_collocation=cand.get("n_collocation", fixed["n_collocation"]), seed=fit_seed,
                        eval_every=10,
                    )  # fmt: skip
                    est = res.estimated_parameters
                    errs = {k: abs(est[k] - p_true[k]) / abs(p_true[k]) for k in case["fit_params"]}
                    finite = all(math.isfinite(v) for v in errs.values())
                    max_err = max(errs.values()) if finite else None
                    row.update(
                        max_rel_error=max_err,
                        recovered=bool(
                            finite
                            and (
                                all(errs[k] <= max(thr, 2 * crb[k]) for k in errs)
                                if crb
                                else max_err <= thr
                            )
                        ),
                        crb_rel=json.dumps(crb) if crb else "",
                        param_errors=json.dumps({k: float(v) for k, v in errs.items()}),
                        min_r_A_estimate=min(est["r"], est["A"]) if "r" in est and "A" in est else None,
                        collapsed=bool("r" in est and "A" in est and min(est["r"], est["A"]) < 1e-3 * min(p_true["r"], p_true["A"])),
                        train_relative_l2=res.train_metrics.get("relative_l2"),
                        train_rmse=res.train_metrics.get("rmse"),
                        train_loss=res.train_loss,
                        noise_floor_ratio=(
                            res.train_metrics["rmse"] / (sigma * n_ref)
                            if sigma > 0 and res.train_metrics.get("rmse") is not None
                            else None
                        ),
                        err_over_crb=max(errs[k] / crb[k] for k in errs) if crb else None,
                        status=res.status,
                    )  # fmt: skip
                except Exception as exc:  # noqa: BLE001  (a failed fit is recorded, not fatal)
                    row.update(
                        max_rel_error=None, recovered=False, param_errors="", min_r_A_estimate=None,
                        collapsed=False, train_relative_l2=None, status=f"failed_{type(exc).__name__}",
                    )  # fmt: skip
                row["wall_seconds"] = round(time.perf_counter() - t0, 3)
                rows.append(row)
    return rows


# ---------------------------------------------------------------- driver


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6e}"
    return value


def write_csv(path, columns, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k)) for k in columns})


def run(config_path, output):
    config_path, output = Path(config_path), Path(output)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    wp04 = load_wp04_runner() if cfg.get("forward") else None
    thr = cfg["inverse"]["recovery_threshold_relative_error"]
    started = time.perf_counter()
    all_f, all_i, summaries = [], [], []
    for cand in cfg["candidates"]:
        t0 = time.perf_counter()
        f_rows = run_forward(cand, cfg, wp04) if wp04 and cand.get("run_forward", True) else []
        i_rows = run_inverse(cand, cfg)
        all_f += f_rows
        all_i += i_rows
        s = summarise_candidate(cand["id"], f_rows, i_rows, thr)
        by_sigma = {}
        for r in i_rows:
            ok, n = by_sigma.get(r["sigma"], (0, 0))
            by_sigma[r["sigma"]] = (ok + bool(r["recovered"]), n + 1)
        s["inverse_by_sigma"] = "; ".join(
            f"sigma={k:g}: {v[0]}/{v[1]}" for k, v in sorted(by_sigma.items())
        )
        summaries.append(s)
        print(
            f"{cand['id']:34s} forward {s['forward_accurate']:>2d}/{s['forward_runs']}  inverse "
            f"{s['inverse_recovered']:>2d}/{s['inverse_fits']} [{s['inverse_by_sigma']}]  score {s['score']:>2d}  collapsed {s['collapsed_fits']}  "
            f"({time.perf_counter() - t0:.0f}s)", flush=True,
        )  # fmt: skip
    winner = choose(summaries)
    for s in summaries:
        s["selected"] = s["candidate"] == winner
    write_csv(output / "forward_runs.csv", FORWARD_COLUMNS, all_f)
    write_csv(output / "inverse_runs.csv", INVERSE_COLUMNS, all_i)
    write_csv(output / "summary.csv", SUMMARY_COLUMNS, summaries)
    import jax

    manifest = {
        "schema_version": 1,
        "scope": cfg.get("scope"),
        "command": f"python experiments/integration/shared_settings.py --config {config_path.as_posix()} --output {output.as_posix()}",
        "config": cfg,
        "jax": jax.__version__,
        "python": sys.version.split()[0],
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "selected": winner,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"SELECTED: {winner}")
    return summaries, winner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    run(args.config, args.output)


if __name__ == "__main__":
    main()
