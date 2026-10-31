"""Small deterministic WP03 studies; data first, plotting in the CLI."""

import time
from dataclasses import asdict
from math import pi, sqrt

import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from .collapse import classify_basin, extinction_time_a, positive_branches, safety_margins
from .models import Parameters
from .solver import solve_trajectory
from .thresholds import (
    checked_a,
    checked_number,
    critical_a,
    critical_b,
    dimensional_thresholds,
    maximum_yields,
    parameters_with,
    sustainable_yield_b,
    threshold_sensitivity,
)


def validate_study_config(config):
    if config.get("schema_version") != 1:
        raise ValueError("Expected WP03 schema_version 1")
    p = Parameters(**config["parameters"])
    if p.H != 0 or p.h != 0:
        raise ValueError("Study base parameters omit harvest; sweeps supply it")
    for key in ("time", "population"):
        if (
            not isinstance(config.get("units", {}).get(key), str)
            or not config["units"][key].strip()
        ):
            raise ValueError("Declare time and population units")
    for key in ("curve_points", "basin_control_points", "basin_initial_points"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or not 3 <= value <= 201:
            raise ValueError(f"{key} must be an integer in [3,201] for this CPU study")
    if not config["a_sweep"]:
        raise ValueError("a_sweep cannot be empty")
    for a in config["a_sweep"]:
        checked_a(a)
    for key in (
        "max_initial_x",
        "max_harvest_fraction",
        "bottleneck_initial_x",
        "runtime_budget_seconds",
    ):
        checked_number(config[key], key, positive=True)
    if config["bottleneck_initial_x"] <= critical_a(p.A / p.K)[0]:
        raise ValueError("Bottleneck starts above x_star")
    for key in ("initial_curve_harvest_fractions", "harvest_curve_initial_x"):
        if not config[key]:
            raise ValueError(f"{key} cannot be empty")
        for value in config[key]:
            checked_number(value, key, nonnegative=True)
    for key in ("bottleneck_fractions_above_fold", "trajectory_fractions_above_fold"):
        if not config[key]:
            raise ValueError(f"{key} cannot be empty")
        for value in config[key]:
            checked_number(value, key, positive=True)
    checked_number(config["root_tolerance"], "root_tolerance", nonnegative=True)
    for value in config["quadrature"].values():
        checked_number(value, "quadrature tolerance", positive=True)
    for value in config["event_validation"].values():
        checked_number(value, "event tolerance", positive=True)
    return p


def build_study(config):
    p = validate_study_config(config)
    start = time.perf_counter()

    def budget():
        if time.perf_counter() - start > config["runtime_budget_seconds"]:
            raise RuntimeError("WP03 runtime budget exceeded; no results published")

    a = p.A / p.K
    star, qc = critical_a(a)
    ec = critical_b(a)
    tables = {
        key: []
        for key in (
            "thresholds",
            "extinction",
            "basins",
            "bottleneck",
            "sensitivity",
            "yield",
            "event_validation",
        )
    }
    for ratio in config["a_sweep"]:
        budget()
        polynomial = Polynomial([0, -1, (1 + ratio) / ratio, -1 / ratio])
        per_capita = Polynomial([-1, (1 + ratio) / ratio, -1 / ratio])
        exact_x, exact_q = critical_a(ratio)
        independent_a = minimize_scalar(
            lambda x: -polynomial(x), bounds=(ratio, 1), method="bounded", options={"xatol": 1e-14}
        )
        independent_b = minimize_scalar(
            lambda x: -per_capita(x), bounds=(ratio, 1), method="bounded", options={"xatol": 1e-14}
        )
        if not independent_a.success or not independent_b.success:
            raise RuntimeError("Independent threshold maximisation did not converge")
        q_error = abs(exact_q + independent_a.fun)
        e_error = abs(critical_b(ratio) + independent_b.fun)
        if q_error > 1e-10 * max(1, exact_q) or e_error > 1e-10 * max(1, critical_b(ratio)):
            raise RuntimeError("Independent threshold maximum check failed")
        tables["thresholds"].append(
            {
                "a": ratio,
                "r": p.r,
                "K": p.K,
                "A": ratio * p.K,
                "x_star": exact_x,
                "q_c": exact_q,
                "e_c": critical_b(ratio),
                "H_c": p.r * p.K * exact_q,
                "h_c": p.r * critical_b(ratio),
                "H_c_logistic": p.r * p.K / 4,
                "q_c_numeric": -independent_a.fun,
                "e_c_numeric": -independent_b.fun,
                "q_error": q_error,
                "e_error": e_error,
                "derivative_residual": float(polynomial.deriv()(exact_x)),
                "status": "passed",
            }
        )

    def extinction_row(kind, fraction, initial):
        budget()
        result = extinction_time_a(
            a=a,
            q=fraction * qc,
            x0=float(initial),
            root_tol=config["root_tolerance"],
            **config["quadrature"],
        )
        tables["extinction"].append(
            {
                "curve": kind,
                "harvest_fraction": fraction,
                "x0": float(initial),
                "q": fraction * qc,
                "H": fraction * qc * p.r * p.K,
                "N0": float(initial) * p.K,
                "outcome": result.basin.outcome,
                "tau_ext": result.tau_ext,
                "t_ext": None if result.tau_ext is None else result.tau_ext / p.r,
                "quadrature_error_tau": result.quadrature_error,
                "method": result.method,
            }
        )

    for fraction in config["initial_curve_harvest_fractions"]:
        for initial in np.linspace(0, config["max_initial_x"], config["curve_points"]):
            extinction_row("vary_initial", fraction, initial)
    for initial in config["harvest_curve_initial_x"]:
        for fraction in np.linspace(0, config["max_harvest_fraction"], config["curve_points"]):
            extinction_row("vary_harvest", float(fraction), initial)

    controls = np.linspace(0, config["max_harvest_fraction"], config["basin_control_points"])
    initials = np.linspace(0, config["max_initial_x"], config["basin_initial_points"])
    grids = {}
    codes = {
        "finite_extinction": 0,
        "initial_extinction": 0,
        "asymptotic_extinction": 1,
        "persistence": 2,
        "equilibrium": 2,
        "boundary_unresolved": 4,
    }
    for model, critical in (("A", qc), ("B", ec)):
        grid = np.empty((len(initials), len(controls)), dtype=int)
        for j, fraction in enumerate(controls):
            budget()
            for i, initial in enumerate(initials):
                label = classify_basin(
                    model,
                    a=a,
                    harvest=float(fraction * critical),
                    x0=float(initial),
                    root_tol=config["root_tolerance"],
                )
                grid[i, j] = (
                    3 if label.outcome == "equilibrium" and initial == 0 else codes[label.outcome]
                )
        grids[model] = grid
        for fraction in (0, 0.5, 1, 1.1):
            roots = positive_branches(model, a=a, harvest=fraction * critical)
            states = (
                [0, a / 2, 1.1] if roots is None else [0, roots[0] / 2, roots[0], roots[1], 1.1]
            )
            for initial in sorted(set(states)):
                label = classify_basin(
                    model,
                    a=a,
                    harvest=fraction * critical,
                    x0=initial,
                    root_tol=config["root_tolerance"],
                )
                tables["basins"].append(asdict(label) | {"control_fraction": fraction})

    target = pi / sqrt(sqrt(1 - a + a * a) / a)
    for fraction in config["bottleneck_fractions_above_fold"]:
        budget()
        q = qc * (1 + fraction)
        result = extinction_time_a(
            a=a, q=q, x0=config["bottleneck_initial_x"], **config["quadrature"]
        )
        tables["bottleneck"].append(
            {
                "fraction_above_fold": fraction,
                "delta_q": q - qc,
                "tau_ext": result.tau_ext,
                "scaled_time": result.tau_ext * sqrt(q - qc),
                "asymptotic_scaled_time": target,
                "method": result.method,
            }
        )

    for row in threshold_sensitivity(p):
        key, name = row["parameter"], row["threshold"]
        step = min(1e-5 * getattr(p, key), 0.1 * (p.K - p.A))
        plus = dimensional_thresholds(parameters_with(p, **{key: getattr(p, key) + step}))[name]
        minus = dimensional_thresholds(parameters_with(p, **{key: getattr(p, key) - step}))[name]
        fd = (plus - minus) / (2 * step)
        if not np.isclose(fd, row["partial"], rtol=2e-7, atol=1e-9):
            raise RuntimeError("Threshold sensitivity finite-difference check failed")
        tables["sensitivity"].append(
            row
            | {"finite_difference": fd, "step": step, "absolute_error": abs(fd - row["partial"])}
        )
    msy = maximum_yields(p)
    numeric_msy = minimize_scalar(
        lambda e: -sustainable_yield_b(a, e)[1],
        bounds=(0, ec),
        method="bounded",
        options={"xatol": 1e-14},
    )
    if not numeric_msy.success or not np.isclose(
        msy["Y_MSY_B"], -numeric_msy.fun * p.r * p.K, rtol=1e-9
    ):
        raise RuntimeError("B MSY numerical maximum check failed")
    for fraction in np.linspace(0, 1, config["curve_points"]):
        x, y = sustainable_yield_b(a, float(fraction * ec))
        tables["yield"].append(
            {
                "effort_fraction": float(fraction),
                "h": float(fraction * ec * p.r),
                "N_upper": p.K * x,
                "realised_yield": p.r * p.K * y,
                "stability": "semistable" if fraction == 1 else "stable_positive",
            }
        )

    # Representative finite paths are checked against two RK45 tolerances
    # and an independently expanded polynomial/DOP853, including the fold.
    lower_subcritical = positive_branches("A", a=a, harvest=0.75 * qc)[0]
    cases = [(0.25, a / 2), (0.75, (a + lower_subcritical) / 2), (1, 0.8 * star)]
    cases += [
        (1 + f, config["bottleneck_initial_x"]) for f in config["trajectory_fractions_above_fold"]
    ]
    for fraction, initial in cases:
        budget()
        result = extinction_time_a(a=a, q=qc * fraction, x0=initial, **config["quadrature"])
        horizon = 1.1 * result.tau_ext + 1
        values = []
        for tol in (1e-8, 1e-10):
            values.append(
                solve_trajectory(
                    "A",
                    a=a,
                    harvest=qc * fraction,
                    x0=initial,
                    tau_end=horizon,
                    rtol=tol,
                    atol=tol / 100,
                    max_step=0.1,
                ).extinction_tau
            )
        polynomial = Polynomial([-qc * fraction, -1, (1 + a) / a, -1 / a])

        def event(t, y):
            return y[0]

        event.terminal, event.direction = True, -1
        independent = solve_ivp(
            lambda t, y: polynomial(y),
            (0, horizon),
            [initial],
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
            max_step=0.1,
            events=event,
        )
        if (
            not independent.success
            or not len(independent.t_events[0])
            or any(v is None for v in values)
        ):
            raise RuntimeError("Independent event integration failed")
        alternate = float(independent.t_events[0][0])
        maximum_error = max(abs(v - result.tau_ext) for v in [*values, alternate])
        tolerance = (
            config["event_validation"]["absolute_tau_tolerance"]
            + config["event_validation"]["relative_tolerance"] * result.tau_ext
        )
        if maximum_error > tolerance:
            raise RuntimeError("Extinction event validation tolerance exceeded")
        tables["event_validation"].append(
            {
                "control_fraction": fraction,
                "x0": initial,
                "quadrature_tau": result.tau_ext,
                "rk45_tau": values[0],
                "tighter_rk45_tau": values[1],
                "dop853_tau": alternate,
                "max_abs_error_tau": maximum_error,
                "tolerance_tau": tolerance,
                "status": "passed",
            }
        )

    trajectories = []
    for fraction in config["trajectory_fractions_above_fold"]:
        budget()
        result = extinction_time_a(a=a, q=qc * (1 + fraction), x0=config["bottleneck_initial_x"])
        solution = solve_trajectory(
            "A",
            a=a,
            harvest=qc * (1 + fraction),
            x0=config["bottleneck_initial_x"],
            tau_end=1.1 * result.tau_ext + 1,
            n_points=301,
        )
        trajectories.append((fraction, solution))
    policies = [
        safety_margins("A", parameters_with(p, H=0.8 * msy["H_c"]), 0.9 * p.K),
        safety_margins("B", parameters_with(p, h=msy["h_MSY"]), 0.9 * p.K),
    ]
    budget()
    return {
        "parameters": p,
        "tables": tables,
        "controls": controls,
        "initials": initials,
        "basin_grids": grids,
        "trajectories": trajectories,
        "msy": msy,
        "policies": policies,
        "wall_seconds_analysis": time.perf_counter() - start,
    }
