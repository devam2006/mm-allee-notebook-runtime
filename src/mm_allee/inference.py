"""Conventional nonlinear least-squares ODE fitting baseline using SciPy.

Serves as an independent, deterministic reference baseline using the exact same
training observations, fixed reference scales, unknown parameters, and bounds as PINN.
"""

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares

from .metrics import errors
from .models import Parameters
from .solver import solve_trajectory


@dataclass(frozen=True)
class FitResult:
    model: str
    fit_params: list[str]
    estimated_parameters: dict[str, float]
    all_parameters: dict[str, float]
    status: str
    success: bool
    cost: float
    optimality: float
    wall_seconds: float
    train_metrics: dict[str, float | None]
    val_metrics: dict[str, float | None]
    test_metrics: dict[str, float | None]
    predicted_trajectory: dict[str, np.ndarray]


def _simulate_trajectory(
    model: str,
    params: dict[str, float],
    initial_pop: float,
    time_points: np.ndarray,
) -> np.ndarray:
    """Simulate trajectory at specified time points."""
    p = Parameters(
        r=float(params["r"]),
        K=float(params["K"]),
        A=float(params["A"]),
        H=float(params.get("H", 0.0)),
        h=float(params.get("h", 0.0)),
    )
    d = p.dimensionless()
    harvest = d["q"] if model == "A" else d["e"]
    t_end = float(time_points[-1])
    tau_eval = p.r * time_points

    traj = solve_trajectory(
        model,
        a=d["a"],
        harvest=harvest,
        x0=initial_pop / p.K,
        tau_end=p.r * t_end,
        tau_eval=tau_eval,
        rtol=1e-8,
        atol=1e-10,
    )
    # Reconstruct population on time_points
    # Note: solve_trajectory returns points up to extinction if an event occurred
    sim_pop = np.zeros_like(time_points, dtype=float)
    if traj.extinction_tau is not None:
        t_ext = traj.extinction_tau / p.r
        valid_mask = time_points <= t_ext
        n_valid = np.sum(valid_mask)
        sim_pop[:n_valid] = traj.x[:n_valid] * p.K
        # If simulated path goes extinct early while data exists, remaining is 0
        sim_pop[n_valid:] = 0.0
    else:
        # Interpolate or match grid directly
        if len(traj.tau) == len(time_points):
            sim_pop = traj.x * p.K
        else:
            sim_pop = np.interp(time_points, traj.tau / p.r, traj.x * p.K)
    return sim_pop


def fit_ode_least_squares(
    model: str,
    *,
    observations: dict[str, Any],
    known_conditions: dict[str, float],
    fit_params: Sequence[str],
    initial_guess: dict[str, float] | None = None,
    parameter_bounds: dict[str, tuple[float, float]] | None = None,
    n_ref: float = 100.0,
    max_nfev: int = 500,
) -> FitResult:
    """Fit unknown parameters using SciPy nonlinear least squares on training observations only.

    Parameters:
    -----------
    model : 'A' or 'B'
    observations : dict containing 'time', 'observed_population', 'train_indices',
                   'val_indices', 'test_indices'
    known_conditions : fixed known parameters (e.g. {'K': 100.0, 'N0': 45.0})
    fit_params : parameter names to estimate (e.g. ['r', 'A', 'H'] or ['r', 'A', 'h'])
    initial_guess : starting values for fit_params
    parameter_bounds : dict of (lower, upper) bounds for fit_params
    n_ref : fixed population reference scale for normalized residuals
    """
    times = np.asarray(observations["time"], dtype=float)
    obs_pop = np.asarray(observations["observed_population"], dtype=float)
    idx_train = observations["train_indices"]
    idx_val = observations["val_indices"]
    idx_test = observations["test_indices"]

    t_train = times[idx_train]
    y_train = obs_pop[idx_train]
    n0 = float(known_conditions.get("known_initial_population", y_train[0]))
    k_val = float(known_conditions.get("known_K", 100.0))

    # Default bounds and starting points respecting physical constraints
    # r > 0, 0 < A < K, H >= 0, h >= 0
    default_bounds = {
        "r": (1e-4, 10.0),
        "K": (1e-2, 500.0),
        "A": (1e-4, k_val - 1e-4 if "K" not in fit_params else 499.0),
        "H": (0.0, 500.0),
        "h": (0.0, 50.0),
    }
    bounds_dict = default_bounds.copy()
    if parameter_bounds:
        bounds_dict.update(parameter_bounds)

    default_guesses = {
        "r": 1.0,
        "K": k_val,
        "A": k_val * 0.3,
        "H": 5.0 if model == "A" else 0.0,
        "h": 0.2 if model == "B" else 0.0,
    }
    guess_dict = default_guesses.copy()
    if initial_guess:
        guess_dict.update(initial_guess)

    fit_params_list = list(fit_params)
    x0 = [guess_dict[p] for p in fit_params_list]
    lower = [bounds_dict[p][0] for p in fit_params_list]
    upper = [bounds_dict[p][1] for p in fit_params_list]

    # Fixed base parameters
    base_params = {
        "r": 1.0,
        "K": k_val,
        "A": 20.0,
        "H": 0.0,
        "h": 0.0,
    }
    for k, v in known_conditions.items():
        if k in base_params and k not in fit_params_list:
            base_params[k] = float(v)

    def residual_fn(vec: np.ndarray) -> np.ndarray:
        current_params = base_params.copy()
        for name, val in zip(fit_params_list, vec, strict=True):
            current_params[name] = float(val)

        # Enforce strict physical validity 0 < A < K
        if current_params["A"] >= current_params["K"]:
            return np.full_like(t_train, 1e4)

        try:
            sim = _simulate_trajectory(model, current_params, n0, t_train)
            res = (sim - y_train) / n_ref
            if not np.all(np.isfinite(res)):
                return np.full_like(t_train, 1e4)
            return res
        except Exception:
            return np.full_like(t_train, 1e4)

    start_time = time.perf_counter()
    try:
        opt_res = least_squares(
            residual_fn,
            x0=x0,
            bounds=(lower, upper),
            method="trf",
            max_nfev=max_nfev,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        wall_time = time.perf_counter() - start_time
        success = bool(opt_res.success)
        opt_status = opt_res.message
        cost = float(opt_res.cost)
        optimality = float(opt_res.optimality)

        est_dict = {name: float(val) for name, val in zip(fit_params_list, opt_res.x, strict=True)}
    except Exception as e:
        wall_time = time.perf_counter() - start_time
        success = False
        opt_status = f"Fit error: {e}"
        cost = float("nan")
        optimality = float("nan")
        est_dict = {name: float(guess_dict[name]) for name in fit_params_list}

    all_p = base_params.copy()
    all_p.update(est_dict)

    # Evaluate predictions on train, val, and test splits
    pred_all = np.zeros_like(times)
    try:
        pred_all = _simulate_trajectory(model, all_p, n0, times)
        train_err = errors(pred_all[idx_train], obs_pop[idx_train])
        val_err = errors(pred_all[idx_val], obs_pop[idx_val])
        test_err = errors(pred_all[idx_test], obs_pop[idx_test])
    except Exception:
        train_err = {"rmse": float("nan"), "mae": float("nan"), "relative_l2": None}
        val_err = {"rmse": float("nan"), "mae": float("nan"), "relative_l2": None}
        test_err = {"rmse": float("nan"), "mae": float("nan"), "relative_l2": None}

    return FitResult(
        model=model,
        fit_params=fit_params_list,
        estimated_parameters=est_dict,
        all_parameters=all_p,
        status="converged" if success else f"failed: {opt_status}",
        success=success,
        cost=cost,
        optimality=optimality,
        wall_seconds=wall_time,
        train_metrics=train_err,
        val_metrics=val_err,
        test_metrics=test_err,
        predicted_trajectory={"time": times, "population": pred_all},
    )
