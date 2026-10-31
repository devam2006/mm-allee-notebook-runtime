"""Reference RK45 solver with explicit physical-domain and extinction semantics."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy.integrate import solve_ivp

from .models import rhs

# SciPy RK45 raises smaller requests to this floor. Reject them so requested
# tolerances remain the effective settings recorded by experiment manifests.
MIN_RK45_RTOL = 100 * np.finfo(float).eps


@dataclass(frozen=True)
class Trajectory:
    tau: np.ndarray
    x: np.ndarray
    extinction_tau: float | None
    model: str
    boundary_policy: str


def solve_trajectory(
    model,
    *,
    a,
    harvest,
    x0,
    tau_end,
    n_points=301,
    rtol=1e-9,
    atol=1e-11,
    max_step=0.1,
    tau_eval=None,
):
    """Return only the physical pre-extinction trajectory, including its event endpoint.

    For A/logistic with positive constant yield, stop at zero; this wrapper does
    not append a post-extinction zero tail. B/zero-yield have no finite zero event.
    Numerical negative roundoff no larger than 10*atol is rounded to zero only in
    returned samples, never in the vector field or used to claim B extinction.
    Optional tau_eval is a strictly increasing finite grid beginning at zero,
    bounded by tau_end. Samples after extinction are removed and its exact
    numerical event endpoint is appended. Internal RK steps remain adaptive.
    """
    if not all(isfinite(v) for v in (a, harvest, x0, tau_end, rtol, atol, max_step)):
        raise ValueError("Inputs must be finite")
    if not 0 < a < 1 or x0 < 0 or tau_end <= 0:
        raise ValueError("Require 0<a<1, x0>=0 and tau_end>0")
    if not isinstance(n_points, int) or isinstance(n_points, bool) or n_points < 2:
        raise ValueError("n_points must be an integer >=2")
    if min(rtol, atol, max_step) <= 0:
        raise ValueError("Solver tolerances and max_step must be positive")
    if rtol < MIN_RK45_RTOL:
        raise ValueError(f"RK45 rtol must be at least {MIN_RK45_RTOL:.17g}")
    rhs(x0, model, a=a, harvest=harvest)  # validate model and harvest
    requested = None if tau_eval is None else np.asarray(tau_eval, dtype=float)
    if requested is not None and (
        requested.ndim != 1
        or requested.size == 0
        or not np.all(np.isfinite(requested))
        or requested[0] != 0
        or requested[-1] > tau_end
        or np.any(np.diff(requested) <= 0)
    ):
        raise ValueError("tau_eval must increase strictly from zero within [0,tau_end]")
    constant_yield = model in {"A", "logistic"} and harvest > 0
    if x0 == 0:
        if constant_yield:
            return Trajectory(np.array([0.0]), np.array([0.0]), 0.0, model, "stop_at_zero")
        t = np.linspace(0, tau_end, n_points) if requested is None else requested.copy()
        return Trajectory(t, np.zeros_like(t), None, model, "zero_equilibrium")

    def vector_field(tau, state):
        return [rhs(state[0], model, a=a, harvest=harvest)]

    def extinction(tau, state):
        return state[0]

    extinction.terminal = True
    extinction.direction = -1
    result = solve_ivp(
        vector_field,
        (0.0, tau_end),
        [x0],
        method="RK45",
        dense_output=True,
        events=extinction if constant_yield else None,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not result.success:
        raise RuntimeError(f"Reference integration failed: {result.message}")
    hit = None
    if constant_yield and len(result.t_events[0]):
        hit = float(result.t_events[0][0])
    stop = tau_end if hit is None else hit
    grid = np.linspace(0, stop, n_points) if requested is None else requested[requested <= stop]
    if hit is not None and grid[-1] != hit:
        grid = np.append(grid, hit)
    population = result.sol(grid)[0]
    if np.min(population) < -10 * atol:
        raise RuntimeError("Reference solver produced negative physical samples; tighten settings")
    population = np.maximum(population, 0.0)
    if hit is not None:
        population[-1] = 0.0
    return Trajectory(
        grid,
        population,
        hit,
        model,
        "stop_at_zero" if constant_yield else "smooth_nonnegative_dynamics",
    )
