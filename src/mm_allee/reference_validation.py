"""Independent numerical checks for WP03; not a general basin classifier."""

import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import quad, solve_ivp

from .datasets import generate_reference, solver_settings
from .solver import MIN_RK45_RTOL


def expanded_field(parameters, model):
    """Independent dimensional polynomial, rather than the production RHS helper."""
    p = parameters
    return Polynomial(
        [
            -p.H if model == "A" else 0,
            -p.r if model == "A" else -(p.r + p.h),
            p.r * (1 / p.A + 1 / p.K),
            -p.r / (p.A * p.K),
        ]
    )


def extinction_quadrature(parameters, initial_population):
    """A only: integrate 1/(-f) if f is strictly negative over the whole path.

    Endpoints and every real interior critical point bound this cubic's maximum.
    A conservative numerical margin rejects near-pole/fold paths; this is not
    interval-arithmetic certification or a replacement for the later basin proof.
    """
    if parameters.H <= 0 or initial_population <= 0:
        return None
    polynomial = expanded_field(parameters, "A")
    candidates = [0.0, initial_population]
    candidates += [
        float(z.real)
        for z in polynomial.deriv().roots()
        if abs(z.imag) < 1e-10 and 0 < z.real < initial_population
    ]
    margin = 1e-10 * max(1, parameters.r * parameters.K, parameters.H)
    if max(float(polynomial(x)) for x in candidates) >= -margin:
        return None
    return float(
        quad(lambda x: -1 / polynomial(x), 0, initial_population, epsabs=1e-11, epsrel=1e-11)[0]
    )


def validate_reference(case, reference, solver=None, population_fraction_K=5e-8, event_atol=1e-7):
    """Compare with refined RK45 and independent dimensional DOP853.

    Refine atol/max_step and reduce rtol only to the supported floor; record
    that effective value so an already accurate baseline remains valid.
    """
    if not (
        np.isfinite(population_fraction_K)
        and population_fraction_K > 0
        and np.isfinite(event_atol)
        and event_atol > 0
    ):
        raise ValueError("Validation tolerances must be finite and positive")
    settings = solver_settings(solver)
    p = reference.parameters
    polynomial = expanded_field(p, reference.model)
    initial = case["initial_population"]
    if initial == 0:
        raise ValueError("Numerical evidence cases require positive initial populations")

    def event(t, state):
        return state[0]

    event.terminal = True
    event.direction = -1
    alternate = solve_ivp(
        lambda t, y: polynomial(y),
        (0, case["time_end"]),
        [initial],
        method="DOP853",
        dense_output=True,
        events=event if reference.model == "A" and p.H > 0 else None,
        rtol=1e-12,
        atol=p.K * 1e-14,
        max_step=settings["max_step"] / (2 * p.r),
    )
    if not alternate.success:
        raise RuntimeError(alternate.message)
    reference_event = None
    if alternate.t_events and len(alternate.t_events[0]):
        reference_event = float(alternate.t_events[0][0])
    if (reference.extinction_time is None) != (reference_event is None):
        raise RuntimeError("RK45/DOP853 disagree on event occurrence")
    # Never extrapolate a numerical reference beyond its physical event.
    mask = reference.time <= alternate.t[-1]
    error = float(
        np.max(np.abs(reference.population[mask] - alternate.sol(reference.time[mask])[0]))
    )
    fine_case = dict(case, time_points=reference.time.tolist())
    refinement_rtol = max(MIN_RK45_RTOL, settings["rtol"] / 100)
    fine = generate_reference(
        fine_case,
        {
            "rtol": refinement_rtol,
            "atol": settings["atol"] / 100,
            "max_step": settings["max_step"] / 2,
        },
    )
    common = reference.time < min(reference.time[-1], fine.time[-1])
    if reference.extinction_time is None:
        common = np.ones(len(reference.time), dtype=bool)
    count = int(common.sum())
    if count < 2:
        raise RuntimeError("Insufficient common samples for convergence check")
    if not np.allclose(
        reference.time[common],
        fine.time[:count],
        rtol=0,
        atol=16 * np.finfo(float).eps * max(1, case["time_end"]),
    ):
        raise RuntimeError("Refinement samples do not align")
    refinement = float(np.max(np.abs(reference.population[common] - fine.population[:count])))
    event_error = (
        None if reference_event is None else abs(reference.extinction_time - reference_event)
    )
    refinement_event_error = (
        None
        if reference.extinction_time is None
        else abs(reference.extinction_time - fine.extinction_time)
    )
    integral = extinction_quadrature(p, initial) if reference.model == "A" else None
    integral_error = None
    if integral is not None and reference.extinction_time is not None:
        integral_error = abs(reference.extinction_time - integral)
    accepted = error / p.K <= population_fraction_K and refinement / p.K <= population_fraction_K
    accepted &= all(
        v is None or v <= event_atol for v in (event_error, refinement_event_error, integral_error)
    )
    return {
        "trajectory_id": reference.trajectory_id,
        "model": reference.model,
        "max_abs_population_error_DOP853": error,
        "max_abs_population_change_tighter_RK45": refinement,
        "population_tolerance": population_fraction_K * p.K,
        "extinction_time": reference.extinction_time,
        "event_difference_DOP853": event_error,
        "event_change_tighter_RK45": refinement_event_error,
        "extinction_quadrature_time": integral,
        "event_difference_quadrature": integral_error,
        "event_tolerance": event_atol,
        "refinement_rtol": refinement_rtol,
        "refinement_rtol_at_floor": bool(refinement_rtol == MIN_RK45_RTOL),
        "status": "passed" if accepted else "failed",
    }
