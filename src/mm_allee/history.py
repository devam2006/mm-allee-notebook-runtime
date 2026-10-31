"""Piecewise-constant harvest histories using the shared event-aware solver.

Repeated boundary times deliberately encode left/right limits and declared
restocking jumps. No population reset occurs at an ordinary control change.
A's absorbing continuation is a shutdown policy, not its original vector field.
B never receives an extinction event or a small-population reset.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .models import rhs
from .solver import solve_trajectory


@dataclass
class History:
    rows: list[dict]
    extinction_times: list[float]


def run_history(
    model,
    *,
    a,
    x0,
    times,
    controls,
    restocking=None,
    n_points=41,
    rtol=1e-9,
    atol=1e-11,
    max_step=0.1,
):
    """Integrate controls[i] on [times[i],times[i+1]].

    restocking maps segment index to an *added* population in x=N/K units.
    It is applied at that segment's start, after recording its pre-jump state.
    """
    times = np.asarray(times, dtype=float)
    controls = np.asarray(controls, dtype=float)
    if model not in {"A", "B"}:
        raise ValueError("History supports A or B")
    if (
        times.ndim != 1
        or controls.ndim != 1
        or len(controls) < 1
        or len(times) != len(controls) + 1
        or not np.all(np.isfinite(times))
        or times[0] != 0
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Times must start at zero and strictly increase; one control per interval")
    if not np.all(np.isfinite(controls)) or np.any(controls < 0):
        raise ValueError("Controls must be finite and nonnegative")
    if not isfinite(x0) or x0 < 0:
        raise ValueError("Initial population must be finite and nonnegative")
    rhs(x0, model, a=a, harvest=float(controls[0]))
    if not isinstance(n_points, int) or isinstance(n_points, bool) or n_points < 2:
        raise ValueError("n_points must be an integer >=2")
    if not all(isfinite(v) and v > 0 for v in (rtol, atol, max_step)):
        raise ValueError("Positive finite solver tolerances required")
    restocking = dict(restocking or {})
    for i, amount in restocking.items():
        if (
            not isinstance(i, int)
            or isinstance(i, bool)
            or not 0 <= i < len(controls)
            or not isfinite(amount)
            or amount <= 0
        ):
            raise ValueError("Restocking requires valid segment indices and positive additions")
    rows, events = [], []
    state = float(x0)
    absorbed = False

    def add(t, x, i, label, shutdown=False):
        control = float(controls[i])
        effective = 0.0 if shutdown else control
        rows.append(
            dict(
                tau=float(t),
                x=float(x),
                segment=i,
                control=control,
                effective_control=effective,
                label=label,
                yield_dimensionless=effective * (x if model == "B" else 1.0),
            )
        )

    for i, control in enumerate(controls):
        start, end = times[i : i + 2]
        if i in restocking:
            add(start, state, i, "restock_before", absorbed)
            state += restocking[i]
            absorbed = False
            add(start, state, i, "restock_after")
        if absorbed:
            add(start, 0, i, "shutdown", True)
            add(end, 0, i, "shutdown", True)
            continue
        sol = solve_trajectory(
            model,
            a=a,
            harvest=float(control),
            x0=state,
            tau_end=float(end - start),
            n_points=n_points,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        for j, (t, x) in enumerate(zip(sol.tau, sol.x, strict=True)):
            label = "segment_start" if j == 0 else "sample"
            add(start + t, x, i, label)
        state = float(sol.x[-1])
        if sol.extinction_tau is not None:
            hit = float(start + sol.extinction_tau)
            rows[-1]["label"] = "extinction"
            events.append(hit)
            absorbed = True
            add(hit, 0, i, "shutdown", True)
            if hit < end:
                add(end, 0, i, "shutdown", True)
    return History(rows, events)


def ramp_schedule(
    critical,
    *,
    steps=12,
    dwell=2.0,
    peak_fraction=1.3,
    peak_hold=12.0,
    recovery_hold=15.0,
    restock_hold=20.0,
):
    """Staircase 0 -> above threshold -> 0, then two zero-control holds.

    The last interval is the optional restocking interval. Both runs share it.
    """
    if (
        not isinstance(steps, int)
        or isinstance(steps, bool)
        or steps < 2
        or not all(
            isfinite(v) and v > 0 for v in (critical, dwell, peak_hold, recovery_hold, restock_hold)
        )
        or not isfinite(peak_fraction)
        or peak_fraction <= 1
    ):
        raise ValueError("Positive durations/threshold, steps>=2 and peak_fraction>1 required")
    up = np.linspace(0, critical * peak_fraction, steps + 1)
    controls = np.r_[up, up[-2::-1], 0.0, 0.0]
    durations = np.full(len(controls), dwell)
    durations[steps] = peak_hold
    durations[-2:] = [recovery_hold, restock_hold]
    return np.r_[0, np.cumsum(durations)], controls, len(controls) - 1
