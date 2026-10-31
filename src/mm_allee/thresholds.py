"""Calculus thresholds and optional dimensional sensitivity/yield quantities."""

from math import isfinite, sqrt
from numbers import Real

from .models import Parameters


def checked_number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f"Invalid {name}")
    return float(value)


def checked_a(a):
    a = checked_number(a, "a")
    if not 0 < a < 1:
        raise ValueError("Require 0<a<1")
    return a


def critical_a(a):
    """Return (x_star,q_c) for Model A; q_c is NOT logistic's 1/4.

    Computing 1-x_star by rationalisation avoids subtracting two near-unit
    numbers in q_c as a approaches 1. x_star itself is still a float.
    """
    a = checked_a(a)
    d = sqrt(1 - a + a * a)
    gap = (1 - a) / (2 - a + d)
    x_star = 1 - gap
    return x_star, x_star * gap * ((1 - a) - gap) / a


def critical_b(a):
    """Dimensionless critical effort; zero remains an equilibrium above it."""
    a = checked_a(a)
    return (1 - a) ** 2 / (4 * a)


def dimensional_thresholds(parameters):
    p = parameters
    x, q = critical_a(p.A / p.K)
    return {
        "N_star": p.K * x,
        "H_c": p.r * p.K * q,
        "h_c": p.r * critical_b(p.A / p.K),
        "H_c_logistic": p.r * p.K / 4,
    }


def threshold_sensitivity(parameters):
    """Partials hold the other DIMENSIONAL r,K,A fixed (envelope theorem for Hc)."""
    p = parameters
    thresholds = dimensional_thresholds(p)
    n = thresholds["N_star"]
    derivatives = {
        "H_c": {
            "r": thresholds["H_c"] / p.r,
            "K": p.r * n**2 * (n / p.A - 1) / p.K**2,
            "A": -p.r * n**2 * (1 - n / p.K) / p.A**2,
        },
        "h_c": {
            "r": thresholds["h_c"] / p.r,
            "K": p.r * (1 / p.A - p.A / p.K**2) / 4,
            "A": p.r * (-p.K / p.A**2 + 1 / p.K) / 4,
        },
    }
    return [
        {
            "threshold": name,
            "parameter": key,
            "partial": derivative,
            "elasticity": getattr(p, key) * derivative / thresholds[name],
        }
        for name, partials in derivatives.items()
        for key, derivative in partials.items()
    ]


def sustainable_yield_b(a, effort):
    """(positive upper equilibrium, realised yield e*x_plus), or None above fold.

    This is an equilibrium-branch quantity, not a prediction for all initial
    populations. At the fold the positive equilibrium is only semistable.
    """
    a = checked_a(a)
    e = checked_number(effort, "effort", nonnegative=True)
    ec = critical_b(a)
    if e > ec:
        return None
    x_plus = (1 + a) / 2 + sqrt(a * (ec - e))
    return x_plus, e * x_plus


def maximum_yields(parameters):
    """A stable-branch yield supremum and B interior MSY; neither guarantees safety."""
    p = parameters
    thresholds = dimensional_thresholds(p)
    n = thresholds["N_star"]
    effort = thresholds["H_c"] / n
    return thresholds | {
        "h_MSY": effort,
        "Y_MSY_B": thresholds["H_c"],
        "B_effort_margin_at_MSY": 1 - effort / thresholds["h_c"],
        "B_basin_boundary_at_MSY": p.K + p.A - n,
        "A_control_margin_at_supremum": 0.0,
    }


def parameters_with(parameters, **updates):
    """Validated dimensional parameter replacement for finite-difference studies."""
    return Parameters(**(vars(parameters) | updates))
