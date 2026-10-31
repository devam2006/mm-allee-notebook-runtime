"""Analytical basin classification and event-free extinction quadrature.

No finite solver horizon is used to decide long-term outcomes. Numerical roots
are used only for the two subcritical A boundaries, with explicit uncertainty.
"""

from dataclasses import dataclass
from math import atan, expm1, isfinite, log, log1p, sin, sqrt, tan

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from .models import growth
from .thresholds import checked_a, checked_number, critical_a, critical_b

ROOT_TOL = 2e-12


def positive_branches(model, *, a, harvest):
    """Return lower/upper positive branches, a double fold pair, or None.

    Only exact floating equality to the calculated critical control is called
    the fold. Adjacent distinct controls are not silently merged. At q=0,
    return (a,1); the separate zero equilibrium is not a positive branch.
    """
    a = checked_a(a)
    control = checked_number(harvest, "harvest", nonnegative=True)
    if model not in {"A", "B"}:
        raise ValueError("Model must be A or B")
    if control == 0:
        return a, 1.0
    if model == "B":
        ec = critical_b(a)
        if control > ec:
            return None
        centre = (1 + a) / 2
        if control == ec:
            return centre, centre
        offset = sqrt(a * (ec - control))
        upper = centre + offset
        # The quadratic has root product a*(1+e). Subtraction at the
        # lower branch loses relative accuracy when a is small.
        return a * (1 + control) / upper, upper
    star, qc = critical_a(a)
    if control > qc:
        return None
    if control == qc:
        return star, star
    if control < 0.9 * qc:
        # Near zero catch, subtracting two O(q_c) Taylor terms destroys the
        # endpoint signs. Here the original product form has exact -control
        # endpoint values and is well conditioned away from the fold.
        def field(x):
            return growth(x, a) - control

        lower = brentq(field, a, star, xtol=np.nextafter(0.0, 1.0), rtol=1e-15)
        upper = brentq(field, star, 1, xtol=np.nextafter(0.0, 1.0), rtol=1e-15)
        return lower, upper
    d = sqrt(1 - a + a * a)

    def equation(offset):
        # Exact cubic Taylor identity about its maximum, avoiding G(x)-q
        # cancellation close to the fold. Solve the offsets before adding star.
        return offset * offset * (d + offset) / a - (qc - control)

    left = brentq(equation, a - star, 0, xtol=np.nextafter(0.0, 1.0), rtol=1e-15)
    right = brentq(equation, 0, 1 - star, xtol=np.nextafter(0.0, 1.0), rtol=1e-15)
    lower, upper = star + left, star + right
    if lower >= upper:
        raise ArithmeticError("Distinct A roots cannot be resolved at float precision")
    return lower, upper


@dataclass(frozen=True)
class Basin:
    model: str
    a: float
    harvest: float
    x0: float
    regime: str
    outcome: str
    limiting_x: float | None
    boundary_x: float | None
    equilibrium_type: str | None
    root_tolerance: float


def classify_basin(model, *, a, harvest, x0, root_tol=ROOT_TOL):
    """Classify exact roots explicitly; flag non-exact starts within root_tol.

    root_tol is an absolute dimensionless uncertainty band around the unstable
    or semistable boundary. It does not alter the critical control or merge
    branches. An exact match to a returned root is recognised as equilibrium.
    """
    a = checked_a(a)
    control = checked_number(harvest, "harvest", nonnegative=True)
    initial = checked_number(x0, "x0", nonnegative=True)
    tolerance = checked_number(root_tol, "root_tol", nonnegative=True)
    roots = positive_branches(model, a=a, harvest=control)
    critical = critical_a(a)[1] if model == "A" else critical_b(a)
    regime = (
        "zero_harvest"
        if control == 0
        else "subcritical"
        if control < critical
        else "critical"
        if control == critical
        else "supercritical"
    )
    boundary = None if roots is None else roots[0]

    def result(outcome, limit, equilibrium=None):
        return Basin(
            model, a, control, initial, regime, outcome, limit, boundary, equilibrium, tolerance
        )

    if initial == 0:
        if model == "A" and control > 0:
            return result("initial_extinction", 0)
        return result("equilibrium", 0, "stable_zero")
    if roots is None:
        return result("finite_extinction" if model == "A" else "asymptotic_extinction", 0)
    lower, upper = roots
    if control > 0 and model == "A" and initial == lower == a:
        # A tiny positive catch can move the root by less than one float ulp.
        # A returned endpoint approximation is not an exact equilibrium.
        return result("boundary_unresolved", None)
    if control > 0 and model == "A" and initial == upper == 1:
        return result("persistence", upper)
    if initial == lower:
        return result("equilibrium", initial, "semistable" if lower == upper else "unstable")
    if initial == upper:
        return result("equilibrium", initial, "stable_positive")
    if abs(initial - lower) <= tolerance:
        return result("boundary_unresolved", None)
    if initial < lower:
        finite = model == "A" and control > 0
        return result("finite_extinction" if finite else "asymptotic_extinction", 0)
    return result("persistence", upper)


@dataclass(frozen=True)
class ExtinctionTime:
    basin: Basin
    tau_ext: float | None
    quadrature_error: float | None
    method: str


def _finite_quad(integrand, lower, upper, absolute, relative):
    output = quad(
        integrand, lower, upper, epsabs=absolute, epsrel=relative, limit=250, full_output=1
    )
    if len(output) != 3:
        raise RuntimeError("Extinction quadrature failed to converge: " + str(output[3]))
    value, error, _ = output
    if not isfinite(value) or value < 0 or not isfinite(error):
        raise ArithmeticError("Invalid extinction quadrature result")
    return float(value), float(error)


def extinction_time_a(*, a, q, x0, epsabs=1e-10, epsrel=1e-10, root_tol=ROOT_TOL):
    """Finite tau_ext, or None with an explicit non-finite/unresolved basin status.

    Analytic basin classification excludes equilibrium poles before integrating.
    The reported QUADPACK error is an estimate, not a rigorous error bound.
    A tangent substitution resolves the supercritical bottleneck when q~q_c.
    At the fold an elementary antiderivative avoids an unresolved endpoint
    peak; quadrature_error is None because no numerical quadrature is used.
    """
    absolute = checked_number(epsabs, "epsabs", positive=True)
    relative = checked_number(epsrel, "epsrel", positive=True)
    basin = classify_basin("A", a=a, harvest=q, x0=x0, root_tol=root_tol)
    if basin.outcome == "initial_extinction":
        return ExtinctionTime(basin, 0.0, 0.0, "initial_boundary")
    if basin.outcome != "finite_extinction":
        return ExtinctionTime(basin, None, None, "not_applicable")
    a, q, x0 = basin.a, basin.harvest, basin.x0
    star, qc = critical_a(a)
    d = sqrt(1 - a + a * a)
    beta = d / a
    delta = q - qc
    if q == qc:
        # q_c-G(x)=(x_star-x)^2*(d-x_star+x)/a. Partial fractions
        # give this positive, cancellation-free integral for x0<x_star.
        gap = star - x0
        d_minus_star = (1 - a) ** 2 / (2 * d + 1 + a)
        value = a * x0 / (d * gap * star) + a / d**2 * log1p(d * x0 / (gap * d_minus_star))
        if not isfinite(value) or value <= 0:
            raise ArithmeticError("Fold extinction time exceeds float resolution")
        return ExtinctionTime(basin, value, None, "fold_closed_form")
    if q < qc:
        lower_root, upper_root = positive_branches("A", a=a, harvest=q)
        if x0 > 0.99 * lower_root:
            # Resolve the logarithmic endpoint growth without subtracting
            # nearly equal production/catch values. Factoring the cubic gives
            # q-G(x)=(x-r_negative)*(lower-x)*(upper-x)/a.
            split = lower_root / 2
            negative_root = -a * q / (lower_root * upper_root)
            first, first_error = _finite_quad(
                lambda x: 1 / (q - growth(x, a)), 0, split, absolute / 2, relative
            )

            def transformed(u):
                x = -lower_root * expm1(-u)
                distance = lower_root * np.exp(-u)
                return a / ((x - negative_root) * (upper_root - lower_root + distance))

            second, second_error = _finite_quad(
                transformed,
                log(2),
                log1p(x0 / (lower_root - x0)),
                absolute / 2,
                relative,
            )
            return ExtinctionTime(
                basin, first + second, first_error + second_error, "boundary_log_quad"
            )
    method = "direct_quad"
    if 0 < delta < 0.1 * qc and x0 > star:
        scale = sqrt(delta / beta)
        lower, upper = atan(-star / scale), atan((x0 - star) / scale)

        def integrand(theta):
            offset = scale * tan(theta)
            return 1 / (sqrt(beta * delta) * (1 + offset * sin(theta) ** 2 / d))

        method = "bottleneck_tangent_quad"
    else:
        lower, upper = 0.0, x0

        def integrand(x):
            offset = x - star
            # Use the stable cubic identity near the maximum only. Farther
            # away, q-G(x) avoids cancellation for tiny positive q at x~0.
            denominator = (
                delta + offset * offset * (d + offset) / a
                if abs(delta) < 0.1 * qc and abs(offset) < 0.2
                else q - growth(x, a)
            )
            if denominator <= 0:
                raise ArithmeticError("Quadrature path too close to an unresolved equilibrium")
            return 1 / denominator

    value, error = _finite_quad(integrand, lower, upper, absolute, relative)
    return ExtinctionTime(basin, value, error, method)


def safety_margins(model, parameters, initial_population):
    """Control and initial-state basin margins, with dimensional population units."""
    p = parameters
    a = p.A / p.K
    control = p.H / (p.r * p.K) if model == "A" else p.h / p.r
    critical = critical_a(a)[1] if model == "A" else critical_b(a)
    basin = classify_basin(model, a=a, harvest=control, x0=initial_population / p.K)
    return {
        "model": model,
        "control_fraction": control / critical,
        "control_margin": 1 - control / critical,
        "basin_boundary_N": None if basin.boundary_x is None else p.K * basin.boundary_x,
        "population_margin": None
        if basin.boundary_x is None
        else initial_population - p.K * basin.boundary_x,
        "outcome": basin.outcome,
    }
