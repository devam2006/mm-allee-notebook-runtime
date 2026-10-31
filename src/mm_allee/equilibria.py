"""Equilibria and stability for the logistic baseline and Model A (constant yield).

Everything here uses the dimensionless variables of the scientific contract:
``x = N/K``, ``tau = r t``, ``a = A/K`` and ``q = H/(rK)``. The vector fields are

* logistic: ``f(x) = x(1-x) - q``
* Model A:  ``f(x) = G(x) - q`` with ``G(x) = x(1-x)(x/a - 1) = x(1-x)(x-a)/a``.

An equilibrium is a root of ``f``. Stability is decided from the sign of ``f`` just
below and just above the root (a phase-line argument), not from ``f'`` alone, so the
repeated root at the fold is classified correctly as semistable. The linearisation
``f'(x_eq)`` is reported alongside and agrees with the sign test at simple roots.

Root finding avoids a generic polynomial solver. The calculus of Section 1 of the
report brackets every root between points where ``f`` changes sign, and Brent's method
(``scipy.optimize.brentq``) refines it to a *relative* tolerance, so small roots near
``x = 0`` keep their sign and magnitude.

Numerical conventions (dimensionless, documented in the report):

* Model A is evaluated internally as ``x(1-x)(x-a)/a - q``. For ``x`` in ``[a, 1]``
  with ``a >= 1/2`` the differences ``1-x`` and ``x-a`` are exact in floating point
  (Sterbenz), so ``G`` keeps full relative accuracy even when ``q_c`` is tiny (``a``
  near 1). It equals ``models.rhs`` algebraically; tests check the parity.
* ``q_c`` uses a rationalised closed form without cancellation (see :func:`fold_a`).
* The integrated topology uses exact equality to the shared computed critical
  control. FOLD_RTOL remains an uncertainty scale, not a root-merging rule.
* Sign probes sit at ``x_eq +/- delta`` with ``delta = min(1e-4, d/4)``, where ``d`` is
  the distance to the nearest other real root of the polynomial (excluded negative
  roots included) and, for positive roots, to the boundary ``x = 0``. A probe can
  therefore never cross another root or leave the physical domain, and the
  classification never depends on whether excluded roots are displayed.

For ``q > 0`` the point ``x = 0`` is **not** an equilibrium of Model A or the logistic
model (``f(0) = -q``). Trajectories reach it in finite time; the solver stops there
(absorbing extinction convention). That boundary is reported separately by
:func:`extinction_boundary`, never inside the equilibrium list.
"""

from dataclasses import asdict, dataclass
from math import isfinite, sqrt

import numpy as np
from scipy.optimize import brentq

from .collapse import positive_branches
from .thresholds import critical_a

EPS = float(np.finfo(float).eps)
FOLD_RTOL = 64 * EPS
BRENT_RTOL = 4 * EPS
BRENT_XTOL = 1e-300  # effectively pure relative termination
BRENT_MAXITER = 500
PROBE_MAX = 1e-4

STABLE = "stable"
UNSTABLE = "unstable"
SEMISTABLE = "semistable"
EXCLUDED = "excluded"


@dataclass(frozen=True)
class Equilibrium:
    """One root of the dimensionless vector field and its classification.

    ``stability`` is ``stable``, ``unstable`` or ``semistable`` for physical roots and
    ``excluded`` for a negative algebraic root. ``attracts_from`` says from which
    side nearby trajectories approach (``both``, ``above``, ``below`` or ``neither``).
    ``left_sign``/``right_sign`` are the signs of ``f`` at the probes; ``left_sign`` is
    ``None`` at the physical boundary ``x = 0`` where only ``x >= 0`` matters.
    """

    model: str
    a: float | None
    q: float
    x: float
    physical: bool
    multiplicity: int
    derivative: float
    left_sign: int | None
    right_sign: int | None
    stability: str
    attracts_from: str
    residual: float
    note: str

    def as_row(self):
        return asdict(self)


def _validate_a(a):
    if not isfinite(a) or not 0 < a < 1:
        raise ValueError("Require 0<a<1")


def _validate_q(q):
    if not isfinite(q) or q < 0:
        raise ValueError("Harvest q must be finite and nonnegative")


def _f_logistic(q):
    def f(x):
        return x * (1 - x) - q

    return f


def _f_a(a, q):
    """Model A vector field in the cancellation-free form ``x(1-x)(x-a)/a - q``."""

    def f(x):
        return x * (1 - x) * ((x - a) / a) - q

    return f


def growth_derivative(x, a):
    """``G'(x) = -3x^2/a + 2(1+a)x/a - 1``, derived by differentiating ``growth``."""
    _validate_a(a)
    return -3 * x**2 / a + 2 * (1 + a) * x / a - 1


def rhs_derivative(x, model, *, a=None, q=0.0):
    """Derivative of the logistic or Model A vector field with respect to ``x``.

    The constant harvest ``q`` does not change the derivative; it is accepted for a
    uniform call signature and validated.
    """
    _validate_q(q)
    if model == "logistic":
        return 1 - 2 * x
    if model == "A":
        return growth_derivative(x, a)
    raise ValueError("rhs_derivative supports model 'logistic' or 'A'")


def fold_a(a):
    """Fold (saddle-node) point of Model A: ``(x_star, q_c)``.

    ``G'(x) = 0`` is the quadratic ``3x^2 - 2(1+a)x + a = 0`` with roots
    ``((1+a) +/- s)/3``, ``s = sqrt(1-a+a^2)``. The larger root lies in ``(a, 1)``
    because ``G'(a) = 1-a > 0`` and ``G'(1) = (a-1)/a < 0``; it is the maximum of ``G``
    on ``[a, 1]``. Rationalising the two small factors,

    ``1 - x_star = (1-a)/(2-a+s)`` and ``x_star - a = (1-a)(1-a+s)/(2-a+s)``,

    so ``q_c = G(x_star) = x_star (1-a)^2 (1-a+s) / (a (2-a+s)^2)``. Every term is a
    sum of positive quantities, so ``q_c`` has full relative accuracy even as
    ``a -> 1`` (where the expanded form
    ``[2s^3 + (1+a)(2a-1)(a-2)]/(27a)`` loses all digits to cancellation).
    This agrees with WP03's ``thresholds.critical_a``.
    """
    return critical_a(a)


def _sign(value):
    return int(np.sign(value))


def _classify(f, deriv, x_eq, others, *, boundary=False):
    """Phase-line classification from one-sided signs of ``f``.

    ``others`` lists every other real root of the polynomial. For a positive root the
    physical boundary ``0`` also limits the probe distance.
    """
    limits = [abs(x_eq - other) for other in others if other != x_eq]
    if x_eq > 0:
        limits.append(x_eq)
    delta = min([PROBE_MAX] + [d / 4 for d in limits])
    right = _sign(f(x_eq + delta))
    left = None if boundary else _sign(f(x_eq - delta))
    # A probe value can only round to exactly zero if the root is extremely flat; the
    # derivative sign then supplies the local direction.
    if right == 0:
        right = _sign(deriv)
    if left == 0:
        left = -_sign(deriv)
    if boundary:
        # Only x >= 0 is physical: the root attracts if trajectories just above move down.
        if right < 0:
            return STABLE, "above", left, right
        return UNSTABLE, "neither", left, right
    if left > 0 and right < 0:
        return STABLE, "both", left, right
    if left < 0 and right > 0:
        return UNSTABLE, "neither", left, right
    if left < 0 and right < 0:
        return SEMISTABLE, "above", left, right
    if left > 0 and right > 0:
        return SEMISTABLE, "below", left, right
    raise RuntimeError(f"Sign probes undetermined near x={x_eq}")


def _records(
    model,
    a,
    q,
    physical_roots,
    f,
    *,
    excluded_roots=(),
    multiplicities=None,
    boundary_roots=(),
    notes=None,
    include_excluded=False,
):
    """Classify physical roots against *all* real roots; output is flag-independent."""
    multiplicities = multiplicities or {}
    notes = notes or {}
    all_roots = list(excluded_roots) + list(physical_roots)
    out = []
    for x_eq in physical_roots:
        deriv = float(rhs_derivative(x_eq, model, a=a, q=q))
        if model == "A" and q > 0 and len(physical_roots) == 2:
            # The factored cubic gives exact branch signs even when a floating
            # sign probe rounds onto an almost-coincident root near a=1.
            lower, upper = physical_roots
            negative = excluded_roots[0]
            direction = 1 if x_eq == lower else -1
            deriv = direction * (x_eq - negative) * (upper - lower) / a
            left, right = -direction, direction
            stability = UNSTABLE if direction > 0 else STABLE
            attracts = "neither" if direction > 0 else "both"
        else:
            stability, attracts, left, right = _classify(
                f, deriv, x_eq, all_roots, boundary=x_eq in boundary_roots
            )
        out.append(
            Equilibrium(
                model=model,
                a=a,
                q=float(q),
                x=float(x_eq),
                physical=True,
                multiplicity=multiplicities.get(x_eq, 1),
                derivative=deriv,
                left_sign=left,
                right_sign=right,
                stability=stability,
                attracts_from=attracts,
                residual=float(abs(f(x_eq))),
                note=notes.get(x_eq, ""),
            )
        )
    if include_excluded:
        for x_eq in excluded_roots:
            if not x_eq < 0:  # pragma: no cover - guarded by _negative_root_a
                raise RuntimeError("Excluded root must be negative")
            out.append(
                Equilibrium(
                    model=model,
                    a=a,
                    q=float(q),
                    x=float(x_eq),
                    physical=False,
                    multiplicity=1,
                    derivative=float(rhs_derivative(x_eq, model, a=a, q=q)),
                    left_sign=None,
                    right_sign=None,
                    stability=EXCLUDED,
                    attracts_from="neither",
                    residual=float(abs(f(x_eq))),
                    note="negative algebraic root; not a population",
                )
            )
    return sorted(out, key=lambda r: r.x)


def _is_critical(q, q_c):
    return q == q_c


def equilibria_logistic(q, *, include_excluded=False):
    """Equilibria of ``dx/dtau = x(1-x) - q``.

    Roots are ``(1 +/- sqrt(1-4q))/2``. The smaller root is computed as ``q/x_plus``
    (product of roots equals ``q``) to avoid cancellation when ``q`` is small.
    Logistic roots are never negative for ``q >= 0``; ``include_excluded`` is accepted
    for a uniform interface.
    """
    _validate_q(q)
    f = _f_logistic(q)
    q_c = 0.25
    if q == 0:
        return _records(
            "logistic",
            None,
            q,
            [0.0, 1.0],
            f,
            boundary_roots=(0.0,),
            notes={0.0: "extinct state; unstable for logistic growth", 1.0: "carrying capacity"},
        )
    if _is_critical(q, q_c):
        return _records(
            "logistic",
            None,
            q,
            [0.5],
            f,
            multiplicities={0.5: 2},
            notes={0.5: "fold: double root at x=1/2 (maximum sustainable constant yield)"},
        )
    if q > q_c:
        return []
    disc = sqrt(1 - 4 * q)
    upper = (1 + disc) / 2
    lower = q / upper
    return _records(
        "logistic",
        None,
        q,
        [lower, upper],
        f,
        notes={lower: "harvest-induced threshold", upper: "persistent state"},
    )


def _brent(f, lo, hi):
    return brentq(f, lo, hi, xtol=BRENT_XTOL, rtol=BRENT_RTOL, maxiter=BRENT_MAXITER)


def _negative_root_a(a, q):
    """Single negative root of ``G(x) = q`` for ``q > 0``.

    For ``x < 0``, ``G(x) > 0`` and ``G'(x) < 0`` so ``G`` decreases from ``+inf`` to 0
    and crosses ``q`` exactly once. Since ``G(x) ~ -x`` near 0 the root is close to
    ``-q`` when ``q`` is small; the bracket starts there and Brent's method runs to a
    relative tolerance, so even a root of size ``1e-15`` stays strictly negative.
    It is a polynomial root but not a population.
    """
    f = _f_a(a, q)
    lo = -2.0 * q
    while f(lo) <= 0:
        lo *= 2
    root = _brent(f, lo, 0.0)  # f(0) = -q < 0, f(lo) > 0
    if not root < 0:
        raise RuntimeError("Negative root bracketing failed")
    return root


def equilibria_a(a, q, *, include_excluded=False):
    """Equilibria of Model A, ``dx/dtau = x(1-x)(x/a-1) - q``.

    Physical cases (proved in the report via the sign and monotonicity of ``G``):

    * ``q = 0``: roots ``0, a, 1`` (stable, unstable, stable).
    * ``0 < q < q_c``: one root in ``(a, x_star)`` (unstable) and one in
      ``(x_star, 1)`` (stable).
    * ``q == q_c``: one double root at ``x_star`` (semistable,
      attracts from above).
    * ``q > q_c``: no physical equilibrium.

    For ``q > 0`` the cubic always has one negative root. It is always used to bound
    the sign probes but is returned only when ``include_excluded=True``, with
    ``physical=False`` and ``stability='excluded'``. Physical records are identical
    either way.
    """
    _validate_a(a)
    _validate_q(q)
    f = _f_a(a, q)
    x_star, q_c = fold_a(a)

    if q == 0:
        return _records(
            "A",
            a,
            q,
            [0.0, a, 1.0],
            f,
            boundary_roots=(0.0,),
            notes={
                0.0: "extinct state; stable under strong Allee growth",
                a: "Allee threshold (minimum viable population)",
                1.0: "carrying capacity",
            },
            include_excluded=include_excluded,
        )

    excluded = [_negative_root_a(a, q)]
    if _is_critical(q, q_c):
        physical = [x_star]
        mult = {x_star: 2}
        notes = {x_star: "fold: double root, one-sided signs decide stability"}
    elif q < q_c:
        lower, upper = positive_branches("A", a=a, harvest=q)
        physical = [lower, upper]
        mult = {}
        notes = {lower: "harvest-shifted Allee threshold", upper: "persistent state"}
    else:
        physical, mult, notes = [], {}, {}
    return _records(
        "A",
        a,
        q,
        physical,
        f,
        excluded_roots=excluded,
        multiplicities=mult,
        notes=notes,
        include_excluded=include_excluded,
    )


def extinction_boundary(model, q):
    """Describe ``x = 0`` for constant-yield models without calling it an equilibrium."""
    _validate_q(q)
    if model not in {"logistic", "A"}:
        raise ValueError("Constant-yield boundary applies to 'logistic' or 'A'")
    if q == 0:
        return {"x": 0.0, "is_equilibrium": True, "policy": "zero_equilibrium"}
    return {
        "x": 0.0,
        "is_equilibrium": False,
        "rhs_at_zero": -float(q),
        "policy": "stop_at_zero (absorbing extinction; harvest shuts down)",
    }


def classify_equilibrium(model, x_eq, *, a=None, q=0.0):
    """Classify a user-supplied root by one-sided signs (convenience for WP02/WP03).

    Raises if ``x_eq`` is not within ``1e-10`` (relative to ``max(1, |x|)``) of a root.
    """
    _validate_q(q)
    if model == "A":
        _validate_a(a)
        records = equilibria_a(a, q)
    elif model == "logistic":
        records = equilibria_logistic(q)
    else:
        raise ValueError("classify_equilibrium supports 'logistic' or 'A'")
    for record in records:
        if abs(record.x - x_eq) <= 1e-10 * max(1.0, abs(x_eq)):
            return record
    raise ValueError(f"x={x_eq} is not an equilibrium of {model} at q={q}")


def branch_rows(model, q_values, *, a=None):
    """Long-form branch data for plots and CSV export.

    One row per physical equilibrium per ``q``. ``branch`` is ``upper``/``lower`` for
    the two populated branches, ``fold`` at the double root and ``zero``/``threshold``/
    ``capacity`` at ``q = 0``.
    """
    rows = []
    for q in q_values:
        q = float(q)
        records = equilibria_a(a, q) if model == "A" else equilibria_logistic(q)
        for i, rec in enumerate(records):
            if q == 0:
                branch = ["zero", "threshold" if model == "A" else "capacity", "capacity"][i]
            elif rec.multiplicity == 2:
                branch = "fold"
            else:
                branch = "lower" if i == 0 and len(records) == 2 else "upper"
            rows.append({**rec.as_row(), "branch": branch})
    return rows
