"""Proportional-effort equilibria; all quantities are dimensionless.

The record mirrors WP01, but names the control e (not q: different units).
Sign labels follow the factorisation f=-x(x-x_minus)(x-x_plus)/a.
Only exact equality to the computed critical effort is called a fold: nearby
representable controls retain their distinct topology, without an absolute band.
"""

from dataclasses import asdict, dataclass
from math import isfinite, sqrt

from .models import rhs
from .thresholds import critical_b


@dataclass(frozen=True)
class EquilibriumB:
    model: str
    a: float
    e: float
    x: float
    physical: bool
    multiplicity: int
    derivative: float
    left_sign: int | None
    right_sign: int
    stability: str
    attracts_from: str
    residual: float
    note: str

    def as_row(self):
        return asdict(self)


def critical_effort(a):
    """Discriminant threshold (1-a)^2/(4a), in h/r units."""
    return critical_b(a)


def equilibria_b(a, e):
    """Return zero and admissible positive roots, ordered by population.

    The lower root uses the product a(1+e) to avoid cancellation at small a.
    At the fold f=-x(x-(1+a)/2)^2/a is negative on both sides.
    """
    ec = critical_effort(a)
    if not isfinite(e) or e < 0:
        raise ValueError("Effort must be finite and nonnegative")

    def record(x, mult, derivative, left, right, stability, attracts, note):
        return EquilibriumB(
            "B",
            a,
            e,
            x,
            True,
            mult,
            derivative,
            left,
            right,
            stability,
            attracts,
            abs(float(rhs(x, "B", a=a, harvest=e))),
            note,
        )

    roots = [record(0.0, 1, -1 - e, None, -1, "stable", "above", "zero equilibrium")]
    if e > ec:
        return roots
    if e == ec:
        roots.append(record((1 + a) / 2, 2, 0.0, -1, -1, "semistable", "above", "fold"))
        return roots
    # ec-e avoids subtracting two independently rounded discriminant terms.
    gap = sqrt(4 * a * (ec - e))
    upper = (1 + a + gap) / 2
    lower = a * (1 + e) / upper
    if e == 0:
        lower, upper = a, 1.0
    if lower >= upper:
        raise ArithmeticError("Positive root separation is below floating-point resolution")
    roots.extend(
        [
            record(lower, 1, lower * gap / a, -1, 1, "unstable", "neither", "basin boundary"),
            record(upper, 1, -upper * gap / a, 1, -1, "stable", "both", "populated state"),
        ]
    )
    return roots
