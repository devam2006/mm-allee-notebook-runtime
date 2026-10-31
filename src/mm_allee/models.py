"""Dimensional and dimensionless vector fields; never clamp the mathematical RHS."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Parameters:
    r: float
    K: float
    A: float
    H: float = 0.0
    h: float = 0.0

    def __post_init__(self):
        if not all(isfinite(v) for v in (self.r, self.K, self.A, self.H, self.h)):
            raise ValueError("Parameters must be finite")
        if self.r <= 0 or self.K <= 0 or not 0 < self.A < self.K:
            raise ValueError("Require r>0, K>0 and 0<A<K")
        if self.H < 0 or self.h < 0:
            raise ValueError("Harvest controls must be nonnegative")

    def dimensionless(self):
        return {"a": self.A / self.K, "q": self.H / (self.r * self.K), "e": self.h / self.r}


def growth(x, a):
    """Allee production in x=N/K, tau=r*t; accepts scalars or NumPy arrays."""
    if not isfinite(a) or not 0 < a < 1:
        raise ValueError("Require 0<a<1")
    return x * (1 - x) * (x / a - 1)


def rhs(x, model, *, a, harvest):
    """q is harvest for logistic/A; e is harvest for B. Negative x aids root finding."""
    if not isfinite(harvest) or harvest < 0:
        raise ValueError("Harvest must be finite and nonnegative")
    if model == "logistic":
        return x * (1 - x) - harvest
    if model == "A":
        return growth(x, a) - harvest
    if model == "B":
        return growth(x, a) - harvest * x
    raise ValueError("Model must be 'logistic', 'A' or 'B'")


def dimensional_rhs(N, model, parameters):
    """Population/time vector field, with consistent dimensional parameters."""
    p = parameters
    d = p.dimensionless()
    harvest = d["e"] if model == "B" else d["q"]
    return p.r * p.K * rhs(N / p.K, model, a=d["a"], harvest=harvest)
