"""Explicit comparison protocols; theoretical basins and finite-horizon yields.

A/logistic roots are reused from WP01, not reimplemented. The curated design
uses a=0.2 and controls well outside WP01's unresolved near-fold rounding band.
"""

from math import isfinite

from scipy.integrate import trapezoid

from .collapse import classify_basin
from .equilibria import equilibria_a, equilibria_logistic, fold_a
from .model_b import critical_effort, equilibria_b
from .solver import solve_trajectory

# A conservative input exclusion for this experiment, independent of the
# equilibrium provider's definition of an exact fold.
COMPARISON_CONTROL_EXCLUSION = 1e-12


def comparison_controls(parameters, x_ref, fraction, protocol):
    """Return dimensionless controls, with reference N=K*x_ref.

    initial_catch uses fraction of A's critical yield as the common starting
    catch. threshold_fraction uses the same fraction of each model's threshold.
    """
    if not all(isfinite(v) and v > 0 for v in (x_ref, fraction)):
        raise ValueError("Positive finite reference population and fraction required")
    a = parameters.A / parameters.K
    # WP01's absolute fold band is known to be unsuitable near a=1.
    if not 0.05 <= a <= 0.95:
        raise ValueError("Comparisons currently validated only for 0.05<=a<=0.95")
    qc = fold_a(a)[1]
    q = fraction * qc
    if q != qc and abs(q - qc) <= COMPARISON_CONTROL_EXCLUSION:
        raise ValueError("A control lies in WP01's unresolved fold tolerance band")
    if protocol == "initial_catch":
        if q != 0.25 and abs(q - 0.25) <= COMPARISON_CONTROL_EXCLUSION:
            raise ValueError("Logistic control lies in WP01's unresolved fold tolerance band")
        return {"A": fraction * qc, "B": fraction * qc / x_ref, "logistic": fraction * qc}
    if protocol == "threshold_fraction":
        if fraction / 4 != 0.25 and abs(fraction / 4 - 0.25) <= COMPARISON_CONTROL_EXCLUSION:
            raise ValueError("Logistic control lies in WP01's unresolved fold tolerance band")
        return {"A": fraction * qc, "B": fraction * critical_effort(a), "logistic": fraction / 4}
    raise ValueError("Unknown comparison protocol")


def compare_policies(
    parameters,
    *,
    x0,
    fraction,
    protocol,
    tau_end=30,
    n_points=1201,
    rtol=1e-9,
    atol=1e-11,
    max_step=0.1,
):
    """Return rows in illustrative population/time units, no empirical claim.

    Outcome is inferred from the phase line, not lack of extinction in a short
    run. At exactly supplied equilibrium roots, report equilibrium explicitly.
    B harvest integral uses sampled trapezoidal quadrature, checked by refinement.
    """
    p = parameters
    a = p.A / p.K
    controls = comparison_controls(p, x0, fraction, protocol)
    rows = []
    for model, control in controls.items():
        records = (
            equilibria_b(a, control)
            if model == "B"
            else equilibria_a(a, control)
            if model == "A"
            else equilibria_logistic(control)
        )
        positive = [v for v in records if v.x > 0]
        boundary = positive[0].x if positive else None
        if model in {"A", "B"}:
            basin = classify_basin(model, a=a, harvest=control, x0=x0)
            boundary = basin.boundary_x
            outcome = (
                "asymptotic_decline" if basin.outcome == "asymptotic_extinction" else basin.outcome
            )
        elif any(x0 == v.x for v in positive):
            outcome = "equilibrium"
        elif boundary is None or x0 < boundary:
            outcome = "asymptotic_decline" if model == "B" else "finite_extinction"
        else:
            outcome = "persistence"
        sol = solve_trajectory(
            model,
            a=a,
            harvest=control,
            x0=x0,
            tau_end=tau_end,
            n_points=n_points,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        stop = sol.extinction_tau if sol.extinction_tau is not None else tau_end
        catch = (control * trapezoid(sol.x, sol.tau) if model == "B" else control * stop) * p.K
        final = 0.0 if sol.extinction_tau is not None else float(sol.x[-1])
        rows.append(
            dict(
                protocol=protocol,
                fraction=fraction,
                model=model,
                initial_population=p.K * x0,
                control=control,
                control_units="h/r" if model == "B" else "H/(rK)",
                H=None if model == "B" else control * p.r * p.K,
                h=control * p.r if model == "B" else None,
                initial_catch=p.r * p.K * control * (x0 if model == "B" else 1),
                critical_initial_population=None if boundary is None else p.K * boundary,
                theoretical_outcome=outcome,
                final_population=p.K * final,
                horizon_time=tau_end / p.r,
                extinction_time=None if sol.extinction_tau is None else sol.extinction_tau / p.r,
                total_catch=catch,
                mean_catch=catch / (tau_end / p.r),
                final_catch=p.r * p.K * control * (final if model == "B" else float(final > 0)),
            )
        )
    return rows
