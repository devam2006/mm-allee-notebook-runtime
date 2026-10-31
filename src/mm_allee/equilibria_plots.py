"""WP01 figures: production/phase-line geometry, branch diagrams, perturbation checks.

Functions return Matplotlib figures and never write files; ``experiments/wp01/run.py``
and the notebook decide where output goes. All quantities are dimensionless
(x = N/K, tau = r t, q = H/(rK)). Fills are opaque so EPS export is faithful.
"""

import matplotlib.pyplot as plt
import numpy as np

from .equilibria import branch_rows, equilibria_a, equilibria_logistic, fold_a
from .models import growth
from .solver import solve_trajectory

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3
INK = "#0b0b0b"
MUTED = "#52514e"
SHADE = "#ebe9e4"  # opaque light fill for the negative-growth band
MARKER = dict(markersize=8, markeredgewidth=1.6, zorder=5)


def _style(ax):
    ax.grid(color="#dddcd8", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _mark(ax, x, y, stability, color):
    if stability == "stable":
        ax.plot(x, y, "o", color=color, markerfacecolor=color, **MARKER)
    elif stability == "unstable":
        ax.plot(x, y, "o", color=color, markerfacecolor="white", **MARKER)
    else:  # semistable: half filled
        ax.plot(
            x,
            y,
            "o",
            color=color,
            markerfacecolor=color,
            fillstyle="left",
            markerfacecoloralt="white",
            **MARKER,
        )


def _stability_legend(ax, loc):
    handles = [
        ax.plot([], [], "o", color=INK, markerfacecolor=INK, label="stable", **MARKER)[0],
        ax.plot([], [], "o", color=INK, markerfacecolor="white", label="unstable", **MARKER)[0],
        ax.plot(
            [],
            [],
            "o",
            color=INK,
            fillstyle="left",
            markerfacecoloralt="white",
            label="semistable (fold)",
            **MARKER,
        )[0],
    ]
    return handles


def _phase_line(ax, y, records, f, color, x_max, constant_yield):
    """Arrows show the sign of dx/dtau between consecutive equilibria."""
    ax.plot([0, x_max], [y, y], color=color, linewidth=1.2, zorder=1)
    xs = [0.0] + [r.x for r in records if r.x > 0] + [x_max]
    for left, right in zip(xs[:-1], xs[1:], strict=True):
        if right - left < 0.04:
            continue
        mid = 0.5 * (left + right)
        direction = np.sign(f(mid))
        span = min(0.06, 0.3 * (right - left))
        ax.annotate(
            "",
            xy=(mid + direction * span, y),
            xytext=(mid - direction * span, y),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6),
        )
    for r in records:
        _mark(ax, r.x, y, r.stability, color)
    if constant_yield:
        ax.plot(0, y, marker="x", color=MUTED, markersize=8, markeredgewidth=2, zorder=6)


def production_figure(a=0.2, logistic_q=(0.1, 0.25, 0.35), allee_q=None):
    """2x2 figure: production curves with harvest lines (top) and phase lines (bottom)."""
    x_star, q_c = fold_a(a)
    if allee_q is None:
        allee_q = (0.25 * q_c, q_c, 1.2 * q_c)
    x = np.linspace(0, 1.08, 600)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), gridspec_kw=dict(height_ratios=[3, 2]))
    (ax_l, ax_a), (ph_l, ph_a) = axes

    # Logistic production and constant harvest lines.
    ax_l.plot(x, x * (1 - x), color=INK, linewidth=2, label=r"production $x(1-x)$")
    for color, q in zip(SERIES, logistic_q, strict=True):
        ax_l.axhline(q, color=color, linewidth=2, label=f"harvest $q={q:g}$")
        for r in equilibria_logistic(q):
            _mark(ax_l, r.x, q, r.stability, color)
    _mark(ax_l, 0.0, 0.0, "unstable", INK)
    _mark(ax_l, 1.0, 0.0, "stable", INK)
    ax_l.axhline(0, color=MUTED, linewidth=0.8)
    ax_l.set(
        xlim=(-0.02, 1.08),
        ylim=(-0.08, 0.64),
        xlabel=r"population $x=N/K$",
        ylabel=r"rate (per unit $\tau=rt$)",
        title=r"(a) Logistic: $dx/d\tau = x(1-x) - q$",
    )
    ax_l.legend(
        handles=ax_l.get_legend_handles_labels()[0] + _stability_legend(ax_l, None),
        loc="upper right",
        fontsize=8.5,
        framealpha=1,
    )

    # Allee production with negative band below a.
    ax_a.axvspan(0, a, color=SHADE, zorder=0, label=r"$G<0$ for $0<x<a$")
    ax_a.plot(x, growth(x, a), color=INK, linewidth=2, label=r"production $G(x)=x(1-x)(x/a-1)$")
    for color, q in zip(SERIES, allee_q, strict=True):
        name = r"$q_c$" if q == q_c else f"{q / q_c:.2g}" + r"$\,q_c$"
        ax_a.axhline(q, color=color, linewidth=2, label=f"harvest $q=${name} ({q:.3f})")
        for r in equilibria_a(a, q):
            _mark(ax_a, r.x, q, r.stability, color)
    for r in equilibria_a(a, 0.0):
        _mark(ax_a, r.x, 0.0, r.stability, INK)
    ax_a.axhline(0, color=MUTED, linewidth=0.8)
    ax_a.annotate(
        rf"fold $x^*={x_star:.3f}$",
        xy=(x_star, q_c),
        xytext=(x_star + 0.04, 1.55 * q_c),
        arrowprops=dict(arrowstyle="->", color=MUTED),
        color=INK,
        fontsize=9,
    )
    ax_a.annotate(rf"$a={a:g}$", xy=(a, 0), xytext=(a + 0.03, -0.12), color=INK, fontsize=9)
    top = 2.5 * q_c
    ax_a.set(
        xlim=(-0.02, 1.08),
        ylim=(-0.13 * top, top),
        xlabel=r"population $x=N/K$",
        ylabel=r"rate (per unit $\tau=rt$)",
        title=rf"(b) Strong Allee (Model A, $a={a:g}$): $dx/d\tau=G(x)-q$",
    )
    ax_a.legend(loc="upper left", fontsize=8.5, framealpha=1)

    # Phase lines.
    for ax, qs, model in ((ph_l, logistic_q, "logistic"), (ph_a, allee_q, "A")):
        for i, (color, q) in enumerate(zip(SERIES, qs, strict=True)):
            if model == "logistic":
                recs, f = equilibria_logistic(q), (lambda z, q=q: z * (1 - z) - q)
            else:
                recs, f = equilibria_a(a, q), (lambda z, q=q: growth(z, a) - q)
            _phase_line(ax, -i, recs, f, color, 1.08, constant_yield=q > 0)
            ax.text(1.1, -i, f"q={q:.3g}", va="center", fontsize=9, color=INK)
        ax.set(
            xlim=(-0.02, 1.2), ylim=(-len(qs) + 0.4, 0.6), yticks=[], xlabel=r"population $x=N/K$"
        )
        ax.spines["left"].set_visible(False)
    ph_l.set_title("(c) Logistic phase lines (arrows: direction of change)")
    ph_a.set_title(r"(d) Model A phase lines;  $\times$ = $x=0$ is reached in finite time")
    for ax in axes.flat:
        _style(ax)
    for ax in (ph_l, ph_a):
        ax.grid(False)
    fig.tight_layout()
    return fig


def branch_figure(a=0.2, n_q=400):
    """Branch (bifurcation) diagrams: logistic (left) and Model A (right)."""
    x_star, q_c = fold_a(a)
    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, model, qc, xf in ((ax_l, "logistic", 0.25, 0.5), (ax_a, "A", q_c, x_star)):
        # Dense near the fold so the branches meet visibly; exclude q_c itself here.
        q = np.unique(
            np.concatenate(
                [np.linspace(0, qc, n_q, endpoint=False), qc - np.geomspace(1e-9, 1e-2, 60) * qc]
            )
        )
        q = q[q < qc]
        rows = branch_rows(model, q, a=a if model == "A" else None)
        upper = [(r["q"], r["x"]) for r in rows if r["branch"] in ("upper", "capacity")]
        lower = [
            (r["q"], r["x"])
            for r in rows
            if r["branch"] in ("lower", "threshold")
            or (model == "logistic" and r["branch"] == "zero")
        ]
        upper, lower = np.array(sorted(upper)), np.array(sorted(lower))
        ax.plot(
            upper[:, 0],
            upper[:, 1],
            color=SERIES[0],
            linewidth=2.2,
            label="stable equilibrium (persistent state)",
        )
        ax.plot(
            lower[:, 0],
            lower[:, 1],
            color=SERIES[1],
            linewidth=2.2,
            linestyle=(0, (5, 3)),
            label="unstable equilibrium (collapse threshold)",
        )
        _mark(ax, qc, xf, "semistable", INK)
        ax.annotate(
            f"fold (semistable)\n$q_c={qc:.4g}$, $x^*={xf:.4g}$",
            xy=(qc, xf),
            xytext=(
                qc * (0.45 if model == "logistic" else 0.3),
                0.5 if model == "logistic" else 0.63,
            ),
            fontsize=9,
            color=INK,
            arrowprops=dict(arrowstyle="->", color=MUTED),
        )
        if model == "A":
            ax.plot(
                [0],
                [0],
                "o",
                color=SERIES[0],
                markerfacecolor=SERIES[0],
                **MARKER,
                label=r"$x=0$ stable equilibrium only at $q=0$",
            )
            ax.plot(
                [1e-9, 1.25 * qc],
                [0, 0],
                color=MUTED,
                linewidth=3,
                linestyle=(0, (1, 1.5)),
                label=r"$x=0$ for $q>0$: extinction boundary (not an equilibrium)",
            )
        ax.axvspan(qc, 1.25 * qc, color=SHADE, zorder=0)
        ax.text(
            1.125 * qc,
            0.22,
            "no positive\nequilibrium:\ncollapse from\nevery start",
            ha="center",
            fontsize=8.5,
            color=INK,
        )
        ax.set(
            xlim=(-0.012 * qc, 1.25 * qc),
            ylim=(-0.05, 1.08),
            xlabel=r"harvest $q=H/(rK)$",
            ylabel=r"equilibrium population $x=N/K$",
        )
        _style(ax)
    ax_l.set_title(r"(a) Logistic: $x_\pm=(1\pm\sqrt{1-4q})/2$")
    ax_a.set_title(rf"(b) Model A, $a={a:g}$ (roots from Brent bracketing)")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(
        *ax_a.get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8.5, framealpha=1
    )
    return fig


def perturbation_figure(
    a=0.2, q=0.3, delta=0.01, tau_end=12.0, *, rtol=1e-9, atol=1e-11, max_step=0.1
):
    """Trajectories started just above/below each Model A equilibrium (RK45).

    ``rtol``, ``atol`` and ``max_step`` are passed unchanged to
    :func:`mm_allee.solver.solve_trajectory` for every trajectory.
    """
    lower, upper = equilibria_a(a, q)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    starts = [
        (upper.x + delta, SERIES[0], "-", r"$x_+ + \delta$"),
        (upper.x - delta, SERIES[0], (0, (5, 3)), r"$x_+ - \delta$"),
        (lower.x + delta, SERIES[2], "-", r"$x_- + \delta$"),
        (lower.x - delta, SERIES[1], "-", r"$x_- - \delta$"),
    ]
    for x0, color, style, label in starts:
        traj = solve_trajectory(
            "A",
            a=a,
            harvest=q,
            x0=x0,
            tau_end=tau_end,
            n_points=600,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        suffix = (
            "" if traj.extinction_tau is None else rf" (hits 0 at $\tau={traj.extinction_tau:.2f}$)"
        )
        ax.plot(traj.tau, traj.x, color=color, linestyle=style, linewidth=2, label=label + suffix)
        if traj.extinction_tau is not None:
            ax.plot(
                traj.extinction_tau, 0, marker="x", color=MUTED, markersize=9, markeredgewidth=2
            )
    ax.axhline(upper.x, color=MUTED, linewidth=1)
    ax.axhline(lower.x, color=MUTED, linewidth=1, linestyle=(0, (2, 2)))
    ax.text(tau_end, upper.x + 0.02, rf"$x_+={upper.x:.3f}$ stable", ha="right", fontsize=9)
    ax.text(tau_end, lower.x + 0.02, rf"$x_-={lower.x:.3f}$ unstable", ha="right", fontsize=9)
    ax.set(
        xlim=(0, tau_end),
        ylim=(-0.03, 1.02),
        xlabel=r"time $\tau = rt$",
        ylabel=r"population $x=N/K$",
        title=rf"Model A, $a={a:g}$, $q={q:g}$: perturbations $\delta={delta:g}$ (RK45, rtol={rtol:g})",
    )
    ax.legend(loc="lower right", fontsize=8.5, framealpha=1)
    _style(ax)
    fig.tight_layout()
    return fig
