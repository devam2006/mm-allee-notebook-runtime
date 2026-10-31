"""Differentiable ODE right-hand sides and the physics residual (dimensionless x, tau).

Nothing here casts to Python floats: ``a`` and ``harvest`` may be numbers or JAX arrays
(for WP05 they are traced arrays being optimised), and gradients flow to both.
"""

import jax.numpy as jnp

from .autodiff import derivative

MODELS = ("logistic", "A", "B")


def jax_rhs(x, model, *, a, harvest):
    """JAX twin of ``mm_allee.models.rhs`` (q for logistic/A, e for B).

    No range validation: ``a`` or ``harvest`` may be traced values during optimisation.
    """
    if model == "logistic":
        return x * (1 - x) - harvest
    if model == "A":
        return x * (1 - x) * (x / a - 1) - harvest
    if model == "B":
        return x * (1 - x) * (x / a - 1) - harvest * x
    raise ValueError("Model must be 'logistic', 'A' or 'B'")


def collocation_residual(x_fn, tau, model, *, a, harvest):
    """Return ``(x, dx/dtau - f(x))`` at the times ``tau`` (shape (n,)).

    ``x_fn`` maps one scalar physical time to the scalar state. The derivative is with
    respect to ``tau`` itself, so any internal time rescaling is differentiated through.
    """
    tau = jnp.asarray(tau, dtype=jnp.float64)
    x = jnp.vectorize(x_fn)(tau)
    dx = derivative(x_fn, tau)
    return x, dx - jax_rhs(x, model, a=a, harvest=harvest)


def physics_loss(x_fn, tau, model, *, a, harvest):
    """``MSE(dx/dtau - f(x))`` over the collocation times. The initial condition is hard-coded
    in ``x_fn`` (see ``network.x_hat``), so this is the entire training loss."""
    _, residual = collocation_residual(x_fn, tau, model, a=a, harvest=harvest)
    return jnp.mean(residual**2)
