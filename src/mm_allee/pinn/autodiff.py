"""Elementwise time derivative with JAX; this is the only autodiff primitive the PINN needs."""

import jax

jax.config.update("jax_enable_x64", True)  # float64 everywhere; must precede any array creation


def derivative(fn, time):
    """d fn / d time at each sample of ``time`` (shape (n,)), for a scalar function ``fn``.

    ``fn`` maps one scalar time to one scalar value, so samples cannot mix. The
    derivative is exact reverse-mode autodiff of whatever ``fn`` closes over
    (network weights, physical parameters), so gradients of anything built from it
    with respect to those arrays remain available (no detaching, no float casts).
    """
    return jax.vmap(jax.grad(fn))(time)
