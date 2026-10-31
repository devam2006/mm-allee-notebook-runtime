"""Small seeded MLP x_hat(tau) with fixed input scaling and a hard initial condition."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from . import autodiff  # noqa: F401  (enables float64 before any array is created)

ACTIVATIONS = {"tanh": jnp.tanh, "swish": jax.nn.silu, "sigmoid": jax.nn.sigmoid}


@dataclass(frozen=True)
class NetworkConfig:
    """Everything needed to rebuild a network; stored inside checkpoints."""

    layers: tuple = (1, 32, 32, 1)  # input, hidden..., output
    t_end: float = 10.0  # fixed input scale: the network sees s = tau / t_end in [0, 1]
    x0: float = 0.8
    seed: int = 11
    activation: str = "tanh"  # tanh | swish | sigmoid

    def validate(self):
        if len(self.layers) < 3 or self.layers[0] != 1 or self.layers[-1] != 1:
            raise ValueError("layers must look like (1, h1, ..., 1)")
        if min(self.layers) < 1 or self.t_end <= 0:
            raise ValueError("Require positive layer sizes and t_end>0")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}")


def init_params(config: NetworkConfig):
    """Glorot-normal weights, zero biases; reproducible from ``config.seed`` alone."""
    config.validate()
    keys = jax.random.split(jax.random.PRNGKey(config.seed), len(config.layers) - 1)
    params = []
    for key, n_in, n_out in zip(keys, config.layers[:-1], config.layers[1:]):
        scale = jnp.sqrt(2.0 / (n_in + n_out))
        params.append((scale * jax.random.normal(key, (n_in, n_out)), jnp.zeros(n_out)))
    return params


def mlp(params, s, activation="tanh"):
    """N(s) for a scalar ``s``; ``activation`` hidden layers, linear output."""
    act = ACTIVATIONS[activation]
    h = jnp.reshape(s, (1,))
    for weight, bias in params[:-1]:
        h = act(h @ weight + bias)
    weight, bias = params[-1]
    return (h @ weight + bias)[0]


def x_hat(params, tau, config: NetworkConfig):
    """Hard-constraint solution: x_hat(tau) = x0 + s*N(s), s = tau/t_end.

    ``x_hat(0) = x0`` exactly for any weights, so no initial-condition loss is needed.
    ``tau`` is physical dimensionless time, so autodiff with respect to ``tau`` applies
    the chain rule dx/dtau = (1/t_end) d/ds [x0 + s N(s)] automatically.
    """
    s = tau / config.t_end
    return config.x0 + s * mlp(params, s, config.activation)
