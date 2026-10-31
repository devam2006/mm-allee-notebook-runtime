"""Physics-Informed Neural Network (PINN) subpackage for mm_allee.

Shared JAX neural engine: WP04 owns forward engine; WP05 extends for inverse inference.
"""

from .autodiff import derivative
from .inverse import (
    InversePINNResult,
    InversePINNTrainer,
    TrainableParameters,
    load_inverse_checkpoint,
    save_inverse_checkpoint,
)
from .network import NetworkConfig, init_params, mlp, x_hat
from .residual import collocation_residual, jax_rhs, physics_loss
from .train_forward import (
    ForwardConfig,
    ForwardResult,
    evaluate_prediction,
    load_checkpoint,
    predict,
    save_checkpoint,
    train_forward,
)

__all__ = [
    "derivative",
    "NetworkConfig",
    "init_params",
    "mlp",
    "x_hat",
    "collocation_residual",
    "jax_rhs",
    "physics_loss",
    "ForwardConfig",
    "ForwardResult",
    "train_forward",
    "predict",
    "evaluate_prediction",
    "save_checkpoint",
    "load_checkpoint",
    "TrainableParameters",
    "InversePINNTrainer",
    "InversePINNResult",
    "save_inverse_checkpoint",
    "load_inverse_checkpoint",
]
