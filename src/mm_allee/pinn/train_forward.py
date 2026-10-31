"""Forward PINN trainer (JAX): physics loss only, hard initial condition, Adam -> L-BFGS.

The shared engine for WP04 and WP05. No solver solution is ever used as a label. WP05
extends it through ``physical`` (starting values of ``a``/``harvest`` that are then trained
jointly with the weights, always inside their allowed ranges, see ``train_forward``).

Run statuses (``ForwardResult.status``):
    ok                 training ran to the end of its iteration budget (finite loss)
    budget_exhausted   the wall-clock cap stopped training early
    nonfinite_loss     the loss or the weights became NaN/inf
The experiment runners add two more when scoring: ``invalid_prediction`` (the network output
contains NaN/inf, so no error score exists) and ``failed`` (training raised an exception).
"""

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

from ..metrics import errors
from .network import NetworkConfig, init_params, x_hat
from .residual import MODELS, collocation_residual, physics_loss

CHECKPOINT_SCHEMA = 2  # 1 was the retired PyTorch format; extra optional keys do not change it
PHYSICAL_KEYS = ("a", "harvest")
HARVEST_FLOOR = 1e-6  # a harvest start below this is lifted to it (raw = 0 has zero gradient)


@dataclass(frozen=True)
class ForwardConfig:
    """One forward problem plus its training recipe. All fields are JSON-serialisable."""

    model: str = "B"
    a: float = 0.2
    harvest: float = 0.3  # q for logistic/A, e for B
    x0: float = 0.8
    t_end: float = 10.0  # for A: the pre-extinction horizon, never beyond the event
    layers: tuple = (1, 32, 32, 1)
    seed: int = 11
    activation: str = "tanh"  # tanh | swish | sigmoid
    n_collocation: int = 100  # fixed uniform grid on [0, t_end], both ends included
    adam_iters: int = 1000  # 2/7 of the 3500-iteration budget
    lbfgs_iters: int = 2500
    lr: float = 1e-2  # Adam learning rate
    time_budget_s: float | None = None  # wall-clock cap; status becomes "budget_exhausted"

    def validate(self):
        if self.model not in MODELS:
            raise ValueError("model must be 'logistic', 'A' or 'B'")
        if not 0 < self.a < 1 or self.harvest < 0 or self.x0 < 0 or self.t_end <= 0:
            raise ValueError("Require 0<a<1, harvest>=0, x0>=0, t_end>0")
        if self.n_collocation < 2:
            raise ValueError("n_collocation must be >=2")
        if self.adam_iters < 0 or self.lbfgs_iters < 0 or self.lr <= 0:
            raise ValueError("Require nonnegative iteration counts and lr>0")
        self.network_config().validate()

    def network_config(self):
        return NetworkConfig(
            layers=tuple(self.layers),
            t_end=self.t_end,
            x0=self.x0,
            seed=self.seed,
            activation=self.activation,
        )


@dataclass
class ForwardResult:
    params: list
    config: ForwardConfig
    physical: dict = field(default_factory=dict)  # trained a/harvest (valid values) if hooked
    history: dict = field(default_factory=lambda: {"loss": [], "phase": []})
    wall_seconds: float = 0.0
    status: str = "ok"  # see the module docstring
    lbfgs: dict | None = None  # {"status", "message", "iterations"} of the L-BFGS-B exit


# ---- physical parameters: unconstrained internal value <-> valid physical value ---------


def _to_raw(name, value):
    """Valid physical value -> unconstrained internal value that the optimiser updates."""
    value = float(value)
    if name == "harvest":
        if not math.isfinite(value) or value < 0:
            raise ValueError("physical harvest must be finite and >= 0")
        return math.sqrt(max(value, HARVEST_FLOOR))  # harvest = raw**2 >= 0
    if name == "a":
        if not 0 < value < 1:
            raise ValueError("physical a must satisfy 0 < a < 1")
        return math.log(value / (1.0 - value))  # a = sigmoid(raw) in (0, 1)
    raise ValueError(f"unknown physical parameter {name!r}; allowed: {PHYSICAL_KEYS}")


def _to_physical(name, raw):
    """Unconstrained internal value -> valid physical value, recomputed in every loss call so
    the range holds after every Adam step and every L-BFGS iteration (gradients flow through)."""
    return raw**2 if name == "harvest" else jax.nn.sigmoid(raw)


def train_forward(config: ForwardConfig, *, physical=None, tau=None):
    """Train ``x_hat`` on the ODE residual alone (initial condition is hard-coded).

    ``physical`` may map ``"a"``/``"harvest"`` to starting values that are optimised together
    with the weights; otherwise the config values are fixed constants. A trained ``harvest``
    is always >= 0 and a trained ``a`` always in (0, 1): the optimisers update internal values
    (harvest = raw**2, a = sigmoid(raw)) and the loss converts them on every evaluation, in both
    Adam and L-BFGS. Starting values must already be valid (``ValueError`` otherwise); a harvest
    start below 1e-6 (including exactly 0) is lifted to 1e-6. ``result.physical`` holds the
    trained physical values.
    ``tau`` optionally replaces the default uniform collocation grid (1-D, within
    ``[0, t_end]``); used by diagnostics, the frozen sweeps use the default.
    """
    config.validate()
    ncfg = config.network_config()
    if tau is None:
        tau = jnp.linspace(0.0, config.t_end, config.n_collocation)
    else:
        tau = jnp.asarray(tau, dtype=jnp.float64)
        if tau.ndim != 1 or tau.size < 2 or float(tau.min()) < 0 or float(tau.max()) > config.t_end:
            raise ValueError("tau must be a 1-D array of >=2 times inside [0, t_end]")
    raw = {k: jnp.asarray(_to_raw(k, v), dtype=jnp.float64) for k, v in (physical or {}).items()}
    theta = (init_params(ncfg), raw)

    def physical_values(raw_values):
        return {k: _to_physical(k, v) for k, v in raw_values.items()}

    def loss(theta):
        params, raw_values = theta
        phys = physical_values(raw_values)
        a = phys.get("a", config.a)
        harvest = phys.get("harvest", config.harvest)
        return physics_loss(
            lambda t: x_hat(params, t, ncfg), tau, config.model, a=a, harvest=harvest
        )

    result = ForwardResult(params=theta[0], config=config)
    start = time.perf_counter()

    def over_budget():
        return bool(config.time_budget_s) and time.perf_counter() - start > config.time_budget_s

    # ---- phase 1: Adam, a short push into a good basin ----
    zeros = jax.tree_util.tree_map(jnp.zeros_like, theta)
    m, v = zeros, zeros
    b1, b2, eps = 0.9, 0.999, 1e-8

    @jax.jit
    def adam_step(theta, m, v, t):
        value, grad = jax.value_and_grad(loss)(theta)
        m = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1 - b1) * g, m, grad)
        v = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, v, grad)
        theta = jax.tree_util.tree_map(
            lambda p, m_, v_: (
                p - config.lr * (m_ / (1 - b1**t)) / (jnp.sqrt(v_ / (1 - b2**t)) + eps)
            ),
            theta,
            m,
            v,
        )
        return theta, m, v, value

    for step in range(1, config.adam_iters + 1):
        theta, m, v, value = adam_step(theta, m, v, step)
        result.history["loss"].append(float(value))
        result.history["phase"].append("adam")
        if not math.isfinite(result.history["loss"][-1]):
            result.status = "nonfinite_loss"
            break
        if over_budget():
            result.status = "budget_exhausted"
            break

    # ---- phase 2: L-BFGS takes over from the Adam solution ----
    if result.status == "ok" and config.lbfgs_iters:
        flat0, unravel = ravel_pytree(theta)
        flat_vg = jax.jit(jax.value_and_grad(lambda flat: loss(unravel(flat))))

        def fun(flat):
            value, grad = flat_vg(jnp.asarray(flat))
            return float(value), np.asarray(grad, dtype=np.float64)

        def callback(intermediate_result):
            result.history["loss"].append(float(intermediate_result.fun))
            result.history["phase"].append("lbfgs")
            if over_budget():
                result.status = "budget_exhausted"
                raise StopIteration

        fit = minimize(
            fun,
            np.asarray(flat0, dtype=np.float64),
            jac=True,
            method="L-BFGS-B",
            callback=callback,
            options={
                "maxiter": config.lbfgs_iters,
                "maxfun": 20 * config.lbfgs_iters,
                "maxcor": 50,
                "ftol": 0.0,
                "gtol": 0.0,
            },
        )
        # Recorded, not interpreted: with ftol=gtol=0 an "ABNORMAL" exit (status 2) is the usual
        # benign stop once the loss cannot decrease further at floating-point precision.
        result.lbfgs = {
            "status": int(fit.status),
            "message": str(fit.message),
            "iterations": int(fit.nit),
        }
        if np.all(np.isfinite(fit.x)):
            theta = unravel(jnp.asarray(fit.x))
        else:
            result.status = "nonfinite_loss"

    result.params = theta[0]
    result.physical = {k: float(val) for k, val in physical_values(theta[1]).items()}
    if result.status == "ok" and not math.isfinite(float(loss(theta))):
        result.status = "nonfinite_loss"
    result.wall_seconds = time.perf_counter() - start
    return result


def predict(params, config: ForwardConfig, tau):
    """NumPy in, NumPy out."""
    ncfg = config.network_config()
    fn = jax.jit(jax.vmap(lambda t: x_hat(params, t, ncfg)))
    return np.asarray(fn(jnp.asarray(np.asarray(tau, dtype=float))))


def physics_residual(params, config: ForwardConfig, tau, physical=None):
    """ODE residual ``dx/dtau - f(x)`` of a trained (or loaded) network at the times ``tau``.

    Uses the trained ``physical`` values where given (for example ``result.physical``) and the
    config constants otherwise, so the result is the same before and after save/load.
    """
    ncfg = config.network_config()
    physical = physical or {}
    _, residual = collocation_residual(
        lambda t: x_hat(params, t, ncfg),
        tau,
        config.model,
        a=physical.get("a", config.a),
        harvest=physical.get("harvest", config.harvest),
    )
    return np.asarray(residual)


def evaluate_prediction(prediction, reference, *, norm_floor=1e-8, near_zero=0.05):
    """Trajectory errors against a reference on the same grid.

    A prediction containing NaN or +-inf is *invalid*: it is never repaired or replaced, every
    error score is None and ``valid`` is False (the runners label it ``invalid_prediction``).
    For a valid prediction ``relative_l2`` is None only when the reference norm is below
    ``norm_floor``. ``max_abs_near_zero`` is the largest absolute error at reference values
    below ``near_zero`` (None if there are none), the scientific contract's near-zero error.
    ``boundary_violations`` counts negative or non-finite predictions.
    """
    pred = np.asarray(prediction, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if pred.shape != ref.shape or pred.size == 0:
        raise ValueError("Metric inputs must have identical nonempty shapes")
    finite = np.isfinite(pred)
    out = {
        "boundary_violations": int(np.sum(~finite | (pred < 0))),
        "n_invalid": int(np.sum(~finite)),
    }
    if not finite.all():
        return {
            **out,
            "valid": False,
            "rmse": None,
            "mae": None,
            "relative_l2": None,
            "max_abs": None,
            "max_abs_near_zero": None,
        }
    out.update(errors(pred, ref))
    out["valid"] = True
    delta = np.abs(pred - ref)
    out["max_abs"] = float(np.max(delta))
    near = ref < near_zero
    out["max_abs_near_zero"] = float(np.max(delta[near])) if near.any() else None
    if np.linalg.norm(ref) < norm_floor:
        out["relative_l2"] = None
    return out


def save_checkpoint(path, result):
    """Write config, weights, trained physical parameters, status and L-BFGS exit to ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for i, (weight, bias) in enumerate(result.params):
        arrays[f"W{i}"], arrays[f"b{i}"] = np.asarray(weight), np.asarray(bias)
    meta = {
        "schema": CHECKPOINT_SCHEMA,
        "config": asdict(result.config),
        "physical": result.physical,
        "status": result.status,
        "lbfgs": result.lbfgs,
    }
    np.savez(path, meta=np.array(json.dumps(meta)), **arrays)


def load_checkpoint(path):
    """Return the saved model as a ``ForwardResult``: weights, config, the learned physical
    parameters (``.physical``), the training status and the L-BFGS exit (``.lbfgs``). The loss
    history and runtime are not saved, so they come back empty / zero."""
    with np.load(Path(path), allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta["schema"] != CHECKPOINT_SCHEMA:
            raise ValueError(f"Unsupported checkpoint schema {meta['schema']}")
        n_layers = len([k for k in data.files if k.startswith("W")])
        params = [(jnp.asarray(data[f"W{i}"]), jnp.asarray(data[f"b{i}"])) for i in range(n_layers)]
    config = ForwardConfig(**{**meta["config"], "layers": tuple(meta["config"]["layers"])})
    return ForwardResult(
        params=params,
        config=config,
        physical={k: float(v) for k, v in meta.get("physical", {}).items()},
        status=meta.get("status", "ok"),
        lbfgs=meta.get("lbfgs"),
    )
