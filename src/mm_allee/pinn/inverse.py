"""Inverse PINN trainer (JAX): physical parameter estimation under strong Allee effect.

Extends the shared WP04 JAX neural engine for inverse parameter estimation:
  - Constrained physical parameter transforms with nonzero gradients everywhere:
      r > 0: r = softplus(raw_r)
      0 < A < K: A = K * sigmoid(raw_A) => dimensionless a = A / K in (0, 1)
      H >= 0: H = softplus(raw_H) (Model A)
      h >= 0: h = softplus(raw_h) (Model B)
  - Hard initial condition inherited from WP04 network x_hat: x(0) == x0 exactly.
  - Strict data-leakage guards: training loss uses training observations only;
    validation observations are evaluated strictly without gradient for checkpoint
    selection and early stopping; test observations are held out until inference.
  - Two-phase joint optimization: Phase 1 (Adam) for rapid basin discovery,
    followed by Phase 2 (L-BFGS) for fine second-order parameter convergence.
  - Seeded network and parameter initialization via jax.random.PRNGKey(seed).
  - Explicit convergence and failure status tracking.
"""

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

from ..metrics import errors
from .autodiff import derivative
from .network import NetworkConfig, init_params, x_hat

CHECKPOINT_SCHEMA_INVERSE = 2


def _inv_softplus(y: float) -> float:
    """Inverse softplus: x = ln(exp(y) - 1)."""
    y_clamped = max(float(y), 1e-6)
    if y_clamped > 20.0:
        return y_clamped
    return math.log(math.expm1(y_clamped))


def _inv_sigmoid(y: float) -> float:
    """Inverse sigmoid (logit): x = ln(y / (1 - y))."""
    y_clamped = min(max(float(y), 1e-6), 1.0 - 1e-6)
    return math.log(y_clamped / (1.0 - y_clamped))


class TrainableParameters:
    """Constrained physical parameter module for Models A and B in JAX.

    Enforces physical constraints without detaching gradients:
      - r > 0: r = softplus(raw_r)
      - K > 0: K = softplus(raw_K) (if fitted, otherwise fixed known_K)
      - 0 < A < K: A = K * sigmoid(raw_A), so dimensionless a = A / K in (0, 1)
      - H >= 0: H = softplus(raw_H) (Model A constant yield)
      - h >= 0: h = softplus(raw_h) (Model B proportional effort)

    Trainable parameters receive continuous, strictly nonzero gradients everywhere.
    """

    def __init__(
        self,
        model: Literal["A", "B"],
        fit_params: Sequence[str],
        known_conditions: dict[str, float] | None = None,
        initial_guesses: dict[str, float] | None = None,
    ):
        if model not in ("A", "B"):
            raise ValueError(f"model must be 'A' or 'B', got '{model}'")

        self.model: Literal["A", "B"] = model
        self.fit_params: list[str] = list(fit_params)
        known = known_conditions or {}
        guesses = initial_guesses or {}

        # Default standard baseline physical values
        default_p = {
            "r": 1.0,
            "K": float(known.get("known_K", 100.0)),
            "A": 20.0,
            "H": 5.0 if model == "A" else 0.0,
            "h": 0.2 if model == "B" else 0.0,
        }
        for k, v in guesses.items():
            default_p[k] = float(v)

        self.fixed_values: dict[str, float] = {}
        self.raw_params: dict[str, jnp.ndarray] = {}

        # Initialize r
        if "r" in self.fit_params:
            self.raw_params["r"] = jnp.array(_inv_softplus(default_p["r"]), dtype=jnp.float64)
        else:
            self.fixed_values["r"] = float(known.get("r", default_p["r"]))

        # Initialize K
        if "K" in self.fit_params:
            self.raw_params["K"] = jnp.array(_inv_softplus(default_p["K"]), dtype=jnp.float64)
        else:
            self.fixed_values["K"] = float(known.get("known_K", default_p["K"]))

        # Initialize A: constrained 0 < A < K
        if "A" in self.fit_params:
            k_ref = self.fixed_values.get("K", default_p["K"])
            ratio = min(max(default_p["A"] / k_ref, 0.01), 0.99)
            self.raw_params["A"] = jnp.array(_inv_sigmoid(ratio), dtype=jnp.float64)
        else:
            self.fixed_values["A"] = float(known.get("A", default_p["A"]))

        # Initialize harvest control
        if model == "A":
            if "H" in self.fit_params:
                self.raw_params["H"] = jnp.array(_inv_softplus(default_p["H"]), dtype=jnp.float64)
            else:
                self.fixed_values["H"] = float(known.get("H", default_p["H"]))
            self.fixed_values["h"] = 0.0
        elif model == "B":
            if "h" in self.fit_params:
                self.raw_params["h"] = jnp.array(_inv_softplus(default_p["h"]), dtype=jnp.float64)
            else:
                self.fixed_values["h"] = float(known.get("h", default_p["h"]))
            self.fixed_values["H"] = 0.0

    def transform(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Differentiable mapping from unconstrained raw parameters to physical domain."""
        # r > 0
        if "r" in self.fit_params:
            r = jax.nn.softplus(raw["r"])
        else:
            r = jnp.asarray(self.fixed_values["r"], dtype=jnp.float64)

        # K > 0
        if "K" in self.fit_params:
            K = jax.nn.softplus(raw["K"])
        else:
            K = jnp.asarray(self.fixed_values["K"], dtype=jnp.float64)

        # 0 < A < K: A = K * sigmoid(raw_A)
        if "A" in self.fit_params:
            a_frac = jax.nn.sigmoid(raw["A"])
            A = K * a_frac
        else:
            A = jnp.asarray(self.fixed_values["A"], dtype=jnp.float64)

        # Harvest parameters
        if self.model == "A":
            if "H" in self.fit_params:
                H = jax.nn.softplus(raw["H"])
            else:
                H = jnp.asarray(self.fixed_values["H"], dtype=jnp.float64)
            h = jnp.asarray(0.0, dtype=jnp.float64)
        else:
            if "h" in self.fit_params:
                h = jax.nn.softplus(raw["h"])
            else:
                h = jnp.asarray(self.fixed_values["h"], dtype=jnp.float64)
            H = jnp.asarray(0.0, dtype=jnp.float64)

        return {"r": r, "K": K, "A": A, "H": H, "h": h}

    def __call__(self, raw: dict[str, Any] | None = None) -> dict[str, Any]:
        """Compute differentiable physical parameters satisfying domain constraints."""
        target_raw = self.raw_params if raw is None else raw
        return self.transform(target_raw)

    def get_physical_parameters(self, raw: dict[str, Any] | None = None) -> dict[str, float]:
        """Extract physical parameters as standard Python floats."""
        p_tensors = self(raw)
        return {k: float(v) for k, v in p_tensors.items()}


@dataclass
class InversePINNResult:
    """Complete results and diagnostic metadata from inverse PINN parameter estimation."""

    model: str
    fit_params: list[str]
    estimated_parameters: dict[str, float]
    all_parameters: dict[str, float]
    # "converged" | "validation_stopped" | "budget_exhausted" | "nonfinite_loss" | "lbfgs_failed"
    # (``status`` describes the Adam phase and whether the run finished; how the L-BFGS phase ended
    # is in ``optimizer``: a status of ``converged`` does NOT mean L-BFGS reached a tolerance)
    status: str
    success: bool
    final_epoch: int
    wall_seconds: float
    train_loss: float
    val_loss: float
    train_metrics: dict[str, Any]
    val_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    predicted_trajectory: dict[str, np.ndarray]
    training_history: dict[str, list]
    network_params: list = field(default_factory=list)
    network_config: NetworkConfig | None = None
    # L-BFGS exit: lbfgs_outcome is one of not_run | tolerance_reached | iteration_limit |
    # abnormal_exit | budget_exhausted | interrupted | error (with the exception in ``error``)
    optimizer: dict[str, Any] = field(default_factory=dict)


class InversePINNTrainer:
    """Physics-Informed Neural Network inverse trainer using JAX.

    Features:
      - Uses shared WP04 x_hat with exact hard initial condition x(0) == x0.
      - Autodiff time derivatives computed via mm_allee.pinn.autodiff.derivative.
      - Two-phase joint optimizer: Phase 1 (Adam) with separate learning rates
        for network weights and physical parameters, followed by Phase 2 (L-BFGS)
        for rapid second-order refinement.
      - Strict split discipline: training observations ONLY in backprop loss,
        validation observations ONLY for model selection and early stopping,
        test observations strictly held out for final evaluation.
      - Reproducible initialization from seed.
    """

    def __init__(
        self,
        network_config: NetworkConfig | None = None,
        trainable_params: TrainableParameters | None = None,
        lr_net: float = 1e-3,
        lr_params: float = 1e-2,
        lambda_data: float = 1.0,
        lambda_phys: float = 0.1,
        time_budget_s: float | None = None,
        lbfgs_iters: int = 200,
        lbfgs_ftol: float = 1e-12,
        lbfgs_gtol: float = 1e-8,
        validation_selection: bool = True,
        # Backward compatibility with positional (net, tp) constructor
        *args,
        **kwargs,
    ):
        if len(args) > 0 and trainable_params is None:
            # Handle InversePINNTrainer(net, tp, ...)
            trainable_params = args[0]

        self.user_network_config = network_config
        self.params_mod = trainable_params
        self.lr_net = float(lr_net)
        self.lr_params = float(lr_params)
        self.lambda_data = float(lambda_data)
        self.lambda_phys = float(lambda_phys)
        self.time_budget_s = time_budget_s
        self.lbfgs_iters = int(lbfgs_iters)
        self.lbfgs_ftol = float(lbfgs_ftol)  # 0 disables the stop, as in the forward trainer
        self.lbfgs_gtol = float(lbfgs_gtol)
        # True (default, unchanged behaviour): early-stop on validation loss, restore the best
        # validation checkpoint and accept L-BFGS only if validation loss does not worsen by >5%.
        # False: train the full budget, keep the final state and always accept a finite L-BFGS result.
        self.validation_selection = bool(validation_selection)

    def fit(
        self,
        observations: Any,
        known_conditions: dict[str, float] | None = None,
        epochs: int = 1000,
        patience: int = 50,
        n_collocation: int = 100,
        seed: int = 11,
        eval_every: int = 10,
        adam_iters: int | None = None,
        lbfgs_iters: int | None = None,
    ) -> InversePINNResult:
        """Run inverse parameter estimation on observation splits."""
        start_time = time.perf_counter()
        known = known_conditions or {}

        def over_budget() -> bool:
            return (
                bool(self.time_budget_s) and (time.perf_counter() - start_time) > self.time_budget_s
            )

        # 1. Parse observation arrays and partition splits
        times_all = np.asarray(observations["time"], dtype=np.float64)
        obs_pop_all = np.asarray(observations["observed_population"], dtype=np.float64)
        splits = np.asarray(observations["split"])

        idx_train = np.where(splits == "train")[0]
        idx_val = np.where(splits == "val")[0]
        idx_test = np.where(splits == "test")[0]

        if len(idx_train) == 0:
            raise ValueError("Observation table contains no training records")

        t_train_np = times_all[idx_train]
        y_train_np = obs_pop_all[idx_train]

        t_val_np = times_all[idx_val]
        y_val_np = obs_pop_all[idx_val]

        # Reference scales
        k_val = float(known.get("known_K", 100.0))
        n0 = float(known.get("known_initial_population", obs_pop_all[0]))
        x0 = n0 / k_val
        t_end = float(times_all.max())

        # 2. Build network configuration and initialize weights reproducible from seed
        layers = (1, 32, 32, 1)
        activation = "tanh"
        if self.user_network_config is not None:
            if hasattr(self.user_network_config, "layers"):
                layers = tuple(self.user_network_config.layers)
            if hasattr(self.user_network_config, "activation"):
                activation = self.user_network_config.activation

        ncfg = NetworkConfig(
            layers=layers,
            t_end=t_end,
            x0=x0,
            seed=seed,
            activation=activation,
        )
        ncfg.validate()

        # Seed network weights
        net_params = init_params(ncfg)

        # Ensure TrainableParameters module exists
        if self.params_mod is None:
            model_type = str(known.get("model", "A"))
            self.params_mod = TrainableParameters(
                model=model_type,
                fit_params=["r", "A", "H"] if model_type == "A" else ["r", "A", "h"],
                known_conditions=known,
            )

        # Initialize physical parameters raw tensors
        raw_p = {
            k: jnp.asarray(v, dtype=jnp.float64) for k, v in self.params_mod.raw_params.items()
        }

        # Collocation points across full physical time window [0, t_end]
        t_colloc = jnp.linspace(0.0, t_end, n_collocation, dtype=jnp.float64)

        # Training data arrays (normalized by K)
        t_train = jnp.asarray(t_train_np, dtype=jnp.float64)
        y_train_norm = jnp.asarray(y_train_np / k_val, dtype=jnp.float64)

        # Validation data arrays
        t_val = jnp.asarray(t_val_np, dtype=jnp.float64)
        y_val_norm = jnp.asarray(y_val_np / k_val, dtype=jnp.float64)

        # 3. Define differentiable loss function
        model = self.params_mod.model
        tp = self.params_mod
        l_data_w = self.lambda_data
        l_phys_w = self.lambda_phys

        def loss_fn(state):
            weights, raw = state
            phys = tp.transform(raw)
            r = phys["r"]
            K = phys["K"]
            A = phys["A"]
            H = phys["H"]
            h = phys["h"]
            a = A / K

            # Supervised Data Loss strictly on training observations
            x_pred_train = jax.vmap(lambda t: x_hat(weights, t, ncfg))(t_train)
            loss_data = jnp.mean((x_pred_train - y_train_norm) ** 2)

            # Physics ODE residual on collocation points
            x_col = jax.vmap(lambda t: x_hat(weights, t, ncfg))(t_colloc)
            dx_col = derivative(lambda t: x_hat(weights, t, ncfg), t_colloc)

            # Physical right-hand side in dimensionless state x = N / K
            growth = r * x_col * (1.0 - x_col) * (x_col / a - 1.0)
            if model == "A":
                rhs_val = growth - H / K
            else:
                rhs_val = growth - h * x_col

            res = dx_col - rhs_val
            loss_phys = jnp.mean(res**2)

            total_loss = l_data_w * loss_data + l_phys_w * loss_phys
            return total_loss, (loss_data, loss_phys)

        # 4. Phase 1: Joint pure JAX Adam optimizer with separate learning rates
        state = (net_params, raw_p)
        m_net = jax.tree_util.tree_map(jnp.zeros_like, net_params)
        v_net = jax.tree_util.tree_map(jnp.zeros_like, net_params)
        m_p = jax.tree_util.tree_map(jnp.zeros_like, raw_p)
        v_p = jax.tree_util.tree_map(jnp.zeros_like, raw_p)

        b1, b2, eps = 0.9, 0.999, 1e-8
        lr_net = self.lr_net
        lr_p = self.lr_params

        @jax.jit
        def adam_step(st, m_n, v_n, m_param, v_param, step_idx):
            (loss_val, (ld, lp)), (g_n, g_param) = jax.value_and_grad(loss_fn, has_aux=True)(st)

            # Update network Adam state
            m_n = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1.0 - b1) * g, m_n, g_n)
            v_n = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1.0 - b2) * g * g, v_n, g_n)
            updated_net = jax.tree_util.tree_map(
                lambda p, m_, v_: (
                    p
                    - lr_net
                    * (m_ / (1.0 - b1**step_idx))
                    / (jnp.sqrt(v_ / (1.0 - b2**step_idx)) + eps)
                ),
                st[0],
                m_n,
                v_n,
            )

            # Update physical parameters Adam state
            m_param = jax.tree_util.tree_map(
                lambda m_, g: b1 * m_ + (1.0 - b1) * g, m_param, g_param
            )
            v_param = jax.tree_util.tree_map(
                lambda v_, g: b2 * v_ + (1.0 - b2) * g * g, v_param, g_param
            )
            updated_p = jax.tree_util.tree_map(
                lambda p, m_, v_: (
                    p
                    - lr_p
                    * (m_ / (1.0 - b1**step_idx))
                    / (jnp.sqrt(v_ / (1.0 - b2**step_idx)) + eps)
                ),
                st[1],
                m_param,
                v_param,
            )

            return (updated_net, updated_p), m_n, v_n, m_param, v_param, loss_val, ld, lp

        # 5. Training loop with validation early stopping and budget caps
        history = {
            "epoch": [],
            "loss_train": [],
            "loss_val": [],
            "loss_data": [],
            "loss_phys": [],
            "phase": [],
        }
        for pname in tp.fit_params:
            history[f"param_{pname}"] = []

        @jax.jit
        def _validation_loss(weights):
            pred_val = jax.vmap(lambda t: x_hat(weights, t, ncfg))(t_val)
            return jnp.mean((pred_val - y_val_norm) ** 2)

        def validation_loss(weights):  # compiled: it is also called once per L-BFGS iteration
            return float(_validation_loss(weights))

        last_val_loss = float("nan")  # no validation measurement yet
        best_val_loss = float("inf")
        best_state = state
        patience_counter = 0
        status = "converged"
        success = True

        n_adam = epochs if adam_iters is None else int(adam_iters)
        n_lbfgs = self.lbfgs_iters if lbfgs_iters is None else int(lbfgs_iters)

        for epoch in range(1, n_adam + 1):
            if over_budget():
                status = "budget_exhausted"
                break

            state, m_net, v_net, m_p, v_p, tr_loss, d_loss, p_loss = adam_step(
                state, m_net, v_net, m_p, v_p, epoch
            )
            tr_loss_f = float(tr_loss)

            if not math.isfinite(tr_loss_f):
                status = "nonfinite_loss"
                success = False
                break

            # Measure the validation loss periodically (strictly no gradient). This happens whether
            # or not validation is used to select a checkpoint; only the selection and the early
            # stop depend on ``validation_selection``.
            val_loss_f = best_val_loss if self.validation_selection else last_val_loss
            if len(idx_val) > 0 and (epoch % eval_every == 0 or epoch == n_adam):
                val_loss_f = validation_loss(state[0])
                last_val_loss = val_loss_f

                if self.validation_selection:
                    if val_loss_f < best_val_loss:
                        best_val_loss = val_loss_f
                        best_state = state
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= (patience // eval_every):
                            status = "validation_stopped"
                            break

            history["epoch"].append(epoch)
            history["loss_train"].append(tr_loss_f)
            history["loss_val"].append(val_loss_f)
            history["loss_data"].append(float(d_loss))
            history["loss_phys"].append(float(p_loss))
            history["phase"].append("adam")

            curr_phys = tp.get_physical_parameters(state[1])
            for pname in tp.fit_params:
                history[f"param_{pname}"].append(curr_phys[pname])

        if not self.validation_selection:
            best_state = state  # no validation-based checkpoint: continue from the last Adam state

        optimizer = {
            "lbfgs_outcome": "not_run",
            "lbfgs_status": None,
            "lbfgs_message": None,
            "lbfgs_iterations": 0,
            "error": None,
        }

        # 6. Phase 2: L-BFGS takes over from the best Adam checkpoint
        if status in ("converged", "validation_stopped") and n_lbfgs > 0 and not over_budget():
            flat0, unravel = ravel_pytree(best_state)
            flat_vg = jax.jit(jax.value_and_grad(lambda flat: loss_fn(unravel(flat))[0]))

            def lbfgs_obj(flat):
                val, grad = flat_vg(jnp.asarray(flat))
                return float(val), np.asarray(grad, dtype=np.float64)

            lbfgs_iter = [0]
            last_val_seen = [last_val_loss]

            def lbfgs_callback(intermediate_result):
                l_val = float(intermediate_result.fun)
                history["epoch"].append(len(history["epoch"]) + 1)
                history["loss_train"].append(l_val)
                lbfgs_iter[0] += 1
                if self.validation_selection:
                    val_now = best_val_loss
                elif len(idx_val) > 0 and lbfgs_iter[0] % eval_every == 0:
                    # measured on the same schedule as the Adam phase (every eval_every iterations)
                    val_now = validation_loss(unravel(jnp.asarray(intermediate_result.x))[0])
                    last_val_seen[0] = val_now
                else:
                    val_now = last_val_seen[0]  # carried between evaluations
                history["loss_val"].append(val_now)
                history["loss_data"].append(float("nan"))
                history["loss_phys"].append(float("nan"))
                history["phase"].append("lbfgs")
                if over_budget():
                    raise StopIteration

            try:
                fit = minimize(
                    lbfgs_obj,
                    np.asarray(flat0, dtype=np.float64),
                    jac=True,
                    method="L-BFGS-B",
                    callback=lbfgs_callback,
                    options={
                        "maxiter": n_lbfgs,
                        "maxfun": 20 * n_lbfgs,
                        "maxcor": 50,
                        "ftol": self.lbfgs_ftol,
                        "gtol": self.lbfgs_gtol,
                    },
                )
                # SciPy L-BFGS-B exit: 0 tolerance reached, 1 iteration/evaluation limit, 2 abnormal
                # (for example a failed line search). With ftol = gtol = 0 the limit is the usual end.
                optimizer.update(
                    lbfgs_status=int(fit.status),
                    lbfgs_message=str(fit.message),
                    lbfgs_iterations=int(fit.nit),
                    lbfgs_outcome={
                        0: "tolerance_reached",
                        1: "iteration_limit",
                        2: "abnormal_exit",
                    }.get(int(fit.status), "abnormal_exit"),
                )
                if np.all(np.isfinite(fit.x)):
                    lbfgs_candidate = unravel(jnp.asarray(fit.x))
                    if self.validation_selection and len(idx_val) > 0:
                        pred_val_lbfgs = jax.vmap(lambda t: x_hat(lbfgs_candidate[0], t, ncfg))(
                            t_val
                        )
                        val_loss_lbfgs = float(jnp.mean((pred_val_lbfgs - y_val_norm) ** 2))
                        # Accept L-BFGS solution if it does not degrade validation loss
                        if val_loss_lbfgs <= best_val_loss * 1.05:
                            best_state = lbfgs_candidate
                            best_val_loss = val_loss_lbfgs
                            status = "converged"
                    else:
                        best_state = lbfgs_candidate
                        status = "converged"
                else:
                    status = "nonfinite_loss"
            except (StopIteration, KeyboardInterrupt):
                optimizer["lbfgs_outcome"] = "budget_exhausted" if over_budget() else "interrupted"
                if over_budget():
                    status = "budget_exhausted"
            except Exception as exc:  # the Adam state is kept as a labelled fallback
                optimizer.update(lbfgs_outcome="error", error=f"{type(exc).__name__}: {exc}")
                status = "lbfgs_failed"
                success = False

        wall_time = time.perf_counter() - start_time

        # Restore best checkpoint state
        if status in ("converged", "validation_stopped", "budget_exhausted", "lbfgs_failed"):
            state = best_state

        # validation loss of the model that is actually returned
        final_val_loss = validation_loss(state[0]) if len(idx_val) > 0 else float("nan")

        final_phys = tp.get_physical_parameters(state[1])
        self.params_mod.raw_params = {k: jnp.asarray(v) for k, v in state[1].items()}
        estimated_p = {p: final_phys[p] for p in tp.fit_params}

        # 7. Final evaluation on the entire observation trajectory
        t_all_jax = jnp.asarray(times_all, dtype=jnp.float64)
        x_pred_all = jax.vmap(lambda t: x_hat(state[0], t, ncfg))(t_all_jax)
        pred_pop_all = np.asarray(x_pred_all * k_val, dtype=np.float64)

        train_err = errors(pred_pop_all[idx_train], obs_pop_all[idx_train])
        val_err = (
            errors(pred_pop_all[idx_val], obs_pop_all[idx_val])
            if len(idx_val) > 0
            else {"rmse": float("nan"), "mae": float("nan"), "relative_l2": None}
        )
        test_err = (
            errors(pred_pop_all[idx_test], obs_pop_all[idx_test])
            if len(idx_test) > 0
            else {"rmse": float("nan"), "mae": float("nan"), "relative_l2": None}
        )

        return InversePINNResult(
            model=model,
            fit_params=tp.fit_params,
            estimated_parameters=estimated_p,
            all_parameters=final_phys,
            status=status,
            success=success,
            final_epoch=len(history["epoch"]),
            wall_seconds=wall_time,
            train_loss=history["loss_train"][-1] if history["loss_train"] else float("nan"),
            val_loss=final_val_loss,
            train_metrics=train_err,
            val_metrics=val_err,
            test_metrics=test_err,
            predicted_trajectory={"time": times_all, "population": pred_pop_all},
            training_history=history,
            network_params=state[0],
            network_config=ncfg,
            optimizer=optimizer,
        )


def save_inverse_checkpoint(path: str | Path, result: InversePINNResult) -> None:
    """Save inverse result, network weights, and parameters to a portable NPZ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for i, (weight, bias) in enumerate(result.network_params):
        arrays[f"W{i}"] = np.asarray(weight)
        arrays[f"b{i}"] = np.asarray(bias)

    meta = {
        "schema": CHECKPOINT_SCHEMA_INVERSE,
        "model": result.model,
        "fit_params": result.fit_params,
        "estimated_parameters": result.estimated_parameters,
        "all_parameters": result.all_parameters,
        "status": result.status,
        "final_epoch": result.final_epoch,
        "wall_seconds": result.wall_seconds,
        "train_loss": result.train_loss,
        "val_loss": result.val_loss,
        "network_config": asdict(result.network_config) if result.network_config else {},
    }
    np.savez(path, meta=np.array(json.dumps(meta)), **arrays)


def load_inverse_checkpoint(path: str | Path) -> tuple[list, NetworkConfig, dict[str, float]]:
    """Load network weights, config, and estimated physical parameters from NPZ file."""
    with np.load(Path(path), allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("schema") != CHECKPOINT_SCHEMA_INVERSE:
            raise ValueError(f"Unsupported inverse checkpoint schema {meta.get('schema')}")
        n_layers = len([k for k in data.files if k.startswith("W")])
        params = [(jnp.asarray(data[f"W{i}"]), jnp.asarray(data[f"b{i}"])) for i in range(n_layers)]
        cfg_dict = meta["network_config"]
        cfg = NetworkConfig(**{**cfg_dict, "layers": tuple(cfg_dict["layers"])})
    return params, cfg, meta["estimated_parameters"]
