"""Clean synthetic references only; WP05 owns noise, splits and inverse fitting."""

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np

from .models import Parameters
from .solver import MIN_RK45_RTOL, solve_trajectory

DEFAULT_SOLVER = {"rtol": 1e-9, "atol": 1e-11, "max_step": 0.1}
REFERENCE_COLUMNS = (
    "observation_id",
    "trajectory_id",
    "model",
    "time",
    "population",
    "is_synthetic",
)


@dataclass(frozen=True)
class ReferenceTrajectory:
    trajectory_id: str
    model: str
    parameters: Parameters
    time: np.ndarray
    population: np.ndarray
    extinction_time: float | None
    boundary_policy: str


def solver_settings(settings=None):
    result = DEFAULT_SOLVER | (settings or {})
    if set(result) != set(DEFAULT_SOLVER):
        raise ValueError("Solver settings are rtol, atol and max_step (dimensionless)")
    if any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in result.values()):
        raise ValueError("Solver settings must be finite and positive")
    if result["rtol"] < MIN_RK45_RTOL:
        raise ValueError(f"RK45 rtol must be at least {MIN_RK45_RTOL:.17g}")
    return result


def generate_reference(case, solver=None):
    """Generate one deterministic dimensional A/B trajectory, with no noise/splits.

    time_points, when supplied, are dimensional and begin at zero. Otherwise
    n_points evenly cover [0,min(time_end,extinction_time)]. Only physical
    pre-event data and the event endpoint are returned; no absorbing tail.
    """
    name = case["trajectory_id"]
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", name):
        raise ValueError("trajectory_id must be a short lowercase identifier")
    model = case["model"]
    if model not in {"A", "B"}:
        raise ValueError("Reference cases must be Model A or B")
    p = Parameters(**case["parameters"])
    if (model == "A" and p.h != 0) or (model == "B" and p.H != 0):
        raise ValueError("A uses H only; B uses h only")
    initial, end = case["initial_population"], case["time_end"]
    if any(isinstance(v, bool) or not isfinite(v) for v in (initial, end)):
        raise ValueError("Initial population and time_end must be finite numbers")
    d = p.dimensionless()
    times = case.get("time_points")
    tau_eval = None if times is None else p.r * np.asarray(times, dtype=float)
    solution = solve_trajectory(
        model,
        a=d["a"],
        harvest=d["q"] if model == "A" else d["e"],
        x0=initial / p.K,
        tau_end=p.r * end,
        n_points=case.get("n_points", 101),
        tau_eval=tau_eval,
        **solver_settings(solver),
    )
    return ReferenceTrajectory(
        name,
        model,
        p,
        solution.tau / p.r,
        solution.x * p.K,
        None if solution.extinction_tau is None else solution.extinction_tau / p.r,
        solution.boundary_policy,
    )


def validate_config(config):
    if config.get("schema_version") != 1:
        raise ValueError("Expected schema_version 1")
    for key in ("time", "population"):
        unit = config.get("units", {}).get(key)
        scale = config.get("fixed_reference_scales", {}).get(key)
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError(f"Declare {key} units")
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not isfinite(scale)
            or scale <= 0
        ):
            raise ValueError(f"Declare a finite positive fixed {key} reference scale")
    solver_settings(config.get("solver"))
    cases = config.get("cases", [])
    if not cases:
        raise ValueError("At least one case is required")
    names = [c["trajectory_id"] for c in cases]
    if len(names) != len(set(names)):
        raise ValueError("trajectory_id values must be unique")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_reference_files(output, references, config):
    """Write clean CSVs, declared known conditions and separate evaluation truth.

    manifest/config/truth are evaluation metadata, not an inverse trainer input.
    conditions.json deliberately omits generating r,A,H/h and dimensionless scales.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    conditions = {
        "schema_version": 1,
        "is_synthetic": True,
        "units": config["units"],
        "fixed_reference_scales": config["fixed_reference_scales"],
        "inverse_design": "Known K and initial population; WP05 chooses fitted subset",
        "trajectories": {},
    }
    truth = {"schema_version": 1, "evaluation_only": True, "trajectories": {}}
    paths = []
    for ref in references:
        path = output / f"{ref.trajectory_id}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(REFERENCE_COLUMNS)
            for i, (time, population) in enumerate(zip(ref.time, ref.population, strict=True)):
                writer.writerow(
                    [
                        f"{ref.trajectory_id}:{i:05d}",
                        ref.trajectory_id,
                        ref.model,
                        format(float(time), ".17g"),
                        format(float(population), ".17g"),
                        "true",
                    ]
                )
        paths.append(path)
        conditions["trajectories"][ref.trajectory_id] = {
            "model": ref.model,
            "known_K": ref.parameters.K,
            "known_initial_population": float(ref.population[0]),
            "csv": path.name,
        }
        truth["trajectories"][ref.trajectory_id] = {
            "parameters": vars(ref.parameters),
            "extinction_time": ref.extinction_time,
            "termination": "extinction_event"
            if ref.extinction_time is not None
            else "horizon_reached_no_event",
            "boundary_policy": ref.boundary_policy,
            "first_time": float(ref.time[0]),
            "last_time": float(ref.time[-1]),
            "n_observations": len(ref.time),
        }
    for name, value in (("conditions.json", conditions), ("truth.json", truth)):
        path = output / name
        write_json(path, value)
        paths.append(path)
    return paths
