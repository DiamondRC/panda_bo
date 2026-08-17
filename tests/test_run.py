"""Tests for the run script's configuration and helpers.

Importing this module at all is part of the test: run.py used to execute its
whole body on import (clearing the terminal, loading data from an absolute path,
reassigning sys.stdout, prompting for input), which broke pytest collection.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from bayesian_pid import run
from bayesian_pid.panda import validate_bounds

DATA_AVAILABLE = os.path.isdir(run.DATA_PATH)
needs_data = pytest.mark.skipif(
    not DATA_AVAILABLE, reason=f"data directory {run.DATA_PATH} not present"
)


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------


def test_module_exposes_main_without_running_it() -> None:
    assert callable(run.main)


def test_configuration_constants_are_present() -> None:
    for name in (
        "TUNE_PARAMS",
        "FIXED_PARAMS",
        "SCORE_AXES",
        "DT",
        "DT_INV",
        "MAX_OUTPUT",
        "MAX_INTEGRAL",
    ):
        assert hasattr(run, name), name


# ---------------------------------------------------------------------------
# Configuration consistency
# ---------------------------------------------------------------------------


def test_tuning_bounds_are_writable_to_the_device() -> None:
    """Catches an edit to TUNE_PARAMS that BO could not transfer to hardware."""
    validate_bounds(run.TUNE_PARAMS)


def test_fixed_params_are_writable_to_the_device() -> None:
    validate_bounds({k: (v, v) for k, v in run.FIXED_PARAMS.items()})


def test_expected_device_config_matches_the_live_registers() -> None:
    """These were read from bl99p-mo-panda-03; the simulation must mirror them.

    dt_i and dt_inv_i are independent registers, both 1, matching the PMAC C
    reference which has no dt at all.
    """
    assert run.EXPECTED_DEVICE_CONFIG == {
        "dt": 1.0,
        "dt_inv": 1.0,
        "ktot": 1.0,
        "max_output": 3932.0,
        "max_integral": 28000.0,
        "dir_toggle": 0.0,
    }


def test_expected_config_is_derived_from_the_module_constants() -> None:
    """The two must not be able to drift apart."""
    assert run.EXPECTED_DEVICE_CONFIG["dt"] == run.DT
    assert run.EXPECTED_DEVICE_CONFIG["dt_inv"] == run.DT_INV
    assert run.EXPECTED_DEVICE_CONFIG["max_output"] == run.MAX_OUTPUT
    assert run.EXPECTED_DEVICE_CONFIG["max_integral"] == run.MAX_INTEGRAL


def test_every_tuned_gain_has_a_device_field() -> None:
    from bayesian_pid.panda import PANDA_PID_FIELDS

    for name in run.TUNE_PARAMS:
        assert name in PANDA_PID_FIELDS, name


def test_tuned_and_fixed_params_do_not_overlap() -> None:
    """An overlap would make the fixed value silently win over the search."""
    assert not set(run.TUNE_PARAMS) & set(run.FIXED_PARAMS)


def test_bounds_tensor_follows_param_order() -> None:
    bounds = run.make_bounds(run.TUNE_PARAMS)
    assert bounds.shape == (2, len(run.TUNE_PARAMS))
    for i, (lo, hi) in enumerate(run.TUNE_PARAMS.values()):
        assert bounds[0, i].item() == lo
        assert bounds[1, i].item() == hi


# ---------------------------------------------------------------------------
# Results directory
# ---------------------------------------------------------------------------


def test_results_dir_is_timestamped_so_runs_do_not_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: every run used to write into one fixed directory."""
    monkeypatch.setattr(run, "RESULTS_ROOT", str(tmp_path / "results"))
    monkeypatch.setattr(run, "TIMESTAMP_RESULTS", True)
    directory = run.make_results_dir()
    assert os.path.isdir(directory)
    assert os.path.basename(directory).startswith("run_")
    assert str(tmp_path) in directory


def test_results_dir_can_be_pinned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = str(tmp_path / "fixed")
    monkeypatch.setattr(run, "RESULTS_ROOT", root)
    monkeypatch.setattr(run, "TIMESTAMP_RESULTS", False)
    assert run.make_results_dir() == root
    assert os.path.isdir(root)


# ---------------------------------------------------------------------------
# Data loading and objective construction
# ---------------------------------------------------------------------------


@needs_data
def test_plant_matrices_have_compatible_shapes() -> None:
    a1, a2, b, c = run.load_plant()
    assert a1.shape == (3, 3)
    assert a2.shape == (3, 3)
    assert b.shape == (3, 3)
    assert c.shape == (3,)


@needs_data
def test_trajectory_is_padded_to_three_axes() -> None:
    trajectory = run.load_trajectory()
    assert trajectory.ndim == 2
    assert trajectory.shape[1] == 3
    # x carries the demand; y and z are zero-padded.
    assert np.any(trajectory[:, 0] != 0)
    assert not np.any(trajectory[:, 1:])


@needs_data
def test_objective_scores_the_known_device_gains() -> None:
    """End-to-end: the gains currently on the PandA should score sensibly."""
    param_names = list(run.TUNE_PARAMS)
    objective = run.build_sim_objective(
        run.load_trajectory(), run.load_plant(), param_names
    )
    device_gains = {
        "kp": 0.8671094452078429,
        "kv": 2.0,
        "ki": 0.007852857458780095,
        "kvff": 4.7250680531512765,
        "kaff": 214.82588340071794,
        "kpff1": 0.0,
        "kpff0": 0.0,
    }
    x = torch.tensor([[device_gains[n] for n in param_names]], dtype=torch.double)
    y, _ = objective(x)
    value = float(y)
    assert np.isfinite(value)
    assert 0.0 < value < 1.0, f"expected sub-count tracking error, got {value}"


# ---------------------------------------------------------------------------
# Cold simulation, warm hardware — the framework's core contract
# ---------------------------------------------------------------------------


def _noop(*args: Any, **kwargs: Any) -> None:
    """Typed stand-in for callbacks the contract tests do not care about."""
    return None


def _zero_objective(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, None]:
    return torch.zeros(1, 1, dtype=torch.double), None


def _true(*args: Any, **kwargs: Any) -> bool:
    return True


def test_simulation_is_a_cold_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sim must never be seeded with prior knowledge of the stage.

    This is what makes the pipeline general rather than a solution for one
    stage: run fresh, it discovers gains for hardware it has never seen. Any
    warm start here would flatter the result with an answer it was handed.
    """
    captured: dict[str, Any] = {}

    def fake_run_mc(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"run_results": [{"best_y": 1.0, "best_x_phys": torch.zeros(2)}]}

    monkeypatch.setattr(run, "run_mc_optimisation", fake_run_mc)
    monkeypatch.setattr(run, "plot_mc_result", _noop)
    monkeypatch.setattr(run, "RUN_MC", True)

    run.run_simulation(
        _zero_objective,
        ["a", "b"],
        run.make_bounds({"a": (0.0, 1.0), "b": (0.0, 1.0)}),
        ".",
        _noop,
    )
    assert "warm_start_x" not in captured, (
        "the simulated search must start cold; seeding it bakes stage-specific "
        "knowledge into a framework meant to generalise"
    )


def test_single_run_simulation_is_also_a_cold_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run_one(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"best_y": 1.0, "best_x_phys": torch.zeros(2)}

    monkeypatch.setattr(run, "run_one_optimization", fake_run_one)
    monkeypatch.setattr(run, "plot_single_result", _noop)
    monkeypatch.setattr(run, "RUN_MC", False)

    run.run_simulation(
        _zero_objective,
        ["a", "b"],
        run.make_bounds({"a": (0.0, 1.0), "b": (0.0, 1.0)}),
        ".",
        _noop,
    )
    assert captured.get("warm_start_x") is None


def test_hardware_phase_warm_starts_from_the_simulation_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prior knowledge enters only here, so live gains are always sim-vetted."""
    captured: dict[str, Any] = {}

    def fake_run_one(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"best_y": 1.0, "best_x_phys": torch.zeros(2)}

    monkeypatch.setattr(run, "health_check", _true)
    monkeypatch.setattr(run, "make_hardware_evaluator", _noop)
    monkeypatch.setattr(run, "run_one_optimization", fake_run_one)

    sim_result = {
        "eval_x_unit": torch.tensor([[0.3, 0.4], [0.9, 0.9]], dtype=torch.double),
        "eval_y": np.array([0.5, 9.0]),
    }
    run.run_hardware_phase(
        sim_result,
        ["a", "b"],
        run.make_bounds({"a": (0.0, 1.0), "b": (0.0, 1.0)}),
        np.zeros((4, 3)),
        "debug.png",
        _noop,
    )

    warm = captured.get("warm_start_x")
    assert warm is not None, "hardware must start from the simulation's best point"
    # The best sim evaluation was (0.3, 0.4) with objective 0.5.
    np.testing.assert_allclose(warm[0].numpy(), [0.3, 0.4])


def test_hardware_restart_policy_never_allows_random_gains() -> None:
    """A restart must re-seed near vetted points, not draw over the full box."""
    assert run.HW_RESTART_POLICY in ("warm_region", "stop")
