"""Tests for the BO loop, focused on hardware safety.

The load-bearing property here: when a warm start is supplied, the run is a
hardware run, and no evaluation may ever be a gain set drawn from the full
space. A TuRBO restart used to break that.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

import bayesian_pid.optimiser as opt
from bayesian_pid.optimiser import (
    _warm_start_design,  # pyright: ignore[reportPrivateUsage]
    run_mc_optimisation,
    run_one_optimization,
    select_warm_start,
)

DIM = 2
BOUNDS = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)


def quadratic(
    pid_tensor: torch.Tensor, return_details: bool = False
) -> tuple[torch.Tensor, None]:
    """Cheap analytic objective with its minimum at (0.25, 0.25)."""
    if pid_tensor.ndim == 1:
        pid_tensor = pid_tensor.unsqueeze(0)
    x = pid_tensor.detach().cpu().numpy()
    y = ((x - 0.25) ** 2).sum(axis=1)
    return torch.tensor(y, dtype=torch.double).unsqueeze(-1), None


def fake_result(points: list[list[float]], values: list[float]) -> dict[str, Any]:
    return {
        "eval_x_unit": torch.tensor(points, dtype=torch.double),
        "eval_y": np.asarray(values, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# select_warm_start
# ---------------------------------------------------------------------------


def test_k_of_one_returns_the_single_best_point() -> None:
    """k=1 reproduces the previously intended behaviour."""
    result = fake_result([[0.1, 0.1], [0.9, 0.9], [0.5, 0.5]], [5.0, 1.0, 3.0])
    ws = select_warm_start(result, k=1, verbose=False)
    assert ws.shape == (1, 2)
    np.testing.assert_allclose(ws[0].numpy(), [0.9, 0.9])


def test_selected_points_are_ordered_best_first() -> None:
    result = fake_result([[0.1, 0.1], [0.9, 0.9], [0.5, 0.5]], [5.0, 1.0, 3.0])
    ws = select_warm_start(result, k=3, min_separation=0.0, verbose=False)
    np.testing.assert_allclose(ws[0].numpy(), [0.9, 0.9])
    np.testing.assert_allclose(ws[1].numpy(), [0.5, 0.5])


def test_min_separation_skips_clustered_points() -> None:
    """Clustered top-k leave the GP's ARD lengthscales unidentifiable."""
    result = fake_result(
        [[0.50, 0.50], [0.501, 0.501], [0.502, 0.502], [0.9, 0.9]],
        [1.0, 1.1, 1.2, 4.0],
    )
    ws = select_warm_start(result, k=2, min_separation=0.05, verbose=False)
    assert ws.shape == (2, 2)
    separation = float(torch.norm(ws[0] - ws[1]).item())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    assert separation >= 0.05
    np.testing.assert_allclose(ws[1].numpy(), [0.9, 0.9])


def test_tops_up_when_too_few_points_are_separated() -> None:
    """Must still return k points rather than silently returning fewer."""
    result = fake_result([[0.5, 0.5], [0.501, 0.501], [0.502, 0.502]], [1.0, 1.1, 1.2])
    ws = select_warm_start(result, k=3, min_separation=0.5, verbose=False)
    assert ws.shape == (3, 2)


def test_never_returns_duplicate_points_when_topping_up() -> None:
    result = fake_result([[0.5, 0.5], [0.501, 0.501], [0.502, 0.502]], [1.0, 1.1, 1.2])
    ws = select_warm_start(result, k=3, min_separation=0.5, verbose=False)
    unique = {tuple(row) for row in ws.numpy().tolist()}
    assert len(unique) == 3


def test_rejects_invalid_k() -> None:
    result = fake_result([[0.5, 0.5]], [1.0])
    with pytest.raises(ValueError, match="k must be at least 1"):
        select_warm_start(result, k=0, verbose=False)


def test_rejects_empty_result() -> None:
    with pytest.raises(ValueError, match="no evaluations"):
        select_warm_start(fake_result([], []), k=1, verbose=False)


# ---------------------------------------------------------------------------
# _warm_start_design: the guarantee that nothing unvetted is generated
# ---------------------------------------------------------------------------


def test_design_returns_the_warm_start_points_first() -> None:
    ws = torch.tensor([[0.3, 0.3], [0.7, 0.7]], dtype=torch.double)
    gen = torch.Generator().manual_seed(0)
    design = _warm_start_design(ws, 5, 0.2, gen, torch.double, "cpu")
    assert design.shape == (5, 2)
    np.testing.assert_allclose(design[:2].numpy(), ws.numpy())


def test_design_keeps_every_extra_point_within_the_radius() -> None:
    """Top-up points must be perturbations of vetted gains, not fresh draws."""
    ws = torch.tensor([[0.5, 0.5]], dtype=torch.double)
    gen = torch.Generator().manual_seed(0)
    design = _warm_start_design(ws, 50, 0.1, gen, torch.double, "cpu")
    deviation = (design - ws[0]).abs().max().item()
    assert deviation <= 0.1 + 1e-12


def test_design_stays_inside_the_unit_cube() -> None:
    ws = torch.tensor([[0.02, 0.99]], dtype=torch.double)
    gen = torch.Generator().manual_seed(0)
    design = _warm_start_design(ws, 50, 0.3, gen, torch.double, "cpu")
    assert design.min().item() >= 0.0
    assert design.max().item() <= 1.0


def test_design_truncates_rather_than_dropping_warm_start_points() -> None:
    ws = torch.tensor([[0.3, 0.3], [0.7, 0.7], [0.9, 0.9]], dtype=torch.double)
    gen = torch.Generator().manual_seed(0)
    design = _warm_start_design(ws, 2, 0.2, gen, torch.double, "cpu")
    assert design.shape == (2, 2)
    np.testing.assert_allclose(design.numpy(), ws[:2].numpy())


# ---------------------------------------------------------------------------
# run_one_optimization
# ---------------------------------------------------------------------------


def test_n_init_is_honoured_with_a_warm_start() -> None:
    """Regression: warm_start_x used to silently override n_init down to its length.

    run.py asked for n_init=10 and got 1, leaving the GP fitted to a single point.
    """
    ws = torch.tensor([[0.5, 0.5]], dtype=torch.double)
    result = run_one_optimization(
        quadratic,
        BOUNDS,
        dim=DIM,
        total_budget=8,
        n_init=8,
        seed=1,
        verbose=False,
        warm_start_x=ws,
    )
    assert len(result["eval_y"]) == 8


def test_warm_start_points_are_evaluated_verbatim() -> None:
    ws = torch.tensor([[0.4, 0.6]], dtype=torch.double)
    result = run_one_optimization(
        quadratic,
        BOUNDS,
        dim=DIM,
        total_budget=4,
        n_init=4,
        seed=1,
        verbose=False,
        warm_start_x=ws,
    )
    np.testing.assert_allclose(result["eval_x_unit"][0].numpy(), [0.4, 0.6])


def test_rejects_unknown_restart_policy() -> None:
    with pytest.raises(ValueError, match="Unknown restart_policy"):
        run_one_optimization(
            quadratic, BOUNDS, dim=DIM, total_budget=2, restart_policy="nonsense"
        )


def test_without_warm_start_the_full_space_is_still_searched() -> None:
    """Simulation runs must keep their global Sobol coverage."""
    result = run_one_optimization(
        quadratic, BOUNDS, dim=DIM, total_budget=12, n_init=12, seed=1, verbose=False
    )
    spread = np.ptp(result["eval_x_unit"].numpy(), axis=0)
    assert (spread > 0.5).all(), "Sobol design should cover the space broadly"


@pytest.fixture
def collapsing_trust_region(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force TuRBO to restart after a single non-improving evaluation.

    A real collapse needs ~49 consecutive failures, and every BO step fits a GP,
    so reproducing that honestly would make this test minutes long. Shrinking
    both thresholds reaches the same restart code path in a few evaluations.
    """
    real = opt.TurboState

    def make(**kwargs: Any) -> Any:
        state = real(**kwargs)
        state.failure_tolerance = 1  # one failure halves the trust region
        state.length_min = 0.7  # and one halving from 0.8 triggers a restart
        return state

    monkeypatch.setattr(opt, "TurboState", make)


def test_restarts_never_leave_the_warm_start_region(
    collapsing_trust_region: None,
) -> None:
    """The safety-critical property.

    A restart used to draw n_init Sobol points over the full parameter box and
    write them straight to the device. Every evaluated point must now stay near
    a vetted warm-start point.
    """
    centre = np.array([0.5, 0.5])
    radius = 0.1
    ws = torch.tensor(centre.reshape(1, -1), dtype=torch.double)

    # Record which evaluations came from an initial design. Those are the ones
    # that used to be full-space Sobol draws; acquisition points are free to
    # roam within the trust region, which is expected behaviour.
    init_points: list[NDArray[np.float64]] = []
    restarts: set[int] = set()

    def capture(
        eval_count: int,
        total: int,
        x_phys: torch.Tensor,
        y: float,
        best_y: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        extra = extra or {}
        if extra.get("restart") is not None:
            restarts.add(extra["restart"])
        if extra.get("phase") == "init":
            init_points.append(x_phys.detach().cpu().numpy().flatten())

    run_one_optimization(
        quadratic,
        BOUNDS,
        dim=DIM,
        # Just enough to force a second restart; each BO step fits a GP, so
        # extra budget here costs real wall-clock in CI.
        total_budget=14,
        n_init=4,
        seed=1,
        verbose=False,
        warm_start_x=ws,
        restart_policy="warm_region",
        warm_restart_radius=radius,
        progress_fn=capture,
    )

    assert max(restarts) > 1, "test needs at least one restart to be meaningful"
    deviation = np.abs(np.array(init_points) - centre).max()
    assert deviation <= radius + 1e-12, (
        f"an initial-design point sat {deviation} from the warm start, so it "
        f"came from a full-space draw rather than the vetted region"
    )


def test_restart_policy_stop_halts_at_collapse(collapsing_trust_region: None) -> None:
    ws = torch.tensor([[0.5, 0.5]], dtype=torch.double)
    result = run_one_optimization(
        quadratic,
        BOUNDS,
        dim=DIM,
        total_budget=40,
        n_init=4,
        seed=1,
        verbose=False,
        warm_start_x=ws,
        restart_policy="stop",
    )
    assert len(result["eval_y"]) < 40, "should stop before exhausting the budget"
    assert result["best_x_phys"] is not None


# ---------------------------------------------------------------------------
# run_mc_optimisation CSV labelling
# ---------------------------------------------------------------------------


def test_csv_columns_follow_param_names(tmp_path: Path) -> None:
    """Regression: columns were hardcoded to seven gains in a fixed order.

    Reordering or resizing TUNE_PARAMS silently mislabelled them.
    """
    names = ["alpha", "beta"]
    run_mc_optimisation(
        quadratic,
        BOUNDS,
        dim=DIM,
        n_runs=2,
        total_budget=4,
        n_init=4,
        verbose=False,
        save_csv=True,
        results_dir=str(tmp_path),
        param_names=names,
    )
    import pandas as pd

    summary = pd.read_csv(tmp_path / "turbo_mc_summary.csv")
    assert "best_alpha" in summary.columns
    assert "best_beta" in summary.columns
    assert not any(str(c).startswith("best_K") for c in summary.columns)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportUnknownVariableType]

    evals = pd.read_csv(tmp_path / "turbo_mc_all_evaluations.csv")
    assert "alpha" in evals.columns
    assert "beta" in evals.columns


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_replays_exactly() -> None:
    """The headline guarantee: seed in, same run out, bit for bit.

    Not merely the final value — every evaluated point, in order. This holds
    only at DEFAULT_BO_THREADS == 1; above one thread BLAS reorders its
    reductions and the search diverges within ~13 iterations.
    """
    kwargs: dict[str, Any] = {
        "dim": DIM,
        "seed": 3,
        "total_budget": 25,
        "n_init": 5,
        "verbose": False,
    }
    first = run_one_optimization(quadratic, BOUNDS, **kwargs)
    second = run_one_optimization(quadratic, BOUNDS, **kwargs)

    np.testing.assert_array_equal(first["eval_y"], second["eval_y"])
    np.testing.assert_array_equal(
        first["eval_x_phys"].numpy(), second["eval_x_phys"].numpy()
    )
    assert first["best_y"] == second["best_y"]


def test_different_seeds_do_not_replay() -> None:
    """Guards the test above against passing for the wrong reason — e.g. an
    objective flat enough that every run looks identical."""
    kwargs: dict[str, Any] = {
        "dim": DIM,
        "total_budget": 25,
        "n_init": 5,
        "verbose": False,
    }
    a = run_one_optimization(quadratic, BOUNDS, seed=3, **kwargs)
    b = run_one_optimization(quadratic, BOUNDS, seed=4, **kwargs)
    assert not np.array_equal(a["eval_y"], b["eval_y"])


# ---------------------------------------------------------------------------
# Parallel MC
# ---------------------------------------------------------------------------


def test_parallel_mc_matches_sequential_exactly() -> None:
    """The load-bearing property: n_jobs is a scheduling choice, not a
    numerical one. Seeds are independent, so running them in worker processes
    must return bit-identical results to running them in a loop. This is what
    lets the cores be used without giving up reproducibility — unlike raising
    the thread count, which does change the answer.
    """
    kwargs: dict[str, Any] = {
        "dim": DIM,
        "n_runs": 3,
        "seed0": 1,
        "total_budget": 12,
        "n_init": 4,
        "verbose": False,
        "save_csv": False,
        "torch_threads": 1,
    }
    sequential = run_mc_optimisation(quadratic, BOUNDS, n_jobs=1, **kwargs)
    parallel = run_mc_optimisation(quadratic, BOUNDS, n_jobs=3, **kwargs)

    np.testing.assert_array_equal(
        sequential["final_best_values"], parallel["final_best_values"]
    )
    np.testing.assert_array_equal(sequential["best_curves"], parallel["best_curves"])


def test_parallel_results_are_ordered_by_seed_not_completion() -> None:
    """Workers finish in whatever order they finish; the aggregate must not
    depend on that, or the per-seed CSV columns would be mislabelled."""
    result = run_mc_optimisation(
        quadratic,
        BOUNDS,
        dim=DIM,
        n_runs=3,
        seed0=7,
        total_budget=8,
        n_init=4,
        verbose=False,
        save_csv=False,
        n_jobs=3,
        torch_threads=1,
    )
    assert [r["seed"] for r in result["run_results"]] == [7, 8, 9]


def test_n_jobs_is_capped_at_the_number_of_runs() -> None:
    """Asking for more workers than seeds must not spawn idle processes."""
    result = run_mc_optimisation(
        quadratic,
        BOUNDS,
        dim=DIM,
        n_runs=2,
        total_budget=8,
        n_init=4,
        verbose=False,
        save_csv=False,
        n_jobs=64,
        torch_threads=1,
    )
    assert len(result["run_results"]) == 2


def test_zero_jobs_is_rejected() -> None:
    with pytest.raises(ValueError, match="n_jobs must be at least 1"):
        run_mc_optimisation(
            quadratic,
            BOUNDS,
            dim=DIM,
            n_runs=1,
            total_budget=4,
            n_init=4,
            verbose=False,
            save_csv=False,
            n_jobs=0,
        )


def test_thread_cap_does_not_leak_out_of_a_run() -> None:
    """run_one_optimization sets a global; it must put it back."""
    before = torch.get_num_threads()
    run_one_optimization(
        quadratic,
        BOUNDS,
        dim=DIM,
        seed=1,
        total_budget=6,
        n_init=3,
        verbose=False,
        torch_threads=1,
    )
    assert torch.get_num_threads() == before


def test_param_names_length_must_match_dim(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="param_names has 3 entries but dim is 2"):
        run_mc_optimisation(
            quadratic,
            BOUNDS,
            dim=DIM,
            n_runs=1,
            total_budget=2,
            n_init=2,
            verbose=False,
            save_csv=False,
            results_dir=str(tmp_path),
            param_names=["a", "b", "c"],
        )
