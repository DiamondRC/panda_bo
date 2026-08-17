"""Tests for the BO objective, following_error_rms.

The 1-D case is the one that crashed the last live hardware run: the simulator
passes (n, 3) arrays but the hardware evaluator streams a single encoder column.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from bayesian_pid.metrics import (
    DEFAULT_WINDOW,
    PlateauWindow,
    diverged_at,
    following_error_rms,
    following_error_score,
)

Array = NDArray[np.float64]
Signals = tuple[Array, Array]

N = 1000


@pytest.fixture
def signals() -> Signals:
    """A 1-D setpoint/position pair with a known, non-trivial error."""
    rng = np.random.RandomState(0)
    setpoints = np.linspace(0.0, 100.0, N)
    positions = setpoints + rng.normal(0.0, 0.5, N)
    return setpoints, positions


def test_accepts_1d_hardware_arrays(signals: Signals) -> None:
    """Regression: hardware streams 1-D arrays and used to raise IndexError."""
    setpoints, positions = signals
    assert following_error_rms(setpoints, positions) > 0.0


def test_1d_and_single_column_2d_agree(signals: Signals) -> None:
    """The simulation and hardware paths must score identical data identically."""
    setpoints, positions = signals
    assert following_error_rms(setpoints, positions) == following_error_rms(
        setpoints[:, None], positions[:, None]
    )


def test_axis_selection_picks_the_named_column(signals: Signals) -> None:
    """axes=(0,) on multi-axis data must equal scoring that axis alone."""
    setpoints, positions = signals
    multi_sp = np.column_stack([setpoints, setpoints * 2, setpoints * 3])
    multi_pos = np.column_stack([positions, positions * 2, positions * 3])
    assert following_error_rms(multi_sp, multi_pos, axes=(0,)) == pytest.approx(
        following_error_rms(setpoints, positions)
    )
    # Axis 1's error is scaled by 2, so its RMS is too.
    assert following_error_rms(multi_sp, multi_pos, axes=(1,)) == pytest.approx(
        2 * following_error_rms(setpoints, positions)
    )


def test_axes_none_scores_every_column(signals: Signals) -> None:
    """Default behaviour combines all axes, as the pre-existing 3-D objective did."""
    setpoints, positions = signals
    multi_sp = np.column_stack([setpoints, setpoints])
    multi_pos = np.column_stack([positions, positions])
    # Two identical columns: combined RMS equals the single-column RMS.
    assert following_error_rms(multi_sp, multi_pos) == pytest.approx(
        following_error_rms(setpoints, positions)
    )


def test_out_of_range_axis_raises_rather_than_misscoring(signals: Signals) -> None:
    """A 3-axis config against single-encoder hardware must fail loudly."""
    setpoints, positions = signals
    with pytest.raises(ValueError, match="out of range"):
        following_error_rms(setpoints, positions, axes=(0, 1, 2))


def test_empty_axes_raises(signals: Signals) -> None:
    setpoints, positions = signals
    with pytest.raises(ValueError, match="at least one axis"):
        following_error_rms(setpoints, positions, axes=())


def test_mismatched_shapes_raise(signals: Signals) -> None:
    setpoints, positions = signals
    with pytest.raises(ValueError, match="same shape"):
        following_error_rms(setpoints, positions[: N // 2])


def test_rejects_3d_input(signals: Signals) -> None:
    setpoints, _ = signals
    with pytest.raises(ValueError, match="must be 1-D"):
        following_error_rms(setpoints.reshape(-1, 1, 1), setpoints.reshape(-1, 1, 1))


def test_window_selects_the_expected_samples() -> None:
    """Only samples inside the window contribute."""
    setpoints = np.zeros(N)
    positions = np.zeros(N)
    window = PlateauWindow(start_frac=0.5, end_frac=0.6)
    # Error outside the window must be ignored entirely.
    positions[:500] = 1000.0
    positions[600:] = 1000.0
    assert following_error_rms(setpoints, positions, window=window) == 0.0
    # Error inside it must not be.
    positions[550] = 4.0
    expected = np.sqrt((4.0**2) / 100)
    assert following_error_rms(setpoints, positions, window=window) == pytest.approx(
        expected
    )


def test_rms_equals_sqrt_of_variance_plus_squared_mean(signals: Signals) -> None:
    """The documented identity RMS = sqrt(std^2 + mean^2)."""
    setpoints, positions = signals
    i0 = int(N * DEFAULT_WINDOW.start_frac)
    i1 = int(N * DEFAULT_WINDOW.end_frac)
    err = (setpoints - positions)[i0:i1]
    assert following_error_rms(setpoints, positions) == pytest.approx(
        np.sqrt(err.std() ** 2 + err.mean() ** 2)
    )


def test_perfect_tracking_scores_zero() -> None:
    setpoints = np.linspace(0.0, 100.0, N)
    assert following_error_rms(setpoints, setpoints.copy()) == 0.0


def test_degenerate_window_raises(signals: Signals) -> None:
    setpoints, positions = signals
    with pytest.raises(ValueError, match="no samples"):
        following_error_rms(
            setpoints, positions, window=PlateauWindow(start_frac=0.7, end_frac=0.3)
        )


# ---------------------------------------------------------------------------
# Divergence-aware BO objective
# ---------------------------------------------------------------------------


def controlled(n: int = 1000) -> tuple[Array, Array]:
    sp = np.concatenate([np.linspace(0, 100, n // 2), np.full(n - n // 2, 100.0)])
    return sp, sp - 0.5


def diverging(
    n: int = 1000, escape: int = 400, rate: float = 1.5
) -> tuple[Array, Array]:
    """A run that tracks, then leaves the demand envelope at `escape`."""
    sp = np.concatenate([np.linspace(0, 100, n // 2), np.full(n - n // 2, 100.0)])
    pos = sp.copy()
    pos[escape:] = 200.0 * rate ** np.arange(n - escape).clip(max=40)
    return sp, pos


def test_diverged_at_is_none_for_a_controlled_run() -> None:
    sp, pos = controlled()
    assert diverged_at(sp, pos) is None


def test_diverged_at_finds_the_escape_sample() -> None:
    sp, pos = diverging(escape=400)
    index = diverged_at(sp, pos)
    assert index is not None
    assert index >= 400


def test_score_equals_plateau_rms_when_controlled() -> None:
    """Controlled runs must score exactly as before, so results stay comparable."""
    sp, pos = controlled()
    assert following_error_score(sp, pos) == following_error_rms(sp, pos)


def test_score_of_any_diverging_run_exceeds_every_controlled_run() -> None:
    sp_c, pos_c = controlled()
    sp_d, pos_d = diverging()
    assert following_error_score(sp_d, pos_d) > following_error_score(sp_c, pos_c)
    assert following_error_score(sp_d, pos_d) > 1.0e6


def test_score_orders_diverging_runs_by_how_long_they_survived() -> None:
    """The gradient that the plateau RMS alone cannot provide.

    Plateau RMS is exactly constant once control saturates for the whole window,
    so BO has nothing to follow. Surviving longer must score better.
    """
    early = following_error_score(*diverging(escape=200))
    late = following_error_score(*diverging(escape=800))
    assert late < early, "a run that stayed controlled longer must score better"


def test_score_is_not_flat_across_diverging_runs() -> None:
    """Regression: eight unrelated diverging points all scored 3.53812e6."""
    scores = [following_error_score(*diverging(escape=e)) for e in (300, 400, 500, 600)]
    assert len(set(scores)) == len(scores), "diverging runs must not all score alike"


def test_score_rejects_a_non_positive_floor() -> None:
    sp, pos = controlled()
    with pytest.raises(ValueError, match="diverged_floor must be positive"):
        following_error_score(sp, pos, diverged_floor=0.0)


def test_gain_vector_packs_in_the_order_the_fast_simulator_expects() -> None:
    from bayesian_pid.validation import GAIN_ORDER, gain_vector

    vector = gain_vector({"kp": 1.0, "kv": 2.0, "ki": 3.0})
    assert GAIN_ORDER.index("kp") == 0
    np.testing.assert_allclose(vector[:3], [1.0, 2.0, 3.0])


def test_gain_vector_defaults_ktot_to_one_not_zero() -> None:
    """A zero ktot would mute the controller entirely."""
    from bayesian_pid.validation import GAIN_ORDER, gain_vector

    vector = gain_vector({"kp": 1.0})
    assert vector[GAIN_ORDER.index("ktot")] == 1.0
    assert vector[GAIN_ORDER.index("kd")] == 0.0


def test_gain_vector_rejects_an_unknown_gain() -> None:
    from bayesian_pid.validation import gain_vector

    with pytest.raises(ValueError, match="unknown gains"):
        gain_vector({"kp": 1.0, "not_a_gain": 2.0})
