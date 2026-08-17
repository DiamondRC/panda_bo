"""
Hardware and simulation BO objective: RMS of following error in the plateau window.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class PlateauWindow:
    start_frac: float  # fraction into total samples where plateau begins
    end_frac: float  # fraction into total samples where plateau ends


# Inset inside the flat top of the demand trajectory, so the score never picks up
# the ramp shoulders. For data/xonly_trajectory.npy (140,000 samples, 0 -> 500,000
# counts) the true flat top runs from sample 39,185 to 100,815, i.e. fractions
# 0.2799 to 0.7201; the window below sits just inside that with a small margin.
DEFAULT_WINDOW = PlateauWindow(start_frac=0.289, end_frac=0.714)


def _as_2d(array: np.ndarray, name: str) -> np.ndarray:
    """Normalise (n,) or (n, k) input to (n, k). Anything else is an error.

    The simulator scores 3-axis (n, 3) arrays; the hardware evaluator streams a
    single encoder column and produces (n,). Both are valid here.
    """
    arr = np.asarray(array)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise ValueError(
        f"{name} must be 1-D (n,) or 2-D (n, n_axes), got shape {arr.shape}"
    )


def _select(array: np.ndarray, axes: Sequence[int] | None, name: str) -> np.ndarray:
    """Keep only the requested axis columns, raising on an index that isn't there.

    Failing loudly matters: a 3-axis config against single-encoder hardware
    would otherwise silently score the wrong thing.
    """
    if axes is None:
        return array
    axes = tuple(axes)
    if not axes:
        raise ValueError("axes must select at least one axis")
    n_axes = array.shape[1]
    bad = [a for a in axes if not -n_axes <= a < n_axes]
    if bad:
        raise ValueError(
            f"axes {bad} out of range for {name} with {n_axes} axis/axes. "
            f"Hardware streams a single column, so only axis 0 is available "
            f"unless extra position fields are configured."
        )
    return array[:, axes]


def diverged_at(
    setpoints: np.ndarray,
    positions: np.ndarray,
    axes: Sequence[int] | None = None,
    factor: float = 2.0,
) -> int | None:
    """First sample where the position leaves `factor` x the demand's envelope.

    Returns None if the run stayed inside it for the whole trajectory.
    """
    sp = _select(_as_2d(setpoints, "setpoints"), axes, "setpoints")
    pos = _select(_as_2d(positions, "positions"), axes, "positions")
    envelope = factor * float(np.abs(sp).max())
    if envelope == 0.0:
        return None
    outside = np.nonzero((np.abs(pos) > envelope).any(axis=1))[0]
    return int(outside[0]) if outside.size else None


def following_error_score(
    setpoints: np.ndarray,
    positions: np.ndarray,
    window: PlateauWindow = DEFAULT_WINDOW,
    axes: Sequence[int] | None = None,
    divergence_factor: float = 2.0,
    diverged_floor: float = 1.0e6,
) -> float:
    """BO objective: plateau RMS, with an ordering restored where it goes flat.

    `following_error_rms` alone is *exactly* constant across every run whose
    control saturates for the whole scored window: the plant is then driven by a
    constant input and the gains stop affecting the response at all. Measured on
    the 3.5v plant, eight unrelated Sobol points all scored 3.53812e6 to six
    significant figures. That flat region covers essentially the entire search
    space, so a GP has nothing to learn from and BO degenerates into random
    search.

    So when a run diverges this returns a value that is
      * always above `diverged_floor`, hence above any controlled run, and
      * decreasing as the run survives longer and tracks better before going.

    Both of those signals do vary with the gains where the plateau RMS does not
    (relative spread 0.039 and 0.49 respectively against 0.0), and both point
    toward the stable basin, so they give BO a gradient to follow into it.

    Controlled runs are scored exactly as before, so results stay comparable.
    """
    if diverged_floor <= 0.0:
        raise ValueError(f"diverged_floor must be positive, got {diverged_floor}")

    index = diverged_at(setpoints, positions, axes=axes, factor=divergence_factor)
    if index is None:
        return following_error_rms(setpoints, positions, window=window, axes=axes)

    sp = _select(_as_2d(setpoints, "setpoints"), axes, "setpoints")
    pos = _select(_as_2d(positions, "positions"), axes, "positions")
    n = sp.shape[0]

    # Fraction of the trajectory the run stayed controlled for; larger is better.
    survived = max(index / n, 1.0 / n)
    # How well it tracked while it was still controlled; smaller is better.
    pre_error = sp[:index] - pos[:index]
    pre_rms = float(np.sqrt(np.mean(pre_error**2))) if index > 0 else 0.0

    return diverged_floor / survived + pre_rms


def following_error_rms(
    setpoints: np.ndarray,
    positions: np.ndarray,
    window: PlateauWindow = DEFAULT_WINDOW,
    axes: Sequence[int] | None = None,
) -> float:
    """
    RMS of (setpoint - position) in the plateau window, over the selected axes.

    RMS = sqrt(mean(err²)) = sqrt(std² + mean²), so it penalises both
    variance (jitter) and mean offset (steady-state bias) without needing
    a tuning parameter. Lower is better; BO minimises this value.

    Accepts either 1-D (n,) or 2-D (n, n_axes) arrays, so the same objective
    scores the 3-axis simulator and the single-encoder hardware stream.

    axes:
        None      -> score every column present.
        Sequence  -> score only these column indices, e.g. (0,) for x only.

    Selecting the same axes on both paths is what makes the simulation and
    hardware objectives numerically comparable, and therefore what makes the
    simulation-to-hardware warm start meaningful. An axis index that does not
    exist in the supplied data raises rather than silently scoring the wrong
    thing.
    """
    sp = _as_2d(setpoints, "setpoints")
    pos = _as_2d(positions, "positions")

    if sp.shape != pos.shape:
        raise ValueError(
            f"setpoints and positions must have the same shape, "
            f"got {sp.shape} and {pos.shape}"
        )

    sp = _select(sp, axes, "setpoints")
    pos = _select(pos, axes, "positions")

    n = sp.shape[0]
    i0 = int(n * window.start_frac)
    i1 = int(n * window.end_frac)
    if i1 <= i0:
        raise ValueError(
            f"plateau window [{window.start_frac}, {window.end_frac}] selects no "
            f"samples from {n} samples"
        )

    err = sp[i0:i1] - pos[i0:i1]
    return float(np.sqrt(np.mean(err**2)))
