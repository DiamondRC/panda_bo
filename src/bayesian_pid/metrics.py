"""
Hardware and simulation BO objective: RMS of following error in the plateau window.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PlateauWindow:
    start_frac: float  # fraction into total samples where plateau begins
    end_frac: float  # fraction into total samples where plateau ends


# Derived from the live_utils trapezoid params used by the GUI and hardware:
#   zero_pad=1.5, slope_len=1.0, total_length=10
#   plateau = [2.5 s, 7.5 s]  =>  fractions [0.25, 0.75]
DEFAULT_WINDOW = PlateauWindow(start_frac=0.289, end_frac=0.714)


def following_error_rms(
    setpoints: np.ndarray,
    positions: np.ndarray,
    window: PlateauWindow = DEFAULT_WINDOW,
) -> float:
    """
    RMS of (setpoint - position) in the plateau window.

    RMS = sqrt(mean(err²)) = sqrt(std² + mean²), so it penalises both
    variance (jitter) and mean offset (steady-state bias) without needing
    a tuning parameter. Lower is better; BO minimises this value.
    """

    n = len(setpoints)
    i0 = int(n * window.start_frac)
    i1 = int(n * window.end_frac)
    err = setpoints[i0:i1, :] - positions[i0:i1, :]
    return float(np.sqrt(np.mean(err**2)))
