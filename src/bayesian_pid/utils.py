import math
from dataclasses import dataclass

import numpy as np
import torch

# ============================================================
# Utilities
# ============================================================


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def unnormalize(x_unit: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """
    Map X from [0,1]^d to physical bounds.
    bounds shape: (2, d), where bounds[0] = lower, bounds[1] = upper
    """
    return bounds[0] + (bounds[1] - bounds[0]) * x_unit


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")
    success_counter: int = 0
    success_tolerance: int = 5
    best_value: float = float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state, y_next):
    y_next_best = y_next.min().item()

    # Improvement threshold
    improvement_tol = 1e-3 * max(1.0, abs(state.best_value))

    if y_next_best < state.best_value - improvement_tol:
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0

    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = min(state.best_value, y_next_best)

    if state.length < state.length_min:
        state.restart_triggered = True

    return state
