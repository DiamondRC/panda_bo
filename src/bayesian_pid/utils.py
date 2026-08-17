import math
import random
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

# ============================================================
# Utilities
# ============================================================


# Threads torch may use inside a BO loop. This is 1 for REPRODUCIBILITY, not for
# speed: a seeded run must replay exactly, and above one thread it does not.
#
# Why. BLAS splits a reduction across threads and sums the partials in a
# different grouping, so the GP fit lands a few hundred ULPs away. That would be
# harmless if the loop were not a feedback loop — but every candidate is chosen
# from a GP fitted to every previous candidate, so the perturbation compounds.
# Traced on seed 1, 1 thread against 8:
#
#     iter  n   lengthscale rel diff   objective (1 thr)   objective (8 thr)
#     0-9   10-19             0        identical           identical
#     10    20         1.6e-13         243.31176           243.31176
#     12    22         2.3e-08         7.6025131           7.6025135
#     16    26         1.2e-05         0.32796257          0.32798085
#     22    32         1.5e-04         641.66              2694.41
#
# 1e-13 to 1e-2 in thirteen iterations. The Sobol design and its objectives are
# bit-identical throughout — the seeding never depended on threads — and the
# divergence begins exactly at n=20, consistent with BLAS only parallelising a
# reduction above a size threshold.
#
# The amplification cannot be suppressed; it is what a chaotic search does to any
# perturbation, whether it comes from thread count, a different BLAS build or a
# new CPU. The only fix is to not create the perturbation.
#
# Cost: one thread is free below n~200 and ~2.2x slower on the fit at n=300
# (395 ms against 178 ms at 8 threads). Take parallelism from
# `run_mc_optimisation(n_jobs=...)` instead — separate processes, one thread
# each, which is both reproducible and scales past what threads managed.
DEFAULT_BO_THREADS = 1


@contextmanager
def torch_thread_limit(n_threads: int | None) -> Generator[None]:
    """Cap torch's intra-op thread pool for the duration of the block.

    `None` leaves the current setting alone. The previous value is restored on
    exit, so importing this library never silently reconfigures a caller's torch.

    Setting this to 1 is what makes a seeded run replay exactly, and it is
    sufficient on its own: verified bit-identical across repeat runs and against
    a hostile `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8`
    environment, on the full trace (Sobol design, objectives, GP noise,
    lengthscales, candidates). torch's setting overrides the environment.

    Above 1 the thread count changes results, not merely speed — see
    DEFAULT_BO_THREADS. Raise it only for a one-off run whose exact numbers you
    do not need to reproduce, and never mid-comparison.
    """
    if n_threads is None:
        yield
        return
    if n_threads < 1:
        raise ValueError(f"n_threads must be at least 1, got {n_threads}")
    previous = torch.get_num_threads()
    torch.set_num_threads(n_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def set_seed(seed: int) -> None:
    """Seed every RNG a run can draw from, so MC seeds are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    # torch's stubs leave manual_seed's parameter untyped.
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]


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
    # Overwritten in __post_init__ from dim and batch_size; the sentinel is only
    # here so the field has a default. It was annotated `int` while defaulting to
    # a float NaN, which pyright flags.
    failure_tolerance: int = -1
    success_counter: int = 0
    success_tolerance: int = 5
    best_value: float = float("inf")
    restart_triggered: bool = False

    def __post_init__(self) -> None:
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state: TurboState, y_next: torch.Tensor) -> TurboState:
    """Grow, shrink or flag-for-restart the trust region after an evaluation."""
    y_next_best = float(y_next.min().item())

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
