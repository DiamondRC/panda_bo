"""Tests for the closed-loop simulator.

All tests use a short synthetic trajectory and a small plant so the suite stays
fast; the real 140,000-sample trajectory takes ~2 s per run.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from bayesian_pid.controller import PID
from bayesian_pid.sim import simulate_pid, simulate_pid_fast

Array = NDArray[np.float64]
Plant = tuple[Array, Array, Array, Array]

N_AXES = 3
N_SAMPLES = 200
DT = 1e-4


@pytest.fixture
def plant() -> Plant:
    """A mildly damped, weakly coupled second-order plant."""
    a1 = np.eye(N_AXES) * 1.8
    a2 = np.eye(N_AXES) * -0.81
    b = np.eye(N_AXES) * 0.05
    c = np.zeros(N_AXES)
    return a1, a2, b, c


@pytest.fixture
def trajectory() -> Array:
    """A ramp into a plateau, mirroring the shape of the real demand."""
    ramp = np.linspace(0.0, 100.0, N_SAMPLES // 2)
    plateau = np.full(N_SAMPLES - N_SAMPLES // 2, 100.0)
    x = np.concatenate([ramp, plateau])
    return np.column_stack([x, np.zeros(N_SAMPLES), np.zeros(N_SAMPLES)])


class ZeroController:
    """Controller protocol implementation that always outputs zero."""

    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self, n_axes: int = 3, initial_setpoint: Array | None = None) -> None:
        self.reset_calls += 1
        self.n_axes = n_axes

    def update(self, setpoint: Array, pv: Array) -> Array:
        return np.zeros(self.n_axes)


def test_zero_control_reproduces_the_open_loop_plant(
    plant: Plant, trajectory: Array
) -> None:
    """With u = 0 the recursion must be exactly x[k+1] = a1 x[k] + a2 x[k-1] + c."""
    a1, a2, b, c = plant
    actual, _, control = simulate_pid(ZeroController(), trajectory, a1, a2, b, c)

    expected = np.zeros_like(actual)
    expected[0], expected[1] = trajectory[0], trajectory[1]
    for k in range(1, N_SAMPLES - 1):
        expected[k + 1] = a1 @ expected[k] + a2 @ expected[k - 1] + c

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(control, 0.0)


def test_controller_is_reset_before_the_run(plant: Plant, trajectory: Array) -> None:
    """State from a previous trajectory must not leak into the next."""
    a1, a2, b, c = plant
    controller = ZeroController()
    simulate_pid(controller, trajectory, a1, a2, b, c)
    simulate_pid(controller, trajectory, a1, a2, b, c)
    assert controller.reset_calls == 2


def test_repeated_runs_are_deterministic(plant: Plant, trajectory: Array) -> None:
    """The same PID re-run on the same trajectory must give identical output."""
    a1, a2, b, c = plant
    limit = np.full(N_AXES, 1000.0)
    pid = PID(
        kp=1.0, kv=0.01, ki=0.5, kd=0.0, kvff=0.1, dt=DT, u_min=-limit, u_max=limit
    )
    first = simulate_pid(pid, trajectory, a1, a2, b, c)
    second = simulate_pid(pid, trajectory, a1, a2, b, c)
    for a, b_ in zip(first, second, strict=True):
        np.testing.assert_allclose(a, b_)


def test_first_two_samples_are_seeded_from_the_demand(
    plant: Plant, trajectory: Array
) -> None:
    """The plant model needs two initial states, taken open-loop from the demand."""
    a1, a2, b, c = plant
    actual, _, _ = simulate_pid(ZeroController(), trajectory, a1, a2, b, c)
    np.testing.assert_allclose(actual[0], trajectory[0])
    np.testing.assert_allclose(actual[1], trajectory[1])


def test_final_sample_of_history_is_written(plant: Plant, trajectory: Array) -> None:
    """Regression: control_hist[-1] used to be an unwritten zero.

    It fed a spurious zero-voltage count into the control histogram.
    """
    a1, a2, b, c = plant
    limit = np.full(N_AXES, 1000.0)
    pid = PID(
        kp=5.0, kv=0.0, ki=0.0, kd=0.0, kvff=0.0, dt=DT, u_min=-limit, u_max=limit
    )
    _, error_hist, control_hist = simulate_pid(pid, trajectory, a1, a2, b, c)

    assert control_hist[-1].any(), "final control sample must be simulated"
    assert error_hist[-1].any(), "final error sample must be simulated"


def test_no_unwritten_gaps_in_control_history(plant: Plant, trajectory: Array) -> None:
    """Every sample from index 1 onward must come from an actual controller call.

    Index 0 is excluded by design: it is seeded open-loop, so no control exists.
    """
    a1, a2, b, c = plant
    limit = np.full(N_AXES, 1000.0)
    # A constant positional feedforward guarantees a non-zero output at every
    # step, so any exact zero can only be an unwritten slot.
    pid = PID(
        kp=0.0,
        kv=0.0,
        ki=0.0,
        kd=0.0,
        kvff=0.0,
        kpff1=1.0,
        dt=DT,
        u_min=-limit,
        u_max=limit,
    )
    traj = trajectory.copy()
    traj[:, 0] += 10.0  # keep the demand away from zero throughout
    _, _, control_hist = simulate_pid(pid, traj, a1, a2, b, c)

    unwritten = np.nonzero(~control_hist[1:].any(axis=1))[0]
    assert unwritten.size == 0, f"unwritten control samples at {unwritten + 1}"


def test_error_history_matches_demand_minus_position(
    plant: Plant, trajectory: Array
) -> None:
    a1, a2, b, c = plant
    limit = np.full(N_AXES, 1000.0)
    pid = PID(
        kp=2.0, kv=0.0, ki=0.0, kd=0.0, kvff=0.0, dt=DT, u_min=-limit, u_max=limit
    )
    actual, error_hist, _ = simulate_pid(pid, trajectory, a1, a2, b, c)
    # Interior samples: error[k] is measured against the position at step k.
    for k in (1, N_SAMPLES // 2, N_SAMPLES - 2):
        np.testing.assert_allclose(error_hist[k], trajectory[k] - actual[k])


def test_control_is_clamped_to_the_output_limits(
    plant: Plant, trajectory: Array
) -> None:
    a1, a2, b, c = plant
    limit = np.full(N_AXES, 5.0)
    pid = PID(
        kp=1e6, kv=0.0, ki=0.0, kd=0.0, kvff=0.0, dt=DT, u_min=-limit, u_max=limit
    )
    _, _, control_hist = simulate_pid(pid, trajectory, a1, a2, b, c)
    assert control_hist.max() <= 5.0
    assert control_hist.min() >= -5.0


def test_output_shapes_match_the_trajectory(plant: Plant, trajectory: Array) -> None:
    a1, a2, b, c = plant
    actual, error_hist, control_hist = simulate_pid(
        ZeroController(), trajectory, a1, a2, b, c
    )
    for arr in (actual, error_hist, control_hist):
        assert arr.shape == trajectory.shape


# ---------------------------------------------------------------------------
# Compiled simulator
# ---------------------------------------------------------------------------


def gains_vector(
    kp: float = 0.0,
    kv: float = 0.0,
    ki: float = 0.0,
    kd: float = 0.0,
    kvff: float = 0.0,
    kaff: float = 0.0,
    kpff1: float = 0.0,
    kpff0: float = 0.0,
    ktot: float = 1.0,
) -> Array:
    return np.array([kp, kv, ki, kd, kvff, kaff, kpff1, kpff0, ktot])


@pytest.mark.parametrize(
    "gains",
    [
        {"kp": 0.867, "kv": 2.0, "ki": 0.00785, "kvff": 4.725, "kaff": 214.8},
        {"kp": 1.7, "kv": 0.4, "ki": 1.2, "kd": 0.3, "kvff": 8.0, "kaff": 120.0},
        {"kp": 0.05, "kv": 1.9, "ki": 0.0, "kvff": 0.0, "kaff": 0.0, "kpff1": 0.004},
        {"kp": 2.0, "kv": 2.0, "ki": 2.0, "kvff": 10.0, "kaff": 300.0, "ktot": 0.5},
    ],
)
def test_fast_simulator_matches_the_reference_exactly(
    plant: Plant, trajectory: Array, gains: dict[str, float]
) -> None:
    """simulate_pid_fast must stay bit-identical to PID + simulate_pid.

    PID is the readable reference and the thing the VHDL tests pin; the compiled
    path is a ~40x speed-up of the same algorithm, so any divergence between
    them is a silent correctness bug in the BO inner loop.
    """
    a1, a2, b, c = plant
    limit = np.full(N_AXES, 500.0)
    pid = PID(
        kp=gains.get("kp", 0.0),
        kv=gains.get("kv", 0.0),
        ki=gains.get("ki", 0.0),
        kd=gains.get("kd", 0.0),
        kvff=gains.get("kvff", 0.0),
        kaff=gains.get("kaff", 0.0),
        kpff1=gains.get("kpff1", 0.0),
        kpff0=gains.get("kpff0", 0.0),
        ktot=gains.get("ktot", 1.0),
        dt=1.0,
        dt_inv=1.0,
        u_min=-limit,
        u_max=limit,
        integral_limit=1e4,
    )
    ref = simulate_pid(pid, trajectory, a1, a2, b, c)
    fast = simulate_pid_fast(
        gains_vector(**gains), trajectory, a1, a2, b, c, 1.0, 1.0, -limit, limit, 1e4
    )
    for name, r, f in zip(("positions", "errors", "control"), ref, fast, strict=True):
        np.testing.assert_array_equal(r, f, err_msg=f"{name} differ")


def test_fast_simulator_honours_dt_and_dt_inv_independently(
    plant: Plant, trajectory: Array
) -> None:
    a1, a2, b, c = plant
    limit = np.full(N_AXES, 500.0)
    pid = PID(
        kp=0.0,
        kv=1.0,
        ki=1.0,
        kd=0.0,
        kvff=0.0,
        dt=0.25,
        dt_inv=8.0,
        u_min=-limit,
        u_max=limit,
    )
    ref = simulate_pid(pid, trajectory, a1, a2, b, c)
    fast = simulate_pid_fast(
        gains_vector(kv=1.0, ki=1.0),
        trajectory,
        a1,
        a2,
        b,
        c,
        0.25,
        8.0,
        -limit,
        limit,
        np.inf,
    )
    np.testing.assert_array_equal(ref[0], fast[0])
