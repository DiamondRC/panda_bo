from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, overload

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    _F = TypeVar("_F", bound=Callable[..., Any])

    # numba ships no type information, so a bare import erases the signature of
    # everything it decorates. Declaring it as signature-preserving keeps the
    # compiled simulators typed. Covers both @njit and @njit(cache=True).
    @overload
    def njit(func: _F) -> _F: ...

    @overload
    def njit(*, cache: bool = False) -> Callable[[_F], _F]: ...

    def njit(func: _F | None = None, *, cache: bool = False) -> _F | Callable[[_F], _F]:
        raise NotImplementedError

else:
    from numba import njit

Array = NDArray[np.float64]


class Controller(Protocol):
    """What simulate_pid needs from a controller. `PID` satisfies this."""

    def reset(self, n_axes: int = ..., initial_setpoint: Any = ...) -> None: ...

    def update(self, setpoint: Array, pv: Array) -> Array: ...


# ============================================================
# PID simulation: given proposed parameters,
# simulate control for the whole trajectory brett's PID version
# ============================================================
@njit
def simulate_pid_vhdl_like_numba(
    params: Array,
    trajectory: Array,
    a1: Array,
    a2: Array,
    b: Array,
    c: Array,
    dt: float,
    u_min: Array,
    u_max: Array,
    max_integral: float,
) -> tuple[Array, Array, Array]:
    """
    params = [ktot, kp, kv, ki, kd, kvff, kaff, kpff0, kpff1]
    """

    ktot = params[0]
    kp = params[1]
    kv = params[2]
    ki = params[3]
    kd = params[4]
    kvff = params[5]
    kaff = params[6]
    kpff0 = params[7]
    kpff1 = params[8]

    t = trajectory.shape[0]

    actual_pos = np.zeros((t, 3))
    control_hist = np.zeros((t, 3))
    error_hist = np.zeros((t, 3))

    x0 = trajectory[0].copy()
    x1 = trajectory[1].copy()

    actual_pos[0] = x0
    actual_pos[1] = x1

    integral = np.zeros(3)
    prev_error = trajectory[1] - x1
    prev_setpoint = trajectory[0].copy()
    prev_des_vel = np.zeros(3)

    dt_inv = 1.0 / dt

    for k in range(1, t - 1):
        setpoint = trajectory[k]
        error = setpoint - x1

        # Actual velocity
        act_vel = (x1 - x0) * dt_inv

        # Desired velocity and acceleration-like feedforward
        des_vel = (setpoint - prev_setpoint) * dt_inv
        des_acc_like = des_vel - prev_des_vel

        # Integral term
        integral += ki * dt * error

        for j in range(3):
            if integral[j] > max_integral:
                integral[j] = max_integral
            elif integral[j] < -max_integral:
                integral[j] = -max_integral

        # Derivative of error
        d_error = (error - prev_error) * dt_inv

        # Positional feedforward
        pos_ff = kpff1 * setpoint + kpff0 * np.abs(setpoint) * setpoint

        u = (
            kp * error
            - kv * act_vel
            + integral
            + kd * d_error
            + kvff * des_vel
            + kaff * des_acc_like
            + pos_ff
        )

        u = ktot * u

        # Output clamp
        for j in range(3):
            if u[j] > u_max[j]:
                u[j] = u_max[j]
            elif u[j] < u_min[j]:
                u[j] = u_min[j]

        x_next = a1 @ x1 + a2 @ x0 + b @ u + c

        actual_pos[k + 1] = x_next
        control_hist[k] = u
        error_hist[k] = error

        x0 = x1.copy()
        x1 = x_next.copy()

        prev_error = error.copy()
        prev_setpoint = setpoint.copy()
        prev_des_vel = des_vel.copy()

    return actual_pos, error_hist, control_hist


# ============================================================
# PID simulation: given proposed parameters, simulate control for the whole trajectory
# Standard PID version
# ============================================================


@njit
def simulate_pid_tracking_linear_numba(
    pid_params: Array,
    trajectory: Array,
    a_1: Array,
    a_2: Array,
    b: Array,
    c: Array,
    dt: float,
    u_min: Array,
    u_max: Array,
    integral_limit: float,
) -> tuple[Array, Array, Array]:
    t = trajectory.shape[0]

    kp = np.empty(3)
    ki = np.empty(3)
    kd = np.empty(3)

    if pid_params.size == 3:
        for i in range(3):
            kp[i] = pid_params[0]
            ki[i] = pid_params[1]
            kd[i] = pid_params[2]
    else:
        kp[0], ki[0], kd[0] = pid_params[0], pid_params[1], pid_params[2]
        kp[1], ki[1], kd[1] = pid_params[3], pid_params[4], pid_params[5]
        kp[2], ki[2], kd[2] = pid_params[6], pid_params[7], pid_params[8]

    actual_pos = np.zeros((t, 3))
    error_hist = np.zeros((t, 3))
    control_hist = np.zeros((t, 3))

    x0 = trajectory[0].copy()
    x1 = trajectory[1].copy()

    actual_pos[0] = x0
    actual_pos[1] = x1

    integ = np.zeros(3)
    prev_error = trajectory[1] - x1

    for k in range(1, t - 1):
        error = trajectory[k] - x1

        for j in range(3):
            integ[j] += error[j] * dt
            if integ[j] > integral_limit:
                integ[j] = integral_limit
            elif integ[j] < -integral_limit:
                integ[j] = -integral_limit

        deriv = (error - prev_error) / dt

        u = kp * error + ki * integ + kd * deriv

        for j in range(3):
            if u[j] > u_max[j]:
                u[j] = u_max[j]
            elif u[j] < u_min[j]:
                u[j] = u_min[j]

        error_hist[k] = error
        control_hist[k] = u

        x_next = a_1 @ x1 + a_2 @ x0 + b @ u + c

        x0 = x1.copy()
        x1 = x_next.copy()

        actual_pos[k + 1] = x1
        prev_error = error.copy()

    return actual_pos, error_hist, control_hist


# Deliberately NOT cache=True. Persisting compiled code to __pycache__ saves
# ~1-2 s of compile time per process, but a stale cache hit silently runs
# superseded code — which is not a trade worth making in the BO inner loop.
@njit
def simulate_pid_fast(
    gains: Array,
    trajectory: Array,
    a1: Array,
    a2: Array,
    b: Array,
    c: Array,
    dt: float,
    dt_inv: float,
    u_min: Array,
    u_max: Array,
    integral_limit: float,
    integral_gate: int = 0,
) -> tuple[Array, Array, Array]:
    """Compiled equivalent of PID + simulate_pid, for the BO inner loop.

    gains = [kp, kv, ki, kd, kvff, kaff, kpff1, kpff0, ktot].

    integral_gate mirrors `PID.integral_gate`, as an int so numba can specialise
    on it: 0 = "vhdl" (frozen while the demand is stationary, what the device
    does), 1 = "pmac" (integrate unless saturated), 2 = "pmac_zv" (integrate
    only while stationary). See PID for why this distinction is load-bearing.

    The interpreted version spends ~98% of an evaluation in per-sample numpy
    calls on length-3 arrays; compiling the loop removes that overhead and runs
    ~33x faster, bit-for-bit identical. `PID` remains the readable reference and
    the thing the VHDL tests check — this must track it.
    """
    kp, kv, ki, kd = gains[0], gains[1], gains[2], gains[3]
    kvff, kaff, kpff1, kpff0, ktot = gains[4], gains[5], gains[6], gains[7], gains[8]

    t, n = trajectory.shape
    actual_pos = np.zeros((t, n))
    error_hist = np.zeros((t, n))
    control_hist = np.zeros((t, n))

    x0 = trajectory[0].copy()
    x1 = trajectory[1].copy()
    actual_pos[0] = x0
    actual_pos[1] = x1

    prev_pv = np.zeros(n)
    prev_error = np.zeros(n)
    prev_setpoint = trajectory[0].copy()
    prev_vel_desired = np.zeros(n)
    integral = np.zeros(n)
    prev_sum = np.zeros(n)

    u = np.zeros(n)
    # Runs to t-1 rather than t-2 so the final sample of the histories holds a
    # real control value; the plant is only stepped while there is a slot to
    # write it into. Matches simulate_pid's trailing controller call.
    for k in range(1, t):
        setpoint = trajectory[k]
        for j in range(n):
            error = setpoint[j] - x1[j]
            velocity = (x1[j] - prev_pv[j]) * dt_inv
            vel_desired = (setpoint[j] - prev_setpoint[j]) * dt_inv
            derivative = (error - prev_error[j]) * dt_inv

            # VHDL STAGE_4/5 gate, per axis. See the docstring for the modes.
            if integral_gate == 0:
                moving_ok = vel_desired != 0.0
            elif integral_gate == 2:
                moving_ok = vel_desired == 0.0
            else:
                moving_ok = True
            if prev_sum[j] < u_max[j] and moving_ok:
                integral[j] += ki * dt * error
                if integral[j] > integral_limit:
                    integral[j] = integral_limit
                elif integral[j] < -integral_limit:
                    integral[j] = -integral_limit

            ff = (
                kvff * vel_desired
                + kaff * (vel_desired - prev_vel_desired[j])
                + kpff1 * setpoint[j]
                + kpff0 * abs(setpoint[j]) * setpoint[j]
            )
            sum_int = kp * error - kv * velocity + integral[j] + kd * derivative + ff
            scaled = ktot * sum_int
            if scaled > u_max[j]:
                u[j] = u_max[j]
            elif scaled < u_min[j]:
                u[j] = u_min[j]
            else:
                u[j] = scaled

            prev_sum[j] = sum_int
            prev_pv[j] = x1[j]
            prev_error[j] = error
            prev_vel_desired[j] = vel_desired
            error_hist[k, j] = error

        control_hist[k] = u
        prev_setpoint = setpoint.copy()
        if k < t - 1:
            x_next = a1 @ x1 + a2 @ x0 + b @ u + c
            actual_pos[k + 1] = x_next
            x0 = x1.copy()
            x1 = x_next.copy()

    # Sample 0 is seeded open-loop from the demand, so its error is genuinely
    # zero and no control action exists for it.
    error_hist[0] = trajectory[0] - actual_pos[0]

    return actual_pos, error_hist, control_hist


def simulate_pid(
    controller: Controller,
    trajectory: Array,
    a1: Array,
    a2: Array,
    b: Array,
    c: Array,
) -> tuple[Array, Array, Array]:
    """Simulate a trajectory using any controller with reset() and update() methods.

    the controller protocol: reset(n_axes, initial_setpoint) then
    update(setpoint, pv) -> np.ndarray of shape (n_axes,).
    """
    t, n = trajectory.shape
    actual_pos = np.zeros((t, n))
    error_hist = np.zeros((t, n))
    control_hist = np.zeros((t, n))

    x0, x1 = trajectory[0].copy(), trajectory[1].copy()
    actual_pos[0], actual_pos[1] = x0, x1

    controller.reset(n_axes=n, initial_setpoint=trajectory[0])

    for k in range(1, t - 1):
        u = controller.update(trajectory[k], x1)
        x_next = a1 @ x1 + a2 @ x0 + b @ u + c
        actual_pos[k + 1] = x_next
        control_hist[k] = u
        error_hist[k] = trajectory[k] - x1
        x0, x1 = x1.copy(), x_next.copy()

    # The loop stops one short of the end because it writes actual_pos[k + 1].
    # Run the controller once more (without stepping the plant) so the final
    # sample holds a real control value rather than an unsimulated zero, which
    # would otherwise show up as a spurious zero-voltage count downstream.
    error_hist[t - 1] = trajectory[t - 1] - x1
    control_hist[t - 1] = controller.update(trajectory[t - 1], x1)

    # actual_pos[0:2] is seeded open-loop from the demand because the plant model
    # needs two initial states, so sample 0 has a genuine zero error and no
    # control action at all. control_hist[0] stays zero and is not meaningful.
    error_hist[0] = trajectory[0] - actual_pos[0]

    return actual_pos, error_hist, control_hist
