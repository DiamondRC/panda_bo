import numpy as np
from numba import njit


# ============================================================
# PID simulation: given proposed parameters,
# simulate control for the whole trajectory brett's PID version
# ============================================================
@njit
def simulate_pid_vhdl_like_numba(
    params,
    trajectory,
    a1,
    a2,
    b,
    c,
    dt,
    u_min,
    u_max,
    max_integral,
):
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
    pid_params,
    trajectory,
    a_1,
    a_2,
    b,
    c,
    dt,
    u_min,
    u_max,
    integral_limit,
):
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


def simulate_pid(controller, trajectory, a1, a2, b, c):
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

    return actual_pos, error_hist, control_hist
