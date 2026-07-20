import numpy as np


class PID:
    """Vectorized PID controller matching the brett_pid algorithm.

    Works on numpy arrays of shape (n_axes,). Call reset() before each
    trajectory run, then update() once per timestep.

    Any class with reset(n_axes) and update(setpoint, pv) -> ndarray
    satisfies the controller protocol expected by simulate_pid().
    """

    def __init__(
        self,
        kp: float,
        kv: float,
        ki: float,
        kd: float,
        kvff: float,
        kaff: float = 0.0,
        kpff1: float = 0.0,
        kpff0: float = 0.0,
        dt: float = 1e-4,
        pv_scale: float = 1.0,
        u_min: np.ndarray | None = None,
        u_max: np.ndarray | None = None,
        integral_limit: float | None = None,
    ) -> None:
        self.kp = kp
        self.kv = kv
        self.ki = ki
        self.kd = kd
        self.kvff = kvff
        self.kaff = kaff
        self.kpff1 = kpff1
        self.kpff0 = kpff0
        self.inv_dt = 1.0 / dt
        self.pv_scale = pv_scale
        self.u_min = u_min
        self.u_max = u_max
        self.integral_limit = integral_limit

    def reset(
        self,
        n_axes: int = 3,
        initial_setpoint: np.ndarray | None = None,
    ) -> None:
        """Zero all controller state. Call once before each trajectory run."""
        self.prev_pv = np.zeros(n_axes)
        self.prev_error = np.zeros(n_axes)
        self.prev_setpoint = (
            initial_setpoint.copy()
            if initial_setpoint is not None
            else np.zeros(n_axes)
        )
        self.prev_vel_desired = np.zeros(n_axes)
        self.integral = np.zeros(n_axes, dtype=np.float64)
        self._saturated = False

    def update(self, setpoint: np.ndarray, pv: np.ndarray) -> np.ndarray:
        """Compute one control step. Both inputs are shape (n_axes,)."""
        setpoint = np.asarray(setpoint, dtype=np.float64)
        pv_scaled = np.asarray(pv, dtype=np.float64) * self.pv_scale

        error = setpoint - pv_scaled
        velocity = pv_scaled - self.prev_pv  # per-sample, no dt
        vel_desired = setpoint - self.prev_setpoint  # per-sample, no dt
        derivative = (error - self.prev_error) * self.inv_dt  # per-second

        # Anti-windup: only integrate when not at positive saturation
        # and the setpoint is actually moving (matches brett_pid gating logic)
        if not self._saturated and np.any(vel_desired != 0):
            self.integral += self.ki * error
            if self.integral_limit is not None:
                np.clip(
                    self.integral,
                    -self.integral_limit,
                    self.integral_limit,
                    out=self.integral,
                )

        ff = (
            self.kvff * vel_desired
            + self.kaff * (vel_desired - self.prev_vel_desired)
            + self.kpff1 * setpoint
            + self.kpff0 * np.abs(setpoint) * setpoint
        )

        u = (
            self.kp * error
            - self.kv * velocity
            + self.integral
            + self.kd * derivative
            + ff
        )

        if self.u_min is not None and self.u_max is not None:
            np.clip(u, self.u_min, self.u_max, out=u)
            # Mirror brett_pid: flag saturation only on positive limit
            self._saturated = bool(np.any(u >= self.u_max))

        self.prev_pv = pv_scaled.copy()
        self.prev_error = error.copy()
        self.prev_vel_desired = vel_desired.copy()
        self.prev_setpoint = setpoint.copy()
        return u.copy()
