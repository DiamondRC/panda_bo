import numpy as np


class PID:
    """
    Vectorised PID + feedforward controller modelling the brett_pid FPGA block.

    This is a floating-point model of Brett's PID.
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
        ktot: float = 1.0,
        dt: float = 1.0,
        dt_inv: float = 1.0,
        pv_scale: float = 1.0,
        u_min: np.ndarray | None = None,
        u_max: np.ndarray | None = None,
        integral_limit: float | None = None,
        dir_toggle: bool = False,
        integral_gate: str = "vhdl",
    ) -> None:
        """
        ktot: the VHDL's k_tot output gain. The device holds this at 1, so it is
            not tuned; it exists only so the model does not silently diverge if
            the device is ever configured otherwise.
        dt, dt_inv: the VHDL's dt_i and dt_inv_i input ports. These are
            INDEPENDENT registers, not reciprocals of each other and not derived
            from pid_period_i — the servo rate is set separately by
            pid_period_i. dt scales the integral term (ki * dt * error) and
            dt_inv scales the velocity, desired-velocity and derivative terms.

            Both default to 1.0, which makes this reduce exactly to the PMAC C
            reference (PIDpositionalFF), where the integral has no dt and the
            velocities are per servo cycle. Set them only to whatever the device
            actually holds. Deriving dt_inv as 1/dt rescales kv, kvff, kaff and
            ki by four orders of magnitude and makes tuned gains meaningless on
            hardware.
        pv_scale: measurement scaling, the VHDL's PV_SCALE (0.256 to convert
            256 pm encoder counts to nm). Leave at 1.0 when the plant model was
            identified in the same units as the demand.
        dir_toggle: the VHDL's dir_toggle_i, which negates the output. Matching
            the VHDL, it is applied only when the output is not clamped.
        integral_gate: when the integrator is allowed to accumulate. This is the
            one place brett_pid_ff.vhd and the PMAC C reference genuinely
            disagree, and the disagreement is exactly inverted while holding
            position.

            "vhdl"  (default, matches the device) — STAGE_4/5:
                    `sum_int < max_output_i and v_des_cal /= 0`.
                    The integrator is FROZEN whenever the demand is stationary.
            "pmac"  — the C's `ctrl_out < MaxDac && (SwZvInt != 1 ||
                    DesVelZero == 1)` with SwZvInt = 0: integrate whenever the
                    output is not saturated, including while holding.
            "pmac_zv" — the same C expression with SwZvInt = 1: integrate ONLY
                    while the demand is stationary. Note that both C settings
                    integrate during a hold; only the VHDL does not.

            This matters for a measured symptom: holding at a constant offset
            from the setpoint is what a frozen integrator predicts, because the
            residual is then held by the proportional term alone (kp = 0.004 on
            this stage, so the residual is large). It is not what thermal drift
            predicts, which would appear as slow movement rather than a fixed
            offset and would be nulled by a running integrator.
        """
        if integral_gate not in ("vhdl", "pmac", "pmac_zv"):
            raise ValueError(
                f"integral_gate must be 'vhdl', 'pmac' or 'pmac_zv', "
                f"got {integral_gate!r}"
            )
        self.kp = kp
        self.kv = kv
        self.ki = ki
        self.kd = kd
        self.kvff = kvff
        self.kaff = kaff
        self.kpff1 = kpff1
        self.kpff0 = kpff0
        self.ktot = ktot
        self.dt = dt
        self.dt_inv = dt_inv
        self.pv_scale = pv_scale
        self.u_min = u_min
        self.u_max = u_max
        self.integral_limit = integral_limit
        self.dir_toggle = dir_toggle
        self.integral_gate = integral_gate

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
        # VHDL resets sum_int to 0 on init, so the limit gate starts open.
        self.prev_sum = np.zeros(n_axes, dtype=np.float64)

    def update(self, setpoint: np.ndarray, pv: np.ndarray) -> np.ndarray:
        """Compute one control step. Both inputs are shape (n_axes,)."""
        setpoint = np.asarray(setpoint, dtype=np.float64)
        pv_scaled = np.asarray(pv, dtype=np.float64) * self.pv_scale

        error = setpoint - pv_scaled
        velocity = (pv_scaled - self.prev_pv) * self.dt_inv
        vel_desired = (setpoint - self.prev_setpoint) * self.dt_inv
        derivative = (error - self.prev_error) * self.dt_inv

        # Anti-windup limit, common to every gate mode: the previous tick's
        # unscaled sum must be below the (positive) output limit.
        if self.u_max is not None:
            below_limit = self.prev_sum < self.u_max
        else:
            below_limit = np.ones_like(error, dtype=bool)

        # Per-axis, because the FPGA instantiates one brett_pid per axis.
        if self.integral_gate == "vhdl":
            gate = below_limit & (vel_desired != 0.0)
        elif self.integral_gate == "pmac_zv":
            gate = below_limit & (vel_desired == 0.0)
        else:  # "pmac"
            gate = below_limit

        integral = self.integral + self.ki * self.dt * error
        if self.integral_limit is not None:
            integral = np.clip(integral, -self.integral_limit, self.integral_limit)
        self.integral = np.where(gate, integral, self.integral)

        ff = (
            self.kvff * vel_desired
            + self.kaff * (vel_desired - self.prev_vel_desired)
            + self.kpff1 * setpoint
            + self.kpff0 * np.abs(setpoint) * setpoint
        )

        # VHDL STAGE_6/7: sum_int is the pre-k_tot sum, and is what the next
        # tick's integral gate compares against max_output_i.
        sum_int = (
            self.kp * error
            - self.kv * velocity
            + self.integral
            + self.kd * derivative
            + ff
        )

        u = self.ktot * sum_int

        if self.u_min is not None and self.u_max is not None:
            # VHDL DONE stage: dir_toggle is inside the else branch, so a clamped
            # output keeps the limit's sign rather than being negated.
            toggled = -u if self.dir_toggle else u
            u = np.where(
                u > self.u_max,
                self.u_max,
                np.where(u < self.u_min, self.u_min, toggled),
            )
        elif self.dir_toggle:
            u = -u

        self.prev_sum = np.asarray(sum_int, dtype=np.float64).copy()
        self.prev_pv = pv_scaled.copy()
        self.prev_error = error.copy()
        self.prev_vel_desired = vel_desired.copy()
        self.prev_setpoint = setpoint.copy()
        return np.asarray(u, dtype=np.float64).copy()
