"""
Tests for the PID controller.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from bayesian_pid.controller import PID

Array = NDArray[np.float64]

DT = 0.5  # VHDL dt_i     (independent register)
DT_INV = 8.0  # VHDL dt_inv_i (independent register, NOT 1/dt)
N_AXES = 3


Gains = dict[str, float]


def make_pid(**gains: float) -> PID:
    """A PID with every gain zeroed except those named, and no limits."""
    base: Gains = {
        "kp": 0.0,
        "kv": 0.0,
        "ki": 0.0,
        "kd": 0.0,
        "kvff": 0.0,
        "kaff": 0.0,
        "kpff1": 0.0,
        "kpff0": 0.0,
    }
    base.update(gains)
    pid = make_pid_from(base)
    pid.reset(n_axes=N_AXES)
    return pid


def make_pid_from(
    gains: Gains,
    *,
    ktot: float = 1.0,
    u_min: Array | None = None,
    u_max: Array | None = None,
    integral_limit: float | None = None,
    dir_toggle: bool = False,
) -> PID:
    """Build a PID from a name->value mapping without **-unpacking.

    Explicit keywords keep the call checkable; ** unpacking of a dict[str, float]
    hides which parameter each value lands on.
    """
    return PID(
        kp=gains.get("kp", 0.0),
        kv=gains.get("kv", 0.0),
        ki=gains.get("ki", 0.0),
        kd=gains.get("kd", 0.0),
        kvff=gains.get("kvff", 0.0),
        kaff=gains.get("kaff", 0.0),
        kpff1=gains.get("kpff1", 0.0),
        kpff0=gains.get("kpff0", 0.0),
        ktot=ktot,
        dt=DT,
        dt_inv=DT_INV,
        u_min=u_min,
        u_max=u_max,
        integral_limit=integral_limit,
        dir_toggle=dir_toggle,
    )


def vec(*values: float) -> Array:
    return np.array(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# Individual terms: the dt scaling of each, taken from the VHDL
# ---------------------------------------------------------------------------


def test_proportional_term_has_no_dt_scaling() -> None:
    """VHDL: p_mul = kp_i * pos_err. No dt anywhere."""
    pid = make_pid(kp=2.0)
    u = pid.update(vec(10.0, 0.0, 0.0), vec(4.0, 0.0, 0.0))
    assert u[0] == pytest.approx(2.0 * 6.0)


def test_velocity_term_scales_with_inv_dt() -> None:
    """VHDL: vel = (pos_store - prev_pos) * dt_inv, summed as -kv * vel."""
    pid = make_pid(kv=3.0)
    pid.update(vec(0.0, 0.0, 0.0), vec(1.0, 0.0, 0.0))  # establish prev_pv = 1.0
    u = pid.update(vec(0.0, 0.0, 0.0), vec(5.0, 0.0, 0.0))
    assert u[0] == pytest.approx(-3.0 * (5.0 - 1.0) * DT_INV)


def test_desired_velocity_feedforward_scales_with_inv_dt() -> None:
    """VHDL: v_des_cal = (set_store - prev_set) * dt_inv, then kvff * v_des_cal."""
    pid = make_pid(kvff=0.5)
    pid.update(vec(2.0, 0.0, 0.0), vec(2.0, 0.0, 0.0))  # prev_setpoint = 2.0
    u = pid.update(vec(7.0, 0.0, 0.0), vec(7.0, 0.0, 0.0))
    assert u[0] == pytest.approx(0.5 * (7.0 - 2.0) * DT_INV)


def test_acceleration_feedforward_uses_desired_velocity_difference() -> None:
    """VHDL: a_des_sub = v_des_cal - prev_v_des, then kaff * a_des_sub."""
    pid = make_pid(kaff=2.0)
    pid.update(vec(1.0, 0.0, 0.0), vec(1.0, 0.0, 0.0))  # v_des = 1/dt
    u = pid.update(vec(4.0, 0.0, 0.0), vec(4.0, 0.0, 0.0))  # v_des = 3/dt
    expected_prev_v_des = 1.0 * DT_INV
    expected_v_des = 3.0 * DT_INV
    assert u[0] == pytest.approx(2.0 * (expected_v_des - expected_prev_v_des))


def test_integral_accumulates_ki_times_dt_times_error() -> None:
    """VHDL: i_mul_dt = ki_i * dt_i, then i_mul_err = pos_err * i_mul_dt."""
    pid = make_pid(ki=4.0)
    # Setpoint must move for the VHDL gate to open.
    u = pid.update(vec(10.0, 0.0, 0.0), vec(0.0, 0.0, 0.0))
    assert u[0] == pytest.approx(4.0 * DT * 10.0)


def test_derivative_scales_with_inv_dt() -> None:
    """VHDL: d_mul_dt = kd_i * dt_inv, then d_mul_err = d_mul_dt * d_err."""
    pid = make_pid(kd=1.5)
    pid.update(vec(2.0, 0.0, 0.0), vec(0.0, 0.0, 0.0))  # error 2.0
    u = pid.update(vec(9.0, 0.0, 0.0), vec(0.0, 0.0, 0.0))  # error 9.0
    assert u[0] == pytest.approx(1.5 * (9.0 - 2.0) * DT_INV)


def test_dt_and_dt_inv_are_independent_registers() -> None:
    """Regression: dt_inv must never be derived as 1/dt.

    brett_pid_ff.vhd takes dt_i and dt_inv_i as separate input ports, and the
    servo rate is set elsewhere by pid_period_i. Tying them together rescales
    kv/kvff/kaff against ki by orders of magnitude, so gains tuned in
    simulation stop meaning anything on the device.
    """
    pid = PID(kp=0.0, kv=1.0, ki=1.0, kd=0.0, kvff=0.0, dt=0.25, dt_inv=100.0)
    pid.reset(n_axes=1)
    pid.update(vec(0.0), vec(1.0))  # prev_pv = 1.0, no integration (demand still)
    u = pid.update(vec(0.0), vec(3.0))

    # velocity term uses dt_inv only; integral is gated off by a still demand.
    assert u[0] == pytest.approx(-1.0 * (3.0 - 1.0) * 100.0)
    assert pid.integral[0] == 0.0


def test_defaults_reduce_to_the_pmac_c_reference() -> None:
    """With dt = dt_inv = 1 the algorithm is the C: no dt anywhere.

    The C does `Integrator += Ki * PosError` and treats ActVel/DesVel as
    per-servo-cycle deltas.
    """
    pid = PID(kp=0.0, kv=0.0, ki=3.0, kd=0.0, kvff=0.0)
    pid.reset(n_axes=1)
    assert pid.dt == 1.0
    assert pid.dt_inv == 1.0
    # Moving demand opens the gate; increment is exactly Ki * PosError.
    u = pid.update(vec(10.0), vec(0.0))
    assert u[0] == pytest.approx(3.0 * 10.0)


def test_positional_feedforward_is_linear_plus_signed_square() -> None:
    """VHDL: kpff1 * set_store + kpff0 * abs(set_store) * set_store."""
    pid = make_pid(kpff1=0.5, kpff0=0.25)
    u = pid.update(vec(-4.0, 0.0, 0.0), vec(-4.0, 0.0, 0.0))
    assert u[0] == pytest.approx(0.5 * -4.0 + 0.25 * 4.0 * -4.0)


def test_pv_scale_applied_to_measurement() -> None:
    """VHDL: ri_nm = real_input * PV_SCALE before the error is formed."""
    pid = PID(kp=1.0, kv=0.0, ki=0.0, kd=0.0, kvff=0.0, dt=DT, pv_scale=0.256)
    pid.reset(n_axes=N_AXES)
    u = pid.update(vec(0.0, 0.0, 0.0), vec(1000.0, 0.0, 0.0))
    assert u[0] == pytest.approx(-0.256 * 1000.0)


# ---------------------------------------------------------------------------
# k_tot: held at 1 on the device, but must be modelled in the right place
# ---------------------------------------------------------------------------


def test_ktot_defaults_to_one() -> None:
    """The device holds k_tot at 1, so the default must be a no-op."""
    assert PID(kp=1.0, kv=0.0, ki=0.0, kd=0.0, kvff=0.0).ktot == 1.0


def test_ktot_scales_the_whole_sum() -> None:
    """VHDL STAGE_8: sca_mul = k_tot_i * sum_int, i.e. after every term is summed."""
    gains = {"kp": 2.0, "kvff": 0.5, "kpff1": 0.25}
    plain = make_pid(**gains)
    scaled = make_pid_from(gains, ktot=3.0)
    scaled.reset(n_axes=N_AXES)
    sp, pv = vec(5.0, 1.0, 0.0), vec(2.0, 0.0, 0.0)
    assert scaled.update(sp, pv)[0] == pytest.approx(3.0 * plain.update(sp, pv)[0])


def test_ktot_applied_before_output_clamp() -> None:
    """VHDL clamps scale_out (post-k_tot), not sum_int."""
    u_max = vec(10.0, 10.0, 10.0)
    pid = PID(
        kp=1.0,
        kv=0.0,
        ki=0.0,
        kd=0.0,
        kvff=0.0,
        dt=DT,
        dt_inv=DT_INV,
        ktot=5.0,
        u_min=-u_max,
        u_max=u_max,
    )
    pid.reset(n_axes=N_AXES)
    # sum_int = 4 is under the limit, but k_tot * 4 = 20 is over it.
    u = pid.update(vec(4.0, 0.0, 0.0), vec(0.0, 0.0, 0.0))
    assert u[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# The integral gate — the part the VHDL and the PMAC C disagree on
# ---------------------------------------------------------------------------


def test_integral_frozen_while_demand_is_stationary() -> None:
    """VHDL gate is `v_des_cal /= 0`: no integration while the demand is still.

    Across the plateau the objective scores, this means ki cannot provide
    closed-loop correction. The value accumulated during the ramp does persist
    as a constant offset, so ki still influences the score.
    """
    pid = make_pid(ki=1.0)
    pid.update(vec(5.0, 0.0, 0.0), vec(0.0, 0.0, 0.0))  # moving: integrates
    frozen = pid.integral.copy()
    for _ in range(50):
        pid.update(vec(5.0, 0.0, 0.0), vec(0.0, 0.0, 0.0))  # stationary demand
    np.testing.assert_allclose(pid.integral, frozen)


def test_integral_accumulates_while_demand_is_moving() -> None:
    pid = make_pid(ki=1.0)
    for step in range(1, 5):
        pid.update(vec(float(step), 0.0, 0.0), vec(0.0, 0.0, 0.0))
    # error equals the setpoint each step, so the sum is ki*dt*(1+2+3+4)
    assert pid.integral[0] == pytest.approx(1.0 * DT * 10.0)


def test_integral_gate_is_per_axis() -> None:
    """The FPGA instantiates one brett_pid per axis, so gates must not couple.

    Axis 0's demand moves while axis 1's is stationary; axis 1 must stay frozen.
    """
    pid = make_pid(ki=1.0)
    pid.update(vec(1.0, 7.0, 0.0), vec(0.0, 0.0, 0.0))  # both axes move once
    after_first = pid.integral.copy()
    for step in range(2, 20):
        # only axis 0 keeps moving; axis 1 holds at 7.0
        pid.update(vec(float(step), 7.0, 0.0), vec(0.0, 0.0, 0.0))
    assert pid.integral[0] > after_first[0]
    assert pid.integral[1] == pytest.approx(after_first[1])


def test_saturation_gate_is_one_sided_and_per_axis() -> None:
    """VHDL: `sum_int < max_output_i` only — the negative rail is not gated."""
    u_max = vec(10.0, 10.0, 10.0)
    pid = make_pid_from({"kp": 1.0, "ki": 1.0}, u_min=-u_max, u_max=u_max)
    pid.reset(n_axes=N_AXES)
    # Axis 0 drives sum_int above +u_max; axis 1 drives it below -u_max.
    pid.update(vec(100.0, -100.0, 0.0), vec(0.0, 0.0, 0.0))
    frozen_hi, frozen_lo = pid.integral[0], pid.integral[1]
    pid.update(vec(200.0, -200.0, 0.0), vec(0.0, 0.0, 0.0))
    assert pid.integral[0] == pytest.approx(frozen_hi), "positive rail must gate"
    assert pid.integral[1] != pytest.approx(frozen_lo), "negative rail is not gated"


def test_integral_clamped_to_limit() -> None:
    pid = PID(kp=0.0, kv=0.0, ki=1e6, kd=0.0, kvff=0.0, dt=DT, integral_limit=2.0)
    pid.reset(n_axes=N_AXES)
    for step in range(1, 20):
        pid.update(vec(float(step), 0.0, 0.0), vec(0.0, 0.0, 0.0))
    assert pid.integral[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Output stage
# ---------------------------------------------------------------------------


def test_output_clamped_to_limits() -> None:
    u_max = vec(10.0, 10.0, 10.0)
    pid = PID(
        kp=1.0,
        kv=0.0,
        ki=0.0,
        kd=0.0,
        kvff=0.0,
        dt=DT,
        dt_inv=DT_INV,
        u_min=-u_max,
        u_max=u_max,
    )
    pid.reset(n_axes=N_AXES)
    u = pid.update(vec(500.0, -500.0, 0.0), vec(0.0, 0.0, 0.0))
    assert u[0] == pytest.approx(10.0)
    assert u[1] == pytest.approx(-10.0)


def test_dir_toggle_negates_only_the_unclamped_output() -> None:
    """VHDL applies dir_toggle inside the else branch of the clamp."""
    u_max = vec(10.0, 10.0, 10.0)
    pid = PID(
        kp=1.0,
        kv=0.0,
        ki=0.0,
        kd=0.0,
        kvff=0.0,
        dt=DT,
        dt_inv=DT_INV,
        u_min=-u_max,
        u_max=u_max,
        dir_toggle=True,
    )
    pid.reset(n_axes=N_AXES)
    u = pid.update(vec(3.0, 500.0, 0.0), vec(0.0, 0.0, 0.0))
    assert u[0] == pytest.approx(-3.0), "unclamped output is negated"
    assert u[1] == pytest.approx(10.0), "clamped output keeps the limit's sign"


def test_reset_clears_all_state() -> None:
    pid = make_pid(kp=1.0, ki=1.0, kd=1.0, kv=1.0, kvff=1.0, kaff=1.0)
    first = [
        pid.update(vec(float(k), 1.0, 0.0), vec(0.5 * k, 0.0, 0.0)) for k in range(5)
    ]
    pid.reset(n_axes=N_AXES)
    second = [
        pid.update(vec(float(k), 1.0, 0.0), vec(0.5 * k, 0.0, 0.0)) for k in range(5)
    ]
    np.testing.assert_allclose(np.array(first), np.array(second))


# ---------------------------------------------------------------------------
# Whole-algorithm check against an independent transcription of the VHDL
# ---------------------------------------------------------------------------


def vhdl_reference(
    gains: Gains,
    setpoints: Array,
    positions: Array,
    dt: float,
    dt_inv: float,
    u_max: float,
    max_integral: float,
) -> Array:
    """Single-axis transcription of brett_pid_ff.vhd, stage by stage.

    Written from the VHDL rather than from controller.py, so agreement between
    the two is meaningful.
    """
    integral = 0.0
    sum_int = 0.0  # reset to 0 on init, so the gate starts open
    prev_pos = 0.0
    prev_set = 0.0
    prev_err = 0.0
    prev_v_des = 0.0
    out: list[float] = []

    for sp, pos in zip(setpoints, positions, strict=True):
        pos_err = sp - pos  # on trigger
        vel = (pos - prev_pos) * dt_inv  # INITIAL
        v_des = (sp - prev_set) * dt_inv  # INITIAL
        d_err = pos_err - prev_err  # INITIAL

        # STAGE_4 / STAGE_5: gate on the previous sum_int and on v_des
        if sum_int < u_max and v_des != 0.0:
            integral += gains["ki"] * dt * pos_err
            integral = min(max(integral, -max_integral), max_integral)

        # STAGE_6: sum every scaled term
        sum_int = (
            gains["kp"] * pos_err
            - gains["kv"] * vel
            + integral
            + gains["kd"] * dt_inv * d_err
            + gains["kvff"] * v_des
            + gains["kaff"] * (v_des - prev_v_des)
            + gains["kpff1"] * sp
            + gains["kpff0"] * abs(sp) * sp
        )

        # STAGE_8 / STAGE_9 / DONE: k_tot then clamp
        scale_out = gains.get("ktot", 1.0) * sum_int
        out.append(min(max(scale_out, -u_max), u_max))

        prev_pos, prev_set, prev_err, prev_v_des = pos, sp, pos_err, v_des

    return np.array(out)


def test_matches_vhdl_reference() -> None:
    gains = {
        "kp": 1.3,
        "kv": 0.7,
        "ki": 2.5,
        "kd": 0.4,
        "kvff": 3.1,
        "kaff": 180.0,
        "kpff1": 0.02,
        "kpff0": 0.003,
    }
    rng = np.random.RandomState(0)
    n = 300
    # A ramp into a plateau, mirroring the real demand: exercises both the
    # moving and the stationary branch of the integral gate.
    setpoints = np.concatenate([np.linspace(0, 50, 100), np.full(200, 50.0)])
    positions = setpoints + rng.normal(0, 0.5, n)
    u_max, max_integral = 1000.0, 1e9

    expected = vhdl_reference(
        gains, setpoints, positions, DT, DT_INV, u_max, max_integral
    )

    limit = np.full(N_AXES, u_max)
    pid = make_pid_from(gains, u_min=-limit, u_max=limit, integral_limit=max_integral)
    pid.reset(n_axes=N_AXES)
    actual = np.array(
        [
            pid.update(np.full(N_AXES, sp), np.full(N_AXES, pos))[0]
            for sp, pos in zip(setpoints, positions, strict=True)
        ]
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)
