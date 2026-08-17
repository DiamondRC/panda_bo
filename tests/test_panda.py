"""Tests for the PandA fixed-point gain formats.

These bound what the device can actually hold. Tuning outside them produces an
optimum that cannot be written to the FPGA, which is how the previous search
space ended up with a best ki that no KI_I register could represent.
"""

import pytest

import bayesian_pid.panda as panda_mod
from bayesian_pid.panda import (
    PANDA_CONFIG_FIELDS,
    PANDA_PID_FIELDS,
    PANDA_PID_FORMATS,
    check_device_config,
    field_range,
    field_resolution,
    quantise,
    validate_bounds,
)

# Registers read from bl99p-mo-panda-03. The simulation must mirror these
# exactly or gains tuned against it will not transfer.
LIVE_DEVICE_CONFIG: dict[str, float] = {
    "pid_period": 12500.0,
    "dt": 1.0,
    "dt_inv": 1.0,
    "ktot": 1.0,
    "max_output": 3932.0,
    "max_integral": 28000.0,
    "dir_toggle": 0.0,
}


def _fake_read_device_config(host: str) -> dict[str, float]:  # noqa: ARG001
    """Stand-in for the live read, so these tests need no device."""
    return dict(LIVE_DEVICE_CONFIG)


def test_every_field_is_a_32_bit_word() -> None:
    """global_constants.vhd sizes every gain to PANDA_PORT_SIZE."""
    for name, (int_bits, frac_bits) in PANDA_PID_FORMATS.items():
        assert int_bits + frac_bits == 32, name


def test_every_tunable_gain_has_a_device_field() -> None:
    """A gain with a format but no field could never be written."""
    for name in PANDA_PID_FORMATS:
        if name == "k_tot":  # held at 1, deliberately not exposed for tuning
            continue
        assert name in PANDA_PID_FIELDS, name


@pytest.mark.parametrize(
    ("name", "expected_max"),
    [
        ("kp", 64.0),
        ("kv", 64.0),
        ("ki", 64.0),
        ("kd", 64.0),
        ("kvff", 64.0),
        ("kaff", 2048.0),
        ("kpff1", 64.0),
        ("kpff0", 64.0),
        ("k_tot", 2.0),
    ],
)
def test_field_ranges_match_the_vhdl_constants(name: str, expected_max: float) -> None:
    lo, hi = field_range(name)
    assert lo == -expected_max
    assert hi == pytest.approx(expected_max, rel=1e-6)
    assert hi < expected_max, "top of a signed range is one step below the power of two"


def test_resolutions_match_the_fractional_bit_counts() -> None:
    assert field_resolution("kp") == pytest.approx(2.0**-25)
    assert field_resolution("kaff") == pytest.approx(2.0**-20)
    assert field_resolution("k_tot") == pytest.approx(2.0**-30)


def test_quantise_snaps_to_the_representable_grid() -> None:
    step = field_resolution("kp")
    assert quantise("kp", 1.0 + 0.4 * step) == pytest.approx(1.0)
    assert quantise("kp", 1.0 + 0.6 * step) == pytest.approx(1.0 + step)


def test_quantise_is_idempotent() -> None:
    once = quantise("kaff", 214.8258834)
    assert quantise("kaff", once) == once


def test_quantise_saturates_at_the_field_limits() -> None:
    lo, hi = field_range("kp")
    assert quantise("kp", 1e9) == pytest.approx(hi)
    assert quantise("kp", -1e9) == pytest.approx(lo)


def test_quantise_preserves_exactly_representable_values() -> None:
    assert quantise("kp", 0.5) == 0.5
    assert quantise("kp", 0.0) == 0.0


def test_validate_bounds_accepts_the_full_device_range() -> None:
    validate_bounds({name: (0.0, field_range(name)[1]) for name in PANDA_PID_FIELDS})


def test_validate_bounds_rejects_an_unrepresentable_upper_bound() -> None:
    """Regression: the previous best ki of ~78.5 exceeds KI_I's +-64."""
    with pytest.raises(ValueError, match="exceed the device range"):
        validate_bounds({"ki": (0.0, 78.5)})


def test_validate_bounds_rejects_an_unrepresentable_lower_bound() -> None:
    with pytest.raises(ValueError, match="exceed the device range"):
        validate_bounds({"kaff": (-5000.0, 1.0)})


def test_validate_bounds_rejects_a_range_below_the_resolution() -> None:
    """A range that quantises entirely to zero would waste the whole budget."""
    with pytest.raises(ValueError, match="below the device resolution"):
        validate_bounds({"kp": (0.0, 1e-12)})


def test_validate_bounds_rejects_an_unknown_gain() -> None:
    with pytest.raises(ValueError, match="no PandA field format known"):
        validate_bounds({"not_a_gain": (0.0, 1.0)})


def test_quantise_reproduces_the_live_device_registers() -> None:
    """The fixed-point model must land on exactly what the FPGA stores.

    Left column is what the simulation produced; right is what was read back
    from BRETT_PID after those values were written.
    """
    written_vs_readback = [
        ("kp", 0.8671094452078429, 0.8671094477),
        ("kv", 2.0, 2.0),
        ("ki", 0.007852857458780095, 0.007852852345),
        ("kvff", 4.7250680531512765, 4.725068063),
        ("kaff", 214.82588340071794, 214.8258839),
    ]
    for name, written, readback in written_vs_readback:
        assert quantise(name, written) == pytest.approx(
            readback, abs=field_resolution(name)
        ), name


def test_dt_and_dt_inv_are_both_one_on_the_device() -> None:
    """Guards the assumption the whole gain convention rests on.

    dt_i and dt_inv_i are independent registers, both 1 on the device, matching
    the PMAC C reference which has no dt. Deriving dt_inv as 1/dt rescales
    kv/kvff/kaff and ki by orders of magnitude.
    """
    assert LIVE_DEVICE_CONFIG["dt"] == 1.0
    assert LIVE_DEVICE_CONFIG["dt_inv"] == 1.0
    assert LIVE_DEVICE_CONFIG["ktot"] == 1.0


def test_pid_period_gives_ten_kilohertz() -> None:
    rate = panda_mod.PANDA_SERVO_CLOCK_HZ / LIVE_DEVICE_CONFIG["pid_period"]
    assert rate == pytest.approx(10_000.0)


def test_every_config_field_is_named() -> None:
    for key in LIVE_DEVICE_CONFIG:
        assert key in PANDA_CONFIG_FIELDS


def test_check_device_config_passes_when_everything_agrees(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(panda_mod, "read_device_config", _fake_read_device_config)
    assert check_device_config("fake-host", dict(LIVE_DEVICE_CONFIG)) is True
    assert "FAIL" not in capsys.readouterr().out


def test_check_device_config_flags_a_mismatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression: run.py clamped at 1000 while the device clamps at 3932."""
    monkeypatch.setattr(panda_mod, "read_device_config", _fake_read_device_config)
    stale = dict(LIVE_DEVICE_CONFIG)
    stale["max_output"] = 1000.0
    stale["max_integral"] = 1e9
    assert check_device_config("fake-host", stale) is False
    out = capsys.readouterr().out
    assert "MAX_OUTPUT_I" in out
    assert "MAX_INTEGRAL_I" in out


def test_check_device_config_reports_when_nothing_is_readable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_config(host: str) -> dict[str, float]:  # noqa: ARG001
        return {}

    monkeypatch.setattr(panda_mod, "read_device_config", no_config)
    assert check_device_config("fake-host", dict(LIVE_DEVICE_CONFIG)) is False
    assert "not readable" in capsys.readouterr().out


def test_validate_bounds_reports_every_problem_at_once() -> None:
    with pytest.raises(ValueError) as excinfo:
        validate_bounds({"ki": (0.0, 100.0), "kp": (0.0, 100.0)})
    assert "ki" in str(excinfo.value)
    assert "kp" in str(excinfo.value)
