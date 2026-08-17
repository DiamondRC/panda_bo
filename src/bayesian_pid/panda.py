"""PandA device address, field mapping, and pre-run health check."""

import asyncio
import socket

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import Get

_PANDA_PORT = 8888  # pandablocks control channel


def _tcp_reachable(host: str, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, _PANDA_PORT), timeout=timeout):
            return True
    except OSError:
        return False


DEBUG_HEALTH_CHECK: bool = True  # set False to skip the pre-run field sanity check

PANDA_HOST: str = "bl99p-mo-panda-03"  # device IP/hostname; "" skips hardware phase

# Mapping from lowercase gain names (matching TUNE_PARAMS keys) to PandABlocks fields.
PANDA_PID_FIELDS: dict[str, str] = {
    "kp": "BRETT_PID.KP_I",
    "kv": "BRETT_PID.KV_I",
    "ki": "BRETT_PID.KI_I",
    "kd": "BRETT_PID.KD_I",
    "kvff": "BRETT_PID.KVFF_I",
    "kaff": "BRETT_PID.KAFF_I",
    "kpff1": "BRETT_PID.KPFF1_I",
    "kpff0": "BRETT_PID.KPFF0_I",
}

# Fixed-point format of each gain field, taken from the FPGA's
# global_constants.vhd: (integer_bits_including_sign, fractional_bits). Every
# field is a signed 32-bit word, so the two always sum to 32.
#
# These bound what the device can actually hold. A gain outside the range here
# cannot be written at all, and one below the resolution is indistinguishable
# from zero — so tuning outside these limits produces gains that will never
# reproduce on hardware.
PANDA_PID_FORMATS: dict[str, tuple[int, int]] = {
    "kp": (7, 25),  # KP_I_INT = 6 + 1, KP_I_FRAC = 25
    "kv": (7, 25),
    "ki": (7, 25),
    "kd": (7, 25),
    "kvff": (7, 25),
    "kaff": (12, 20),  # KAFF_I_INT = 11 + 1, KAFF_I_FRAC = 20
    "kpff1": (7, 25),  # KP1FF_I
    "kpff0": (7, 25),  # KP0FF_I
    "k_tot": (2, 30),  # K_TOT_I_INT = 1 + 1, K_TOT_I_FRAC = 30
}

_WORD_BITS = 32


def field_range(name: str) -> tuple[float, float]:
    """Smallest and largest value the device can hold for this gain."""
    int_bits, frac_bits = PANDA_PID_FORMATS[name]
    scale = float(1 << frac_bits)
    half = 1 << (int_bits + frac_bits - 1)
    return -half / scale, (half - 1) / scale


def field_resolution(name: str) -> float:
    """Smallest representable step for this gain."""
    return 2.0 ** -PANDA_PID_FORMATS[name][1]


def quantise(name: str, value: float) -> float:
    """Snap a gain to the device's representable grid, saturating at its range.

    Applied when writing gains so that what the optimiser records is what the
    device actually runs. Not applied inside the simulation loop: the coarsest
    resolution is ~1e-6, far below the level at which the objective responds.
    """
    lo, hi = field_range(name)
    step = field_resolution(name)
    return min(max(round(value / step) * step, lo), hi)


def full_positive_range(name: str) -> tuple[float, float]:
    """Search range covering every non-negative value the device can hold.

    The top of a signed fixed-point field is one step below the power of two
    (e.g. KP_I tops out just under 64), so deriving the bound rather than
    writing it out avoids an off-by-one-step that validate_bounds would reject.
    """
    return 0.0, field_range(name)[1]


def validate_bounds(tune_params: dict[str, tuple[float, float]]) -> None:
    """Raise if any search bound falls outside what the device can represent.

    Catches at startup the failure mode where BO spends its whole budget finding
    an optimum that cannot then be written to the FPGA.
    """
    problems: list[str] = []
    for name, (lo, hi) in tune_params.items():
        if name not in PANDA_PID_FORMATS:
            problems.append(f"{name}: no PandA field format known")
            continue
        dev_lo, dev_hi = field_range(name)
        if lo < dev_lo or hi > dev_hi:
            problems.append(
                f"{name}: bounds [{lo!r}, {hi!r}] exceed the device range "
                f"[{dev_lo!r}, {dev_hi!r}]. Use panda.full_positive_range({name!r}) "
                f"for the widest writable range."
            )
        if hi != 0 and abs(hi) < field_resolution(name):
            problems.append(
                f"{name}: upper bound {hi} is below the device resolution "
                f"{field_resolution(name):.3e}, so the whole range quantises to zero"
            )
    if problems:
        raise ValueError(
            "TUNE_PARAMS cannot be represented on the PandA:\n  "
            + "\n  ".join(problems)
        )


PANDA_SERVO_CLOCK_HZ = 125_000_000  # PandA master clock

# Device configuration registers that the simulation must mirror exactly.
# These are not tuned; they define the algorithm the FPGA runs. Confirmed on
# bl99p-mo-panda-03: PID_PERIOD_I=12500 (10 kHz), DT_I=1, DT_INV_I=1, K_TOT_I=1,
# MAX_OUTPUT_I=3932, MAX_INTEGRAL_I=28000, DIR_TOGGLE_I=0.
PANDA_CONFIG_FIELDS: dict[str, str] = {
    "pid_period": "BRETT_PID.PID_PERIOD_I",
    "dt": "BRETT_PID.DT_I",
    "dt_inv": "BRETT_PID.DT_INV_I",
    "ktot": "BRETT_PID.K_TOT_I",
    "max_output": "BRETT_PID.MAX_OUTPUT_I",
    "max_integral": "BRETT_PID.MAX_INTEGRAL_I",
    "dir_toggle": "BRETT_PID.DIR_TOGGLE_I",
}


def read_device_config(host: str) -> dict[str, float]:
    """Read the BRETT_PID configuration registers the simulation must mirror.

    Unreadable fields are omitted rather than defaulted, so a caller can tell
    "not read" from "read as zero".
    """
    raw = asyncio.run(_read_fields_async(host, list(PANDA_CONFIG_FIELDS.values())))
    out: dict[str, float] = {}
    for name, field in PANDA_CONFIG_FIELDS.items():
        value = raw[field]
        if isinstance(value, Exception):
            continue
        try:
            out[name] = _to_float(value)
        except (TypeError, ValueError):
            continue
    return out


def check_device_config(host: str, expected: dict[str, float]) -> bool:
    """Report the device's configuration and flag anything run.py disagrees with.

    Returns True when every expected value matches. A mismatch here means the
    simulation is modelling a different controller than the one on the device,
    so tuned gains will not transfer — dt/dt_inv rescale the velocity and
    integral terms, and the output/integral limits decide where it saturates.
    """
    config = read_device_config(host)
    if not config:
        print("  [WARN]  BRETT_PID config registers not readable")
        return False

    agreed = True
    for name, value in sorted(config.items()):
        line = f"  [OK  ]  {PANDA_CONFIG_FIELDS[name]:<24}  {value:g}"
        if name == "pid_period" and value > 0:
            line += f"   ({PANDA_SERVO_CLOCK_HZ / value:.0f} Hz)"
        if name in expected and expected[name] != value:
            agreed = False
            line = (
                f"  [FAIL]  {PANDA_CONFIG_FIELDS[name]:<24}  {value:g}"
                f"   run.py uses {expected[name]:g}"
            )
        print(line)

    if not agreed:
        print(
            "\nThe simulation does not match the device configuration above; "
            "tuned gains will not transfer until they agree."
        )
    return agreed


def _to_float(value: object) -> float:
    """Convert a PandA field response to a float.

    Responses arrive as untyped objects, so narrow before converting rather than
    calling float() on an `object`.
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"cannot convert {type(value).__name__} to float")


async def _read_fields_async(host: str, fields: list[str]) -> dict[str, object]:
    results: dict[str, object] = {}
    async with AsyncioClient(host) as client:
        for field in fields:
            try:
                results[field] = await client.send(Get(field))
            except Exception as exc:
                results[field] = exc
    return results


def health_check(
    host: str,
    param_names: list[str],
    prompt: str = "Continue anyway? [y/N]: ",
    expected_config: dict[str, float] | None = None,
) -> bool:
    """
    Read current PID field values from the device and report their status.

    expected_config: the configuration run.py is simulating (dt, dt_inv, ktot,
        max_output, max_integral, dir_toggle). When given, any disagreement with
        the device is reported as a failure, because it means the simulation is
        modelling a different controller than the one being tuned.

    Returns True to proceed, False to abort.
    Skipped entirely when DEBUG_HEALTH_CHECK is False.
    """
    if not DEBUG_HEALTH_CHECK:
        return True

    print(f"\n--- PandA health check ({host}) ---")

    reachable = _tcp_reachable(host)
    tag = "OK  " if reachable else "FAIL"
    print(f"  [{tag}]  connectivity to {host}:{_PANDA_PORT}")

    if not reachable:
        print(f"\nDevice {host} is not reachable on port {_PANDA_PORT}.")
        try:
            choice = input(prompt).strip().lower()
        except (KeyboardInterrupt, EOFError):
            choice = "n"
        return choice == "y"

    active_fields = [PANDA_PID_FIELDS[n] for n in param_names if n in PANDA_PID_FIELDS]
    results = asyncio.run(_read_fields_async(host, active_fields))

    bad = {f: v for f, v in results.items() if isinstance(v, Exception)}

    for field, val in results.items():
        tag = "OK  " if field not in bad else "FAIL"
        detail = str(val) if field not in bad else f"ERROR: {val}"
        print(f"  [{tag}]  {field:<22}  {detail}")

    if expected_config is not None:
        print("--- device configuration ---")
        if not check_device_config(host, expected_config):
            bad["config"] = ValueError("device configuration mismatch")

    if not bad:
        print("All fields readable.\n")
        return True

    print(f"\n{len(bad)} field(s) could not be read.")
    try:
        choice = input(prompt).strip().lower()
    except (KeyboardInterrupt, EOFError):
        choice = "n"
    return choice == "y"


def ask_run_hardware() -> bool:
    """Prompt the user after simulation to confirm they want hardware tuning."""
    try:
        choice = (
            input(
                "\nSimulation complete. "
                "Begin real-hardware tuning on the device? [y/N]: "
            )
            .strip()
            .lower()
        )
    except (KeyboardInterrupt, EOFError):
        choice = "n"
    return choice == "y"


def ask_run_mode() -> str:
    """Ask whether to run simulation first or go direct to hardware.

    Returns 'sim', 'direct', or 'quit'.
    """
    print("\nRun mode:")
    print("  [S]  Simulation -> optional hardware tuning")
    print("  [D]  Direct hardware tuning (reads current PandA values as start point)")
    print("  [Q]  Quit")
    try:
        choice = input("Choice [S/D/Q]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return "quit"
    if choice in ("d", "direct"):
        return "direct"
    if choice in ("q", "quit"):
        return "quit"
    return "sim"


def read_pid_values(host: str, param_names: list[str]) -> dict[str, float]:
    """Read current PID gain values from the device.

    Returns a dict mapping param name to float. Falls back to 0.0 with a
    warning if a field cannot be read.
    """
    fields = {n: PANDA_PID_FIELDS[n] for n in param_names if n in PANDA_PID_FIELDS}
    raw = asyncio.run(_read_fields_async(host, list(fields.values())))
    out: dict[str, float] = {}
    for name, field in fields.items():
        val = raw[field]
        if isinstance(val, Exception):
            print(f"  [WARN] Could not read {field}: {val}. Using 0.0.")
            out[name] = 0.0
            continue
        try:
            out[name] = _to_float(val)
        except (TypeError, ValueError):
            print(f"  [WARN] {field} returned {val!r}, not a number. Using 0.0.")
            out[name] = 0.0
    return out
