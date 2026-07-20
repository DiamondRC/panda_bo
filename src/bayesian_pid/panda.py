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
) -> bool:
    """
    Read current PID field values from the device and report their status.

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
        else:
            out[name] = float(val)
    return out
