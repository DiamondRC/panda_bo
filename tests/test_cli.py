import subprocess
import sys

from bayesian_pid import __version__


def test_cli_version() -> None:
    cmd = [sys.executable, "-m", "bayesian_pid", "--version"]
    assert subprocess.check_output(cmd).decode().strip() == __version__
