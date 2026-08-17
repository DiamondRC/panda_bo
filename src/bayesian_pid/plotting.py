# matplotlib types its Axes/Figure methods with `**kwargs: Unknown`, so every
# call here reads as "partially unknown" under strict mode regardless of what we
# annotate. Scoped to this module, which is nothing but plotting calls.
# pyright: reportUnknownMemberType=false
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from bayesian_pid.metrics import DEFAULT_WINDOW, PlateauWindow


def _save(fig: Figure, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


def plot_single_result(result: dict[str, Any], save_dir: str = ".") -> None:
    """Plot the best-so-far curve for one optimisation result."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(result["best_curve"]) + 1), result["best_curve"])
    ax.set_xlabel("Function evaluation")
    ax.set_ylabel("Best objective so far")
    ax.set_title(f"Best-so-far curve: {result['method']}, seed={result['seed']}")
    ax.grid(True)
    fig.tight_layout()
    fname = f"{result['method']}_seed{result['seed']}_best_curve.png"
    _save(fig, os.path.join(save_dir, fname))


def plot_mc_result(
    mc_results: dict[str, Any],
    uncertainty: str = "std",
    save_dir: str = ".",
) -> None:
    """
    Plot the MC-averaged best-so-far curve.

    uncertainty:
        "std" -> shaded mean ± standard deviation;
        "sem" -> shaded mean ± standard error;
        None  -> mean curve only.
    """
    os.makedirs(save_dir, exist_ok=True)
    x = np.arange(1, len(mc_results["mean_best_curve"]) + 1)
    mean = mc_results["mean_best_curve"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, mean, label="Mean best so far")

    if uncertainty == "std":
        spread = mc_results["std_best_curve"]
        ax.fill_between(x, mean - spread, mean + spread, alpha=0.2, label="±1 std")
    elif uncertainty == "sem":
        spread = mc_results["sem_best_curve"]
        ax.fill_between(x, mean - spread, mean + spread, alpha=0.2, label="±1 SEM")

    ax.set_xlabel("Function evaluation")
    ax.set_ylabel("Best objective so far")
    ax.set_title(f"MC-averaged best-so-far curve: {mc_results['method']}")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, f"{mc_results['method']}_mc_best_curve.png"))


def plot_hardware_trajectory(
    setpoints: np.ndarray,
    positions: np.ndarray,
    metric: float,
    path: str,
    window: PlateauWindow = DEFAULT_WINDOW,
) -> None:
    """Save a debug plot of hardware setpoints vs recorded position.

    Overwrites `path` on every call. Two stacked subplots:
      top   — setpoints and positions with plateau window marked
      bottom — following error with plateau window and metric annotated
    """
    n = len(setpoints)
    samples = np.arange(n)
    i0 = int(n * window.start_frac)
    i1 = int(n * window.end_frac)
    err = setpoints - positions

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax_top.plot(samples, setpoints, label="Setpoint", linewidth=0.8)
    ax_top.plot(samples, positions, label="Position", linewidth=0.8, alpha=0.8)
    ax_top.axvline(
        i0, color="grey", linestyle="--", linewidth=0.8, label="Plateau window"
    )
    ax_top.axvline(i1, color="grey", linestyle="--", linewidth=0.8)
    ax_top.set_ylabel("Encoder counts")
    ax_top.legend(fontsize=8)
    ax_top.grid(True, linewidth=0.4)

    ax_bot.plot(samples, err, linewidth=0.8, color="tab:red")
    ax_bot.axvline(i0, color="grey", linestyle="--", linewidth=0.8)
    ax_bot.axvline(i1, color="grey", linestyle="--", linewidth=0.8)
    ax_bot.axhline(0, color="black", linewidth=0.5)
    ax_bot.set_ylabel("Following error (counts)")
    ax_bot.set_xlabel("Sample")
    ax_bot.grid(True, linewidth=0.4)
    ax_bot.text(
        0.99,
        0.95,
        f"plateau std = {metric:.4g}",
        transform=ax_bot.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.6},
    )

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
