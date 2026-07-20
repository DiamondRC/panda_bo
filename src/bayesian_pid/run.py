#!/usr/bin/env python3
"""
Created on Fri Jun 12 10:21:53 2026

@author: Xingchi Liu

TurBO with second order state transistion model as the response

Use the one-direction trajectory as the demand

Here the PID is based on Brett's implementation'
@author: ubuntu
"""

import os
import sys
import time
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch

from bayesian_pid.hardware import HardwareEvaluator
from bayesian_pid.metrics import DEFAULT_WINDOW
from bayesian_pid.optimiser import run_mc_optimisation, run_one_optimization
from bayesian_pid.panda import (
    PANDA_HOST,
    PANDA_PID_FIELDS,
    ask_run_hardware,
    ask_run_mode,
    health_check,
    read_pid_values,
)
from bayesian_pid.plotting import plot_mc_result, plot_single_result
from bayesian_pid.validation import _as_pid_vector, evaluate

os.system("clear")

# ============================================================
# Data
# ============================================================

DATA_PATH = "/workspaces/bayesian_pid/data/"

# trajectory learned form 1v_1 data set
# A1 = np.load(f"{DATA_PATH}/A1.npy")
# A2 = np.load(f"{DATA_PATH}/A2.npy")
# b = np.load(f"{DATA_PATH}/B.npy")
# c = np.load(f"{DATA_PATH}/c.npy")

# trajectory learned form 3.5v_1 data set

A1 = np.load(f"{DATA_PATH}/A1_3.5v.npy")
A2 = np.load(f"{DATA_PATH}/A2_3.5v.npy")
b = np.load(f"{DATA_PATH}/B_3.5v.npy")
c = np.load(f"{DATA_PATH}/c_3.5v.npy")


# ============================================================
# Load trajectory and pad 0 to make it a 3D trajectory
# ============================================================


# pos_ref = np.load(f"{DATA_PATH}/scaled_trajectory_25nm.npy")
pos_ref = np.load(f"{DATA_PATH}/xonly_trajectory.npy")
pos_ref = np.concatenate([pos_ref.reshape(-1, 1), np.zeros((len(pos_ref), 2))], axis=1)


# ---------------------------------------------

device = torch.device("cpu")
dtype = torch.double

u_min = 100 * np.array([-10, -10, -10])
u_max = 100 * np.array([10, 10, 10])
max_integral = 1e9

# -------------------------------------------------------
# Declare which PID gains to auto-tune (and their bounds)
# vs which to hold fixed.
# This is the only place to edit
# when changing the active search space.
# Gains are in per-sample units (matching brett_pid / hardware).
# -------------------------------------------------------
TUNE_PARAMS: dict[str, tuple[float, float]] = {
    # name:  (lo,   hi)
    # "kp": (0.0, 10.0),
    # "kv": (0.0, 2.0),
    # "ki": (0.0, 0.002),
    "kp": (0.0, 2.0),
    "kv": (0.0, 2.0),
    "ki": (0.0, 2.0),
    "kvff": (0.0, 10.0),
    "kaff": (0.0, 300.0),
    "kpff1": (0.0, 1.0),
    "kpff0": (0.0, 1.0),
}

# Hold vals constant whilst others are tuned
FIXED_PARAMS: dict[str, float] = {
    "kd": 0.0,
    # "kaff":  0.0,
    # "kpff1": 0.0,
    # "kpff0": 0.0,
    # "kvff": 3.7,
    # "kaff": 200.0,
    # "kpff1": 0.002,
    # "kpff0": 0.0,
}

param_names = list(TUNE_PARAMS.keys())
dim = len(param_names)
bounds_phys = torch.tensor(
    [[lo for lo, _ in TUNE_PARAMS.values()], [hi for _, hi in TUNE_PARAMS.values()]],
    dtype=dtype,
    device=device,
)

sim_evaluate_fn = partial(
    evaluate,
    trajectory=pos_ref,
    A1=A1,
    A2=A2,
    b=b,
    c=c,
    u_min=u_min,
    u_max=u_max,
    max_integral=max_integral,
    param_names=param_names,
    fixed_params=FIXED_PARAMS,
)


def _make_progress_fn(names: list[str]):
    def _progress(eval_count, total, x_phys, y, best_y, extra=None):
        extra = extra or {}
        phase = extra.get("phase", "bo")
        restart = extra.get("restart", None)
        tr = extra.get("tr_length", None)
        x_flat = x_phys.detach().cpu().numpy().flatten()
        params_str = "  ".join(
            f"{n}={v:.4g}" for n, v in zip(names, x_flat, strict=False)
        )
        restart_str = f" R{restart}" if restart else ""
        tr_str = f"  TR={tr:.4f}" if tr is not None else ""
        t_str = f"  t={extra['eval_time']:.1f}s" if "eval_time" in extra else ""
        print(
            f"[{phase:4s} | {restart_str} | {eval_count:4d}/{total}]: "
            f"rms_err={y:.4g} | best={best_y:.4g}{tr_str}{t_str} | {params_str}",
            flush=True,
        )

    return _progress


progress_fn = _make_progress_fn(param_names)

run_mode = "sim"
if PANDA_HOST:
    run_mode = ask_run_mode()
    if run_mode == "quit":
        raise SystemExit(0)
    if run_mode == "sim" and not health_check(
        PANDA_HOST, param_names, prompt="Continue with simulation? [y/N]: "
    ):
        raise SystemExit(0)

method = "Turbo"  # change to "standard_bo" for ordinary BO without trust region
run_mc = True  # False: one optimisation run; True: MC runs over different seeds
n_mc_runs = 3  # How many MC runs
seed0 = 1  # MC seeds will be seed0, seed0+1, ..., seed0+n_mc_runs-1
total_budget = 200  # total number of function evaluations per simulated run
LIVE_BUDGET = 100  # total number of function evaluations per live run
n_init = 10  # initial Sobol points per run/restart
save_mc_csv = True
mc_results_dir = "xonly_trajectory_turbo_mc_results"

os.makedirs(mc_results_dir, exist_ok=True)
_log_file = open(os.path.join(mc_results_dir, "run.log"), "a")  # noqa: SIM115
_debug_plot_path = os.path.join(mc_results_dir, "hw_trajectory_debug.png")


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)

t_start = time.time()
print(f"\n=== Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

print("--- Tuned parameters ---")
for _name, (_lo, _hi) in TUNE_PARAMS.items():
    print(f"  {_name:<8}  [{_lo}, {_hi}]")
if FIXED_PARAMS:
    print("--- Fixed parameters ---")
    for _name, _val in FIXED_PARAMS.items():
        print(f"  {_name:<8}  {_val}")
print("--- Run config ---")
print(f"  method       {method}")
print(f"  mc_runs      {n_mc_runs}  (seeds {seed0}-{seed0 + n_mc_runs - 1})")
print(f"  sim_budget   {total_budget} evals  (n_init={n_init})")
print(f"  live_budget  {LIVE_BUDGET} evals  (n_init=1, warm-started from sim best)")
print(
    f"  plateau      [{DEFAULT_WINDOW.start_frac}, {DEFAULT_WINDOW.end_frac}] "
    f"(fraction of trajectory)"
)
print(f"  run_mode     {run_mode}")
print(f"  results_dir  {mc_results_dir}")
print()


# ============================================================
# Direct-to-hardware mode: skip simulation, warm-start from current PandA values
# ============================================================
if run_mode == "direct":
    if not health_check(
        PANDA_HOST, param_names, prompt="Continue to hardware tuning? [y/N]: "
    ):
        raise SystemExit(0)

    print("\n--- Reading current PandA values as warm start ---")
    current_vals = read_pid_values(PANDA_HOST, param_names)
    for _n, _v in current_vals.items():
        print(f"  {_n:<8}  {_v}")

    _lo = bounds_phys[0].numpy()
    _hi = bounds_phys[1].numpy()
    _raw = np.array([current_vals.get(_n, 0.0) for _n in param_names])
    _warm_unit = np.clip((_raw - _lo) / (_hi - _lo), 0.0, 1.0)
    _warm_start_x = torch.tensor(_warm_unit, dtype=dtype, device=device).unsqueeze(0)

    _hw_pid_fields = {
        n: PANDA_PID_FIELDS[n] for n in param_names if n in PANDA_PID_FIELDS
    }
    _pgen_trajectory = [str(int(np.round(v))) for v in pos_ref[:, 0]]

    _hw_evaluate_fn = HardwareEvaluator(
        host=PANDA_HOST,
        pid_fields=_hw_pid_fields,
        trajectory=_pgen_trajectory,
        debug_plot_path=_debug_plot_path,
    )

    _hw_result = run_one_optimization(
        _hw_evaluate_fn,
        bounds_phys,
        dim=dim,
        dtype=dtype,
        device=device,
        method=method,
        seed=seed0,
        total_budget=LIVE_BUDGET,
        n_init=1,
        verbose=True,
        warm_start_x=_warm_start_x,
        progress_fn=progress_fn,
    )
    print("Direct hardware BO best:", _as_pid_vector(_hw_result["best_x_phys"], dim))


# ============================================================
# Simulation mode
# ============================================================
if run_mode == "sim" and run_mc:
    mc_results = run_mc_optimisation(
        sim_evaluate_fn,
        bounds_phys,
        dim=dim,
        method=method,
        dtype=dtype,
        device=device,
        n_runs=n_mc_runs,
        seed0=seed0,
        total_budget=total_budget,
        n_init=n_init,
        verbose=False,
        save_csv=save_mc_csv,
        results_dir=mc_results_dir,
        progress_fn=progress_fn,
    )
    plot_mc_result(mc_results, uncertainty="std", save_dir=mc_results_dir)

    # Use the best individual MC run for the final trajectory re-simulation below.
    result = min(mc_results["run_results"], key=lambda r: r["best_y"])
elif run_mode == "sim":
    result = run_one_optimization(
        sim_evaluate_fn,
        bounds_phys,
        dim=dim,
        method=method,
        dtype=dtype,
        device=device,
        seed=seed0,
        total_budget=total_budget,
        n_init=n_init,
        verbose=True,
        progress_fn=progress_fn,
    )
    plot_single_result(result, save_dir=mc_results_dir)

if run_mode == "sim":
    print("\nOptimisation finished.")
    print("Best PID found:", _as_pid_vector(result["best_x_phys"], dim))
    print("Best average tracking error:", result["best_y"])

    best_x_phys = result["best_x_phys"].to(dtype=dtype, device=device)

    # ============================================================
    # Re-simulate best PID and plot
    # ============================================================

    _, best_details = sim_evaluate_fn(best_x_phys, return_details=True)
    best_run = best_details[0]
    best_actual = best_run["actual_pos"]
    best_voltage = best_run["control_hist"]

    os.makedirs(mc_results_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pos_ref[:, 0], label="Reference x")
    ax.plot(best_actual[:, 0], label="Tracked x")
    ax.set_xlabel("Time step")
    ax.set_ylabel("x")
    ax.set_title("X-axis tracking with BO-tuned PID")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(
        os.path.join(mc_results_dir, "x_tracking.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    pos_err = np.linalg.norm(best_actual - pos_ref, axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pos_err)
    ax.set_title("Position tracking error over time")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Euclidean error")
    fig.tight_layout()
    fig.savefig(
        os.path.join(mc_results_dir, "pos_error_euclidean.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    pos_err = best_actual[:, 0] - pos_ref[:, 0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pos_err)
    ax.set_title("Position tracking error over time (x only)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("x error")
    fig.tight_layout()
    fig.savefig(
        os.path.join(mc_results_dir, "pos_error_x.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(best_voltage[:, 0], bins=100)
    ax.set_title("Control voltage histogram")
    ax.set_xlabel("Voltage")
    fig.tight_layout()
    fig.savefig(
        os.path.join(mc_results_dir, "voltage_hist.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    print(f"\nPlots saved to: {mc_results_dir}/")

    # ============================================================
    # Phase 2: continue optimisation on real hardware
    # ============================================================

    if PANDA_HOST and ask_run_hardware():
        if health_check(
            PANDA_HOST, param_names, prompt="Continue to hardware tuning? [y/N]: "
        ):
            hw_pid_fields = {
                n: PANDA_PID_FIELDS[n] for n in param_names if n in PANDA_PID_FIELDS
            }

            # pos_ref[:, 0] is already in PGEN integer counts (max ~500,000)
            pgen_trajectory = [str(int(np.round(v))) for v in pos_ref[:, 0]]

            hw_evaluate_fn = HardwareEvaluator(
                host=PANDA_HOST,
                pid_fields=hw_pid_fields,
                trajectory=pgen_trajectory,
                debug_plot_path=_debug_plot_path,
            )

            # Warm-start: seed hardware BO with the simulation best point (unit cube).
            sim_best_unit = (result["best_x_phys"] - bounds_phys[0]) / (
                bounds_phys[1] - bounds_phys[0]
            )

            hw_result = run_one_optimization(
                hw_evaluate_fn,
                bounds_phys,
                dim=dim,
                dtype=dtype,
                device=device,
                method=method,
                seed=seed0,
                total_budget=LIVE_BUDGET,
                n_init=1,
                verbose=True,
                warm_start_x=sim_best_unit.unsqueeze(0),
                progress_fn=progress_fn,
            )
            print("Hardware BO best:", _as_pid_vector(hw_result["best_x_phys"], dim))

print(f"\n=== Total run time: {time.time() - t_start:.1f}s ===")
