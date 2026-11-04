# pyright: reportUnknownMemberType=false
"""
Created on October 2026

@author: Xingchi Liu

check the a given tuning's following error

we have 2 tunings from Brett and one from BO
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from bayesian_pid.metrics import diverged_at, following_error_rms
from bayesian_pid.run import (
    DTYPE,
    TUNE_PARAMS,
    build_sim_objective,
    load_plant,
    load_trajectory,
)

# brett's tuning for trapezoid A
# engineer_params = torch.tensor(
#     [
#         [
#             0.004,  # kp
#             0.9,  # kv
#             0.0002,  # ki
#             3.7,  # kvff
#             100.0,  # kaff
#             0.00196,  # kpff1
#             0.0,  # kpff0
#         ]
#     ],
#     dtype=DTYPE,
# )
# trapezoid B tests, x axis
# engineer_params = torch.tensor(
#     [
#         [
#             0.004,  # kp
#             0.9,  # kv
#             0.0001,  # ki
#             3.55,  # kvff
#             0.0,  # kaff
#             0.001,  # kpff1
#             -2e-11,  # kpff0
#         ]
#     ],
#     dtype=DTYPE,
# )

# one point find by BO
engineer_params = torch.tensor(
    [
        [
            2,  # kp
            2,  # kv
            0.276942929,  # ki
            8.94608781,  # kvff
            0.0,  # kaff
            2.7299e-3,  # kpff1
            0.0,  # kpff0
        ]
    ],
    dtype=DTYPE,
)

# Load exactly the same plant and reference trajectory
plant = load_plant()
trajectory = load_trajectory()
# trajectory = trajectory * 2


# Build the same objective used by control_land.py
param_names = list(TUNE_PARAMS.keys())
evaluate_fn = build_sim_objective(trajectory, plant, param_names)

# Evaluate the engineer's parameters ONCE
score, details = evaluate_fn(
    engineer_params,
    return_details=True,
)

actual_pos = details[0]["actual_pos"]
control_hist = details[0]["control_hist"]

# Original optimisation score
print("Following error score:", score.item())

# Actual x-axis plateau RMS
rms_x = following_error_rms(trajectory, actual_pos, axes=(0,))
print("X-axis plateau RMS:", rms_x)

# Divergence check
index = diverged_at(trajectory, actual_pos, axes=(0,))
print("X-axis divergence index:", index)

import bayesian_pid.control_land as control_land
import bayesian_pid.metrics as metrics
import bayesian_pid.validation as validation
from bayesian_pid.metrics import following_error_score

print("\n--- Scoring diagnostics ---")

print("Metrics module:", metrics.__file__)
print("Validation module:", validation.__file__)
print("Control land module:", control_land.__file__)

print("SCORE_AXES:", control_land.SCORE_AXES)
print("Trajectory shape:", trajectory.shape)
print("Actual position shape:", actual_pos.shape)

score_direct = following_error_score(
    trajectory,
    actual_pos,
    axes=(0,),
)

rms_direct = following_error_rms(
    trajectory,
    actual_pos,
    axes=(0,),
)

print("Direct following_error_score:", score_direct)
print("Direct following_error_rms:", rms_direct)
print("Objective returned by evaluate_fn:", score.item())

if index is not None:
    n = len(trajectory)
    survived = max(index / n, 1.0 / n)

    pre_error = trajectory[:index, 0] - actual_pos[:index, 0]
    pre_rms = float(np.sqrt(np.mean(pre_error**2)))

    expected_score = 1e6 / survived + pre_rms

    print("Divergence index:", index)
    print("Survival fraction:", survived)
    print("Pre-divergence RMS:", pre_rms)
    print("Expected divergence score:", expected_score)

# Confirm y/z control signals are disabled
print("Max |u_y|:", np.max(np.abs(control_hist[:, 1])))
print("Max |u_z|:", np.max(np.abs(control_hist[:, 2])))


ideal_x = np.asarray(trajectory[:, 0], dtype=np.float64)
actual_x = np.asarray(actual_pos[:, 0], dtype=np.float64)
error_x = ideal_x - actual_x

t = np.arange(len(ideal_x))

# ------------------------------------------------------------------
# Plot 1: ideal vs actual x-axis position
# ------------------------------------------------------------------

plt.figure(figsize=(10, 4))
plt.plot(ideal_x, label="Ideal x")
plt.plot(actual_x, label="Actual x")
plt.xlabel("Time step")
plt.ylabel("Position")
plt.title("Ideal vs Actual X-axis Position")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("x_tracking_test9.png", dpi=300)
plt.show()

# ------------------------------------------------------------------
# Plot 2: x-axis tracking error
# ------------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(t, error_x, label="x tracking error")
plt.xlabel("Time step")
plt.ylabel("Error (ideal - actual)")
plt.title("X-axis Tracking Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("x_tracking_test10.png", dpi=300)
plt.show()
