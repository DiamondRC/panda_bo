"""Compare nine identified plants using the repository's unchanged controller.

Place in the panda_bo repository root and run in its existing Python environment:
    python compare_nine_plants.py --trajectory data/xonly_trajectory.npy
    python compare_nine_plants.py --models 4 8 9 --trajectory my_scan.npy

Inspected branch: feature/pid-bo--update-Sep
Inspected commit: 7abe9c0cee78b411361975dcbf1a244e38de0d7d

IMPORTANT MODEL CONVENTIONS
- The simulator uses x_next = a1 @ current + a2 @ previous + b @ u + c.
- The attachment uses A1 @ previous + A2 @ current: these are swapped below.
- This retains the simulator's immediate input timing. The attachment writes
  x(k+2) = A1*x(k) + A2*x(k+1) + B*u(k) + C. If u(k) literally means an
  additional one-sample input delay, these tests do NOT implement that delay.
  Confirm the identification code's indexing before interpreting physical fit.
- No offsets, units, sample-rate conversions or equilibrium corrections are
  inferred. The supplied biases are used literally, with the original two
  initial positions seeded from the trajectory. Absolute-coordinate models
  must be given trajectories in the same coordinate system used for fitting.
- All axes remain controlled exactly as in the inspected branch; x-only scoring
  does not disable y/z actuation. No controller code is copied or modified.
- DT/DT_INV are controller registers, NOT the physical sampling period.
- The matrices have only the precision present in the supplied text file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Make src-layout imports work when this script is placed at the repo root.
if (Path(__file__).resolve().parent / "src" / "bayesian_pid").is_dir():
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Edit these seven values, or supply --gains in exactly this order.
PARAM_NAMES = ("kp", "kv", "ki", "kvff", "kaff", "kpff1", "kpff0")
ENGINEER_PARAMS = (0.004, 0.9, 0.0002, 3.7, 100.0, 0.00196, 0.0)  # trapezoid A
# ENGINEER_PARAMS = (0.004, 0.9, 0.0001, 3.55, 0.0, 0.001, -2e-11)  # trapezoid B
# ENGINEER_PARAMS = (2, 2, 0.276942929, 8.94608781, 0.0, 2.7299e-3, 0.0)  # TurBO

MODEL_TEXT = r"""
==================================================
x(k+1) = A*x(k) + B*u(k)  [LS]
CV rmse_test_10 (training folds): mean=5846.13, std=3305.45, score(mean+std)=9151.58
Held-out test group: 3V_all_axis_1.txt, 3V_all_axis_2.txt
Held-out test rmse_test_10: 8534.82
--------------------------------------------------
A =
        0.998111  -0.000584608    6.2275e-08
     0.000370507      0.998451   2.91939e-07
    -0.000399632  -0.000232037             1

B =
         4.66566      0.228011      0.518934
       -0.947169       3.85028     -0.165148
        0.482403      -0.64257       2.79113


==================================================
x(k+1) = A*x(k) + B*u(k) + C  [LS + bias]
CV rmse_test_10 (training folds): mean=5777.05, std=3333.22, score(mean+std)=9110.27
Held-out test group: 3V_all_axis_1.txt, 3V_all_axis_2.txt
Held-out test rmse_test_10: 8340.8
--------------------------------------------------
A =
        0.998117  -0.000616944  -0.000132099
     0.000355674      0.998529    0.00031662
    -0.000315597    -0.0006705      0.998208

B =
         4.66606       0.26034      0.637448
       -0.948112        3.7729      -0.44881
        0.487747     -0.204194       4.39816

C =
        -6340.59
         15176.2
        -85977.5


==================================================
x(k+2) = A1*x(k) + A2*x(k+1) + B*u(k)  [LS 2-step]
CV rmse_test_10 (training folds): mean=126.218, std=84.7088, score(mean+std)=210.926
Held-out test group: 3V_all_axis_1.txt, 3V_all_axis_2.txt
Held-out test rmse_test_10: 182.403
--------------------------------------------------
A1 =
       -0.997988   1.99701e-05   2.23421e-05
     1.20273e-05     -0.998084   4.44155e-05
     6.58739e-05  -3.31142e-06     -0.999968

A2 =
         1.99793  -1.99648e-05  -2.23419e-05
    -1.19594e-05       1.99802  -4.44097e-05
    -6.82618e-05   1.97358e-05       1.99997

B =
       0.0632533  -0.000480103  -0.000561348
      0.00113887     0.0608126   0.000170385
     -0.00118924    -0.0155925   2.09056e-06


==================================================
*** SELECTED via leave-one-group-out CV on training groups ***
x(k+2) = A1*x(k) + A2*x(k+1) + B*u(k) + C  [LS 2-step + bias]
CV rmse_test_10 (training folds): mean=31.0639, std=18.5655, score(mean+std)=49.6294
Held-out test group: 3V_all_axis_1.txt, 3V_all_axis_2.txt
Held-out test rmse_test_10: 33.3781
--------------------------------------------------
A1 =
       -0.997988   2.06562e-05   3.95192e-05
     1.20187e-05     -0.998084   4.49009e-05
      3.1608e-05   7.33933e-05     -0.998047

A2 =
         1.99793  -2.07909e-05  -4.00705e-05
    -1.19502e-05       1.99802  -4.49107e-05
    -3.17769e-05  -7.26237e-05       1.99799

B =
       0.0632611  -0.000353449    -1.919e-05
      0.00113909     0.0608162   0.000185707
    -0.000306957    -0.0014323     0.0606166

C =
        -26.4554
       -0.747666
        -2957.78


==================================================
x(k+1) = A*x(k) + B*u(k)  [TLS v2]
CV rmse_test_10 (training folds): mean=16249.9, std=9392.81, score(mean+std)=25642.7
Held-out test group: 3V_all_axis_1.txt, 3V_all_axis_2.txt
Held-out test rmse_test_10: 21054.2
--------------------------------------------------
A =
        0.987625   -0.00383478   5.69116e-07
     0.000284243      0.987641   2.25385e-06
     0.000602843    0.00189971             1

B =
         30.9666       3.97316      0.209135
        -3.42125       30.0846     -0.904184
        -2.68823      -5.44977       29.8691


==================================================
x(k+1) = A*x(k) + B*u(k) + C  [TLS v2 + bias]
CV rmse_test_10 (training folds): mean=17183.5, std=9750.55, score(mean+std)=26934.1
Held-out test group: 3V_all_axis_1.txt, 3V_all_axis_2.txt
Held-out test rmse_test_10: 23657.6
--------------------------------------------------
A =
        0.987626   -0.00389757  -0.000209235
     0.000258355      0.988576    0.00312568
     0.000861106   -0.00742496       0.96884

B =
         30.9928       4.04927      0.214274
        -3.81164       28.9515     -0.980677
         1.20632       5.85346       30.6322

C =
        -10065.7
          149852
    -1.49494e+06


==================================================
x(k+2) = A1*x(k) + A2*x(k+1) + B*u(k)  [TLS v2 2-step]
CV rmse_test_10 (training folds): mean=126.225, std=84.7748, score(mean+std)=211
Held-out test group: 3V_all_axis_1.txt, 3V_all_axis_2.txt
Held-out test rmse_test_10: 182.385
--------------------------------------------------
A1 =
       -0.997996   1.84044e-05    2.1938e-05
     1.08999e-05     -0.998092   4.30279e-05
     6.73838e-05  -1.35541e-05      -1.00001

A2 =
         1.99793  -1.83941e-05  -2.19378e-05
    -1.08341e-05       1.99802  -4.30221e-05
    -6.97589e-05   3.00146e-05       2.00001

B =
       0.0632403  -0.000486209    -0.0005667
      0.00114306     0.0608009   0.000167765
     -0.00119639    -0.0156301  -0.000116292


==================================================
x(k+2) = A1*x(k) + A2*x(k+1) + B*u(k) + C  [TLS v2 2-step + bias]
CV rmse_test_10 (training folds): mean=49.4492, std=29.5027, score(mean+std)=78.9519
Held-out test group: 3V_all_axis_1.txt, 3V_all_axis_2.txt
Held-out test rmse_test_10: 60.2838
--------------------------------------------------
A1 =
       -0.997997   2.14469e-05   9.18693e-05
     1.15004e-05     -0.998093   1.02488e-05
     2.17347e-05   9.48591e-05     -0.997519

A2 =
         1.99794  -2.19977e-05  -9.40752e-05
    -1.14716e-05       1.99803  -9.20901e-06
    -2.12928e-05  -9.83956e-05       1.99744

B =
        0.063272   2.15505e-05    0.00160559
      0.00112818     0.0605629   -0.00085046
    -6.53409e-05    0.00246287     0.0772891

C =
        -105.838
         49.6097
        -3771.33

==================================================
x(k+2) = A1*x(k) + A2*x(k+1) + B*u(k) + C  [TLS v2 2-step + bias]
Dataset: 3'5V_all_axis_1.txt
RMSE (10-step test): 63.0097
--------------------------------------------------
A1 =
         -0.9981   0.000122585  -0.000129175
     5.90452e-05     -0.998135   0.000258765
    -0.000471752   0.000760301     -0.997585

A2 =
         1.99804  -0.000122562   0.000129756
    -5.87013e-05       1.99807  -0.000259202
      0.00047125  -0.000762287       1.99751

B =
       0.0624236  -0.000442052   -0.00161649
     0.000526989     0.0611837    0.00110821
    -0.000220577    0.00428184     0.0743826

C =
         27.7922
        -21.2492
        -3713.22
"""


def load_models(text=MODEL_TEXT):
    """Read all nine entries, including the two differently fitted TLS+bias models."""
    models = []
    for block in re.split(r"={10,}", text):
        equation = next(
            (line for line in block.splitlines() if line.startswith("x(k")), None
        )
        if equation is None:
            continue
        matrices = {}
        for match in re.finditer(
            r"^(A1|A2|A|B|C) =\s*\n((?:[ \t]+[^\n]+\n?){3})", block, re.M
        ):
            key, rows = match.groups()
            arr = np.array(
                [[float(v) for v in row.split()] for row in rows.splitlines()]
            )
            matrices[key] = arr.reshape(3) if key == "C" else arr
        if "B" not in matrices:
            raise ValueError(f"Missing B in {equation}")
        if "A" in matrices:
            current, previous = matrices["A"], np.zeros((3, 3))
        else:
            current, previous = matrices["A2"], matrices["A1"]
        plant = tuple(
            np.ascontiguousarray(a, dtype=np.float64)
            for a in (current, previous, matrices["B"], matrices.get("C", np.zeros(3)))
        )
        if any(a.shape != (3, 3) for a in plant[:3]) or plant[3].shape != (3,):
            raise ValueError(f"Invalid matrix shapes: {equation}")
        label = equation.split("[")[-1].rstrip("]")
        if "Dataset:" in block:
            label += " (3.5V dataset)"
        models.append((len(models) + 1, label, plant))
    if len(models) != 9:
        raise ValueError(f"Expected nine models, parsed {len(models)}")
    return models


def read_trajectory(path):
    data = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1):
        data = np.column_stack((data.reshape(-1), np.zeros((data.size, 2))))
    if data.ndim != 2 or data.shape[1] != 3 or len(data) < 3:
        raise ValueError(
            f"Trajectory must be (N,), (N,1) or (N,3), N>=3; got {data.shape}"
        )
    if not np.isfinite(data).all():
        raise ValueError("Trajectory contains nonfinite entries")
    return np.ascontiguousarray(data)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        help=".npy file; omitted: control_land's configured file",
    )
    parser.add_argument(
        "--models",
        type=int,
        nargs="+",
        choices=range(1, 10),
        default=list(range(1, 10)),
    )
    parser.add_argument(
        "--gains", type=float, nargs=7, default=ENGINEER_PARAMS, metavar="GAIN"
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--save-histories",
        action="store_true",
        help="Save full position/error/control arrays for each model",
    )
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()
    models = load_models()
    if args.list_models:
        for number, label, _ in models:
            print(f"{number}: {label}")
        return

    import matplotlib
    import torch

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from bayesian_pid import control_land as cl
    from bayesian_pid import metrics, sim

    if set(cl.TUNE_PARAMS) != set(PARAM_NAMES):
        raise ValueError(
            f"This script expects these seven tuned parameters: {PARAM_NAMES}"
        )
    overlap = set(cl.FIXED_PARAMS) & set(PARAM_NAMES)
    if overlap:
        raise ValueError(f"FIXED_PARAMS would override supplied gains: {overlap}")
    if not np.isfinite(args.gains).all():
        raise ValueError("All gains must be finite")
    trajectory_path = args.trajectory or Path(cl.DATA_PATH) / cl.TRAJECTORY_FILE
    trajectory = read_trajectory(trajectory_path)
    out = args.out or Path("plant_comparison_results") / datetime.now().strftime(
        "run_%Y%m%d_%H%M%S_%f"
    )
    out.mkdir(parents=True, exist_ok=True)
    gain_by_name = dict(zip(PARAM_NAMES, args.gains, strict=True))
    param_names = list(cl.TUNE_PARAMS)
    tensor = torch.tensor([[gain_by_name[k] for k in param_names]], dtype=cl.DTYPE)
    axes = cl.SCORE_AXES
    window = metrics.DEFAULT_WINDOW
    simulator = getattr(sim.simulate_pid_fast, "py_func", sim.simulate_pid_fast)
    config = {
        "trajectory": str(trajectory_path.resolve()),
        "trajectory_shape": list(trajectory.shape),
        "trajectory_sha256": hashlib.sha256(trajectory.tobytes()).hexdigest(),
        "gains": gain_by_name,
        "fixed_params": cl.FIXED_PARAMS,
        "MAX_OUTPUT": cl.MAX_OUTPUT,
        "MAX_INTEGRAL": cl.MAX_INTEGRAL,
        "DT": cl.DT,
        "DT_INV": cl.DT_INV,
        "SCORE_AXES": axes,
        "window": vars(window),
        "control_land_module": cl.__file__,
        "sim_module": sim.__file__,
        "simulator_sha256": hashlib.sha256(
            inspect.getsource(simulator).encode()
        ).hexdigest(),
        "models": args.models,
        "input_timing": "unchanged simulator; no additional input delay",
    }
    (out / "configuration.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2))
    # print(
    #     "NOTE: all axes retain branch controller behaviour; scoring axes do not disable actuation."
    # )
    # print(
    #     "NOTE: second-order input timing uses the existing simulator (see script docstring)"
    # )
    # print(
    #     "NOTE: default scoring window may not be a plateau for a different trajectory."
    # )
    rows = []
    selected = [m for m in models if m[0] in args.models]
    fig_all, ax_all = plt.subplots(
        len(selected), 1, figsize=(11, 3 * len(selected)), squeeze=False
    )
    t = np.arange(len(trajectory))
    for panel, (number, label, plant) in enumerate(selected):
        evaluate_fn = cl.build_sim_objective(trajectory, plant, param_names)
        with np.errstate(over="ignore", invalid="ignore"):
            scores, details = evaluate_fn(tensor, return_details=True)
            actual = details[0]["actual_pos"]
            controls = details[0]["control_hist"]
            error = (
                trajectory - actual
            )  # Includes final sample; avoid padded error histories.
            metrics.following_error_score(trajectory, actual, axes=axes)
            rms = metrics.following_error_rms(trajectory, actual, axes=axes)
            rms_x = metrics.following_error_rms(trajectory, actual, axes=(0,))
            full_x = float(np.sqrt(np.mean(error[:, 0] ** 2)))
        divergence = metrics.diverged_at(trajectory, actual, axes=axes)
        divergence_x = metrics.diverged_at(trajectory, actual, axes=(0,))
        finite = bool(np.isfinite(actual).all() and np.isfinite(controls).all())
        objective = float(scores.item())
        # if not np.isclose(objective, score_direct, equal_nan=True):
        #     raise RuntimeError(
        #         "Objective and direct score disagree: check the local validation.py"
        #     )
        row = {
            "model_id": number,
            "model": label,
            "status": "nonfinite"
            if not finite
            else ("diverged" if divergence is not None else "within_envelope"),
            "objective_score": objective,
            "plateau_rms_scored_axes": rms,
            "plateau_rms_x": rms_x,
            "full_trajectory_rms_x": full_x,
            "divergence_index_scored_axes": divergence,
            "divergence_index_x": divergence_x,
            "max_abs_u_x": float(np.max(np.abs(controls[:, 0]))),
            "max_abs_u_y": float(np.max(np.abs(controls[:, 1]))),
            "max_abs_u_z": float(np.max(np.abs(controls[:, 2]))),
        }
        rows.append(row)
        print(
            f"Model {number}: {label} | {row['status']} | x plateau RMS={rms_x:.8g} | score={objective:.8g} | divergence={divergence}",
            flush=True,
        )
        prefix = out / f"model_{number:02d}"
        if args.save_histories:
            np.savez_compressed(
                str(prefix) + "_histories.npz",
                trajectory=trajectory,
                actual_pos=actual,
                following_error=error,
                control_hist=controls,
            )
        fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        axs[0].plot(t, trajectory[:, 0], label="Demand x")
        axs[0].plot(t, actual[:, 0], label="Actual x", alpha=0.8)
        axs[0].set_ylabel("Position (input units)")
        axs[1].plot(t, error[:, 0], label="Demand - actual x")
        axs[1].set_ylabel("Following error")
        for j, name in enumerate("xyz"):
            axs[2].plot(t, controls[:, j], label=f"u_{name}")
        axs[2].set_ylabel("Control (model units)")
        axs[2].set_xlabel("Sample index")
        for ax in axs:
            ax.axvspan(
                int(len(t) * window.start_frac),
                int(len(t) * window.end_frac),
                color="grey",
                alpha=0.15,
            )
            if divergence is not None:
                ax.axvline(divergence, color="red", linestyle=":")
            ax.legend()
            ax.grid(alpha=0.3)
        fig.suptitle(
            f"Model {number}: {label}\nx plateau RMS={rms_x:.6g}; score={objective:.6g}; {row['status']}"
        )
        fig.tight_layout()
        fig.savefig(str(prefix) + "_tracking.png", dpi=160)
        if not args.show:
            plt.close(fig)
        ax = ax_all[panel, 0]
        ax.plot(t, error[:, 0], linewidth=0.8)
        ax.set_title(f"{number}: {label} | x plateau RMS={rms_x:.6g} | {row['status']}")
        ax.set_ylabel("X error")
        ax.grid(alpha=0.3)
        # Write after every completed model so partial results survive interruption.
        with (out / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    ax_all[-1, 0].set_xlabel("Sample index")
    fig_all.tight_layout()
    fig_all.savefig(out / "all_models_x_error.png", dpi=160)
    print(f"Results saved to {out.resolve()}")
    if args.show:
        plt.show()
    else:
        plt.close(fig_all)


if __name__ == "__main__":
    main()
