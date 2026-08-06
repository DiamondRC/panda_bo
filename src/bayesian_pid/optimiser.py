import os
import time

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from torch.quasirandom import SobolEngine

from bayesian_pid.utils import TurboState, set_seed, unnormalize, update_state
from bayesian_pid.validation import _append_evaluations, _as_pid_vector, build_gp_model


# ============================================================
# One BO loop to find the best PID parameters
# ============================================================
def run_one_optimization(
    evaluate_fn,
    bounds_phys,
    dim,
    dtype=torch.double,
    device="cpu",
    method="turbo",
    seed=1,
    total_budget=300,
    n_init=10,
    verbose=True,
    warm_start_x=None,
    progress_fn=None,
):
    """
    Run one optimisation trial.

    Returns a dictionary containing:
        best_y, best_x_unit, best_x_phys,
        eval_x_unit, eval_x_phys, eval_y,
        best_curve, history.

    The best_curve has length equal to the number of evaluations and is directly
    suitable for averaging across MC seeds.
    """
    method = method.lower()
    if method in ["bo", "normal_bo"]:
        method = "standard_bo"
    if method not in ["turbo", "standard_bo"]:
        raise ValueError(f"Unknown method: {method}. Use 'turbo' or 'standard_bo'.")

    set_seed(seed)

    eval_count = 0
    restart_id = 0

    global_best_y = float("inf")
    global_best_x_unit = None
    global_best_x_phys = None

    eval_x_unit = []
    eval_x_phys = []
    eval_y = []
    history = []

    if method == "turbo":
        # ========================================================
        # Restart TuRBO BO loop
        # ========================================================
        while eval_count < total_budget:
            restart_id += 1
            if verbose:
                print(f"\nStarting TuRBO restart {restart_id} | seed={seed}")

            # ----------------------------------------------------
            # Initial design for this restart
            # ----------------------------------------------------
            n_init_this = min(n_init, total_budget - eval_count)

            if warm_start_x is not None and restart_id == 1:
                train_x = warm_start_x.to(dtype=dtype, device=device)
                n_init_this = train_x.shape[0]
            else:
                sobol = SobolEngine(
                    dimension=dim,
                    scramble=True,
                    seed=seed + restart_id,
                )
                train_x = sobol.draw(n_init_this).to(dtype=dtype, device=device)

            train_x_phys = unnormalize(train_x, bounds_phys)
            train_y_list = []
            running_best = global_best_y
            print(f"n_init_this:{n_init_this}")
            print(f"running_best: {running_best}")
            for i in range(n_init_this):
                _t0 = time.time()
                y_i, _ = evaluate_fn(train_x_phys[i : i + 1])
                _eval_time = time.time() - _t0
                train_y_list.append(y_i)
                eval_count += 1
                running_best = min(running_best, float(y_i.item()))
                if progress_fn is not None:
                    progress_fn(
                        eval_count,
                        total_budget,
                        train_x_phys[i : i + 1],
                        float(y_i.item()),
                        running_best,
                        {
                            "phase": "init",
                            "restart": restart_id,
                            "eval_time": _eval_time,
                        },
                    )
            train_y = torch.cat(train_y_list, dim=0)
            _append_evaluations(
                eval_x_unit, eval_x_phys, eval_y, train_x, train_x_phys, train_y
            )

            best_idx = torch.argmin(train_y)
            restart_best_y = train_y[best_idx].item()
            restart_best_x_unit = train_x[best_idx].detach().clone()
            restart_best_x_phys = unnormalize(restart_best_x_unit, bounds_phys)

            if restart_best_y < global_best_y:
                global_best_y = restart_best_y
                global_best_x_unit = restart_best_x_unit.clone()
                global_best_x_phys = restart_best_x_phys.clone()

            state = TurboState(
                dim=dim,
                batch_size=1,
                best_value=train_y.min().item(),
            )

            if verbose:
                print(
                    f"Restart {restart_id} initial best: {train_y.min().item():.6f} | "
                    f"Global best: {global_best_y:.6f}"
                )

            # ----------------------------------------------------
            # Local TuRBO loop
            # ----------------------------------------------------
            while (not state.restart_triggered) and (eval_count < total_budget):
                if verbose:
                    print(
                        f"Eval {eval_count}/{total_budget} | "
                        f"Restart {restart_id} | "
                        f"Restart best: {train_y.min().item():.6f} | "
                        f"Global best: {global_best_y:.6f} | "
                        f"TR length: {state.length:.4f}"
                    )

                gp, y = build_gp_model(train_x, train_y, dim, verbose=verbose)

                acq = LogExpectedImprovement(
                    model=gp,
                    best_f=y.min(),
                    maximize=False,
                )

                # Trust region centre: best point in this restart
                x_center = train_x[y.argmin(), :].clone()

                # Trust region shape from GP lengthscales
                weights = gp.covar_module.base_kernel.lengthscale.squeeze().detach()
                weights = weights / weights.mean()
                weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

                tr_lb = torch.clamp(
                    x_center - weights * state.length / 2.0,
                    0.0,
                    1.0,
                )
                tr_ub = torch.clamp(
                    x_center + weights * state.length / 2.0,
                    0.0,
                    1.0,
                )

                candidate, acq_value = optimize_acqf(
                    acq_function=acq,
                    bounds=torch.stack([tr_lb, tr_ub]),
                    q=1,
                    num_restarts=20,
                    raw_samples=512,
                )

                candidate_phys = unnormalize(candidate, bounds_phys)
                _t0 = time.time()
                y_candidate, details = evaluate_fn(candidate_phys, return_details=True)
                _eval_time = time.time() - _t0

                if verbose:
                    print("Candidate unit:", candidate.detach().cpu().numpy())
                    print("Candidate PID:", candidate_phys.detach().cpu().numpy())
                    print("Candidate value:", y_candidate.item())

                # Update TuRBO state using raw objective
                state = update_state(state, y_candidate)

                train_x = torch.cat([train_x, candidate], dim=0)
                train_y = torch.cat([train_y, y_candidate], dim=0)

                eval_count += 1
                _append_evaluations(
                    eval_x_unit,
                    eval_x_phys,
                    eval_y,
                    candidate,
                    candidate_phys,
                    y_candidate,
                )

                if y_candidate.item() < global_best_y:
                    global_best_y = y_candidate.item()
                    global_best_x_unit = candidate.detach().clone()
                    global_best_x_phys = candidate_phys.detach().clone()

                if progress_fn is not None:
                    progress_fn(
                        eval_count,
                        total_budget,
                        candidate_phys,
                        float(y_candidate.item()),
                        global_best_y,
                        {
                            "phase": "bo",
                            "restart": restart_id,
                            "tr_length": float(state.length),
                            "eval_time": _eval_time,
                        },
                    )

                history.append(
                    {
                        "seed": seed,
                        "method": "turbo",
                        "eval": eval_count,
                        "restart": restart_id,
                        "candidate_unit": candidate.detach().cpu().numpy().tolist(),
                        "candidate_pid": candidate_phys.detach().cpu().numpy().tolist(),
                        "obj": float(y_candidate.item()),
                        "global_best": float(global_best_y),
                        "tr_length": float(state.length),
                    }
                )

            if verbose:
                print(
                    f"Restart {restart_id} finished. "
                    f"Best in this restart: {train_y.min().item():.6f}. "
                    f"Global best so far: {global_best_y:.6f}."
                )

    elif method == "standard_bo":
        # ========================================================
        # Standard BO loop over the full unit box
        # ========================================================
        if verbose:
            print(f"\nStarting standard BO without trust region | seed={seed}")

        n_init_this = min(n_init, total_budget)

        # Use seed + 1 to match the first TuRBO restart initial design.
        sobol = SobolEngine(
            dimension=dim,
            scramble=True,
            seed=seed + 1,
        )

        if warm_start_x is not None:
            train_x = warm_start_x.to(dtype=dtype, device=device)
            n_init_this = train_x.shape[0]
        else:
            train_x = sobol.draw(n_init_this).to(dtype=dtype, device=device)
        train_x_phys = unnormalize(train_x, bounds_phys)
        train_y_list = []
        running_best = float("inf")
        for i in range(n_init_this):
            _t0 = time.time()
            y_i, _ = evaluate_fn(train_x_phys[i : i + 1])
            _eval_time = time.time() - _t0
            train_y_list.append(y_i)
            eval_count += 1
            running_best = min(running_best, float(y_i.item()))
            if progress_fn is not None:
                progress_fn(
                    eval_count,
                    total_budget,
                    train_x_phys[i : i + 1],
                    float(y_i.item()),
                    running_best,
                    {"phase": "init", "eval_time": _eval_time},
                )
        train_y = torch.cat(train_y_list, dim=0)
        _append_evaluations(
            eval_x_unit, eval_x_phys, eval_y, train_x, train_x_phys, train_y
        )

        best_idx = torch.argmin(train_y)
        global_best_y = train_y[best_idx].item()
        global_best_x_unit = train_x[best_idx].detach().clone()
        global_best_x_phys = unnormalize(global_best_x_unit, bounds_phys)

        if verbose:
            print(
                f"Initial best: {global_best_y:.6f} | "
                f"Initial evaluations: {eval_count}/{total_budget}"
            )

        while eval_count < total_budget:
            if verbose:
                print(
                    f"Eval {eval_count}/{total_budget} | "
                    f"Current best: {train_y.min().item():.6f} | "
                    f"Global best: {global_best_y:.6f}"
                )

            gp, y = build_gp_model(train_x, train_y, dim, verbose=verbose)

            acq = LogExpectedImprovement(
                model=gp,
                best_f=y.min(),
                maximize=False,
            )

            candidate, acq_value = optimize_acqf(
                acq_function=acq,
                bounds=torch.stack(
                    [
                        torch.zeros(dim, dtype=dtype, device=device),
                        torch.ones(dim, dtype=dtype, device=device),
                    ]
                ),
                q=1,
                num_restarts=20,
                raw_samples=512,
            )

            candidate_phys = unnormalize(candidate, bounds_phys)
            _t0 = time.time()
            y_candidate, details = evaluate_fn(candidate_phys, return_details=True)
            _eval_time = time.time() - _t0

            if verbose:
                print("Candidate unit:", candidate.detach().cpu().numpy())
                print("Candidate PID:", candidate_phys.detach().cpu().numpy())
                print("Candidate value:", y_candidate.item())

            train_x = torch.cat([train_x, candidate], dim=0)
            train_y = torch.cat([train_y, y_candidate], dim=0)

            eval_count += 1
            _append_evaluations(
                eval_x_unit, eval_x_phys, eval_y, candidate, candidate_phys, y_candidate
            )

            if y_candidate.item() < global_best_y:
                global_best_y = y_candidate.item()
                global_best_x_unit = candidate.detach().clone()
                global_best_x_phys = candidate_phys.detach().clone()

            if progress_fn is not None:
                progress_fn(
                    eval_count,
                    total_budget,
                    candidate_phys,
                    float(y_candidate.item()),
                    global_best_y,
                    {"phase": "bo", "eval_time": _eval_time},
                )

            history.append(
                {
                    "seed": seed,
                    "method": "standard_bo",
                    "eval": eval_count,
                    "candidate_unit": candidate.detach().cpu().numpy().tolist(),
                    "candidate_pid": candidate_phys.detach().cpu().numpy().tolist(),
                    "obj": float(y_candidate.item()),
                    "global_best": float(global_best_y),
                }
            )

        if verbose:
            print(
                f"Standard BO finished. "
                f"Best value: {train_y.min().item():.6f}. "
                f"Global best: {global_best_y:.6f}."
            )

    eval_y_array = np.asarray(eval_y, dtype=np.float64)
    best_curve = np.minimum.accumulate(eval_y_array)

    result = {
        "method": method,
        "seed": seed,
        "best_y": float(global_best_y),
        "best_x_unit": global_best_x_unit.detach().cpu().reshape(-1),
        "best_x_phys": global_best_x_phys.detach().cpu().reshape(-1),
        "eval_x_unit": torch.stack(eval_x_unit),
        "eval_x_phys": torch.stack(eval_x_phys),
        "eval_y": eval_y_array,
        "best_curve": best_curve,
        "history": history,
    }

    return result


# ============================================================
# MC runs of BO loops
# ============================================================
def run_mc_optimisation(
    evaluate_fn,
    bounds_phys,
    dim,
    method="turbo",
    dtype=torch.double,
    device="cpu",
    n_runs=10,
    seed0=1,
    total_budget=300,
    n_init=10,
    verbose=False,
    save_csv=True,
    results_dir="bo_mc_results",
    progress_fn=None,
):
    """
    Run repeated optimisation trials with different seeds and aggregate the curves.

    One-line usage example:
        mc_results = run_mc_optimisation(method=method, n_runs=10, seed0=1)

    Returns:
        run_results       list of individual run dictionaries;
        seeds             array of seeds used;
        best_curves        shape (n_runs, total_budget), best-so-far objective curves;
        mean_best_curve    MC mean of best-so-far curves;
        std_best_curve     MC standard deviation;
        sem_best_curve     MC standard error of the mean;
        final_best_values  final best objective from each run.
    """
    seeds = np.arange(seed0, seed0 + n_runs, dtype=int)
    run_results = []
    best_curves = []

    for run_idx, run_seed in enumerate(seeds):
        print(
            f"\n========== MC run {run_idx + 1}/{n_runs} | "
            f"method={method} | seed={run_seed} =========="
        )
        result = run_one_optimization(
            evaluate_fn,
            bounds_phys,
            dim=dim,
            method=method,
            dtype=dtype,
            device=device,
            seed=int(run_seed),
            total_budget=total_budget,
            n_init=n_init,
            verbose=verbose,
            progress_fn=progress_fn,
        )
        run_results.append(result)
        best_curves.append(result["best_curve"])
        print(
            f"MC run {run_idx + 1}/{n_runs} finished | "
            f"best={result['best_y']:.6f} | "
            f"PID={_as_pid_vector(result['best_x_phys'], dim)}"
        )

    # In normal use all curves have length total_budget. This trim protects against
    # accidental early stopping or a changed budget in future experiments.
    min_len = min(len(curve) for curve in best_curves)
    if min_len != total_budget:
        print(
            f"[Warning] Some curves are shorter than total_budget. "
            f"Aggregating first {min_len} evaluations."
        )

    best_curves = np.vstack([curve[:min_len] for curve in best_curves])
    mean_best_curve = best_curves.mean(axis=0)
    std_best_curve = (
        best_curves.std(axis=0, ddof=1) if n_runs > 1 else np.zeros(min_len)
    )
    sem_best_curve = std_best_curve / np.sqrt(n_runs)
    final_best_values = best_curves[:, -1]

    mc_results = {
        "method": method,
        "seeds": seeds,
        "run_results": run_results,
        "best_curves": best_curves,
        "mean_best_curve": mean_best_curve,
        "std_best_curve": std_best_curve,
        "sem_best_curve": sem_best_curve,
        "final_best_values": final_best_values,
    }

    if save_csv:
        os.makedirs(results_dir, exist_ok=True)

        summary_df = pd.DataFrame(
            {
                "run": np.arange(1, n_runs + 1),
                "seed": seeds,
                "method": method,
                "final_best": final_best_values,
                "best_Kp": [
                    _as_pid_vector(r["best_x_phys"], dim)[0] for r in run_results
                ],
                "best_Kv": [
                    _as_pid_vector(r["best_x_phys"], dim)[1] for r in run_results
                ],
                "best_Ki": [
                    _as_pid_vector(r["best_x_phys"], dim)[2] for r in run_results
                ],
                "best_Kvff": [
                    _as_pid_vector(r["best_x_phys"], dim)[3] for r in run_results
                ],
                "best_Kaff": [
                    _as_pid_vector(r["best_x_phys"], dim)[4] for r in run_results
                ],
                "best_Kpff1": [
                    _as_pid_vector(r["best_x_phys"], dim)[5] for r in run_results
                ],
                "best_Kpff0": [
                    _as_pid_vector(r["best_x_phys"], dim)[6] for r in run_results
                ],
            }
        )
        summary_path = os.path.join(results_dir, f"{method}_mc_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        curve_df = pd.DataFrame({"eval": np.arange(1, min_len + 1)})
        for i, run_seed in enumerate(seeds):
            curve_df[f"seed_{run_seed}"] = best_curves[i]
        curve_df["mean"] = mean_best_curve
        curve_df["std"] = std_best_curve
        curve_df["sem"] = sem_best_curve
        curve_path = os.path.join(results_dir, f"{method}_mc_best_curves.csv")
        curve_df.to_csv(curve_path, index=False)

        all_eval_rows = []
        for run_idx, r in enumerate(run_results):
            x_phys_np = r["eval_x_phys"].numpy()
            for eval_idx, y_val in enumerate(r["eval_y"]):
                all_eval_rows.append(
                    {
                        "run": run_idx + 1,
                        "seed": int(r["seed"]),
                        "method": r["method"],
                        "eval": eval_idx + 1,
                        "Kp": x_phys_np[eval_idx, 0],
                        "Kv": x_phys_np[eval_idx, 1],
                        "Ki": x_phys_np[eval_idx, 2],
                        "Kvff": x_phys_np[eval_idx, 3],
                        "Kaff": x_phys_np[eval_idx, 4],
                        "Kpff1": x_phys_np[eval_idx, 5],
                        "Kpff0": x_phys_np[eval_idx, 6],
                        "obj": y_val,
                        "best_so_far": r["best_curve"][eval_idx],
                    }
                )
        all_eval_path = os.path.join(results_dir, f"{method}_mc_all_evaluations.csv")
        pd.DataFrame(all_eval_rows).to_csv(all_eval_path, index=False)

        print(f"\nSaved MC summary to: {summary_path}")
        print(f"Saved MC best curves to: {curve_path}")
        print(f"Saved all evaluations to: {all_eval_path}")

    return mc_results
