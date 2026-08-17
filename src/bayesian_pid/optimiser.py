import multiprocessing
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
import torch

# botorch and pandas ship no type information for these entry points.
from botorch.acquisition import (
    LogExpectedImprovement,  # pyright: ignore[reportUnknownVariableType]
)
from botorch.optim import optimize_acqf  # pyright: ignore[reportUnknownVariableType]
from numpy.typing import NDArray
from torch.quasirandom import SobolEngine

from bayesian_pid.utils import (
    DEFAULT_BO_THREADS,
    TurboState,
    set_seed,
    torch_thread_limit,
    unnormalize,
    update_state,
)
from bayesian_pid.validation import as_pid_vector, build_gp_model


def _append_evaluations(
    eval_x_unit: list[torch.Tensor],
    eval_x_phys: list[torch.Tensor],
    eval_y: list[float],
    x_unit: torch.Tensor,
    x_phys: torch.Tensor,
    y: torch.Tensor,
) -> None:
    """Append possibly batched evaluations as individual rows.

    Keeps the per-evaluation record flat, which is what the plots and CSVs want.
    """
    x_unit_cpu = x_unit.detach().cpu()
    x_phys_cpu = x_phys.detach().cpu()
    y_cpu = y.detach().cpu().view(-1)

    for i in range(y_cpu.numel()):
        eval_x_unit.append(x_unit_cpu[i].clone())
        eval_x_phys.append(x_phys_cpu[i].clone())
        eval_y.append(float(y_cpu[i].item()))


# ============================================================
# Warm-start selection
# ============================================================
def select_warm_start(
    result: dict[str, Any],
    k: int = 1,
    min_separation: float = 0.05,
    verbose: bool = True,
) -> torch.Tensor:
    """Pick k good, well-separated evaluated points to warm-start another run.

    Used to seed hardware BO from a completed simulation run, so every live
    evaluation is a gain set the simulation already vetted rather than a random
    draw. k=1 returns the single best point.

    Points are taken best-first, skipping any that lie within `min_separation`
    (Euclidean, unit cube) of one already chosen. Without that filter the top-k
    cluster on top of each other, leaving the GP's ARD lengthscales
    unidentifiable and the trust region arbitrarily shaped — which is the same
    degenerate fit a single point gives.

    Returns a (k, dim) tensor of unit-cube points.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")

    x_unit = result["eval_x_unit"]
    y = np.asarray(result["eval_y"], dtype=np.float64)
    if len(y) == 0:
        raise ValueError("result contains no evaluations to warm-start from")

    order = np.argsort(y)
    chosen: list[int] = []
    for idx in order:
        candidate = x_unit[idx]
        if all(
            float(torch.norm(candidate - x_unit[c]).item())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            >= min_separation
            for c in chosen
        ):
            chosen.append(int(idx))
        if len(chosen) == k:
            break

    if len(chosen) < k:
        # Not enough well-separated points; top up best-first regardless.
        if verbose:
            print(
                f"[warm start] only {len(chosen)} of {k} points are at least "
                f"{min_separation} apart; topping up with the next best."
            )
        for idx in order:
            if int(idx) not in chosen:
                chosen.append(int(idx))
            if len(chosen) == k:
                break

    if verbose:
        print(
            f"[warm start] selected {len(chosen)} point(s), "
            f"objective {y[chosen].min():.6g} to {y[chosen].max():.6g}"
        )
    return torch.stack([x_unit[i] for i in chosen])


def _warm_start_design(
    warm_start_x: torch.Tensor,
    n_points: int,
    radius: float,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: str | torch.device,
) -> torch.Tensor:
    """Build an initial design of n_points drawn only from the warm-start region.

    The first rows are the warm-start points themselves. Any shortfall is filled
    with uniform perturbations around them, clipped to the unit cube — never
    with a fresh draw over the full space, because on hardware that would mean
    commanding gain sets nothing has vetted.
    """
    ws = warm_start_x.to(dtype=dtype, device=device)
    if n_points <= ws.shape[0]:
        return ws[:n_points]

    extra = n_points - ws.shape[0]
    base = ws[torch.arange(extra, device=device) % ws.shape[0]]
    noise = (
        torch.rand(extra, ws.shape[1], generator=generator, dtype=dtype, device=device)
        * 2.0
        - 1.0
    ) * radius
    return torch.cat([ws, torch.clamp(base + noise, 0.0, 1.0)], dim=0)


# ============================================================
# One BO loop to find the best PID parameters
# ============================================================
def run_one_optimization(
    evaluate_fn: Callable[..., tuple[torch.Tensor, Any]],
    bounds_phys: torch.Tensor,
    dim: int,
    dtype: torch.dtype = torch.double,
    device: str | torch.device = "cpu",
    method: str = "turbo",
    seed: int = 1,
    total_budget: int = 300,
    n_init: int = 10,
    verbose: bool = True,
    warm_start_x: torch.Tensor | None = None,
    restart_policy: str = "warm_region",
    warm_restart_radius: float = 0.2,
    progress_fn: Callable[..., None] | None = None,
    torch_threads: int | None = None,
) -> dict[str, Any]:
    """Run one optimisation trial. See `_run_one_optimization` for the details.

    torch_threads:
        Cap on torch's intra-op thread pool for this run. None uses
        `DEFAULT_BO_THREADS`, which is 1 so that a seeded run replays exactly.
        Anything above 1 changes the search path, not just the speed — see
        `utils.DEFAULT_BO_THREADS`. Parallelise over seeds with
        `run_mc_optimisation(n_jobs=...)` instead.
    """
    threads = DEFAULT_BO_THREADS if torch_threads is None else torch_threads
    with torch_thread_limit(threads):
        return _run_one_optimization(
            evaluate_fn,
            bounds_phys,
            dim=dim,
            dtype=dtype,
            device=device,
            method=method,
            seed=seed,
            total_budget=total_budget,
            n_init=n_init,
            verbose=verbose,
            warm_start_x=warm_start_x,
            restart_policy=restart_policy,
            warm_restart_radius=warm_restart_radius,
            progress_fn=progress_fn,
        )


def _run_one_optimization(
    evaluate_fn: Callable[..., tuple[torch.Tensor, Any]],
    bounds_phys: torch.Tensor,
    dim: int,
    dtype: torch.dtype = torch.double,
    device: str | torch.device = "cpu",
    method: str = "turbo",
    seed: int = 1,
    total_budget: int = 300,
    n_init: int = 10,
    verbose: bool = True,
    warm_start_x: torch.Tensor | None = None,
    restart_policy: str = "warm_region",
    warm_restart_radius: float = 0.2,
    progress_fn: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """
    Run one optimisation trial.

    Returns a dictionary containing:
        best_y, best_x_unit, best_x_phys,
        eval_x_unit, eval_x_phys, eval_y,
        best_curve, history.

    The best_curve has length equal to the number of evaluations and is directly
    suitable for averaging across MC seeds.

    warm_start_x:
        (n, dim) unit-cube points to seed the search, e.g. from
        select_warm_start() on a completed simulation run. When supplied, no
        initial design is ever drawn from the full space — see restart_policy.
    restart_policy:
        What a TuRBO restart does when warm_start_x was supplied.
        "warm_region" — re-seed from the warm-start points, perturbed within
            warm_restart_radius. Keeps TuRBO's ability to escape a local
            minimum without commanding unvetted gains.
        "stop"        — halt at the first trust-region collapse and return the
            best found so far.
        Without warm_start_x, restarts draw Sobol points over the full space as
        before; that is only safe in simulation.
    """
    method = method.lower()
    if method in ["bo", "normal_bo"]:
        method = "standard_bo"
    if method not in ["turbo", "standard_bo"]:
        raise ValueError(f"Unknown method: {method}. Use 'turbo' or 'standard_bo'.")
    if restart_policy not in ["warm_region", "stop"]:
        raise ValueError(
            f"Unknown restart_policy: {restart_policy}. Use 'warm_region' or 'stop'."
        )

    set_seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    eval_count = 0
    restart_id = 0

    global_best_y = float("inf")
    global_best_x_unit: torch.Tensor | None = None
    global_best_x_phys: torch.Tensor | None = None

    eval_x_unit: list[torch.Tensor] = []
    eval_x_phys: list[torch.Tensor] = []
    eval_y: list[float] = []
    history: list[dict[str, Any]] = []

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

            if warm_start_x is not None:
                if restart_id > 1 and restart_policy == "stop":
                    if verbose:
                        print(
                            "Trust region collapsed and restart_policy='stop'; "
                            "returning the best point found so far."
                        )
                    break
                # Honour n_init, but never by drawing from the full space: a
                # warm-started run is a hardware run, and an unvetted gain set
                # goes straight to the device.
                n_init_this = max(n_init_this, warm_start_x.shape[0])
                n_init_this = min(n_init_this, total_budget - eval_count)
                train_x = _warm_start_design(
                    warm_start_x,
                    n_init_this,
                    warm_restart_radius,
                    generator,
                    dtype,
                    device,
                )
                if verbose and train_x.shape[0] < n_init:
                    print(
                        f"Initial design reduced from n_init={n_init} to "
                        f"{train_x.shape[0]} by the remaining budget."
                    )
            else:
                sobol = SobolEngine(
                    dimension=dim,
                    scramble=True,
                    seed=seed + restart_id,
                )
                train_x = sobol.draw(n_init_this).to(dtype=dtype, device=device)

            n_init_this = train_x.shape[0]
            train_x_phys = unnormalize(train_x, bounds_phys)
            train_y_list: list[torch.Tensor] = []
            running_best = global_best_y
            if verbose:
                print(f"Initial design: {n_init_this} point(s)")
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

                candidate, _acq_value = optimize_acqf(
                    acq_function=acq,
                    bounds=torch.stack([tr_lb, tr_ub]),
                    q=1,
                    num_restarts=20,
                    raw_samples=512,
                )

                candidate_phys = unnormalize(candidate, bounds_phys)
                _t0 = time.time()
                y_candidate, _details = evaluate_fn(candidate_phys, return_details=True)
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

        if warm_start_x is not None:
            # Same rule as TuRBO: honour n_init, but only from the warm-start
            # region, never from a fresh draw over the full space.
            n_init_this = min(max(n_init_this, warm_start_x.shape[0]), total_budget)
            train_x = _warm_start_design(
                warm_start_x,
                n_init_this,
                warm_restart_radius,
                generator,
                dtype,
                device,
            )
        else:
            # Use seed + 1 to match the first TuRBO restart initial design.
            sobol = SobolEngine(
                dimension=dim,
                scramble=True,
                seed=seed + 1,
            )
            train_x = sobol.draw(n_init_this).to(dtype=dtype, device=device)

        n_init_this = train_x.shape[0]
        train_x_phys = unnormalize(train_x, bounds_phys)
        train_y_list: list[torch.Tensor] = []
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

            candidate, _acq_value = optimize_acqf(
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
            y_candidate, _details = evaluate_fn(candidate_phys, return_details=True)
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

    if global_best_x_unit is None or global_best_x_phys is None:
        raise RuntimeError(
            f"No evaluation completed in {total_budget} budget, so there is no "
            f"best point to return. Check total_budget and n_init."
        )

    eval_y_array = np.asarray(eval_y, dtype=np.float64)
    best_curve = np.minimum.accumulate(eval_y_array)

    result: dict[str, Any] = {
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
def _mc_worker(job: dict[str, Any]) -> dict[str, Any]:
    """Run one MC seed in a worker process. Module level so it is picklable."""
    return run_one_optimization(**job)


def run_mc_optimisation(
    evaluate_fn: Callable[..., tuple[torch.Tensor, Any]],
    bounds_phys: torch.Tensor,
    dim: int,
    method: str = "turbo",
    dtype: torch.dtype = torch.double,
    device: str | torch.device = "cpu",
    n_runs: int = 10,
    seed0: int = 1,
    total_budget: int = 300,
    n_init: int = 10,
    verbose: bool = False,
    save_csv: bool = True,
    results_dir: str = "bo_mc_results",
    param_names: list[str] | None = None,
    progress_fn: Callable[..., None] | None = None,
    n_jobs: int = 1,
    torch_threads: int | None = None,
) -> dict[str, Any]:
    """
    Run repeated optimisation trials with different seeds and aggregate the curves.

    One-line usage example:
        mc_results = run_mc_optimisation(method=method, n_runs=10, seed0=1)

    n_jobs:
        Seeds to run concurrently, as separate processes. The seeds are
        independent, so this is close to linear in cores. `evaluate_fn` and
        `bounds_phys` are pickled to each worker, so `evaluate_fn` must be
        picklable — a module-level function or a `functools.partial` of one is
        fine, a lambda or closure is not. `progress_fn` cannot cross a process
        boundary and is ignored when n_jobs > 1.
    torch_threads:
        Threads per worker, defaulting to `DEFAULT_BO_THREADS` (1). Leave it
        there: one thread per worker is what makes each seed replay exactly, and
        it also stops `n_jobs` workers collectively oversubscribing the machine.
        `n_jobs` itself does NOT change results — parallel and sequential MC are
        bit-identical at the same thread count, and a test pins that.

    Returns:
        run_results       list of individual run dictionaries;
        seeds             array of seeds used;
        best_curves        shape (n_runs, total_budget), best-so-far objective curves;
        mean_best_curve    MC mean of best-so-far curves;
        std_best_curve     MC standard deviation;
        sem_best_curve     MC standard error of the mean;
        final_best_values  final best objective from each run.
    """
    # CSV columns follow the caller's parameter order rather than a fixed list,
    # so reordering or resizing TUNE_PARAMS cannot silently mislabel them.
    if param_names is None:
        names = [f"p{i}" for i in range(dim)]
    else:
        names = list(param_names)
        if len(names) != dim:
            raise ValueError(f"param_names has {len(names)} entries but dim is {dim}")

    if n_jobs < 1:
        raise ValueError(f"n_jobs must be at least 1, got {n_jobs}")
    n_jobs = min(n_jobs, n_runs)
    if torch_threads is None:
        torch_threads = DEFAULT_BO_THREADS

    seeds = np.arange(seed0, seed0 + n_runs, dtype=int)
    run_results: list[dict[str, Any]] = []
    best_curves: list[NDArray[np.float64]] = []

    def job_for(run_seed: int) -> dict[str, Any]:
        return {
            "evaluate_fn": evaluate_fn,
            "bounds_phys": bounds_phys,
            "dim": dim,
            "method": method,
            "dtype": dtype,
            "device": device,
            "seed": int(run_seed),
            "total_budget": total_budget,
            "n_init": n_init,
            "verbose": verbose,
            "torch_threads": torch_threads,
        }

    if n_jobs == 1:
        for run_idx, run_seed in enumerate(seeds):
            print(
                f"\n========== MC run {run_idx + 1}/{n_runs} | "
                f"method={method} | seed={run_seed} =========="
            )
            result = run_one_optimization(
                **job_for(int(run_seed)), progress_fn=progress_fn
            )
            run_results.append(result)
            best_curves.append(result["best_curve"])
            print(
                f"MC run {run_idx + 1}/{n_runs} finished | "
                f"best={result['best_y']:.6f} | "
                f"PID={as_pid_vector(result['best_x_phys'], dim)}"
            )
    else:
        total_threads = n_jobs * torch_threads
        cpus = os.cpu_count() or 1
        if total_threads > cpus:
            print(
                f"[Warning] n_jobs={n_jobs} x torch_threads={torch_threads} = "
                f"{total_threads} threads on {cpus} CPUs. Oversubscription is "
                f"what made the sequential loop slow; lower one of them."
            )
        if progress_fn is not None:
            print("[Note] progress_fn is not called when n_jobs > 1.")
        print(
            f"\n========== {n_runs} MC runs | method={method} | "
            f"seeds {seeds[0]}-{seeds[-1]} | {n_jobs} workers x "
            f"{torch_threads} thread(s) =========="
        )
        # 'spawn' rather than the Linux default 'fork': the parent has already
        # initialised torch's thread pool and numba's compiled cache, and
        # forking that state into a worker is a documented source of hangs.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=context) as pool:
            futures = {pool.submit(_mc_worker, job_for(int(s))): int(s) for s in seeds}
            by_seed: dict[int, dict[str, Any]] = {}
            for done in as_completed(futures):
                run_seed = futures[done]
                result = done.result()
                by_seed[run_seed] = result
                print(
                    f"MC seed {run_seed} finished "
                    f"({len(by_seed)}/{n_runs}) | best={result['best_y']:.6f} | "
                    f"PID={as_pid_vector(result['best_x_phys'], dim)}",
                    flush=True,
                )
        # Completion order is nondeterministic; report in seed order regardless.
        run_results = [by_seed[int(s)] for s in seeds]
        best_curves = [r["best_curve"] for r in run_results]

    # In normal use all curves have length total_budget. This trim protects against
    # accidental early stopping or a changed budget in future experiments.
    min_len = min(len(curve) for curve in best_curves)
    if min_len != total_budget:
        print(
            f"[Warning] Some curves are shorter than total_budget. "
            f"Aggregating first {min_len} evaluations."
        )

    curve_matrix: NDArray[np.float64] = np.vstack(
        [np.asarray(curve[:min_len], dtype=np.float64) for curve in best_curves]
    )
    mean_best_curve = curve_matrix.mean(axis=0)
    std_best_curve = (
        curve_matrix.std(axis=0, ddof=1) if n_runs > 1 else np.zeros(min_len)
    )
    sem_best_curve = std_best_curve / np.sqrt(n_runs)
    final_best_values = curve_matrix[:, -1]

    mc_results: dict[str, Any] = {
        "method": method,
        "seeds": seeds,
        "run_results": run_results,
        "best_curves": curve_matrix,
        "mean_best_curve": mean_best_curve,
        "std_best_curve": std_best_curve,
        "sem_best_curve": sem_best_curve,
        "final_best_values": final_best_values,
    }

    if save_csv:
        os.makedirs(results_dir, exist_ok=True)

        summary_cols: dict[str, Any] = {
            "run": np.arange(1, n_runs + 1),
            "seed": seeds,
            "method": method,
            "final_best": final_best_values,
        }
        best_vectors = [as_pid_vector(r["best_x_phys"], dim) for r in run_results]
        for j, name in enumerate(names):
            summary_cols[f"best_{name}"] = [vec[j] for vec in best_vectors]
        summary_df = pd.DataFrame(summary_cols)
        summary_path = os.path.join(results_dir, f"{method}_mc_summary.csv")
        summary_df.to_csv(summary_path, index=False)  # pyright: ignore[reportUnknownMemberType]

        curve_df = pd.DataFrame({"eval": np.arange(1, min_len + 1)})
        for i, run_seed in enumerate(seeds):
            curve_df[f"seed_{run_seed}"] = curve_matrix[i]
        curve_df["mean"] = mean_best_curve
        curve_df["std"] = std_best_curve
        curve_df["sem"] = sem_best_curve
        curve_path = os.path.join(results_dir, f"{method}_mc_best_curves.csv")
        curve_df.to_csv(curve_path, index=False)  # pyright: ignore[reportUnknownMemberType]

        all_eval_rows: list[dict[str, Any]] = []
        for run_idx, r in enumerate(run_results):
            x_phys_np = r["eval_x_phys"].numpy()
            for eval_idx, y_val in enumerate(r["eval_y"]):
                row: dict[str, Any] = {
                    "run": run_idx + 1,
                    "seed": int(r["seed"]),
                    "method": r["method"],
                    "eval": eval_idx + 1,
                }
                for j, name in enumerate(names):
                    row[name] = x_phys_np[eval_idx, j]
                row["obj"] = y_val
                row["best_so_far"] = r["best_curve"][eval_idx]
                all_eval_rows.append(row)
        all_eval_path = os.path.join(results_dir, f"{method}_mc_all_evaluations.csv")
        pd.DataFrame(all_eval_rows).to_csv(  # pyright: ignore[reportUnknownMemberType]
            all_eval_path, index=False
        )

        print(f"\nSaved MC summary to: {summary_path}")
        print(f"Saved MC best curves to: {curve_path}")
        print(f"Saved all evaluations to: {all_eval_path}")

    return mc_results
