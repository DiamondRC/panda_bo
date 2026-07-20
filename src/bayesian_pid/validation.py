from typing import Any

import gpytorch
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from bayesian_pid.controller import PID
from bayesian_pid.metrics import following_error_rms
from bayesian_pid.sim import simulate_pid

# ============================================================
# Objective / evaluation function used in the Optimisation loop
# ============================================================


# def evaluate(pid_tensor, trajectory=pos_ref, return_details=False):
def evaluate(
    pid_tensor,
    trajectory,
    a1,
    a2,
    b,
    c,
    u_min,
    u_max,
    max_integral,
    param_names: list[str],
    fixed_params: dict[str, Any],
    return_details: bool = False,
) -> tuple[torch.Tensor, list[dict[str, Any]] | None]:
    """
    Evaluate PID parameters using the learned second-order linear dynamics.
    Objective = mean Euclidean tracking error.
    """
    if pid_tensor.ndim == 1:
        pid_tensor = pid_tensor.unsqueeze(0)

    pid_np = pid_tensor.detach().cpu().numpy()
    values = []
    details = []

    for i in range(pid_np.shape[0]):
        bo_params = pid_np[i]

        pid_kwargs: dict[str, Any] = dict(zip(param_names, bo_params, strict=False))
        pid_kwargs.update(fixed_params)
        pid_kwargs.update(
            {"dt": 1e-4, "u_min": u_min, "u_max": u_max, "integral_limit": max_integral}
        )

        controller = PID(**pid_kwargs)  # type: ignore[arg-type]

        actual_pos, error_hist, control_hist = simulate_pid(
            controller, trajectory, a1, a2, b, c
        )

        plateau_std = following_error_rms(trajectory[:, 0], actual_pos[:, 0])

        values.append(plateau_std)
        details.append(
            {
                "pid": {k: float(pid_kwargs[k]) for k in (*param_names, *fixed_params)},
                "plateau_std": float(plateau_std),
                "actual_pos": actual_pos,
                "error_hist": error_hist,
                "control_hist": control_hist,
            }
        )

    y = torch.tensor(values, dtype=torch.double).unsqueeze(-1)
    if return_details:
        return y, details
    else:
        return y, None


def fit_gp_with_fallback(gp, mll, adam_steps=100, adam_lr=0.05, verbose=True):
    """
    Fit an Exact GP using your original L-BFGS-B path, with the same Adam fallback.
    """

    try:
        fit_gpytorch_mll(mll)

    except Exception as e:
        if verbose:
            print(
                f"\n[Warning] L-BFGS-B failed ({type(e).__name__}: {e}). "
                f"Falling back to Adam."
            )

        mll.train()
        optimizer = torch.optim.Adam(gp.parameters(), lr=adam_lr)

        train_inputs = gp.train_inputs[0]
        train_targets = gp.train_targets

        for _ in range(adam_steps):
            optimizer.zero_grad()
            with (
                gpytorch.settings.cholesky_max_tries(15),
                gpytorch.settings.cholesky_jitter(1e-3),
            ):
                output = gp(train_inputs)
                loss = -mll(output, train_targets).sum()
                loss.backward()
            optimizer.step()

        mll.eval()

    return gp


def build_gp_model(train_x, train_y, dim, verbose=True):
    """
    Build and fit the GP model.

    This intentionally matches the model setup in the existing TuRBO code:
    - objective standardised outside the GP;
    - GaussianLikelihood with small noise constraint;
    - ScaleKernel(MaternKernel nu=2.5, ARD, same lengthscale constraint).
    """
    y_std = train_y.std(correction=0).clamp_min(1e-8)
    y = (train_y - train_y.mean()) / y_std

    likelihood = GaussianLikelihood(
        noise_constraint=Interval(
            1e-8, 1e-4
        )  # increase theupper bound to make GP_fit more stable
    )

    covar_module = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=dim,
            lengthscale_constraint=Interval(0.005, 4.0),
        )
    )

    gp = SingleTaskGP(
        train_x=train_x,
        train_y=y,
        covar_module=covar_module,
        likelihood=likelihood,
    )

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    gp = fit_gp_with_fallback(gp, mll, verbose=verbose)

    return gp, y


def _append_evaluations(eval_x_unit, eval_x_phys, eval_y, x_unit, x_phys, y):
    """
    Append possibly batched evaluations as individual rows for easier plotting/saving.
    """

    x_unit_cpu = x_unit.detach().cpu()
    x_phys_cpu = x_phys.detach().cpu()
    y_cpu = y.detach().cpu().view(-1)

    for i in range(y_cpu.numel()):
        eval_x_unit.append(x_unit_cpu[i].clone())
        eval_x_phys.append(x_phys_cpu[i].clone())
        eval_y.append(float(y_cpu[i].item()))


def _as_pid_vector(x, dim):
    """Return PID tensor/array as a flat length-3 numpy vector [Kp, Ki, Kd]."""
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.size != dim:
        raise ValueError(
            f"Expected PID vector with {dim} values, got shape {np.asarray(x).shape}."
        )
    return arr
