"""PandABlocks hardware evaluator — PCAP streaming pattern."""

import asyncio
from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import Arm, Disarm, Put
from pandablocks.responses import EndData, FrameData, ReadyData

from bayesian_pid.metrics import following_error_score
from bayesian_pid.panda import PANDA_PID_FIELDS, quantise
from bayesian_pid.plotting import plot_hardware_trajectory


def _as_field_tuple(field: str | Sequence[str]) -> tuple[str, ...]:
    """Accept a single PCAP field name or a sequence of them."""
    if isinstance(field, str):
        return (field,)
    return tuple(field)


class HardwareEvaluator:
    def __init__(
        self,
        host: str,
        pid_fields: dict[str, str],
        trajectory: list[str],
        fixed_fields: dict[str, float] | None = None,
        pgen_block: str = "PGEN",
        pid_block: str = "BRETT_PID",
        setpoint_field: str | Sequence[str] | None = None,
        position_field: str | Sequence[str] = "INENC1.VAL.Value",
        axes: Sequence[int] | None = None,
        timeout_s: float = 60.0,
        debug_plot_path: str | None = None,
    ):
        """
        pid_fields: mapping from param name to PandA field, e.g.
            {"kp": "BRETT_PID.KP", "kv": "BRETT_PID.KV", "ki": "BRETT_PID.KI"}
        trajectory: pre-scaled PGEN table as list of string integers
        fixed_fields: gains held constant during tuning, as {param_name: value}.
            These are written to the device on every evaluation alongside the
            tuned gains. Without them the device keeps whatever value it already
            had, so the hardware phase would optimise a different controller
            than the simulation that produced the warm start.
        setpoint_field: PCAP stream column(s) for the PGEN output; defaults to
            f"{pgen_block}.OUT.Value" so it tracks the block name automatically
        position_field: PCAP stream column(s) for the measured position. Pass a
            sequence to record more than one encoder; the columns are stacked in
            the order given, so column i is axis i.
        axes: which axis columns to score, matching the `axes` passed to
            validation.evaluate. None scores every column recorded. With a single
            position field only axis 0 exists.
        """
        self.host = host
        self.pid_fields = pid_fields
        self.fixed_fields = dict(fixed_fields or {})
        self.trajectory = trajectory
        self.pgen_block = pgen_block
        self.pid_block = pid_block
        self.setpoint_fields = _as_field_tuple(
            setpoint_field or f"{pgen_block}.OUT.Value"
        )
        self.position_fields = _as_field_tuple(position_field)
        if len(self.setpoint_fields) != len(self.position_fields):
            raise ValueError(
                f"setpoint_field and position_field must name the same number of "
                f"columns, got {len(self.setpoint_fields)} and "
                f"{len(self.position_fields)}"
            )
        self.axes = axes
        self.timeout_s = timeout_s
        self.debug_plot_path = debug_plot_path
        self._eval_count = 0

    def __call__(
        self, pid_tensor: torch.Tensor, return_details: bool = False
    ) -> tuple[torch.Tensor, None]:
        return asyncio.run(self._evaluate_async(pid_tensor, return_details))

    async def _evaluate_async(
        self, pid_tensor: torch.Tensor, return_details: bool
    ) -> tuple[torch.Tensor, None]:
        if pid_tensor.ndim == 1:
            pid_tensor = pid_tensor.unsqueeze(0)
        pid_np = pid_tensor.detach().cpu().numpy()
        values: list[float] = []
        for i in range(pid_np.shape[0]):
            metric, sp, pos = await self._single_evaluate(pid_np[i])
            self._eval_count += 1
            values.append(metric)
            if self.debug_plot_path is not None:
                plot_hardware_trajectory(sp, pos, metric, self.debug_plot_path)
        y = torch.tensor(values, dtype=torch.double).unsqueeze(-1)
        return y, None

    async def _single_evaluate(
        self, params: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        async with AsyncioClient(self.host) as client:
            # Stop trajectory and reset PID before changing gains
            await client.send(Put(f"{self.pgen_block}.ENABLE", "ZERO"))
            await client.send(Put(f"{self.pid_block}.INIT", "ONE"))

            # Write PID gains — field order must match the BO param order in run.py.
            # Values are snapped to the field's fixed-point grid so the recorded
            # gains are exactly what the FPGA runs.
            for (name, field), val in zip(self.pid_fields.items(), params, strict=True):
                await client.send(Put(field, str(quantise(name, float(val)))))

            # Gains held fixed during tuning must be written too, or the device
            # keeps whatever it had and the hardware phase tunes a different
            # controller than the simulation did.
            for name, val in self.fixed_fields.items():
                await client.send(
                    Put(PANDA_PID_FIELDS[name], str(quantise(name, float(val))))
                )

            # Load trajectory into PGEN table
            await client.send(Put(f"{self.pgen_block}.TABLE", self.trajectory))

            # pandablocks >=0.10: open data port first, arm on ReadyData, then start
            # PGEN. Arming before opening the data port misses StartData in 0.10+.
            setpoints: list[NDArray[np.float64]] = []
            positions: list[NDArray[np.float64]] = []
            pgen_started = False
            try:
                async for frame in client.data(
                    scaled=True, frame_timeout=self.timeout_s
                ):
                    if isinstance(frame, ReadyData):
                        await client.send(Arm())
                        if not pgen_started:
                            await client.send(Put(f"{self.pgen_block}.ENABLE", "ONE"))
                            await client.send(Put(f"{self.pid_block}.INIT", "ZERO"))
                            pgen_started = True
                    elif isinstance(frame, FrameData):
                        # Stack the configured columns so each entry is
                        # (n_samples_in_frame, n_axes), matching the (n, n_axes)
                        # layout the simulator scores.
                        setpoints.append(
                            np.column_stack(
                                [frame.data[f] for f in self.setpoint_fields]
                            )
                        )
                        positions.append(
                            np.column_stack(
                                [frame.data[f] for f in self.position_fields]
                            )
                        )
                    elif isinstance(frame, EndData):
                        break
            except Exception:
                await client.send(Disarm())
                raise
            finally:
                # Always stop trajectory and reset PID on exit
                await client.send(Put(f"{self.pgen_block}.ENABLE", "ZERO"))
                await client.send(Put(f"{self.pid_block}.INIT", "ONE"))

        if not setpoints:
            raise RuntimeError(
                f"No PCAP frames captured from {self.host}. Check that "
                f"{', '.join(self.setpoint_fields + self.position_fields)} are "
                f"enabled in the PCAP capture set."
            )

        sp = np.concatenate(setpoints, axis=0)
        pos = np.concatenate(positions, axis=0)
        # Same composite as the simulation, so the two objectives stay
        # numerically comparable and the warm start transfers.
        return following_error_score(sp, pos, axes=self.axes), sp, pos
