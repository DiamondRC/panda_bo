"""PandABlocks hardware evaluator — PCAP streaming pattern."""

import asyncio

import numpy as np
import torch
from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import Arm, Disarm, Put
from pandablocks.responses import EndData, FrameData, ReadyData

from bayesian_pid.metrics import following_error_rms
from bayesian_pid.plotting import plot_hardware_trajectory


class HardwareEvaluator:
    def __init__(
        self,
        host: str,
        pid_fields: dict,
        trajectory: list,
        pgen_block: str = "PGEN",
        pid_block: str = "BRETT_PID",
        setpoint_field: str | None = None,
        position_field: str = "INENC1.VAL.Value",
        timeout_s: float = 60.0,
        debug_plot_path: str | None = None,
    ):
        """
        pid_fields: mapping from param name to PandA field, e.g.
            {"kp": "BRETT_PID.KP", "kv": "BRETT_PID.KV", "ki": "BRETT_PID.KI"}
        trajectory: pre-scaled PGEN table as list of string integers
        setpoint_field: PCAP stream column for the PGEN output; defaults to
            f"{pgen_block}.OUT.Value" so it tracks the block name automatically
        """
        self.host = host
        self.pid_fields = pid_fields
        self.trajectory = trajectory
        self.pgen_block = pgen_block
        self.pid_block = pid_block
        self.setpoint_field = setpoint_field or f"{pgen_block}.OUT.Value"
        self.position_field = position_field
        self.timeout_s = timeout_s
        self.debug_plot_path = debug_plot_path
        self._eval_count = 0

    def __call__(self, pid_tensor: torch.Tensor, return_details: bool = False) -> tuple:
        return asyncio.run(self._evaluate_async(pid_tensor, return_details))

    async def _evaluate_async(self, pid_tensor, return_details):
        if pid_tensor.ndim == 1:
            pid_tensor = pid_tensor.unsqueeze(0)
        pid_np = pid_tensor.detach().cpu().numpy()
        values = []
        for i in range(pid_np.shape[0]):
            metric, sp, pos = await self._single_evaluate(pid_np[i])
            self._eval_count += 1
            values.append(metric)
            if self.debug_plot_path is not None:
                plot_hardware_trajectory(sp, pos, metric, self.debug_plot_path)
        y = torch.tensor(values, dtype=torch.double).unsqueeze(-1)
        return y, None

    async def _single_evaluate(self, params: np.ndarray) -> float:
        async with AsyncioClient(self.host) as client:
            # Stop trajectory and reset PID before changing gains
            await client.send(Put(f"{self.pgen_block}.ENABLE", "ZERO"))
            await client.send(Put(f"{self.pid_block}.INIT", "ONE"))

            # Write PID gains — field order must match the BO param order in run.py
            for field, val in zip(self.pid_fields.values(), params, strict=True):
                await client.send(Put(field, str(val)))

            # Load trajectory into PGEN table
            await client.send(Put(f"{self.pgen_block}.TABLE", self.trajectory))

            # pandablocks >=0.10: open data port first, arm on ReadyData, then start
            # PGEN. Arming before opening the data port misses StartData in 0.10+.
            setpoints, positions = [], []
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
                        setpoints.append(frame.data[self.setpoint_field])
                        positions.append(frame.data[self.position_field])
                    elif isinstance(frame, EndData):
                        break
            except Exception:
                await client.send(Disarm())
                raise
            finally:
                # Always stop trajectory and reset PID on exit
                await client.send(Put(f"{self.pgen_block}.ENABLE", "ZERO"))
                await client.send(Put(f"{self.pid_block}.INIT", "ONE"))

        sp = np.concatenate(setpoints)
        pos = np.concatenate(positions)
        return following_error_rms(sp, pos), sp, pos
