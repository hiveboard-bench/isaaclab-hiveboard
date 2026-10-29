# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Research 3 circuit-breaker task, retargeted from the ANYmal task of the same scene."""

from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.anymal.circuit_breaker.env import (
    AnymalCircuitBreakerEnvCfg,
    AnymalCircuitBreakerEnvCfg_PLAY,
)
from isaaclab_hiveboard.tasks.franka.common import use_franka

# The FR3 TCP sits at the fingertips, while the closed 2F-140 and Spot jaws
# reach past theirs, so the shared push frames stop short of the paddle for
# the FR3. Push this much deeper toward the panel (object -X).
PUSH_DEPTH = 0.015


def _use_franka_breaker(env_cfg) -> None:
    use_franka(env_cfg, "circuit_breaker", "reset_breaker_root")
    # Include the grasp, actuation and retreat; the nominal floor run takes ~12 s.
    env_cfg.episode_length_s = 20.0
    for frame in env_cfg.scene.target_frame.target_frames:
        if frame.name in ("lever_pivot_below", "lever_pivot_above"):
            x, y, z = frame.offset.pos
            frame.offset = OffsetCfg(pos=(x - PUSH_DEPTH, y, z), rot=frame.offset.rot)


@configclass
class FrankaCircuitBreakerEnvCfg(AnymalCircuitBreakerEnvCfg):
    """Table-mounted FR3 on the shared HiveBoard circuit-breaker scene."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_breaker(self)


@configclass
class FrankaCircuitBreakerEnvCfg_PLAY(AnymalCircuitBreakerEnvCfg_PLAY):
    """Deterministic one-environment FR3 demonstration."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_breaker(self)
