# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Research 3 M8 thread task, retargeted from the ANYmal task of the same scene."""

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.anymal.m8_thread.env import AnymalM8ThreadEnvCfg, AnymalM8ThreadEnvCfg_PLAY
from isaaclab_hiveboard.tasks.franka.common import shift_target_frames, use_franka
from isaaclab_hiveboard.tasks.scenes.screw import SCREW_GRASP_FRAMES
from isaaclab_hiveboard.tasks.scenes.threads import M8_SPEC


def _use_franka_m8_thread(env_cfg) -> None:
    use_franka(env_cfg, "thread", "reset_object_root")
    # Allow all regrasp cycles and the final retreat (~59 s at the nominal pose).
    env_cfg.episode_length_s = 75.0
    shift_target_frames(env_cfg, SCREW_GRASP_FRAMES, M8_SPEC.franka_tip_x - M8_SPEC.grasp_x)


@configclass
class FrankaM8ThreadEnvCfg(AnymalM8ThreadEnvCfg):
    """Table-mounted FR3 on the shared HiveBoard M8 thread scene."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_m8_thread(self)


@configclass
class FrankaM8ThreadEnvCfg_PLAY(AnymalM8ThreadEnvCfg_PLAY):
    """Deterministic one-environment FR3 demonstration."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_m8_thread(self)
