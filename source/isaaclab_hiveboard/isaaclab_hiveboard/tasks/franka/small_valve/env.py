# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Research 3 small-valve task, retargeted from the ANYmal task of the same scene."""

import math

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.anymal.small_valve.env import AnymalSmallValveEnvCfg, AnymalSmallValveEnvCfg_PLAY
from isaaclab_hiveboard.tasks.franka.common import use_franka

# The pads meet the ~34.5 mm-radius handwheel rim at fr3_finger_joint ~0.034 m.
# Closing to 0.0 squeezed 34 mm past it: the first pad to reach a rim knob shoved
# the compliant arm off the wheel while the other pad closed on nothing. Stop
# ~4 mm past contact (use_franka ramps the close).
FINGER_CLOSE = 0.030


def _use_franka_small_valve(env_cfg) -> None:
    # The handwheel points along valve -Y. Mount panel +X along that axis so
    # the floor adapter places the handwheel and its approach above the panel.
    env_cfg.scene.honeycomb.init_state.rot = (0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5))
    use_franka(env_cfg, "small_valve", "reset_valve_root", finger_close=FINGER_CLOSE)
    # The ANYmal task's 8 s ends mid-turn at the FR3's slower valve arc.
    env_cfg.episode_length_s = 20.0


@configclass
class FrankaSmallValveEnvCfg(AnymalSmallValveEnvCfg):
    """Table-mounted FR3 on the shared HiveBoard small-valve scene."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_small_valve(self)


@configclass
class FrankaSmallValveEnvCfg_PLAY(AnymalSmallValveEnvCfg_PLAY):
    """Deterministic one-environment FR3 demonstration."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_small_valve(self)
