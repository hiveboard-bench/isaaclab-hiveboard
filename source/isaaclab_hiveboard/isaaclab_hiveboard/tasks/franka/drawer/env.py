# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Research 3 sliding-drawer task, retargeted from the ANYmal task of the same scene."""

import math

from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.mdp.commands.sequential_pose_command import CuroboPlannedGoToFrameCfg
from isaaclab_hiveboard.tasks.anymal.drawer.configs.commands import FramePoseCommandsCfg, add_drawer_removal_steps
from isaaclab_hiveboard.tasks.anymal.drawer.configs.terminations import DrawerRemovalTerminationsCfg
from isaaclab_hiveboard.tasks.anymal.drawer.env import AnymalDrawerEnvCfg, AnymalDrawerEnvCfg_PLAY
from isaaclab_hiveboard.tasks.anymal.drawer.physics import DRAWER_DECIMATION, DRAWER_SIM_DT
from isaaclab_hiveboard.tasks.franka.common import use_franka

FRANKA_DRAWER_OPEN = 0.018
FRANKA_DRAWER_CLOSE = 0.014


def _use_franka_drawer(env_cfg) -> None:
    env_cfg.episode_length_s = 20.0
    use_franka(
        env_cfg,
        "drawer",
        "reset_object_root",
        finger_close=FRANKA_DRAWER_CLOSE,
        companion_names=("drawer_housing",),
    )
    # The generic FR3 adapter selects 1.5 ms; this contact-guided drawer needs
    # the finer drawer tick while retaining its 75 ms action period.
    env_cfg.sim.dt = DRAWER_SIM_DT
    env_cfg.decimation = DRAWER_DECIMATION
    for frame in env_cfg.scene.target_frame.target_frames:
        if frame.name in ("drawer_grasp", "drawer_pulled"):
            x, y, z = frame.offset.pos
            # The pitched finger tips project past the stock FR3 TCP. Keep
            # the shared 58 mm depth and raise the reference 7 mm so their
            # pads overlap the tab without entering the drawer front.
            frame.offset = OffsetCfg(pos=(x, y, z + 0.007), rot=frame.offset.rot)

    # The stock home puts the fingers below the side guides.
    # Start above them, with a 36 mm jaw gap around the 30 mm tab.
    robot = env_cfg.scene.robot
    env_cfg.scene.robot = robot.replace(
        init_state=robot.init_state.replace(
            joint_pos={
                **robot.init_state.joint_pos,
                "fr3_joint2": -0.70,
                "fr3_finger_joint.*": FRANKA_DRAWER_OPEN,
            }
        )
    )
    hand = env_cfg.actions.gripper_action
    hand.open_command_expr = {"fr3_finger_joint.*": FRANKA_DRAWER_OPEN}
    # Full closure squeezes the light box off the tab and out of the slots.
    # Stop 1 mm per jaw past contact and close gently to retain a real grasp.
    hand.close_speed = 0.008

    half_pitch = math.radians(45.0) / 2.0
    pitch = (0.0, math.sin(half_pitch), 0.0, math.cos(half_pitch))
    commands = env_cfg.commands.pose_command.commands
    for command in commands:
        if isinstance(command, CuroboPlannedGoToFrameCfg):
            command.target_offset_rot = pitch
            command.canonicalize_upward = False
            command.distance_threshold = 0.003
    commands[0].target_offset_pos = (0.0, 0.0, 0.04)
    commands[-1].target_offset_pos = (0.0, 0.0, 0.04)
    commands[2].duration_s = 0.9
    # FR3 tracks the lift more closely than DynaArm; keep its target inside
    # the narrow opening between the lip and upper guide.
    add_drawer_removal_steps(env_cfg.commands.pose_command, exit_height=0.012)


@configclass
class FrankaDrawerEnvCfg(AnymalDrawerEnvCfg):
    """Table-mounted FR3 that lifts and removes the contact-guided drawer."""

    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()
    terminations: DrawerRemovalTerminationsCfg = DrawerRemovalTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _use_franka_drawer(self)


@configclass
class FrankaDrawerEnvCfg_PLAY(AnymalDrawerEnvCfg_PLAY):
    """Deterministic one-environment FR3 demonstration."""

    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()
    terminations: DrawerRemovalTerminationsCfg = DrawerRemovalTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _use_franka_drawer(self)
