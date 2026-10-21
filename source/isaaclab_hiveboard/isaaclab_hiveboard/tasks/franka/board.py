# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Floor-mounted HiveBoard placement and reset ranges for Franka tasks."""

import math

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg

# Honeycomb meshes extend along local +X. Rotate that normal onto world +Z.
FRANKA_BOARD_ROT = (0.0, -math.sqrt(0.5), 0.0, math.sqrt(0.5))
FRANKA_BOARD_DISTANCE = 0.60
FRANKA_OBJECT_POSE_RANGE = {
    "x": (-0.05, 0.05),
    "y": (-0.08, 0.08),
    "z": (0.0, 0.08),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}


def reset_board_pose(env, env_ids, pose_range, velocity_range, asset_cfg, companion_names=()):
    """Sample one board pose and move its separately simulated parts together."""
    primary = env.scene[asset_cfg.name]
    ids = slice(None) if env_ids is None else env_ids
    reference = primary.data.default_root_pose.torch[ids]
    offsets = math_utils.sample_uniform_from_ranges(
        pose_range, ("x", "y", "z", "roll", "pitch", "yaw"), len(reference), device=env.device
    )
    pos = reference[:, :3] + env.scene.env_origins[ids] + offsets[:, :3]
    rot = math_utils.quat_mul(
        reference[:, 3:], math_utils.quat_from_euler_xyz(offsets[:, 3], offsets[:, 4], offsets[:, 5])
    )
    velocity = math_utils.sample_uniform_from_ranges(
        velocity_range, ("x", "y", "z", "roll", "pitch", "yaw"), len(reference), device=env.device
    )
    for name in (asset_cfg.name, *companion_names):
        asset = env.scene[name]
        default = asset.data.default_root_pose.torch[ids]
        local_pos, local_rot = math_utils.subtract_frame_transforms(
            reference[:, :3], reference[:, 3:], default[:, :3], default[:, 3:]
        )
        asset_pos, asset_rot = math_utils.combine_frame_transforms(pos, rot, local_pos, local_rot)
        asset.write_root_pose_to_sim_index(root_pose=torch.cat((asset_pos, asset_rot), dim=-1), env_ids=ids)
        asset.write_root_velocity_to_sim_index(
            root_velocity=asset.data.default_root_vel.torch[ids] + velocity, env_ids=ids
        )


def use_franka_board(env_cfg, object_name, root_reset_event=None, companion_names=()):
    """Place the panel's back on the floor, 60 cm along the robot's +X axis.

    Use the panel's mounting transform rather than assuming every mechanism
    has the same root orientation or depth. Child target frames then follow
    the mechanism automatically. Play variants retain deterministic resets.
    """
    scene = env_cfg.scene
    obj = getattr(scene, object_name)
    tensor = lambda value: torch.tensor([value], dtype=torch.float64)  # noqa: E731
    robot_pos = scene.robot.init_state.pos
    panel_pos = tensor((robot_pos[0] + FRANKA_BOARD_DISTANCE, robot_pos[1], 0.0))
    panel_rot = tensor(FRANKA_BOARD_ROT)
    mount = scene.honeycomb.init_state
    # panel = object * mount, so object = panel * inverse(mount).
    inv_pos, inv_rot = math_utils.subtract_frame_transforms(tensor(mount.pos), tensor(mount.rot))
    pos, rot = math_utils.combine_frame_transforms(panel_pos, panel_rot, inv_pos, inv_rot)
    old_pos, old_rot = tensor(obj.init_state.pos), tensor(obj.init_state.rot)
    for name in (object_name, *companion_names):
        state = getattr(scene, name).init_state
        rel_pos, rel_rot = math_utils.subtract_frame_transforms(old_pos, old_rot, tensor(state.pos), tensor(state.rot))
        new_pos, new_rot = math_utils.combine_frame_transforms(pos, rot, rel_pos, rel_rot)
        state.pos = tuple(new_pos[0].tolist())
        state.rot = tuple(new_rot[0].tolist())

    reset = getattr(env_cfg.events, root_reset_event) if root_reset_event else None
    # Some shared expert configs disable pose noise even outside Play. Franka
    # collection uses the requested bounds; only its explicit Play configs are fixed.
    randomized = not type(env_cfg).__name__.endswith("_PLAY")
    params = {
        "asset_cfg": SceneEntityCfg(object_name),
        "pose_range": dict(FRANKA_OBJECT_POSE_RANGE)
        if randomized
        else {key: (0.0, 0.0) for key in FRANKA_OBJECT_POSE_RANGE},
        "velocity_range": {} if reset is None else dict(reset.params.get("velocity_range", {})),
    }
    func = mdp.reset_root_state_uniform
    if companion_names:
        func = reset_board_pose
        params["companion_names"] = companion_names
    setattr(
        env_cfg.events, root_reset_event or "reset_board_root", EventTermCfg(func=func, mode="reset", params=params)
    )
    use_franka_board_viewer(env_cfg)


def use_franka_board_viewer(env_cfg):
    """Frame the horizontal panel from above the robot's right shoulder."""
    robot_pos = env_cfg.scene.robot.init_state.pos
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (robot_pos[0] + 0.85, robot_pos[1] - 0.75, 0.85)
    env_cfg.viewer.lookat = (robot_pos[0] + FRANKA_BOARD_DISTANCE, robot_pos[1], 0.10)
