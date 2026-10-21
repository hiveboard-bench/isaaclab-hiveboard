# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU checks for floor-mounted Franka boards and expert reset bounds."""

import importlib
from types import SimpleNamespace

import gymnasium as gym
import isaaclab_hiveboard  # noqa: F401
import pytest
import torch
from isaaclab_hiveboard.tasks.franka.board import FRANKA_OBJECT_POSE_RANGE, reset_board_pose

import isaaclab.utils.math as math_utils

TASKS = {
    "BallValve": "ball_valve",
    "CircuitBreaker": "circuit_breaker",
    "HighTorqueValve": "high_torque_valve",
    "SmallValve": "small_valve",
    "Button": "button",
    "Drawer": "drawer_housing",
    "Key": "lock",
    "M8Thread": "thread",
    "M30Thread": "thread",
    "PegInsertion": "peg",
    "ShockAbsorber": "shock_absorber",
    "Lamp": "lamp",
}


def config(robot, task, play=False):
    suffix = "-Play" if play else ""
    module, name = gym.spec(f"Isaac-HiveBoard-{robot}-{task}{suffix}-v0").kwargs["env_cfg_entry_point"].split(":")
    return getattr(importlib.import_module(module), name)()


@pytest.mark.parametrize("task,object_name", TASKS.items())
@pytest.mark.parametrize("play", (False, True))
def test_all_franka_panels_face_up_on_floor_and_use_requested_reset_ranges(task, object_name, play):
    cfg = config("Franka", task, play)
    obj = getattr(cfg.scene, object_name).init_state
    panel = cfg.scene.honeycomb.init_state
    tensor = lambda value: torch.tensor([value], dtype=torch.float64)  # noqa: E731
    pos, rot = math_utils.combine_frame_transforms(
        tensor(obj.pos), tensor(obj.rot), tensor(panel.pos), tensor(panel.rot)
    )
    robot_pos = cfg.scene.robot.init_state.pos
    torch.testing.assert_close(pos, tensor((robot_pos[0] + 0.60, robot_pos[1], 0.0)), atol=1e-7, rtol=0)
    normal = math_utils.quat_apply(rot, tensor((1.0, 0.0, 0.0)))
    torch.testing.assert_close(normal, tensor((0.0, 0.0, 1.0)), atol=1e-7, rtol=0)
    resets = [
        term for term in vars(cfg.events).values() if term is not None and "pose_range" in getattr(term, "params", {})
    ]
    assert len(resets) == 1
    expected = {key: (0.0, 0.0) for key in FRANKA_OBJECT_POSE_RANGE} if play else FRANKA_OBJECT_POSE_RANGE
    assert resets[0].params["pose_range"] == expected
    assert cfg.viewer.origin_type == "env" and cfg.viewer.lookat[2] == 0.10
    if task == "Drawer":
        assert cfg.scene.drawer.init_state.pos == pytest.approx(obj.pos)
        assert cfg.scene.drawer.init_state.rot == pytest.approx(obj.rot)
        assert resets[0].params["companion_names"] == ("drawer_housing",)
    elif task == "BallValve":
        # The downstream expert-bank collector identifies this public reset term.
        assert resets[0].func.__name__ == "reset_root_state_uniform"


@pytest.mark.parametrize("robot", ("Anymal", "Spot"))
def test_franka_configuration_does_not_change_other_robots(robot):
    before = config(robot, "BallValve")
    config("Franka", "BallValve")
    after = config(robot, "BallValve")
    assert before.scene.ball_valve.init_state.pos == after.scene.ball_valve.init_state.pos == (1.0, 0.06, 0.804)
    assert before.scene.ball_valve.init_state.rot == after.scene.ball_valve.init_state.rot == (0.0, 0.0, 1.0, 0.0)
    assert before.events.reset_valve_root.params["pose_range"] == after.events.reset_valve_root.params["pose_range"]


@pytest.mark.parametrize("task,axis", (("SmallValve", (0.0, -1.0, 0.0)), ("HighTorqueValve", (0.0, 0.0, 1.0))))
@pytest.mark.parametrize("play", (False, True))
def test_gate_valve_working_axes_point_upward(task, axis, play):
    cfg = config("Franka", task, play)
    obj = getattr(cfg.scene, TASKS[task]).init_state
    rotation = torch.tensor([obj.rot], dtype=torch.float64)
    working_axis = math_utils.quat_apply(rotation, torch.tensor([axis], dtype=torch.float64))
    torch.testing.assert_close(working_axis, torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64), atol=1e-7, rtol=0)


@pytest.mark.parametrize("task,timeout", (("CircuitBreaker", 20.0), ("M8Thread", 75.0)))
@pytest.mark.parametrize("play", (False, True))
def test_franka_expert_timeouts_allow_complete_sequences(task, timeout, play):
    assert config("Franka", task, play).episode_length_s == timeout


@pytest.mark.parametrize("task", ("SmallValve", "HighTorqueValve", "CircuitBreaker", "M8Thread"))
@pytest.mark.parametrize("robot", ("Anymal", "Spot"))
def test_franka_fixes_preserve_other_robot_mounts_and_timeouts(task, robot):
    before = config(robot, task)
    config("Franka", task)
    after = config(robot, task)
    assert before.scene.honeycomb.init_state.to_dict() == after.scene.honeycomb.init_state.to_dict()
    assert before.episode_length_s == after.episode_length_s
    obj = TASKS[task]
    assert getattr(before.scene, obj).init_state.to_dict() == getattr(after.scene, obj).init_state.to_dict()


@pytest.mark.parametrize("ids", (torch.tensor([1]), slice(1, None), None))
def test_drawer_reset_moves_housing_and_box_together_only_for_selected_environments(ids):
    class Scene(dict):
        env_origins = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

    scene, writes = Scene(), {}
    for name in ("drawer", "drawer_housing"):
        default = torch.tensor([[0.60, 0.0, 0.02, 0.0, 0.0, 0.0, 1.0]]).expand(2, -1)
        scene[name] = SimpleNamespace(
            data=SimpleNamespace(
                default_root_pose=SimpleNamespace(torch=default),
                default_root_vel=SimpleNamespace(torch=torch.zeros(2, 6)),
            ),
            write_root_pose_to_sim_index=lambda name=name, **kw: writes.setdefault(name, {}).update(kw),
            write_root_velocity_to_sim_index=lambda name=name, **kw: writes[name].update(kw),
        )
    env = SimpleNamespace(scene=scene, device="cpu")
    reset_board_pose(
        env,
        ids,
        {"x": (0.05, 0.05), "y": (-0.08, -0.08), "z": (0.08, 0.08)},
        {},
        SimpleNamespace(name="drawer"),
        ("drawer_housing",),
    )
    torch.testing.assert_close(writes["drawer"]["root_pose"], writes["drawer_housing"]["root_pose"])
    expected = torch.tensor([[0.65, -0.08, 0.10], [3.65, -0.08, 0.10]])
    torch.testing.assert_close(writes["drawer"]["root_pose"][:, :3], expected if ids is None else expected[1:])
    assert not writes["drawer"]["root_velocity"].any()
