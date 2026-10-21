# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Retarget a shared HiveBoard task config onto the Franka Research 3.

The Spot and ANYmal tasks share one scene per HiveBoard object and differ only
in robot-specific fields: the articulation, the TCP profile, the cuRobo model,
the action terms, the finger bodies and the arm joint names. :func:`use_franka`
rewrites exactly those fields, so each Franka task is a thin subclass of the
matching ANYmal task instead of another copy of it.
"""

from __future__ import annotations

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import ASSET_DIR, FRANKA_EE, FRANKA_FR3_HIGH_PD_CFG, as_command_offset, make_ee_frame
from isaaclab_hiveboard.mdp.actions import RateLimitedBinaryJointPositionActionCfg
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import GripperCommand
from isaaclab_hiveboard.mdp.events import set_contact_stiffness, set_mimic_constraint_stiffness
from isaaclab_hiveboard.tasks.franka.board import FRANKA_OBJECT_POSE_RANGE as FRANKA_OBJECT_POSE_RANGE
from isaaclab_hiveboard.tasks.franka.board import use_franka_board

FRANKA_ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
FRANKA_FINGER_JOINT_NAMES = ["fr3_finger_joint1", "fr3_finger_joint2"]
FRANKA_FINGER_BODY_NAMES = ["fr3_leftfinger", "fr3_rightfinger"]

FRANKA_CUROBO = {
    "robot_joint_names": FRANKA_ARM_JOINT_NAMES,
    "robot_curobo_yaml": f"{ASSET_DIR}/franka/cumotion/fr3.yaml",
    "robot_urdf": f"{ASSET_DIR}/franka/cumotion/fr3.urdf",
}

# Integrate the stiff drives in MJWarp's implicit solver. Explicit PD at the
# 5 ms task timestep saturates the arm torques and makes the fingers oscillate
# even under a constant open command.
ARM_STIFFNESS = 2000.0
ARM_DAMPING = 200.0
FRANKA_NEWTON_ACTUATORS = {
    "fr3_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["fr3_joint[1-4]"],
        effort_limit_sim=87.0,
        stiffness=ARM_STIFFNESS,
        damping=ARM_DAMPING,
        armature=0.1,
    ),
    "fr3_forearm": ImplicitActuatorCfg(
        joint_names_expr=["fr3_joint[5-7]"],
        effort_limit_sim=12.0,
        stiffness=ARM_STIFFNESS,
        damping=ARM_DAMPING,
        armature=0.05,
    ),
    # Stiff enough to squeeze a handle: closing on a 1 cm part leaves a 5 mm
    # position error per finger, which must still reach the FR3 hand's
    # 70 N continuous grasp force. The damping caps the closing speed near
    # the real hand's ~0.07 m/s so the jaws do not slam light handwheels.
    # Both fingers stay driven: fr3.usd's finger mimic becomes a soft MuJoCo
    # equality in Newton, and a passive follower on it swings past its limits.
    "fr3_hand": ImplicitActuatorCfg(
        joint_names_expr=["fr3_finger_joint.*"],
        effort_limit_sim=70.0,
        stiffness=2e4,
        damping=1e3,
        armature=0.01,
    ),
}

FRANKA_NEWTON_CFG = FRANKA_FR3_HIGH_PD_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=FRANKA_FR3_HIGH_PD_CFG.spawn.replace(activate_contact_sensors=True),
    actuators=FRANKA_NEWTON_ACTUATORS,
)
"""Table-mounted FR3 at the env origin, facing +X."""

# Physics settings follow the website's MuJoCo FR3 model
# (hiveboard-bench.github.io/public/sim/models/fr3.xml: 2 ms step, pad solref
# (0.005, 1), screw equalities solref (0.01, 1) with solimp dmax 0.999).
#
# MuJoCo clamps every solref time constant to >= 2 x the physics step. At 5 ms
# that made the 15 g fingers' contacts so soft they sank 10-30 mm into parts.
# 1.5 ms leaves the finger contacts unclamped. Each task keeps its env step.
FRANKA_PHYSICS_DT = 0.0015
# Damping ratio ~1: solref (0.005, 1). kd=2000 (ratio ~5) made the contact
# ~25x softer once MuJoCo clamped its time constant.
FINGER_CONTACT_KE = 4.0e4
FINGER_CONTACT_KD = 400.0
FINGER_OPEN = 0.04
# The stock binary action jumps the finger target in one step, and the drive
# slams the pads into the part (12 mm into the M30 nut in one 75 ms step).
# Ramp the target at about the real FR3 hand's closing speed instead.
FINGER_CLOSE_SPEED = 0.05
FINGER_SETTLE_S = 0.1
"""Time for the drive to settle once the ramped target reaches the close command [s]."""


def finger_contact_stiffness(object_prim_name: str) -> EventTerm:
    """Stiffen FR3 finger and object contacts (see :func:`set_contact_stiffness`).

    MJWarp's default solref is mass-normalized, so the light fingers and parts let
    the stiff FR3 arm push them tens of millimetres into each other (25 mm on the
    button). Same values as the ANYmal small valve. Needs
    :data:`FRANKA_PHYSICS_DT`: at 5 ms MuJoCo clamps the 5 ms time constant to
    10 ms. Add it after any material randomization event, which re-syncs shape
    properties.

    Args:
        object_prim_name: Leading name of the object prims under the env, e.g.
            ``"Button"``. It is a prefix, so ``"Drawer"`` also stiffens ``DrawerHousing``.
    """
    return EventTerm(
        func=set_contact_stiffness,
        mode="startup",
        params={
            "shape_regex": f"/Robot/fr3_(left|right)finger/|/{object_prim_name}",
            "ke": FINGER_CONTACT_KE,
            "kd": FINGER_CONTACT_KD,
            "solimp": (0.95, 0.99, 0.001),
        },
    )


def use_franka_physics(env_cfg, finger_close: float | None = None) -> None:
    """Simulation settings every FR3 task shares, whatever its scene and actions.

    Runs physics at :data:`FRANKA_PHYSICS_DT` with the env step unchanged,
    stiffens a native screw coupling's equality, and ramps the gripper close.
    Call it after the task's own ``__post_init__`` has set the decimation and
    the gripper action.

    Args:
        env_cfg: The FR3 task config.
        finger_close: Finger close command [m]. ``None`` keeps the task's.
            Stopping a few mm past the part's contact width keeps the squeeze
            even and the pads from sliding off.
    """
    step_dt = env_cfg.sim.dt * env_cfg.decimation
    env_cfg.decimation = max(1, round(step_dt / FRANKA_PHYSICS_DT))
    env_cfg.sim.dt = step_dt / env_cfg.decimation

    pose_command = getattr(env_cfg.commands, "pose_command", None)
    coupling = getattr(pose_command, "screw_coupling", None)
    if coupling is not None and coupling.use_native_mimic_constraint:
        # MuJoCo's default equality (solref 0.02, solimp 0.9/0.95) let a 70 N
        # squeeze slide the M30 nut 20 mm off its pitch.
        env_cfg.events.screw_constraint = EventTerm(
            func=set_mimic_constraint_stiffness,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(coupling.asset_name),
                "solref": (0.01, 1.0),
                "solimp": (0.95, 0.999, 0.001),
            },
        )

    hand = env_cfg.actions.gripper_action
    close = dict(hand.close_command_expr)
    if finger_close is not None:
        close = {"fr3_finger_joint.*": finger_close}
    env_cfg.actions.gripper_action = RateLimitedBinaryJointPositionActionCfg(
        asset_name=hand.asset_name,
        joint_names=hand.joint_names,
        open_command_expr=hand.open_command_expr,
        close_command_expr=close,
        close_speed=FINGER_CLOSE_SPEED,
    )
    # Hold each close until the ramped target arrives, so the arm does not
    # leave with the part only half gripped.
    close_s = (FINGER_OPEN - min(close.values())) / FINGER_CLOSE_SPEED + FINGER_SETTLE_S
    for command in getattr(pose_command, "commands", None) or []:
        if isinstance(command, GripperCommand) and not command.open_gripper and command.duration_s < close_s:
            env_cfg.episode_length_s += close_s - command.duration_s
            command.duration_s = close_s


@configclass
class FrankaJointPositionActionCfg:
    """Absolute FR3 joint positions from cuRobo waypoints, plus a binary hand."""

    arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=FRANKA_ARM_JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )

    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["fr3_finger_joint.*"],
        open_command_expr={"fr3_finger_joint.*": 0.04},
        close_command_expr={"fr3_finger_joint.*": 0.0},
    )


def use_franka(
    env_cfg,
    object_name: str,
    root_reset_event: str | None = None,
    finger_close: float | None = None,
    companion_names: tuple[str, ...] = (),
) -> None:
    """Swap the robot-specific parts of a shared HiveBoard task for the FR3.

    Args:
        env_cfg: A Spot or ANYmal task config, after its own ``__post_init__``.
        object_name: Scene entity of the HiveBoard object, shifted into reach.
        root_reset_event: Event term that randomizes the object root. Its pose
            range uses :data:`FRANKA_OBJECT_POSE_RANGE`; ``_PLAY`` variants
            retain a deterministic nominal pose.
        finger_close: Finger close command [m] (see :func:`use_franka_physics`).
            ``None`` closes fully.
        companion_names: Separately simulated parts mounted on the same board.
    """
    scene = env_cfg.scene
    scene.robot = FRANKA_NEWTON_CFG
    scene.ee_frame = make_ee_frame(FRANKA_EE)
    for sensor_name, body in zip(("finger_contact", "jaw_contact"), FRANKA_FINGER_BODY_NAMES):
        sensor = getattr(scene, sensor_name, None)
        if sensor is not None:
            sensor.prim_path = "{ENV_REGEX_NS}/Robot/" + body

    obj = getattr(scene, object_name)
    use_franka_board(env_cfg, object_name, root_reset_event, companion_names)

    pose_command = env_cfg.commands.pose_command
    pose_command.body_name = FRANKA_EE.body_name
    pose_command.body_offset = as_command_offset(FRANKA_EE)
    for command in pose_command.commands:
        if hasattr(command, "robot_joint_names"):
            for key, value in FRANKA_CUROBO.items():
                setattr(command, key, value)

    env_cfg.actions = FrankaJointPositionActionCfg()
    use_franka_physics(env_cfg, finger_close)

    # Replace any stiffening that targets the legged robots' grippers.
    for name, term in list(vars(env_cfg.events).items()):
        if isinstance(term, EventTerm) and term.func is set_contact_stiffness:
            setattr(env_cfg.events, name, None)
    env_cfg.events.finger_contacts = finger_contact_stiffness(obj.prim_path.rsplit("/", 1)[-1])

    material = getattr(env_cfg.events, "robot_physics_material", None)
    if material is not None:
        material.params["asset_cfg"] = SceneEntityCfg("robot", body_names=FRANKA_FINGER_BODY_NAMES)
    for group in vars(env_cfg.observations).values():
        if not isinstance(group, ObservationGroupCfg):
            continue
        for term in vars(group).values():
            if not isinstance(term, ObservationTermCfg):
                continue
            asset_cfg = term.params.get("asset_cfg")
            if not (isinstance(asset_cfg, SceneEntityCfg) and asset_cfg.name == "robot" and asset_cfg.joint_names):
                continue
            # Anything that is not the arm is the gripper (e.g. ANYmal's
            # ``finger_joint``); the FR3 hand has two mirrored finger joints.
            is_arm = len(asset_cfg.joint_names) > 1
            term.params["asset_cfg"] = SceneEntityCfg(
                "robot",
                joint_names=FRANKA_ARM_JOINT_NAMES if is_arm else FRANKA_FINGER_JOINT_NAMES,
                preserve_order=True,
            )


def shift_target_frames(env_cfg, names: tuple[str, ...], dx: float) -> None:
    """Move the named target frames by ``dx`` along the object's outward +X.

    The FR3 TCP is at the fingertips rather than mid-pad, so its grasp frames
    sit deeper (negative ``dx``) than the shared ANYmal/Spot ones.
    """
    for frame in env_cfg.scene.target_frame.target_frames:
        if frame.name in names:
            x, y, z = frame.offset.pos
            frame.offset = OffsetCfg(pos=(x + dx, y, z), rot=frame.offset.rot)
