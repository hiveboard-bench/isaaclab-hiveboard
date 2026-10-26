import math
from dataclasses import MISSING

import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import (
    ActionTerm,
    ActionTermCfg,
    EventTermCfg,
    ManagerTermBase,
    SceneEntityCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.utils import clone
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from isaaclab_hiveboard import assets
from isaaclab_hiveboard.assets.anymal.bench import ANYMAL_ARM_NEWTON_CFG
from isaaclab_hiveboard.assets.spot.spot import (
    SPOT_ARM_NEWTON_CFG,
    SPOT_ARM_UUC_BODY_PRIM,
    SPOT_ARM_UUC_FNGR_PRIM,
    SPOT_ARM_UUC_JAW_PRIM,
    SPOT_ARM_UUC_SOURCE_PRIM,
)
from isaaclab_hiveboard.mdp.actions import RateLimitedBinaryJointPositionActionCfg
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    WRIST_BRANCH_NEAREST,
    MechanismGoalCfg,
    ScrewJointCouplingCfg,
    SequentialPoseCommandCfg,
    _CuroboPlannedGoToFrameHandler,
    _GoToFrameHandler,
    _RotateFrameHandler,
    register_screw_joint_mimic,
)
from isaaclab_hiveboard.mdp.commands.skills import SkillSet
from isaaclab_hiveboard.mdp.events import (
    apply_articulation_gravcomp,
    set_contact_stiffness,
    set_joint_limit_damping,
)

DYNAMICS = ("friction", "damping", "spring", "breakaway", "armature")
EXPERT_DIVERSITY = ("turn_rate", "reach_scale", "grasp_shift", "grasp_roll", "turn_overshoot")
SPRING_STATIC_RAD = 0.4
BREAKAWAY_WIDTH_RAD = 0.15
IDENTITY = (0.0, 0.0, 0.0, 1.0)
FREE = (-math.inf, math.inf)


@configclass
class RobotCfg:
    name: str = MISSING
    articulation: ArticulationCfg = MISSING
    ee: assets.EndEffectorCfg = MISSING
    pads: tuple[str, str] = MISSING
    contact_shapes: str = MISSING
    board_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    board_rot: tuple[float, float, float, float] = IDENTITY
    planner: dict = MISSING
    wrist_flip: tuple[str, ...] | None = None
    min_duration_s: dict = {"grip": 0.0, "hold": 0.0, "release": 0.0}
    grip_effort: dict = {}
    pose_limits: dict = {k: FREE for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    actuator_groups: tuple[str, ...] = MISSING
    postures: dict = MISSING
    arm_range: tuple[float, float] = (-0.1, 0.1)
    arm_joint_names: tuple[str, ...] = MISSING
    hand_joint_names: tuple[str, ...] = MISSING
    finger_joint_names: tuple[str, ...] = MISSING
    diagnostic_joint: str = MISSING
    dt: float = MISSING
    decimation: int = MISSING
    gripper_joint_names: tuple[str, ...] = MISSING
    gripper_open: dict = MISSING
    gripper_close: dict = MISSING
    gripper_close_speed: float | None = None


@configclass
class AssetCfg:
    name: str = MISSING
    prim: str = MISSING
    usd: str = MISSING
    pos: tuple[float, float, float] = (1.0, 0.0, 0.65)
    rot: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.0)
    viewer: dict = {}
    panel_pos: tuple[float, float, float] = MISSING
    panel_rot: tuple[float, float, float, float] = IDENTITY
    joint: str = "RevoluteJoint"
    joints: dict = MISSING
    closed_end_stop: bool = False
    end_stop_damping: float | None = None
    pitch: float | None = None
    upper_limit: float = 0.0
    axial_reference: float = 0.0
    frame_path: str = MISSING
    frame_rot: tuple[float, float, float, float] = MISSING
    targets: dict = MISSING
    grasp_frame: str = MISSING
    mode: str = "grasp"
    grip_effort_scale: float = 1.0
    method: str | None = None
    dynamics: dict = {}
    stuck_prob: float = 0.0
    stuck_breakaway: tuple[float, float] = (0.0, 0.0)
    robot_material: dict = {}
    mechanism_material: dict = {}
    solver: dict = {}


@configclass
class TaskCfg:
    name: str = MISSING
    asset: AssetCfg = MISSING
    start: float = MISSING
    target: float = MISSING
    tolerance: float = MISSING
    pose: dict = {}
    joint_offset: tuple[float, float] = (0.0, 0.0)
    episode_length_s: float = MISSING


ANYMAL_GROUPS = (
    "dynaarm_shoulder",
    "dynaarm_elbow",
    "dynaarm_forearm",
    "dynaarm_wrist_flex",
    "dynaarm_wrist_rot",
)
ROBOTS = {
    "anymal": RobotCfg(
        name="anymal",
        articulation=ANYMAL_ARM_NEWTON_CFG,
        ee=assets.ANYMAL_EE,
        pads=("robotiq_2f_140/left_inner_finger", "robotiq_2f_140/right_inner_finger"),
        contact_shapes="/robotiq_2f_140/",
        planner={
            "robot_curobo_yaml": "anymal/cumotion/dynaarm.yaml",
            "robot_urdf": "anymal/cumotion/dynaarm.urdf",
        },
        wrist_flip=("dynaarm_forearm_rotation", "dynaarm_wrist_flexion", "dynaarm_wrist_rotation"),
        grip_effort={"gripper": 20.0},
        actuator_groups=ANYMAL_GROUPS,
        postures={
            "home": (0.0, 0.0, 1.7, 0.0, 0.0, 1.5708),
            "raised": (0.0, -0.4, 1.9, 0.0, 0.4, 1.5708),
            "lowered": (0.0, 0.3, 1.9, 0.0, -0.5, 1.5708),
            "left": (0.5, 0.0, 1.7, 0.0, 0.0, 1.5708),
            "right": (-0.5, 0.0, 1.7, 0.0, 0.0, 1.5708),
            "wrist_down": (0.0, 0.2, 1.8, 0.0, -0.8, 1.5708),
            "wrist_up": (0.0, 0.2, 1.8, 0.0, 0.8, 1.5708),
        },
        arm_joint_names=(
            "dynaarm_shoulder_rotation",
            "dynaarm_shoulder_flexion",
            "dynaarm_elbow_flexion",
            "dynaarm_forearm_rotation",
            "dynaarm_wrist_flexion",
            "dynaarm_wrist_rotation",
        ),
        hand_joint_names=(
            "finger_joint",
            "left_outer_finger_joint",
            "left_inner_finger_joint",
            "left_inner_finger_pad_joint",
            "right_outer_knuckle_joint",
            "right_outer_finger_joint",
            "right_inner_finger_joint",
            "right_inner_finger_pad_joint",
        ),
        finger_joint_names=("finger_joint",),
        diagnostic_joint="dynaarm_wrist_flexion",
        dt=0.005,
        decimation=10,
        gripper_joint_names=("finger_joint",),
        gripper_open={"finger_joint": 0.0},
        gripper_close={"finger_joint": 1.5},
        gripper_close_speed=2.0,
    ),
    "franka": RobotCfg(
        name="franka",
        articulation=assets.FRANKA_NEWTON_CFG,
        ee=assets.FRANKA_EE,
        pads=("fr3_leftfinger", "fr3_rightfinger"),
        contact_shapes="/Robot/fr3_(left|right)finger/",
        board_pos=(1.404, 0.06, 1.04),
        board_rot=(0.7071067811865476, 0.0, -0.7071067811865476, 0.0),
        planner={
            "robot_curobo_yaml": "franka/cumotion/fr3.yaml",
            "robot_urdf": "franka/cumotion/fr3.urdf",
        },
        min_duration_s={"grip": 0.7, "hold": 0.7, "release": 0.8},
        pose_limits={
            "x": (-0.05, 0.05),
            "y": (-0.08, 0.08),
            "z": (0.0, 0.08),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
        actuator_groups=("fr3_shoulder", "fr3_forearm"),
        postures={"home": (0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741)},
        arm_joint_names=tuple(f"fr3_joint{i}" for i in range(1, 8)),
        hand_joint_names=("fr3_finger_joint1", "fr3_finger_joint2"),
        finger_joint_names=("fr3_finger_joint1", "fr3_finger_joint2"),
        diagnostic_joint="fr3_joint6",
        dt=0.0015151515151515152,
        decimation=33,
        gripper_joint_names=("fr3_finger_joint.*",),
        gripper_open={"fr3_finger_joint.*": 0.04},
        gripper_close={"fr3_finger_joint.*": 0.01},
        gripper_close_speed=0.05,
    ),
    "spot": RobotCfg(
        name="spot",
        articulation=SPOT_ARM_NEWTON_CFG,
        ee=assets.SPOT_EE.replace(
            source_prim=SPOT_ARM_UUC_SOURCE_PRIM, body_prim=SPOT_ARM_UUC_BODY_PRIM
        ),
        pads=(SPOT_ARM_UUC_FNGR_PRIM, SPOT_ARM_UUC_JAW_PRIM),
        contact_shapes="/Robot/" + SPOT_ARM_UUC_BODY_PRIM + "/",
        planner={
            "robot_curobo_yaml": "spot/cumotion/spot_arm.yaml",
            "robot_urdf": "spot/spot_with_arm.urdf",
        },
        min_duration_s={"grip": 0.3, "hold": 0.5, "release": 0.33},
        grip_effort={"gripper": 15.32},
        pose_limits={k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        actuator_groups=("arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1"),
        postures={"home": (0.0, -0.9, 1.8, 0.0, -0.9, 0.0)},
        arm_range=(0.0, 0.0),
        arm_joint_names=("arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1"),
        hand_joint_names=("arm_f1x",),
        finger_joint_names=("arm_f1x",),
        diagnostic_joint="arm_wr0",
        dt=0.005,
        decimation=15,
        gripper_joint_names=("arm_f1x",),
        gripper_open={"arm_f1x": -1.57},
        gripper_close={"arm_f1x": -0.3},
    ),
}

FACE = (0.0, 0.0, 1.0, 0.0)
ASSETS = {
    "ball_valve": AssetCfg(
        name="ball_valve",
        prim="Valve",
        usd=assets.BALL_VALVE_USD,
        pos=(1.0, 0.06, 0.804),
        viewer={"body_name": "alavanca_pivot"},
        panel_pos=(-0.04, 0.0, 0.0),
        joints={
            "RevoluteJoint": {"friction": 0.02, "dynamic_friction": 0.0, "viscous_friction": 0.0}
        },
        closed_end_stop=True,
        frame_path="Geometry/valvula_esfera/alavanca_pivot",
        frame_rot=(0.0, 0.7071067811865476, 0.7071067811865476, 0.0),
        targets={
            "approaching": {"pos": (0.12, 0.06, 0.0)},
            "lever_pivot": {"pos": (0.03, 0.083944307, 0.0)},
            "rotate_frame": {"pos": (0.03, 0.0, 0.0)},
        },
        grasp_frame="lever_pivot",
        method="lever_turn",
        dynamics={
            "friction": (0.05, 2.0),
            "damping": (0.0, 0.5),
            "breakaway": (0.0, 2.0),
            "armature": (0.001, 0.02),
        },
        stuck_prob=0.25,
        stuck_breakaway=(2.0, 5.0),
    ),
    "small_valve": AssetCfg(
        name="small_valve",
        prim="Valve",
        usd=assets.SMALL_VALVE_USD,
        rot=(0.0, 0.0, -0.70710678, 0.70710678),
        viewer={"body_name": "eixo_trans"},
        panel_pos=(0.0, 0.075, 0.0),
        panel_rot=(0.0, 0.0, 0.70710678, 0.70710678),
        joints={
            "RevoluteJoint": {"friction": 0.0002, "dynamic_friction": 0.0, "viscous_friction": 0.0},
            "PrismaticJoint": {
                "stiffness": 100000.0,
                "damping": 1000.0,
                "effort_limit_sim": 1000.0,
            },
        },
        frame_path="Geometry/valvula_gaveta/pivot_registro/eixo_trans",
        frame_rot=(0.0, 0.70710678, 0.0, 0.70710678),
        targets={
            "approaching": {"pos": (0.0, 0.0, 0.2)},
            "rotate_frame": {"pos": (0.0, 0.0, 0.05)},
        },
        grasp_frame="rotate_frame",
        method="wheel_turn",
        dynamics={"friction": (0.02, 1.0), "damping": (0.0, 0.1), "armature": (0.0005, 0.005)},
        solver={"ls_iterations": 50},
    ),
    "circuit_breaker": AssetCfg(
        name="circuit_breaker",
        prim="CircuitBreaker",
        usd=assets.CIRCUIT_BREAKER_USD,
        viewer={"body_name": "lever_pivot"},
        panel_pos=(-0.04, 0.0, 0.0),
        joints={
            "RevoluteJoint": {"friction": 0.02, "dynamic_friction": 0.0, "viscous_friction": 0.0}
        },
        closed_end_stop=True,
        end_stop_damping=100.0,
        frame_path="Geometry/World",
        frame_rot=FACE,
        targets={
            "approaching": {"pos": (0.1, 0.02, 0.02)},
            "lever_pivot_below": {"pos": (0.03, 0.0, -0.05)},
            "lever_pivot_above": {"pos": (0.03, 0.0, 0.09)},
        },
        grasp_frame="lever_pivot_below",
        mode="push",
        method="push",
        dynamics={
            "friction": (0.01, 0.1),
            "damping": (0.0, 0.005),
            "breakaway": (0.0, 0.1),
            "armature": (0.0005, 0.002),
        },
        stuck_breakaway=(0.1, 0.1),
    ),
    "m30_thread": AssetCfg(
        name="m30_thread",
        prim="Thread",
        usd=assets.M30_THREAD_NEWTON_USD,
        viewer={
            "origin_type": "env",
            "env_index": 0,
            "eye": (0.45, 0.65, 1.0),
            "lookat": (1.0, 0.0, 0.7),
        },
        panel_pos=(-0.02, 0.0, 0.0),
        joints={
            "RevoluteJoint": {
                "damping": 0.05,
                "armature": 0.001,
                "friction": 0.1,
                "effort_limit_sim": 5.0,
            },
            "PrismaticJoint": {"effort_limit_sim": 500.0},
        },
        pitch=0.0035,
        upper_limit=0.025,
        axial_reference=0.007,
        frame_path="Geometry/World",
        frame_rot=FACE,
        targets={
            "approaching": {
                "pos": (0.17, 0.0, 0.0),
                "rot": (0.0, -0.04013179253255973, 0.999194395114446, 0.0),
            },
            "grasp": {
                "pos": (0.056, 0.0, 0.0),
                "rot": (0.0, -0.04013179253255973, 0.999194395114446, 0.0),
            },
            "clear": {
                "pos": (0.086, 0.0, 0.0),
                "rot": (0.0, -0.04013179253255973, 0.999194395114446, 0.0),
            },
            "rotate_frame": {"pos": (0.036, 0.0, 0.0)},
        },
        grasp_frame="grasp",
        mode="contact_grasp",
        grip_effort_scale=0.5,
        method="screw_regrasp",
        dynamics={"friction": (0.05, 0.3), "damping": (0.0, 0.1), "armature": (0.001, 0.003)},
        robot_material={"static_friction_range": (0.8, 0.8), "dynamic_friction_range": (0.6, 0.6)},
        mechanism_material={
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
        },
    ),
    "high_torque_valve": AssetCfg(
        name="high_torque_valve",
        prim="Valve",
        usd=assets.HIGH_TORQUE_VALVE_USD,
        rot=(0.0, -0.70710678, 0.0, 0.70710678),
        viewer={"body_name": "nut"},
        panel_pos=(0.0, 0.0, 0.0),
        panel_rot=(0.0, 0.70710678, 0.0, 0.70710678),
        joints={
            "RevoluteJoint": {"friction": 0.02, "dynamic_friction": 0.0, "viscous_friction": 0.0},
            "PrismaticJoint": {
                "stiffness": 100000.0,
                "damping": 1000.0,
                "effort_limit_sim": 1000.0,
            },
        },
        frame_path="Geometry/World/nut",
        frame_rot=(0.0, 0.70710678, 0.0, 0.70710678),
        targets={
            "approaching": {"pos": (-0.04, 0.0, 0.25)},
            "nut_grasp": {"pos": (-0.05, 0.0, 0.16)},
            "rotate_frame": {"pos": (0.0, 0.0, 0.16)},
        },
        grasp_frame="nut_grasp",
        method="wheel_turn",
        dynamics={"friction": (0.02, 0.02)},
    ),
    "m8_thread": AssetCfg(
        name="m8_thread",
        prim="Thread",
        usd=assets.M8_THREAD_NEWTON_USD,
        rot=FACE,
        viewer={"body_name": "nut_pivot"},
        panel_pos=(-0.0454, 0.0, 0.0),
        joints={
            "RevoluteJoint": {
                "damping": 0.05,
                "armature": 0.001,
                "friction": 0.02,
                "effort_limit_sim": 5.0,
            },
            "PrismaticJoint": {
                "stiffness": 100000.0,
                "damping": 200.0,
                "effort_limit_sim": 500.0,
            },
        },
        pitch=0.00125,
        upper_limit=0.015,
        axial_reference=0.0025,
        frame_path="Geometry/World",
        frame_rot=(0.0, 0.25881904510252074, 0.9659258262890683, 0.0),
        targets={
            "approaching": {"pos": (0.17, 0.0, 0.0)},
            "grasp": {"pos": (0.032, 0.0, 0.0)},
            "clear": {"pos": (0.062, 0.0, 0.0)},
            "rotate_frame": {"pos": (0.012, 0.0, 0.0), "rot": FACE},
        },
        grasp_frame="grasp",
        mode="contact_grasp",
        method="screw_regrasp",
        dynamics={"friction": (0.01, 0.03)},
        robot_material={"static_friction_range": (0.8, 0.8), "dynamic_friction_range": (0.6, 0.6)},
        mechanism_material={
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
        },
    ),
    "peg_insertion": AssetCfg(
        name="peg_insertion",
        prim="Peg",
        usd=assets.PEG_INSERTION_NEWTON_USD,
        rot=FACE,
        viewer={"body_name": "peg"},
        panel_pos=(-0.02, 0.0, 0.0),
        joints={
            "RevoluteJoint": {
                "damping": 0.05,
                "armature": 0.001,
                "friction": 0.02,
                "effort_limit_sim": 5.0,
            },
            "PrismaticJoint": {
                "stiffness": 100000.0,
                "damping": 200.0,
                "effort_limit_sim": 500.0,
            },
        },
        pitch=0.00125,
        upper_limit=0.02,
        axial_reference=0.0025,
        frame_path="Geometry/World",
        frame_rot=FACE,
        targets={
            "approaching": {"pos": (0.17, 0.0, 0.0)},
            "grasp": {"pos": (0.057, 0.0, 0.0)},
            "clear": {"pos": (0.087, 0.0, 0.0)},
            "rotate_frame": {"pos": (0.037, 0.0, 0.0)},
        },
        grasp_frame="grasp",
        mode="contact_grasp",
        method="screw_regrasp",
        dynamics={"friction": (0.01, 0.03)},
        robot_material={"static_friction_range": (0.8, 0.8), "dynamic_friction_range": (0.6, 0.6)},
        mechanism_material={
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
        },
    ),
    "lamp": AssetCfg(
        name="lamp",
        prim="Lamp",
        usd=assets.LAMP_NEWTON_USD,
        pos=(0.8, 0.0, 0.9),
        rot=FACE,
        viewer={"body_name": "lamp_pivot"},
        panel_pos=(-0.0647325, 0.0, 0.0),
        joints={
            "RevoluteJoint": {
                "stiffness": 80.0,
                "damping": 1.0,
                "armature": 0.01,
                "friction": 0.0,
                "dynamic_friction": 0.0,
                "viscous_friction": 0.0,
                "effort_limit_sim": 100.0,
            },
            "PrismaticJoint": {
                "stiffness": 5000.0,
                "damping": 50.0,
                "friction": 0.0,
                "dynamic_friction": 0.0,
                "viscous_friction": 0.0,
                "effort_limit_sim": 100.0,
            },
        },
        pitch=0.0064,
        upper_limit=0.024,
        axial_reference=0.024,
        frame_path="Geometry/World/rotation_pivot/lamp_pivot",
        frame_rot=FACE,
        targets={
            "approaching": {"pos": (0.126, 0.0, 0.0)},
            "lamp_grasp": {"pos": (0.076, 0.0, 0.0)},
            "screw_frame": {"pos": (0.076, 0.0, 0.0)},
        },
        grasp_frame="lamp_grasp",
        mode="contact_grasp",
    ),
    "button": AssetCfg(
        name="button",
        prim="Button",
        usd=assets.BUTTON_NEWTON_USD,
        rot=(0.0, 0.7071067811865476, 0.7071067811865476, 0.0),
        viewer={"body_name": "button_pivot"},
        panel_pos=(-0.041, 0.0, 0.0),
        joints={
            "RevoluteJoint": {"friction": 0.02, "effort_limit_sim": 2.0},
            "PrismaticJoint": {"stiffness": 100.0, "damping": 2.0, "effort_limit_sim": 5.0},
        },
        closed_end_stop=True,
        frame_path="Geometry/World",
        frame_rot=FACE,
        targets={
            "lid_approaching": {
                "pos": (0.1, -0.018, 0.0),
                "rot": (0.0, 0.70710678, 0.70710678, 0.0),
            },
            "lid_grasp": {"pos": (0.02375, -0.018, 0.0), "rot": (0.0, 0.70710678, 0.70710678, 0.0)},
            "rotate_frame": {"pos": (-0.0034325, 0.0304632, 0.0059481859)},
            "button_approaching": {"pos": (0.1, 0.0, 0.0)},
            "button_press": {"pos": (0.002, 0.0, 0.0)},
        },
        grasp_frame="lid_grasp",
        mode="contact_grasp",
        dynamics={"friction": (0.005, 0.05)},
        robot_material={"static_friction_range": (0.8, 0.8), "dynamic_friction_range": (0.6, 0.6)},
        mechanism_material={
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
        },
    ),
    "drawer": AssetCfg(
        name="drawer",
        prim="DrawerHousing",
        usd=assets.DRAWER_HOUSING_NEWTON_USD,
        rot=FACE,
        panel_pos=(-0.02, 0.0, 0.0),
        joints={},
        frame_path="Geometry/base",
        frame_rot=(-0.21643961, 0.0, 0.97629601, 0.0),
        targets={
            "drawer_approaching": {"pos": (0.143595, 0.001, 0.055339)},
            "drawer_grasp": {"pos": (0.071595, 0.001, 0.015339)},
            "drawer_pulled": {"pos": (0.097195, 0.001, 0.015339)},
            "drawer_extract_0": {"pos": (0.098095, 0.001, 0.024339)},
            "drawer_extract_1": {"pos": (0.099595, 0.001, 0.027839)},
            "drawer_extract_2": {"pos": (0.113595, 0.001, 0.027839)},
            "drawer_extract_3": {"pos": (0.141595, 0.001, 0.045339)},
        },
        grasp_frame="drawer_grasp",
        mode="contact_grasp",
        robot_material={"static_friction_range": (0.8, 0.8), "dynamic_friction_range": (0.6, 0.6)},
    ),
    "key": AssetCfg(
        name="key",
        prim="Lock",
        usd=assets.LOCK_NEWTON_USD,
        rot=FACE,
        viewer={"body_name": "lock"},
        panel_pos=(-0.03, 0.0, 0.0),
        joints={
            "RevoluteJoint": {
                "damping": 0.01,
                "friction": 0.02,
                "effort_limit_sim": 2.0,
                "armature": 0.001,
            },
        },
        closed_end_stop=True,
        frame_path="Geometry/lock",
        frame_rot=(0.0, 0.7071067811865475, 0.7071067811865476, 0.0),
        targets={
            "key_approaching": {"pos": (0.148, 0.0, 0.0)},
            "key_entry": {"pos": (0.08, 0.0, 0.0)},
            "key_inserted": {"pos": (0.068, 0.0, 0.0)},
            "rotate_frame": {"pos": (0.068, 0.0, 0.0), "rot": FACE},
        },
        grasp_frame="key_inserted",
        dynamics={"friction": (0.005, 0.05)},
        robot_material={"static_friction_range": (0.8, 0.8), "dynamic_friction_range": (0.6, 0.6)},
        mechanism_material={
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
        },
    ),
    "shock_absorber": AssetCfg(
        name="shock_absorber",
        prim="ShockAbsorber",
        usd=assets.SHOCK_ABSORBER_NEWTON_USD,
        rot=FACE,
        viewer={"body_name": "pin"},
        panel_pos=(-0.02, 0.0, 0.0),
        joint="PrismaticJoint",
        joints={
            "PrismaticJoint": {"damping": 2.0, "friction": 0.5, "effort_limit_sim": 20.0},
        },
        frame_path="Geometry/board",
        frame_rot=(0.0, 0.25881904510252074, 0.9659258262890683, 0.0),
        targets={
            "pin_approaching": {"pos": (0.15, 0.0, 0.035)},
            "pin_grasp": {"pos": (0.058, 0.0, 0.035)},
            "pin_push": {"pos": (0.046, 0.0, 0.035)},
            "press_approaching": {"pos": (0.085, 0.0, 0.035), "rot": FACE},
            "press": {"pos": (0.05, 0.0, 0.035), "rot": FACE},
        },
        grasp_frame="pin_grasp",
        mode="contact_grasp",
        dynamics={"friction": (0.005, 0.05)},
        robot_material={"static_friction_range": (0.8, 0.8), "dynamic_friction_range": (0.6, 0.6)},
        mechanism_material={
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
        },
    ),
}

TASKS = {
    "open_ball_valve": TaskCfg(
        name="open_ball_valve",
        asset=ASSETS["ball_valve"],
        start=0.0,
        target=-math.pi / 2,
        tolerance=0.035,
        pose={"roll": (-math.pi, math.pi)},
        joint_offset=(-0.4, 0.0),
        episode_length_s=17.0,
    ),
    "open_small_valve": TaskCfg(
        name="open_small_valve",
        asset=ASSETS["small_valve"],
        start=0.0,
        target=-math.pi / 2,
        tolerance=0.035,
        pose={"pitch": (-math.pi, math.pi)},
        joint_offset=(-0.4, 0.0),
        episode_length_s=17.0,
    ),
    "open_high_torque_valve": TaskCfg(
        name="open_high_torque_valve",
        asset=ASSETS["high_torque_valve"],
        start=0.0,
        target=-math.pi,
        tolerance=0.035,
        episode_length_s=22.0,
    ),
    "switch_circuit_breaker": TaskCfg(
        name="switch_circuit_breaker",
        asset=ASSETS["circuit_breaker"],
        start=math.pi / 6,
        target=-math.pi / 6,
        tolerance=0.1,
        episode_length_s=9.0,
    ),
    "tighten_m30_thread": TaskCfg(
        name="tighten_m30_thread",
        asset=ASSETS["m30_thread"],
        start=0.0,
        target=-4 * math.pi,
        tolerance=4 * math.pi / 7,
        pose={"roll": (-math.pi, math.pi)},
        episode_length_s=57.0,
    ),
}


def all_ids(env, env_ids):
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)
    return env_ids


def mechanism_asset(env):
    return env.scene[env.cfg.task.asset.name]


def mechanism_joint_id(env):
    return mechanism_asset(env).find_joints(env.cfg.task.asset.joint)[0][0]


def mechanism_position(env):
    return mechanism_asset(env).data.joint_pos.torch[:, mechanism_joint_id(env)]


def opening_sign(env):
    return 1.0 if env.cfg.task.target > env.cfg.task.start else -1.0


def write_mechanism_position(env, env_ids, angle, rate=None):
    asset = env.cfg.task.asset
    mechanism = mechanism_asset(env)
    rate = torch.zeros_like(angle) if rate is None else rate
    joints = [mechanism_joint_id(env)]
    pos, vel = angle[:, None], rate[:, None]
    if asset.pitch is not None:
        ratio = asset.pitch / math.tau
        joints.append(mechanism.find_joints("PrismaticJoint")[0][0])
        pos = torch.cat((pos, asset.axial_reference + ratio * pos), dim=-1)
        vel = torch.cat((vel, ratio * vel), dim=-1)
    mechanism.write_joint_position_to_sim_index(
        position=pos.contiguous(), joint_ids=joints, env_ids=env_ids
    )
    mechanism.write_joint_velocity_to_sim_index(
        velocity=vel.contiguous(), joint_ids=joints, env_ids=env_ids
    )


def sync_coupled_joint(env, env_ids):
    ids = all_ids(env, env_ids)
    if env.cfg.task.asset.pitch is not None:
        write_mechanism_position(env, ids, mechanism_position(env)[ids].clone())


def screw_coupling(asset):
    if asset.pitch is None:
        return None
    return ScrewJointCouplingCfg(
        asset_name=asset.name,
        pitch_m_per_revolution=asset.pitch,
        upper_limit=asset.upper_limit,
        viscous_friction=0.0,
        coulomb_friction=0.0,
        stiction=0.0,
        end_stop_base_damping=0.0,
        end_stop_scale=0.0,
        end_stop_activation_distance=0.0005,
    )


@clone
def spawn_movable_panel(prim_path, cfg, translation=None, orientation=None):
    from pxr import UsdPhysics

    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd_file

    prim = spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    body = UsdPhysics.RigidBodyAPI.Apply(prim)
    body.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(1.0)
    return prim


def reset_panel(env, env_ids, pos, rot):
    ids = all_ids(env, env_ids)
    pose = mechanism_asset(env).data.root_link_pose_w[ids]
    pos, rot = math_utils.combine_frame_transforms(
        pose[:, :3],
        pose[:, 3:7],
        pose.new_tensor(pos).expand(len(ids), -1),
        pose.new_tensor(rot).expand(len(ids), -1),
    )
    panel = env.scene["honeycomb"]
    panel.write_root_pose_to_sim_index(root_pose=torch.cat((pos, rot), dim=-1), env_ids=ids)
    panel.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(len(ids), 6, device=env.device), env_ids=ids
    )


def silence_solver_overflow_warnings(env, env_ids):
    NewtonManager._solver.mjw_model.opt.warn_overflow = False


def reset_joints_from_postures(env, env_ids, postures, position_range, asset_cfg):
    asset = env.scene[asset_cfg.name]
    ids = all_ids(env, env_ids)
    table = torch.tensor(list(postures.values()), device=env.device)
    pick = torch.randint(len(table), (len(ids),), device=env.device)
    noise = torch.empty(len(ids), table.shape[1], device=env.device).uniform_(*position_range)
    limits = asset.data.joint_pos_limits.torch[ids][:, asset_cfg.joint_ids]
    q = (table[pick] + noise).clamp(min=limits[..., 0], max=limits[..., 1])
    asset.write_joint_position_to_sim_index(position=q, joint_ids=asset_cfg.joint_ids, env_ids=ids)
    asset.write_joint_velocity_to_sim_index(
        velocity=torch.zeros_like(q), joint_ids=asset_cfg.joint_ids, env_ids=ids
    )


class randomize_dynamics(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        task = env.cfg.task
        ranges = cfg.params["ranges"]
        assert task.asset.closed_end_stop or (
            ranges["spring"][1] == ranges["breakaway"][1] == cfg.params["stuck_prob"] == 0.0
        ), f"'{task.asset.name}' has no end stop at closed for a spring or seat torque"
        self.mechanism = mechanism_asset(env)
        self.joint_ids = self.mechanism.find_joints(task.asset.joint)[0]
        self.closed_rad = task.start
        self.params = torch.zeros(env.num_envs, len(DYNAMICS), device=env.device)
        stuck_breakaway = cfg.params["stuck_breakaway"]
        scale = [
            max(ranges[name][1], stuck_breakaway[1] if name == "breakaway" else 0.0) or 1.0
            for name in DYNAMICS
        ]
        self.scale = torch.tensor(scale, device=env.device)
        env.dynamics_term = self

    def __call__(self, env, env_ids, ranges, stuck_prob, stuck_breakaway):
        ids = all_ids(env, env_ids)
        values = torch.empty(len(ids), len(DYNAMICS), device=env.device)
        for i, name in enumerate(DYNAMICS):
            values[:, i].uniform_(*ranges[name])
        values[:, 2] = torch.minimum(values[:, 2], values[:, 0] / SPRING_STATIC_RAD)
        stuck = torch.rand(len(ids), device=env.device) < stuck_prob
        if stuck.any():
            n = int(stuck.sum())
            values[stuck, 3] = torch.empty(n, device=env.device).uniform_(*stuck_breakaway)
            closed = torch.full((n, 1), self.closed_rad, device=env.device)
            self.mechanism.write_joint_position_to_sim_index(
                position=closed, joint_ids=self.joint_ids, env_ids=ids[stuck]
            )
            self.mechanism.write_joint_velocity_to_sim_index(
                velocity=torch.zeros_like(closed), joint_ids=self.joint_ids, env_ids=ids[stuck]
            )
        self.write(ids, values)

    def write(self, env_ids, values):
        self.params[env_ids] = values
        self.mechanism.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=values[:, 0:1].contiguous(),
            joint_viscous_friction_coeff=values[:, 1:2].contiguous(),
            joint_ids=self.joint_ids,
            env_ids=env_ids,
        )
        self.mechanism.write_joint_armature_to_sim_index(
            armature=values[:, 4:5].contiguous(), joint_ids=self.joint_ids, env_ids=env_ids
        )


class LoadAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._joint_ids = self._asset.find_joints(env.cfg.task.asset.joint)[0]
        self._closed_rad = env.cfg.task.start
        self._opening_sign = opening_sign(env)
        self._empty = torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def action_dim(self):
        return 0

    @property
    def raw_actions(self):
        return self._empty

    @property
    def processed_actions(self):
        return self._empty

    def process_actions(self, actions):
        pass

    def apply_actions(self):
        term = self._env.dynamics_term
        angle = self._asset.data.joint_pos.torch[:, self._joint_ids[0]]
        spring, breakaway, sign = term.params[:, 2], term.params[:, 3], self._opening_sign
        opening = (sign * (angle - self._closed_rad)).clamp(min=0.0)
        width = (1.0 - opening / BREAKAWAY_WIDTH_RAD).clamp(min=0.0)
        torque = -sign * (spring * opening + breakaway * width)
        self._asset.set_joint_effort_target_index(target=torque[:, None], joint_ids=self._joint_ids)


@configclass
class LoadActionCfg(ActionTermCfg):
    class_type: type = LoadAction
    asset_name: str = MISSING


@configclass
class PhysicsCfg(PresetCfg):
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=600,
            nconmax=300,
            iterations=100,
            ls_iterations=20,
            impratio=10.0,
            ccd_iterations=50,
            use_mujoco_contacts=False,
        ),
        num_substeps=2,
        debug_mode=False,
        use_cuda_graph=False,
    )
    default = newton_mjwarp


@configclass
class ActionsCfg:
    arm_action = JointPositionActionCfg(
        asset_name="robot", scale=1.0, use_default_offset=False, preserve_order=True
    )
    gripper_action = RateLimitedBinaryJointPositionActionCfg(asset_name="robot")
    load = LoadActionCfg()


@configclass
class EventCfg:
    quiet_solver = None
    robot_gravcomp = None
    mechanism_gravcomp = None
    robot_physics_material = None
    mechanism_physics_material = None
    contacts = None
    reset_all = None
    reset_mechanism_root = None
    reset_mechanism_joint = None
    reset_arm = None
    dynamics = None
    reset_panel = None
    end_stops = None


@configclass
class CommandsCfg:
    pose_command = None


@configclass
class HiveBoardEnvCfg(ManagerBasedRLEnvCfg):
    robot: RobotCfg = MISSING
    task: TaskCfg = MISSING
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())


def as_pose(pos, rot):
    return torch.tensor([pos], dtype=torch.float64), torch.tensor([rot], dtype=torch.float64)


def build(cfg, robot_name, task_name):
    robot, task = ROBOTS[robot_name].copy(), TASKS[task_name].copy()
    asset, name = task.asset, task.asset.name
    cfg.robot, cfg.task = robot, task
    cfg.episode_length_s = task.episode_length_s
    cfg.decimation = robot.decimation
    cfg.sim.dt, cfg.sim.render_interval = robot.dt, robot.decimation
    for solver in (cfg.sim.physics.newton_mjwarp, cfg.sim.physics.default):
        solver.solver_cfg = solver.solver_cfg.replace(**asset.solver)
    scene = cfg.scene
    scene.ground = AssetBaseCfg(
        prim_path="/World/Ground",
        collision_group=-1,
        spawn=sim_utils.CuboidCfg(
            size=(20.0, 20.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )
    scene.light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    pos, rot = math_utils.combine_frame_transforms(
        *as_pose(robot.board_pos, robot.board_rot), *as_pose(asset.pos, asset.rot)
    )
    setattr(
        scene,
        name,
        ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/" + asset.prim,
            spawn=sim_utils.UsdFileCfg(
                usd_path=asset.usd,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(pos[0].tolist()),
                rot=tuple(rot[0].tolist()),
                joint_pos={
                    joint: (
                        asset.axial_reference
                        if asset.pitch is not None and joint == "PrismaticJoint"
                        else task.start
                        if joint == asset.joint
                        else 0.0
                    )
                    for joint in asset.joints
                },
                joint_vel={".*": 0.0},
            ),
            actuators={
                joint: ImplicitActuatorCfg(
                    joint_names_expr=[joint],
                    **{"stiffness": 0.0, "damping": 0.0, "effort_limit_sim": 2.0, **values},
                )
                for joint, values in asset.joints.items()
            },
        ),
    )
    pos, rot = math_utils.combine_frame_transforms(
        pos, rot, *as_pose(asset.panel_pos, asset.panel_rot)
    )
    scene.honeycomb = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Honeycomb",
        collision_group=-1,
        spawn=sim_utils.UsdFileCfg(
            func=spawn_movable_panel,
            usd_path=assets.HONEYCOMB_USD,
            scale=(0.001, 0.001, 0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(pos[0].tolist()), rot=tuple(rot[0].tolist())
        ),
    )
    frame_path = "{ENV_REGEX_NS}/" + asset.prim + "/" + asset.frame_path
    scene.target_frame = FrameTransformerCfg(
        prim_path=frame_path,
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=frame_path,
                name=frame,
                offset=OffsetCfg(pos=value["pos"], rot=value.get("rot", asset.frame_rot)),
            )
            for frame, value in asset.targets.items()
        ],
    )
    actuators = dict(robot.articulation.actuators)
    for group, effort in robot.grip_effort.items():
        actuators[group] = actuators[group].replace(effort_limit=effort * asset.grip_effort_scale)
    robot.articulation = robot.articulation.replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene.robot = robot.articulation.replace(actuators=actuators)
    scene.ee_frame = assets.make_ee_frame(robot.ee)
    for sensor, pad in zip(("finger_contact", "jaw_contact"), robot.pads):
        contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + pad,
            update_period=0.0,
            history_length=1,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/" + asset.prim + "/.*"],
        )
        setattr(scene, sensor, contact)
    coupling = screw_coupling(asset)
    if coupling:
        register_screw_joint_mimic(
            coupling, revolute_pos_at_reset=task.start, prismatic_pos_at_reset=asset.axial_reference
        )
    actions = cfg.actions
    actions.arm_action.joint_names = list(robot.arm_joint_names)
    actions.gripper_action.joint_names = list(robot.gripper_joint_names)
    actions.gripper_action.open_command_expr = robot.gripper_open
    actions.gripper_action.close_command_expr = robot.gripper_close
    actions.gripper_action.close_speed = robot.gripper_close_speed
    actions.load.asset_name = name
    events = cfg.events
    events.quiet_solver = EventTermCfg(func=silence_solver_overflow_warnings, mode="startup")
    for key, entity in (("robot", SceneEntityCfg("robot")), ("mechanism", SceneEntityCfg(name))):
        setattr(
            events,
            key + "_gravcomp",
            EventTermCfg(
                func=apply_articulation_gravcomp,
                mode="startup",
                params={"asset_cfg": entity, "gravcomp": 1.0},
            ),
        )
    events.robot_physics_material = EventTermCfg(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[pad.rsplit("/", 1)[-1] for pad in robot.pads]
            ),
            "static_friction_range": (0.3, 0.6),
            "dynamic_friction_range": (0.3, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
            **asset.robot_material,
        },
    )
    events.mechanism_physics_material = EventTermCfg(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(name),
            "static_friction_range": (0.2, 1.0),
            "dynamic_friction_range": (0.2, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
            "make_consistent": True,
            **asset.mechanism_material,
        },
    )
    events.contacts = EventTermCfg(
        func=set_contact_stiffness,
        mode="startup",
        params={
            "ke": 40000.0,
            "kd": 400.0,
            "solimp": (0.95, 0.99, 0.001),
            "shape_regex": f"{robot.contact_shapes}|/{asset.prim}/",
        },
    )
    events.reset_all = EventTermCfg(
        func=base_mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True}
    )
    events.dynamics = EventTermCfg(
        func=randomize_dynamics,
        mode="reset",
        params={
            "ranges": {key: asset.dynamics.get(key, (0.0, 0.0)) for key in DYNAMICS},
            "stuck_prob": asset.stuck_prob,
            "stuck_breakaway": asset.stuck_breakaway,
        },
    )
    events.reset_panel = EventTermCfg(
        func=reset_panel, mode="reset", params={"pos": asset.panel_pos, "rot": asset.panel_rot}
    )
    if asset.end_stop_damping is not None:
        events.end_stops = EventTermCfg(
            func=set_joint_limit_damping,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(name, joint_names=[asset.joint]),
                "kd": asset.end_stop_damping,
            },
        )
    return cfg


class sample_expert_diversity(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.values = torch.zeros(env.num_envs, len(EXPERT_DIVERSITY), device=env.device)
        self.rotate = None
        env.expert_diversity_term = self

    def __call__(self, env, env_ids, ranges, wrist_branch_nearest, grasp_axes, overshoot):
        if self.rotate is None:
            handlers = env.command_manager.get_term("pose_command")._command_handlers
            rotates = [i for i, h in enumerate(handlers) if isinstance(h, _RotateFrameHandler)]
            pushes = (
                []
                if rotates
                else [
                    i
                    for i, h in enumerate(handlers)
                    if isinstance(h, _GoToFrameHandler) and h.cfg.phase == "actuate"
                ]
            )
            actuate = rotates or pushes
            self.rotate = [handlers[i] for i in rotates]
            self.push = [handlers[i] for i in pushes]
            self.reach = [h for h in handlers[: actuate[0]] if isinstance(h, _GoToFrameHandler)]
            self.grasp = self.reach[-1]
            self.retreat = [
                h
                for h in handlers[actuate[-1] + 1 :]
                if isinstance(h, _CuroboPlannedGoToFrameHandler)
            ]

            def goal(handler):
                cfg, name = handler._frame.cfg, handler.cfg.target_frame_name
                frame = next(f for f in cfg.target_frames if f.name == name)
                assert frame.prim_path == cfg.target_frames[0].prim_path
                return math_utils.combine_frame_transforms(
                    *(
                        torch.tensor([x], device=env.device)
                        for x in (
                            frame.offset.pos,
                            frame.offset.rot,
                            handler.cfg.target_offset_pos,
                            handler.cfg.target_offset_rot,
                        )
                    )
                )

            grasp = goal(self.grasp)
            self.moved = self.reach[:-1] + self.push
            self.moved_in_grasp = [
                math_utils.subtract_frame_transforms(*grasp, *goal(h)) for h in self.moved
            ]
        ids = all_ids(env, env_ids)
        n = len(ids)
        values = torch.empty(n, len(EXPERT_DIVERSITY), device=env.device)
        for i, name in enumerate(EXPERT_DIVERSITY[:-1]):
            values[:, i].uniform_(*ranges[name])
        task = env.cfg.task
        friction, _, spring, _, _ = env.dynamics_term.params[ids].unbind(dim=-1)
        torque = friction + spring * abs(task.target - task.start)
        values[:, -1] = overshoot[0] + overshoot[1] * torque
        self.values[ids] = values
        turn_rate, reach_scale, shift, roll, extra = values.unbind(dim=-1)
        for rotate in self.rotate:
            rotate.speed_scale[ids] = turn_rate / rotate.cfg.angular_velocity
            rotate.angle_extra_rad[ids] = extra
        for push in self.push:
            push.speed_scale[ids] = turn_rate / push.cfg.velocity
        for handler in self.reach:
            handler.speed_scale[ids] = reach_scale
        if wrist_branch_nearest:
            for handler in self.reach + self.retreat:
                handler.wrist_branch[ids] = WRIST_BRANCH_NEAREST
                handler.forearm_winding[ids] = 0.0
        shift_axis = torch.tensor(grasp_axes[0], device=env.device).expand(n, 3)
        roll_axis = torch.tensor(grasp_axes[1], device=env.device).expand(n, 3)
        grasp_pos = shift_axis * shift[:, None]
        grasp_quat = math_utils.quat_from_angle_axis(roll, roll_axis)
        self.grasp.env_offset_pos[ids] = grasp_pos
        self.grasp.env_offset_rot[ids] = grasp_quat
        for handler, (k_pos, k_quat) in zip(self.moved, self.moved_in_grasp):
            k_pos, k_quat = k_pos.expand(n, 3), k_quat.expand(n, 4)
            moved = math_utils.combine_frame_transforms(grasp_pos, grasp_quat, k_pos, k_quat)
            handler.env_offset_pos[ids], handler.env_offset_rot[ids] = (
                math_utils.subtract_frame_transforms(k_pos, k_quat, *moved)
            )


def expert(cfg):
    robot, task = cfg.robot, cfg.task
    asset = task.asset
    arm = list(robot.arm_joint_names)
    cfg.actions.arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=arm,
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )
    events = cfg.events
    pose_range = {
        "x": (-0.05, 0.05),
        "y": (-0.06, 0.06),
        "z": (-0.05, 0.05),
        **{axis: (-math.pi / 12, math.pi / 12) for axis in ("roll", "pitch", "yaw")},
        **task.pose,
    }
    events.reset_mechanism_root = EventTermCfg(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                k: (max(low, robot.pose_limits[k][0]), min(high, robot.pose_limits[k][1]))
                for k, (low, high) in pose_range.items()
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg(asset.name),
        },
    )
    events.reset_mechanism_joint = EventTermCfg(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": task.joint_offset,
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg(asset.name, joint_names=[asset.joint]),
        },
    )
    events.reset_arm = EventTermCfg(
        func=reset_joints_from_postures,
        mode="reset",
        params={
            "postures": robot.postures,
            "position_range": robot.arm_range,
            "asset_cfg": SceneEntityCfg("robot", joint_names=arm, preserve_order=True),
        },
    )
    skills = SkillSet(robot_joint_names=arm, **robot.planner)
    planned = {"plan_batch_size": cfg.scene.num_envs}
    goto = {**planned, "wrist_flip_joints": robot.wrist_flip}
    minimum = robot.min_duration_s
    diversity = {
        "turn_rate": (0.25, 0.5),
        "reach_scale": (0.75, 1.3),
        "grasp_shift": (0.0, 0.0),
        "grasp_roll": (0.0, 0.0),
    }
    grasp_axes = ((1.0, 0.0, 0.0),) * 2
    overshoot = (0.0, 0.0)
    max_open_s = math.inf
    if asset.method == "lever_turn":
        diversity.update(grasp_shift=(-0.015, 0.015), grasp_roll=(-0.25, 0.25))
        grasp_axes = ((0.0, 0.0, -1.0),) * 2
        overshoot = (0.03, 0.05)
        if robot.name == "anymal":
            goto["upright_axis"] = (0.0, 1.0, 0.0)
        commands = [
            skills.approach("approaching", chain_with_next=True, **goto),
            skills.engage(asset.grasp_frame, chain_with_next=not minimum["grip"], **goto),
        ]
        if minimum["grip"]:
            commands.append(skills.grip(duration_s=minimum["grip"]))
        commands.append(skills.turn(use_valve_angle=False, **planned))
        if minimum["hold"]:
            commands.append(skills.hold(phase="actuate", duration_s=minimum["hold"]))
        commands.extend(
            [
                skills.release(duration_s=max(0.3, minimum["release"])),
                skills.retreat(
                    "approaching",
                    distance_threshold=0.05,
                    canonicalize_upward=False,
                    hold_current_orientation=True,
                    **goto,
                ),
            ]
        )
    elif asset.method == "wheel_turn":
        diversity.update(grasp_shift=(-0.005, 0.005), grasp_roll=(-0.12, 0.12))
        overshoot = (0.0, 0.05)
        commands = [
            skills.approach("approaching", chain_with_next=True, **goto),
            skills.engage(asset.grasp_frame, **goto),
            skills.grip(duration_s=max(0.3, minimum["grip"])),
            skills.turn(angle_threshold_deg=0.25, **planned),
            skills.release(duration_s=max(0.375, minimum["release"])),
            skills.retreat("approaching", velocity=0.25, hold_current_orientation=True, **goto),
        ]
    elif asset.method == "push":
        diversity.update(turn_rate=(0.1, 0.2), grasp_shift=(-0.01, 0.01), grasp_roll=(-0.3, 0.3))
        grasp_axes = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
        max_open_s = 5.0
        commands = [
            skills.approach("approaching", gripper_open=False, **goto),
            skills.engage(asset.grasp_frame, gripper_open=False, **goto),
            skills.move(
                "lever_pivot_above",
                velocity=0.15,
                distance_threshold=0.02,
                done_when_joint=(asset.name, asset.joint, -1.0, task.target + 0.035),
                **goto,
            ),
            skills.hold(phase="actuate", duration_s=max(0.3, minimum["hold"])),
            skills.retreat(
                "lever_pivot_above",
                target_offset_pos=(-0.07, 0.0, 0.0),
                gripper_open=False,
                velocity=0.15,
                distance_threshold=0.02,
                **goto,
            ),
            skills.retreat(
                "approaching",
                gripper_open=False,
                velocity=0.25,
                hold_current_orientation=True,
                **goto,
            ),
        ]
    elif asset.method == "screw_regrasp":
        diversity["turn_rate"] = (0.5, 0.75)
        turn = {
            **planned,
            "angular_velocity": 0.6,
            "angle_threshold_deg": 2.0,
            "max_ee_rotation_deg": 120.0,
            "screw_pitch_m_per_revolution": asset.pitch,
        }
        commands = [skills.approach("approaching", **goto)]
        for cycle in range(7):
            commands.extend(
                [
                    skills.engage(
                        asset.grasp_frame,
                        velocity=0.1,
                        distance_threshold=0.01,
                        until=MechanismGoalCfg(goal_tolerance=0.1),
                        **goto,
                    ),
                    skills.grip(
                        duration_s=max(0.5, minimum["grip"]),
                        until=MechanismGoalCfg(goal_tolerance=0.1),
                    ),
                    skills.turn(until=MechanismGoalCfg(goal_tolerance=0.1), **turn),
                    skills.release(duration_s=max(0.4, minimum["release"])),
                    skills.retreat(
                        "clear",
                        velocity=0.1,
                        distance_threshold=0.01,
                        hold_current_orientation=cycle == 6,
                        **goto,
                    ),
                ]
            )
        commands.append(skills.retreat("approaching", hold_current_orientation=True, **goto))
    events.expert_diversity = EventTermCfg(
        func=sample_expert_diversity,
        mode="reset",
        params={
            "ranges": diversity,
            "wrist_branch_nearest": bool(robot.wrist_flip),
            "grasp_axes": grasp_axes,
            "overshoot": overshoot,
        },
    )
    events.sync_coupled_joint = EventTermCfg(func=sync_coupled_joint, mode="reset")
    cfg.commands.pose_command = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name=robot.ee.body_name,
        body_offset=SequentialPoseCommandCfg.OffsetCfg(
            pos=tuple(robot.ee.tcp_offset.pos), rot=tuple(robot.ee.tcp_offset.rot)
        ),
        resampling_time_range=(1.0e6, 1.0e6),
        open_task_prob=1.0,
        output_joint_positions=True,
        valve_asset_name=asset.name,
        valve_joint_name=asset.joint,
        valve_joint_open=task.target,
        valve_joint_closed=task.start,
        valve_min_delta_rad=0.35,
        screw_coupling=screw_coupling(asset),
        commands=commands,
    )
    return max_open_s
