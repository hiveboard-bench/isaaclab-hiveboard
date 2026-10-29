# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

NEWTON_GRIPPER_JOINT_NAMES = ["finger_joint"]
ANYMAL_BASE_POS: tuple[float, float, float] = (0.0, 0.0, 0.6)
ANYMAL_HOME_ARM: tuple[float, ...] = (0.0, 0.0, 1.7, 0.0, 0.0, 1.5708)
ANYMAL_NEWTON_JOINT_POS: dict[str, float] = {
    ".*HAA": 0.0,
    ".*F_HFE": 0.4,
    ".*H_HFE": -0.4,
    ".*F_KFE": -0.8,
    ".*H_KFE": 0.8,
    "dynaarm_shoulder_rotation": ANYMAL_HOME_ARM[0],
    "dynaarm_shoulder_flexion": ANYMAL_HOME_ARM[1],
    "dynaarm_elbow_flexion": ANYMAL_HOME_ARM[2],
    "dynaarm_forearm_rotation": ANYMAL_HOME_ARM[3],
    "dynaarm_wrist_flexion": ANYMAL_HOME_ARM[4],
    "dynaarm_wrist_rotation": ANYMAL_HOME_ARM[5],
    "finger_joint": 0.0,
    "right_outer_knuckle_joint": 0.0,
    ".*_inner_finger_joint": 0.0,
    ".*_inner_finger_pad_joint": 0.0,
    ".*_outer_finger_joint": 0.0,
}
ANYMAL_BAKED_USDA = str(Path(__file__).resolve().parent / "usd" / "anymal_d_dynaarm_robotiq.usda")


ANYMAL_ARM_NEWTON_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ANYMAL_BAKED_USDA,
        activate_contact_sensors=True,
        # PhysX: disable_gravity on the root (nested-stop). Newton: per-body
        # gravcomp is applied at startup on every robot link (see events).
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        # USD already articulates from the ANYmal trunk; do not set
        # fix_root_link (PhysX-only writer).
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        semantic_tags=[("class", "robot")],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=ANYMAL_BASE_POS,
        joint_pos=ANYMAL_NEWTON_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "fixed_legs": IdealPDActuatorCfg(
            joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
            effort_limit=90.0,
            stiffness=60.0,
            damping=2.0,
            friction=0.02,
            armature=0.01,
        ),
        "dynaarm_shoulder": IdealPDActuatorCfg(
            joint_names_expr=["dynaarm_shoulder_rotation", "dynaarm_shoulder_flexion"],
            effort_limit=40.0,
            velocity_limit=4.0,
            stiffness=200.0,
            damping=20.0,
        ),
        "dynaarm_elbow": IdealPDActuatorCfg(
            joint_names_expr=["dynaarm_elbow_flexion"],
            effort_limit=40.0,
            velocity_limit=4.0,
            stiffness=200.0,
            damping=20.0,
        ),
        # Roll joints drive tiny reflected inertias (wrist_2 is 48 g); the
        # flat 200/20 group rang them into a ~20 rad/s limit cycle at the
        # 50 Hz control rate. Soft per-joint gains + rotor armature keep the
        # discrete PD stable (kp*dt^2/I < 1 with I >= armature).
        "dynaarm_forearm": IdealPDActuatorCfg(
            joint_names_expr=["dynaarm_forearm_rotation"],
            effort_limit=40.0,
            velocity_limit=4.0,
            stiffness=40.0,
            damping=1.5,
            armature=0.01,
        ),
        # Wrist flexion carries the same light payload as the roll joints
        # above (wrist_2 + flange + the 0.63 kg gripper assembly) and needs the
        # same treatment. On the flat 200/20 gains it held a steady 0.29 rad
        # offset from its commanded angle with the actuator pinned at its
        # 40 N.m ceiling, while every other arm joint tracked to 0.002 rad -
        # gravity on that payload is ~1.3 N.m, so the torque was the discrete
        # PD fighting itself. That 0.29 rad is 16.6 deg of TCP orientation
        # error and ~9 cm of position error, which kept the ball-valve
        # approach from ever reaching its 0.02 m threshold: the sequence
        # stalled at the grasp and no demonstration could succeed. Raising the
        # effort ceiling instead makes it worse (the ringing gets more
        # authority); soft gains plus armature fix it.
        "dynaarm_wrist_flex": IdealPDActuatorCfg(
            joint_names_expr=["dynaarm_wrist_flexion"],
            effort_limit=40.0,
            velocity_limit=4.0,
            stiffness=40.0,
            damping=1.5,
            armature=0.01,
        ),
        "dynaarm_wrist_rot": IdealPDActuatorCfg(
            joint_names_expr=["dynaarm_wrist_rotation"],
            effort_limit=40.0,
            velocity_limit=4.0,
            stiffness=20.0,
            damping=0.9,
            armature=0.01,
        ),
        "gripper": IdealPDActuatorCfg(
            joint_names_expr=list(NEWTON_GRIPPER_JOINT_NAMES),
            effort_limit=40.0,
            velocity_limit=1.5,
            stiffness=80.0,
            damping=4.0,
            armature=0.002,
        ),
        "gripper_linkage": IdealPDActuatorCfg(
            joint_names_expr=[
                "right_outer_knuckle_joint",
                ".*_outer_finger_joint",
                ".*_inner_finger_joint",
                ".*_inner_finger_pad_joint",
            ],
            effort_limit=40.0,
            stiffness=0.0,
            damping=0.1,
            armature=0.002,
        ),
    },
)
"""Kitless Newton ANYmal assembly (baked USD, explicit IdealPD).

Newton MJWarp ignores runtime Implicit stiffness writes, so this config uses
explicit IdealPD (Lab computes torque and forwards it as joint_f), mirroring
the Spot bench config. Gripper gains match the Isaac Sim Robotiq setup.
"""
