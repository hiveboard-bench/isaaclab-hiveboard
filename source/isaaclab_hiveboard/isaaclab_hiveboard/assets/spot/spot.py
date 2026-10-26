# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab_hiveboard.assets import ASSET_DIR
from isaaclab_hiveboard.assets.spot.constants import (
    ARM_ARMATURE,
    ARM_DAMPING,
    ARM_EFFORT_LIMIT,
    ARM_STIFFNESS,
    SPOT_DEFAULT_POS,
    SPOT_STOWED_JOINT_POS,
)


# Kitless urdf-usd-converter output (usd/uuc/). Nested under Geometry/body/...
# Newton MJWarp ignores runtime Implicit stiffness writes, so this config uses
# explicit IdealPD (Lab computes torque and forwards it as joint_f).
SPOT_ARM_UUC_USD = f"{ASSET_DIR}/spot/usd/uuc/spot_with_arm.usda"
SPOT_ARM_UUC_BODY_PRIM = "Geometry/body/arm_link_sh0/arm_link_sh1/arm_link_el0/arm_link_el1/arm_link_wr0/arm_link_wr1"
SPOT_ARM_UUC_WR0_PRIM = "Geometry/body/arm_link_sh0/arm_link_sh1/arm_link_el0/arm_link_el1/arm_link_wr0"
SPOT_ARM_UUC_EL1_PRIM = "Geometry/body/arm_link_sh0/arm_link_sh1/arm_link_el0/arm_link_el1"
SPOT_ARM_UUC_EL0_PRIM = "Geometry/body/arm_link_sh0/arm_link_sh1/arm_link_el0"
SPOT_ARM_UUC_FNGR_PRIM = SPOT_ARM_UUC_BODY_PRIM + "/arm_link_fngr"
SPOT_ARM_UUC_JAW_PRIM = SPOT_ARM_UUC_BODY_PRIM + "/arm_link_jaw"
SPOT_ARM_UUC_SOURCE_PRIM = "Geometry/body"

SPOT_ARM_NEWTON_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=SPOT_ARM_UUC_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True),
        semantic_tags=[("class", "robot")],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=SPOT_DEFAULT_POS,
        joint_pos=SPOT_STOWED_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "fixed_legs": IdealPDActuatorCfg(
            joint_names_expr=["[fh][lr]_(hx|hy|kn)"],
            effort_limit=90.0,
            stiffness=60.0,
            damping=2.0,
            friction=0.02,
            armature=0.01,
        ),
        "arm_sh0": IdealPDActuatorCfg(
            joint_names_expr=["arm_sh0"],
            effort_limit=ARM_EFFORT_LIMIT[0],
            stiffness=ARM_STIFFNESS[0],
            damping=ARM_DAMPING[0],
            armature=ARM_ARMATURE[0],
        ),
        "arm_sh1": IdealPDActuatorCfg(
            joint_names_expr=["arm_sh1"],
            effort_limit=ARM_EFFORT_LIMIT[1],
            stiffness=ARM_STIFFNESS[1],
            damping=ARM_DAMPING[1],
            armature=ARM_ARMATURE[1],
        ),
        "arm_el0": IdealPDActuatorCfg(
            joint_names_expr=["arm_el0"],
            effort_limit=ARM_EFFORT_LIMIT[2],
            stiffness=ARM_STIFFNESS[2],
            damping=ARM_DAMPING[2],
            armature=ARM_ARMATURE[2],
        ),
        "arm_el1": IdealPDActuatorCfg(
            joint_names_expr=["arm_el1"],
            effort_limit=ARM_EFFORT_LIMIT[3],
            stiffness=ARM_STIFFNESS[3],
            damping=ARM_DAMPING[3],
            armature=ARM_ARMATURE[3],
        ),
        "arm_wr0": IdealPDActuatorCfg(
            joint_names_expr=["arm_wr0"],
            effort_limit=ARM_EFFORT_LIMIT[4],
            stiffness=ARM_STIFFNESS[4],
            damping=ARM_DAMPING[4],
            armature=ARM_ARMATURE[4],
        ),
        "arm_wr1": IdealPDActuatorCfg(
            joint_names_expr=["arm_wr1"],
            effort_limit=ARM_EFFORT_LIMIT[5],
            stiffness=ARM_STIFFNESS[5],
            damping=ARM_DAMPING[5],
            armature=ARM_ARMATURE[5],
        ),
        "gripper": IdealPDActuatorCfg(
            joint_names_expr=["arm_f1x"],
            effort_limit=ARM_EFFORT_LIMIT[6],
            stiffness=ARM_STIFFNESS[6],
            damping=ARM_DAMPING[6],
            armature=ARM_ARMATURE[6],
        ),
    },
)
