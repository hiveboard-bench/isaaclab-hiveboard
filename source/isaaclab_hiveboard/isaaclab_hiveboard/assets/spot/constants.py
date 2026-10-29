# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

"""Constant data for Boston Dynamics Spot robot."""

SPOT_DEFAULT_POS: tuple[float, float, float] = (0.0, 0.0, 0.65)

ARM_JOINT_NAMES = [
    "arm_sh0",
    "arm_sh1",
    "arm_el0",
    "arm_el1",
    "arm_wr0",
    "arm_wr1",
    "arm_f1x",
]

FL_JOINT_NAMES = ["fl_hx", "fl_hy", "fl_kn"]
FR_JOINT_NAMES = ["fr_hx", "fr_hy", "fr_kn"]
HL_JOINT_NAMES = ["hl_hx", "hl_hy", "hl_kn"]
HR_JOINT_NAMES = ["hr_hx", "hr_hy", "hr_kn"]
LEG_JOINT_NAMES = {
    "fl": FL_JOINT_NAMES,
    "fr": FR_JOINT_NAMES,
    "hl": HL_JOINT_NAMES,
    "hr": HR_JOINT_NAMES,
}
FEET_NAMES = [f"{leg_name}_foot" for leg_name in LEG_JOINT_NAMES.keys()]

# That is the arm stowed
SPOT_STOWED_JOINT_POS: dict[str, float] = {
    "arm_sh0": 0.0,
    "arm_sh1": -3.1415,
    "arm_el0": 3.1415,
    "arm_el1": 1.5655,
    "arm_wr0": 0.00,
    "arm_wr1": 0.0,
    "arm_f1x": 0,
    "[fh]l_hx": 0.1,  # all left hip_x
    "[fh]r_hx": -0.1,  # all right hip_x
    "f[rl]_hy": 0.9,  # front hip_y
    "h[rl]_hy": 1.1,  # hind hip_y
    ".*_kn": -1.5,  # all knees
}

SPOT_DEFAULT_JOINT_POS: dict[str, float] = {
    "arm_sh0": 0.0,
    "arm_sh1": -0.9,
    "arm_el0": 1.8,
    "arm_el1": 0.0,
    "arm_wr0": -0.9,
    "arm_wr1": 0.0,
    "arm_f1x": -1.54,
    "fl_hx": 0.12,
    "fr_hx": -0.12,
    "hl_hx": 0.12,
    "hr_hx": -0.12,
    "fl_hy": 0.5,
    "fr_hy": 0.5,
    "hl_hy": 0.5,
    "hr_hy": 0.5,
    "fl_kn": -1.0,
    "fr_kn": -1.0,
    "hl_kn": -1.0,
    "hr_kn": -1.0,
}

HIP_EFFORT_LIMIT: float = 45.0
HIP_STIFFNESS: float = 60.0
HIP_DAMPING: float = 1.5
HIP_FRICTION: float = 0.008

KNEE_STIFFNESS: float = 60.0
KNEE_DAMPING: float = 1.5
KNEE_FRICTION: float = 0.180

ARM_EFFORT_LIMIT: tuple[float, ...] = (90.9, 181.8, 90.9, 30.3, 30.3, 30.3, 15.32)
# ARM_STIFFNESS: tuple[float, ...] = (120.0, 120.0, 120.0, 100.0, 100.0, 100.0, 16.0)

# ARM_DAMPING: tuple[float, ...] = (2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.32)
ARM_STIFFNESS: tuple[float, ...] = (
    1020.0,  # sh0
    255.0,  # sh1
    204,  # el0
    102,  # el1
    102,  # wr0
    102,  # wr1
    16,  # f1x
)

ARM_DAMPING: tuple[float, ...] = (
    10.2,
     15.3,
     10.2,  # Shoulder/Elbow
     2.04,
     2.04,
     2.04,  # Wrist
    0.32,  # Finger
)
ARM_ARMATURE: tuple[float, ...] = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.001)
