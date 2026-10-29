# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lazy asset exports."""

from __future__ import annotations

import importlib
from pathlib import Path

ASSET_DIR = str(Path(__file__).resolve().parent)

_EXPORT_MODULES = {
    # Canonical end-effector profiles.
    "ANYMAL_EE": ".end_effector",
    "FRANKA_EE": ".end_effector",
    "SPOT_EE": ".end_effector",
    "EndEffectorCfg": ".end_effector",
    "make_ee_frame": ".end_effector",
    # HiveBoard paths.
    "BALL_VALVE_FRICTION_RING_URDF": ".hiveboard",
    "BALL_VALVE_URDF": ".hiveboard",
    "BALL_VALVE_USD": ".hiveboard",
    "BUTTON_URDF": ".hiveboard",
    "BUTTON_USD": ".hiveboard",
    "BUTTON_NEWTON_USD": ".hiveboard",
    "CIRCUIT_BREAKER_URDF": ".hiveboard",
    "CIRCUIT_BREAKER_USD": ".hiveboard",
    "DRAWER_URDF": ".hiveboard",
    "DRAWER_USD": ".hiveboard",
    "DRAWER_HOUSING_NEWTON_USD": ".hiveboard",
    "DRAWER_BOX_NEWTON_USD": ".hiveboard",
    "HIGH_TORQUE_VALVE_URDF": ".hiveboard",
    "HIGH_TORQUE_VALVE_USD": ".hiveboard",
    "HIVEBOARD_DIR": ".hiveboard",
    "HIVEBOARD_SIM_DIR": ".hiveboard",
    "HONEYCOMB_URDF": ".hiveboard",
    "HONEYCOMB_USD": ".hiveboard",
    "HONEYCOMB_COLLISION_USD": ".hiveboard",
    "KEY_URDF": ".hiveboard",
    "KEY_USD": ".hiveboard",
    "LOCK_NEWTON_USD": ".hiveboard",
    "LAMP_URDF": ".hiveboard",
    "LAMP_USD": ".hiveboard",
    "LAMP_NEWTON_USD": ".hiveboard",
    "SHOCK_ABSORBER_URDF": ".hiveboard",
    "SHOCK_ABSORBER_USD": ".hiveboard",
    "SHOCK_ABSORBER_NEWTON_USD": ".hiveboard",
    "M8_THREAD_NEWTON_USD": ".hiveboard",
    "M30_THREAD_NEWTON_USD": ".hiveboard",
    "PEG_INSERTION_NEWTON_USD": ".hiveboard",
    "SMALL_VALVE_URDF": ".hiveboard",
    "SMALL_VALVE_USD": ".hiveboard",
    # Robot articulations.
    "FRANKA_NEWTON_CFG": ".franka",
}

__all__ = ["ASSET_DIR", *_EXPORT_MODULES]


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value
