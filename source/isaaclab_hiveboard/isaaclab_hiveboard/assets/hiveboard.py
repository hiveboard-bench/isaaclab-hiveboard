# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset paths and resolution helpers for the HiveBoard benchmark."""

import os
from pathlib import Path

# Package assets directory
ASSET_DIR = os.path.abspath(os.path.dirname(__file__))


def _find_hiveboard_dir() -> str:
    current = Path(__file__).resolve().parent
    for _ in range(8):
        candidate = current / "dependencies" / "HiveBoard"
        if candidate.exists() and (candidate / "Simulation").exists():
            return str(candidate)
        current = current.parent
    # Fallback to current directory
    return str(Path(__file__).resolve().parents[4] / "dependencies" / "HiveBoard")


HIVEBOARD_DIR = os.getenv("HIVEBOARD_DIR", _find_hiveboard_dir())
HIVEBOARD_SIM_DIR = os.path.join(HIVEBOARD_DIR, "Simulation")

# HiveBoard Panel
HONEYCOMB_USD = os.path.join(ASSET_DIR, "hiveboard", "honeycomb", "Honeycomb_Panel.usd")
HONEYCOMB_COLLISION_USD = os.path.join(ASSET_DIR, "hiveboard", "honeycomb", "Honeycomb_Panel_Collision.usda")
HONEYCOMB_URDF = os.path.join(HIVEBOARD_SIM_DIR, "Honeycomb", "Honeycomb_Panel.urdf")

# HiveBoard Interactive Objects - Valves
BALL_VALVE_URDF = os.path.join(ASSET_DIR, "hiveboard", "ball_valve", "Ball_Valve.urdf")
# urdf-usd-converter base (usd/uuc/) + baked CoACD overlay. See
# scripts/generate_newton_usd.py.
BALL_VALVE_USD = os.path.join(ASSET_DIR, "hiveboard", "ball_valve", "usd", "Ball_Valve_uuc_newton.usda")
BALL_VALVE_FRICTION_RING_URDF = os.path.join(
    HIVEBOARD_SIM_DIR, "Valves", "Lever Valve", "Ball Valve", "Ball_Valve_Friction_Ring_Set.urdf"
)
HIGH_TORQUE_VALVE_URDF = os.path.join(
    HIVEBOARD_SIM_DIR, "Valves", "Gate Valve", "High Torque Valve", "High_Torque_Valve.urdf"
)
# urdf-usd-converter base (usd/uuc/) + baked CoACD overlay. See
# scripts/generate_newton_usd.py.
HIGH_TORQUE_VALVE_USD = os.path.join(
    ASSET_DIR, "hiveboard", "high_torque_valve", "usd", "High_Torque_Valve_uuc_newton.usda"
)
SMALL_VALVE_URDF = os.path.join(HIVEBOARD_SIM_DIR, "Valves", "Gate Valve", "Small Valve", "Small_Valve.urdf")
SMALL_VALVE_USD = os.path.join(ASSET_DIR, "hiveboard", "small_valve", "usd", "Small_Valve_uuc_newton.usda")

# HiveBoard Interactive Objects - Electrical & Mechanical
CIRCUIT_BREAKER_URDF = os.path.join(HIVEBOARD_SIM_DIR, "Circuit Breaker", "Circuit_Breaker_Assembly.urdf")
# urdf-usd-converter base (usd/uuc/) + baked CoACD overlay. See
# scripts/generate_newton_usd.py.
CIRCUIT_BREAKER_USD = os.path.join(ASSET_DIR, "hiveboard", "circuit_breaker", "usd", "Circuit_Breaker_uuc_newton.usda")
BUTTON_URDF = os.path.join(HIVEBOARD_SIM_DIR, "Button", "Button_Assembly.urdf")
BUTTON_USD = os.path.join(HIVEBOARD_SIM_DIR, "Button", "Button_Assembly.usd")
# urdf-usd-converter output. See scripts/generate_newton_usd.py.
BUTTON_NEWTON_USD = os.path.join(ASSET_DIR, "hiveboard", "button", "usd", "uuc", "Button_Assembly.usda")
DRAWER_URDF = os.path.join(HIVEBOARD_SIM_DIR, "Drawer", "Drawer_Assembly.urdf")
DRAWER_USD = os.path.join(HIVEBOARD_SIM_DIR, "Drawer", "Drawer_Assembly.usd")
# Free box and kinematic housing, each a UUC of one link from
# hiveboard/drawer/Drawer_Assembly.urdf. Shafts and side cuts are primitives.
# See scripts/generate_newton_usd.py.
DRAWER_HOUSING_NEWTON_USD = os.path.join(
    ASSET_DIR, "hiveboard", "drawer", "usd", "Drawer_Housing_uuc_newton.usda"
)
DRAWER_BOX_NEWTON_USD = os.path.join(ASSET_DIR, "hiveboard", "drawer", "usd", "Drawer_Box_uuc_newton.usda")
KEY_URDF = os.path.join(HIVEBOARD_SIM_DIR, "Key", "Key_assembly.urdf")
KEY_USD = os.path.join(HIVEBOARD_SIM_DIR, "Key", "Key_assembly.usd")
# The rigid upstream asset split in two and converted by UUC: the lock (fixed
# housing + turning plug) and the free key the robot carries. See
# hiveboard/key/{Lock_Assembly,Key}.urdf and scripts/generate_newton_usd.py.
LOCK_NEWTON_USD = os.path.join(ASSET_DIR, "hiveboard", "key", "usd", "uuc_lock", "Lock_Assembly.usda")
KEY_NEWTON_USD = os.path.join(ASSET_DIR, "hiveboard", "key", "usd", "uuc_key", "Key.usda")
LAMP_URDF = os.path.join(HIVEBOARD_SIM_DIR, "Lamp", "Lamp_Assembly.urdf")
LAMP_USD = os.path.join(HIVEBOARD_SIM_DIR, "Lamp", "Lamp_Assembly.usd")
LAMP_NEWTON_USD = os.path.join(ASSET_DIR, "hiveboard", "lamp", "usd", "uuc", "Lamp_Assembly.usda")
SHOCK_ABSORBER_URDF = os.path.join(HIVEBOARD_SIM_DIR, "Shock Absorber", "Shock_Absorber_Assembly.urdf")
SHOCK_ABSORBER_USD = os.path.join(HIVEBOARD_SIM_DIR, "Shock Absorber", "Shock_Absorber_Assembly.usd")
# Articulated re-authorings (free pin, threaded nut/peg on a revolute +
# prismatic pair): UUC base, plus a baked CoACD spring for the shock absorber.
# See hiveboard/<name>/*.urdf and scripts/generate_newton_usd.py.
SHOCK_ABSORBER_NEWTON_USD = os.path.join(
    ASSET_DIR, "hiveboard", "shock_absorber", "usd", "Shock_Absorber_Assembly_uuc_newton.usda"
)
M8_THREAD_NEWTON_USD = os.path.join(ASSET_DIR, "hiveboard", "m8_thread", "usd", "uuc", "M8_Assy.usda")
M30_THREAD_NEWTON_USD = os.path.join(ASSET_DIR, "hiveboard", "m30_thread", "usd", "uuc", "M30.usda")
PEG_INSERTION_NEWTON_USD = os.path.join(ASSET_DIR, "hiveboard", "peg_insertion", "usd", "uuc", "Peg_Insertion.usda")
