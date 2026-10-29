# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-robot flange → TCP profiles.

Task ``target_frame`` poses are desired TCP poses in a single convention, so
HiveBoard scenes can be shared across robots. Tune ``tcp_offset`` once per
robot; commands, IK, ``ee_frame``, and reset IK all read it from here.

Canonical TCP (right-handed):
    +X  out of the palm (approach)
    +Z  jaw "up"
    +Y  across the jaws

Quaternions are Isaac Lab ``(x, y, z, w)``.
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass


@configclass
class EndEffectorCfg:
    """Flange body and the offset from that body to the canonical TCP."""

    body_name: str = MISSING  # type: ignore
    """Articulation body used by pose commands and differential IK."""

    source_prim: str = MISSING  # type: ignore
    """Robot-relative prim for the ``ee_frame`` source (e.g. ``body``, ``fr3_link0``)."""

    body_prim: str = MISSING  # type: ignore
    """Robot-relative prim of the flange body (e.g. ``arm_link_wr1``, ``fr3_hand``)."""

    tcp_offset: OffsetCfg = MISSING  # type: ignore
    """Flange → canonical TCP. Tune this once per robot."""


def make_ee_frame(ee: EndEffectorCfg, *, debug_vis: bool = False) -> FrameTransformerCfg:
    """Build the scene ``ee_frame``. Target 0 is always the TCP."""
    from isaaclab_tasks.core.cabinet.cabinet_env_cfg import (  # isort: skip
        FRAME_MARKER_SMALL_CFG,
    )

    tcp = FrameTransformerCfg.FrameCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + ee.body_prim,
        name="ee_tcp",
        offset=OffsetCfg(pos=tuple(ee.tcp_offset.pos), rot=tuple(ee.tcp_offset.rot)),
    )
    return FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + ee.source_prim,
        debug_vis=debug_vis,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[tcp],
    )


SPOT_EE = EndEffectorCfg(
    body_name="arm_link_wr1",
    source_prim="body",
    body_prim="arm_link_wr1",
    tcp_offset=OffsetCfg(pos=(0.21, 0.0, -0.03), rot=(0.0, 0.0, 0.0, 1.0)),
)
"""Spot arm: ``arm_link_wr1`` +X is already the canonical approach axis."""

# Franka hand axes differ from the shared canonical TCP convention.  This
# fixed transform aligns the fingertip approach direction with canonical +X
# and the thin/top hand axis with canonical +Z, so scene target frames can
# stay identical for every robot.
FRANKA_EE = EndEffectorCfg(
    body_name="fr3_hand",
    source_prim="fr3_link0",
    body_prim="fr3_hand",
    tcp_offset=OffsetCfg(
        pos=(0.0, 0.0, 0.1034),
        # A 180 deg roll about TCP X keeps the red X axis pointing forward
        # through the fingertips and flips the blue Z axis from down to up.
        # Stored xyzw: wxyz (0.0, 0.7071, 0.0, 0.7071) -> xyzw (0.7071, 0.0, 0.7071, 0.0).
        rot=(0.7071068, 0.0, 0.7071068, 0.0),
    ),
)
"""Franka FR3 hand transformed into the shared canonical TCP frame."""


# Isaac Lab 2F-140 (UR10e Robotiq_2f_140 payload): palm +Z is finger approach,
# +Y across the jaws. Rotation -90 deg about Y: TCP +X = body +Z, TCP +Y =
# body +Y, TCP +Z = body -X. Translation 0.20 m along body +Z puts the triad
# on the pads (measured on the 2F-140 palm).
ANYMAL_EE = EndEffectorCfg(
    body_name="robotiq_base_link",
    # The baked Newton assembly nests the ANYmal-D trunk one level deeper
    # than the runtime Isaac Sim spawn path: Robot/anymal/base, not Robot/base
    # (the "anymal" Xform is a sibling reference alongside "dynaarm" and
    # "robotiq_2f_140" — see scripts/generate_anymal_newton_usd.py).
    source_prim="anymal/base",
    body_prim="robotiq_2f_140/robotiq_base_link",
    tcp_offset=OffsetCfg(
        pos=(0.0, 0.0, 0.20),
        rot=(0.0, -0.7071068, 0.0, 0.7071068),
    ),
)
"""ANYmal-D DynaArm + Isaac Lab 2F-140 transformed into the shared canonical TCP frame."""
