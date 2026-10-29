# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Weld a rigid tool onto a robot link at spawn time ("glue it in the hand").

The tool USD is referenced under the link prim and joined to it by a
``FixedJoint``, so it becomes one more body of the robot articulation:
contacts on the tool push back on the arm, and its pose is
``robot.data.body_pose_w`` of the tool body. No grasp is simulated, so the
gripper can stay open or close on it without holding anything.

:func:`glue_tool` wraps any plain :class:`~isaaclab.sim.UsdFileCfg` robot
spawn; :func:`tool_pose_in_link` places the tool from a pose in the
canonical TCP frame, so one scene pose serves every robot.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd_file
from isaaclab.sim.utils import clone, create_prim, get_current_stage
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets.end_effector import EndEffectorCfg

Pose = tuple[tuple[float, float, float], tuple[float, float, float, float]]
"""Position [m] and xyzw quaternion."""


def quat_mul(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float, float]:
    """Hamilton product of xyzw quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_apply(q: tuple[float, ...], v: tuple[float, ...]) -> tuple[float, float, float]:
    x, y, z, _ = quat_mul(quat_mul(q, (*v, 0.0)), (-q[0], -q[1], -q[2], q[3]))
    return (x, y, z)


def pose_mul(a: Pose, b: Pose) -> Pose:
    """``a * b``: pose ``b`` expressed in frame ``a``, mapped into ``a``'s parent."""
    pos = tuple(pa + pb for pa, pb in zip(a[0], _quat_apply(a[1], b[0])))
    return pos, quat_mul(a[1], b[1])  # type: ignore[return-value]


def pose_inv(a: Pose) -> Pose:
    q_inv = (-a[1][0], -a[1][1], -a[1][2], a[1][3])
    return tuple(-p for p in _quat_apply(q_inv, a[0])), q_inv  # type: ignore[return-value]


def roll_about_x(angle_deg: float) -> tuple[float, float, float, float]:
    """xyzw rotation of ``angle_deg`` about +X."""
    half = math.radians(angle_deg) / 2.0
    return (math.sin(half), 0.0, 0.0, math.cos(half))


def tool_pose_in_link(ee: EndEffectorCfg, tcp_in_tool: Pose) -> Pose:
    """Tool root pose in ``ee.body_name``'s frame for a TCP at ``tcp_in_tool``.

    Args:
        ee: The robot's canonical TCP profile.
        tcp_in_tool: Where the canonical TCP sits in the tool frame.
    """
    link_tcp = (tuple(ee.tcp_offset.pos), tuple(ee.tcp_offset.rot))
    return pose_mul(link_tcp, pose_inv(tcp_in_tool))  # type: ignore[arg-type]


@configclass
class GluedToolCfg:
    """A rigid tool welded to one robot link."""

    usd_path: str = MISSING  # type: ignore
    """Tool USD. Its default prim must hold exactly one rigid body and no joint to the world."""
    body_name: str = MISSING  # type: ignore
    """Robot link (prim name) the tool is welded to."""
    prim_name: str = "Tool"
    """Name of the tool prim created under the link prim."""
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Tool root position in the link frame [m]."""
    rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Tool root orientation (xyzw) in the link frame."""


def _find_link(robot_root, name: str):
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    matches = [
        prim
        for prim in Usd.PrimRange(robot_root)
        if prim.GetName() == name and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one rigid body named '{name}' under '{robot_root.GetPath()}', found "
            f"{[str(p.GetPath()) for p in matches]}."
        )
    return matches[0]


def attach_glued_tool(robot_prim_path: str, tool: GluedToolCfg) -> None:
    """Reference ``tool`` under its link in the spawned robot and weld it there."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

    stage = get_current_stage()
    link = _find_link(stage.GetPrimAtPath(robot_prim_path), tool.body_name)
    if link.IsInstanceProxy():
        raise RuntimeError(f"Robot link '{link.GetPath()}' is an instance proxy; cannot nest a tool under it.")
    tool_path = str(link.GetPath().AppendChild(tool.prim_name))
    create_prim(tool_path, usd_path=tool.usd_path, stage=stage)
    tool_root = stage.GetPrimAtPath(tool_path)
    xform = UsdGeom.Xformable(tool_root)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*tool.pos))
    x, y, z, w = tool.rot
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))

    bodies = [prim for prim in Usd.PrimRange(tool_root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 1:
        raise RuntimeError(f"Tool '{tool.usd_path}' must hold one rigid body, found {len(bodies)}.")
    body = bodies[0]
    for prim in Usd.PrimRange(tool_root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if prim.IsA(UsdPhysics.Joint):
            raise RuntimeError(f"Tool '{tool.usd_path}' has joint {prim.GetPath()}; it must be a free body.")

    # The weld frame is the tool body's authored pose in the link frame.
    cache = UsdGeom.XformCache()
    body_in_link = cache.GetLocalToWorldTransform(body) * cache.GetLocalToWorldTransform(link).GetInverse()
    body_in_link = body_in_link.RemoveScaleShear()
    joint = UsdPhysics.FixedJoint.Define(stage, f"{tool_path}/GlueJoint")
    joint.CreateBody0Rel().SetTargets([link.GetPath()])
    joint.CreateBody1Rel().SetTargets([body.GetPath()])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(body_in_link.ExtractTranslation()))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(body_in_link.ExtractRotationQuat()))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


@clone
def spawn_robot_with_glued_tool(
    prim_path: str,
    cfg: GluedToolSpawnCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the robot USD, then weld :attr:`GluedToolSpawnCfg.tool` onto it."""
    prim = spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    attach_glued_tool(prim_path, cfg.tool)
    return prim


@configclass
class GluedToolSpawnCfg(sim_utils.UsdFileCfg):
    """A robot :class:`~isaaclab.sim.UsdFileCfg` with a tool welded to one link."""

    func = spawn_robot_with_glued_tool
    tool: GluedToolCfg = MISSING  # type: ignore


def glue_tool(robot: ArticulationCfg, tool: GluedToolCfg) -> ArticulationCfg:
    """Return ``robot`` with ``tool`` welded on at spawn. The robot must spawn from a plain USD file."""
    spawn = robot.spawn
    if type(spawn) is not sim_utils.UsdFileCfg:
        raise TypeError(f"glue_tool needs a plain UsdFileCfg robot spawn, got {type(spawn).__name__}.")
    fields = {f.name: getattr(spawn, f.name) for f in dataclasses.fields(spawn) if f.name != "func"}
    return robot.replace(spawn=GluedToolSpawnCfg(**fields, tool=tool))
