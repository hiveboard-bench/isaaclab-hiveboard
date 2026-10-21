"""Load cuRobo robot descriptions with repo-absolute asset paths."""

from __future__ import annotations

import hashlib
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml

from isaaclab_hiveboard.assets import ASSET_DIR


def _resolve_asset_path(path: str | Path) -> Path:
    path = Path(path).expanduser()

    if not path.is_absolute():
        path = Path(ASSET_DIR) / path

    return path.resolve()


def load_curobo_robot_cfg(
    yaml_path: str | Path,
    urdf_path: str | Path | None = None,
    lock_joints: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Load a cuRobo robot yaml, rebasing relative asset paths onto this repo.

    cuRobo resolves ``urdf_path``/``asset_root_path`` against its own content
    tree. Our yaml lives in the repo, so rewrite those entries to absolute
    paths. The returned dict is passed straight to
    ``InverseKinematicsCfg.create`` (a dict passes ``resolve_config`` through
    unchanged) and must keep the top-level ``robot_cfg`` key.
    """
    payload = yaml.safe_load(_resolve_asset_path(yaml_path).read_text())
    if not isinstance(payload, dict) or "robot_cfg" not in payload:
        raise ValueError(f"Expected a cuRobo robot yaml with a 'robot_cfg' key: {yaml_path}")
    kinematics = payload["robot_cfg"].setdefault("kinematics", {})
    # Drop cuRobo-v1/USD keys the v2 KinematicsLoaderCfg rejects. We solve
    # from the URDF, so none of these apply.
    for legacy_key in (
        "use_usd_kinematics",
        "isaac_usd_path",
        "usd_path",
        "usd_robot_root",
        "usd_flip_joints",
        "usd_flip_joint_limits",
        "use_global_cumul",
    ):
        kinematics.pop(legacy_key, None)
    # v2 names the tool frame list `tool_frames`; v1 yamls carry `ee_link`.
    if "tool_frames" not in kinematics and "ee_link" in kinematics:
        kinematics["tool_frames"] = [kinematics.pop("ee_link")]
    # v2 CSpaceParams calls the seed configuration `default_joint_position`.
    cspace = kinematics.get("cspace")
    if isinstance(cspace, dict) and "default_joint_position" not in cspace and "retract_config" in cspace:
        cspace["default_joint_position"] = cspace.pop("retract_config")
    if urdf_path is not None:
        urdf_path = _resolve_asset_path(urdf_path)
        if not urdf_path.is_file():
            raise FileNotFoundError(f"cuRobo URDF not found: {urdf_path}")
        kinematics["urdf_path"] = str(urdf_path)
        kinematics["asset_root_path"] = str(urdf_path.parent)
    # Joints in cspace must be active or explicitly locked. Our sim welds the
    # body and holds the legs fixed, so lock them at their sim values here.
    if lock_joints:
        locks = kinematics.setdefault("lock_joints", {})
        locks.update(lock_joints)
    return payload


def limit_joint_velocity(robot_cfg: dict[str, Any], max_joint_velocity: dict[str, float]) -> dict[str, Any]:
    """Lower cuRobo's joint speed limits to ``max_joint_velocity`` [rad/s], by joint name.

    cuRobo takes its speed limits from the URDF. This writes a copy of ``urdf_path`` with each named joint's
    limit lowered to ``max_joint_velocity`` (never raised), in the system temp directory, and points
    ``robot_cfg`` at it; mesh paths still resolve against the original ``asset_root_path``. Trajectory
    optimization then plans and times the trajectory within the lower limits. Needs an absolute
    ``urdf_path`` (pass ``urdf_path`` to :func:`load_curobo_robot_cfg`). Modifies and returns ``robot_cfg``.

    ``cspace.velocity_scale`` is not used: cuRobo 0.8 applies it twice (``KinematicsLoader._update_joint_limits``
    and ``CSpaceParams.scale_joint_limits``), so a scale of 0.25 gives a sixteenth of the URDF limit.
    """
    kinematics = robot_cfg["robot_cfg"]["kinematics"]
    urdf_path = Path(kinematics["urdf_path"])
    tree = ET.parse(urdf_path)
    joints = {joint.get("name"): joint for joint in tree.getroot().iter("joint")}
    for name, speed in max_joint_velocity.items():
        limit = joints[name].find("limit") if name in joints else None
        if limit is None or limit.get("velocity") is None:
            raise ValueError(f"The cuRobo URDF has no velocity limit for {name}: {urdf_path}")
        limit.set("velocity", repr(min(float(limit.get("velocity")), float(speed))))
    text = ET.tostring(tree.getroot(), encoding="unicode")
    digest = hashlib.sha1(text.encode()).hexdigest()[:12]
    limited = Path(tempfile.gettempdir()) / "isaaclab_hiveboard_curobo" / f"{urdf_path.stem}_{digest}.urdf"
    if not limited.is_file():
        limited.parent.mkdir(parents=True, exist_ok=True)
        partial = limited.with_suffix(f".{os.getpid()}.tmp")
        partial.write_text(text)
        partial.replace(limited)
    kinematics.setdefault("asset_root_path", str(urdf_path.parent))
    kinematics["urdf_path"] = str(limited)
    return robot_cfg
