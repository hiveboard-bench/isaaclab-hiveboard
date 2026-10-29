# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Portable saved paths, independent of checkout location and working directory."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from isaaclab_hiveboard.assets import ASSET_DIR
from isaaclab_hiveboard.mdp import curobo_robot_cfg
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import CuroboPlannedGoToFrameCfg
from isaaclab_hiveboard.utils.command_setup import decode_command, encode_command


def test_relative_and_absolute_descriptions_load_from_another_working_directory(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    assets.mkdir()
    yaml = assets / "robot.yaml"
    yaml.write_text("robot_cfg:\n  kinematics:\n    cspace:\n      retract_config: [0.0]\n")
    urdf = assets / "robot.urdf"
    urdf.write_text('<robot name="robot"><joint name="arm" type="revolute"><limit velocity="2.0"/></joint></robot>')
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    monkeypatch.setattr(curobo_robot_cfg, "ASSET_DIR", str(assets))
    for yaml_path, urdf_path in (("robot.yaml", "robot.urdf"), (yaml, urdf)):
        cfg = curobo_robot_cfg.load_curobo_robot_cfg(yaml_path, urdf_path)
        kin = cfg["robot_cfg"]["kinematics"]
        assert kin["urdf_path"] == str(urdf)
        assert kin["asset_root_path"] == str(assets)
        assert kin["cspace"]["default_joint_position"] == [0.0]
        # Keep upstream's newer speed-limit support alongside path resolution.
        curobo_robot_cfg.limit_joint_velocity(cfg, {"arm": 0.5})
        joint = ET.parse(kin["urdf_path"]).getroot().find("joint/limit")
        assert float(joint.get("velocity")) == 0.5
        assert kin["asset_root_path"] == str(assets)


def test_saving_commands_keeps_repository_paths_portable_and_external_paths_explicit():
    for yaml, urdf, expected in (
        (
            str(Path(ASSET_DIR) / "franka/cumotion/fr3.yaml"),
            str(Path(ASSET_DIR) / "franka/cumotion/fr3.urdf"),
            "franka/cumotion/fr3.yaml",
        ),
        ("/opt/custom/robot.yaml", "/opt/custom/robot.urdf", "/opt/custom/robot.yaml"),
    ):
        cmd = CuroboPlannedGoToFrameCfg(
            frame_name="target_frame",
            target_frame_name="approaching",
            robot_joint_names=["fr3_joint1"],
            robot_curobo_yaml=yaml,
            robot_urdf=urdf,
        )
        saved = encode_command(cmd)
        assert saved["parameters"]["robot_curobo_yaml"] == expected
        assert decode_command(json.loads(json.dumps(saved))).robot_curobo_yaml == expected
        assert cmd.robot_curobo_yaml == yaml


@pytest.mark.parametrize(
    "path", sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.json")), ids=lambda p: p.stem
)
def test_saved_robot_paths_are_asset_relative_and_exist(path):
    def check(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("robot_curobo_yaml", "robot_urdf") and item:
                    assert not Path(item).is_absolute(), (path, item)
                    assert (Path(ASSET_DIR) / item).is_file(), (path, item)
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)

    check(json.loads(path.read_text()))
