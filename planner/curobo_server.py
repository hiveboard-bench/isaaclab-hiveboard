#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HTTP cuRobo planner for running the HiveBoard heuristics on a real robot.

It plans the way the CuroboPlanned* commands do
(mdp/commands/sequential_pose_command.py): a GoTo is MotionPlanner.plan_pose
(IK + trajectory optimization, no world collision, graph seeding off); a
Rotate retargets the authored TCP arc frame by frame with MotionRetargeter,
seeded from the grasp joints, and is densified where a joint would jump more
than max_joint_step. Unlike Isaac Lab, an arc whose retargeted joints miss
any waypoint by more than max_arc_error / max_arc_angle is refused rather
than executed. It needs only torch, warp and cuRobo, not Isaac Lab:
`just planner-sync` builds planner/.venv and `just planner` serves; a
client only needs HTTP and JSON.

Poses are fr3_hand in fr3_link0 as [x, y, z, qx, qy, qz, qw]; joints are
fr3_joint1..7 in radians. Endpoints (JSON in, JSON out):

  GET  /info        joint names, tool and base frames, the loaded robot files
  POST /fk          {"joints": [[7]...]} -> {"tool_poses": [[7]...]}
  POST /plan/pose   {"start": [7], "goal": [7], "dt", ...} -> {"success", "joints", "tool_poses"}
  POST /plan/arc    {"start": [7], "poses": [[7]...], "dt", ...} -> {"success", "joints", "tool_poses", "densified"}
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import warp

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.motion_retargeter import MotionRetargeter, MotionRetargeterCfg, SequenceGoalToolPose
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose, ToolPoseCriteria

REPO = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    """Import one source file without importing its package (which pulls in Isaac Lab)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """xyzw -> cuRobo's wxyz."""
    return quat_xyzw[..., [3, 0, 1, 2]]


def xyzw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """cuRobo's wxyz -> xyzw."""
    return quat_wxyz[..., [1, 2, 3, 0]]


def limit_joint_speed(joints: torch.Tensor, max_step: torch.Tensor) -> torch.Tensor:
    """Re-time a joint path so no joint moves more than max_step per waypoint, as _limit_joint_speed does."""
    if joints.shape[0] < 2:
        return joints
    step = ((joints[1:] - joints[:-1]).abs() / max_step).amax(dim=-1)
    if bool((step <= 1.0).all()):
        return joints
    tau = torch.cat((step.new_zeros(1), step.clamp(min=1.0).cumsum(dim=0)))
    count = int(math.ceil(float(tau[-1]) - 1.0e-6)) + 1
    s = torch.arange(count, device=tau.device, dtype=tau.dtype).clamp(max=tau[-1])
    idx = torch.searchsorted(tau, s).clamp(1, len(tau) - 1)
    t0, t1 = tau[idx - 1], tau[idx]
    w = ((s - t0) / (t1 - t0)).clamp(0.0, 1.0)[:, None]
    return joints[idx - 1] * (1.0 - w) + joints[idx] * w


def densify(joints: torch.Tensor, max_joint_step: float) -> torch.Tensor:
    """Interpolate every step larger than max_joint_step, as _densify_arc does for the joints."""
    steps = torch.abs(joints[1:] - joints[:-1]).amax(dim=-1)
    pieces = torch.clamp(torch.ceil(steps / max_joint_step), min=1).long().tolist()
    out = [joints[:1]]
    for i, n in enumerate(pieces):
        frac = torch.arange(1, n + 1, device=joints.device, dtype=joints.dtype)[:, None] / n
        out.append(joints[i] + frac * (joints[i + 1] - joints[i]))
    return torch.cat(out)


class CuroboPlanner:
    """cuRobo solvers for one robot, built lazily and cached per setting, used from one thread."""

    def __init__(self, isaaclab_hiveboard: Path, robot_yaml: Path, urdf: Path):
        assets = isaaclab_hiveboard / "source/isaaclab_hiveboard/isaaclab_hiveboard/mdp"
        self.robot_cfg_module = load_module(assets / "curobo_robot_cfg.py", "hiveboard_curobo_robot_cfg")
        self.robot_yaml = robot_yaml
        self.urdf = urdf
        kinematics = self.robot_cfg_module.load_curobo_robot_cfg(robot_yaml, urdf)["robot_cfg"]["kinematics"]
        self.tool_frame = kinematics["tool_frames"][0]
        self.base_link = kinematics.get("base_link", "")
        self.joint_names = list(kinematics["cspace"]["joint_names"])
        self.device = torch.device("cuda")
        self.cache = {}

        warp.init()

    def _robot_cfg(self, joint_velocity):
        """Robot config, with the URDF joint speed limits lowered when asked."""
        cfg = self.robot_cfg_module.load_curobo_robot_cfg(self.robot_yaml, self.urdf)
        if joint_velocity:
            cfg = self.robot_cfg_module.limit_joint_velocity(cfg, dict(zip(self.joint_names, joint_velocity)))
        return cfg

    def _cached(self, key: tuple, build):
        """Build a solver once per key, outside inference mode, and log how long it took."""
        if key not in self.cache:
            start = time.perf_counter()
            with torch.inference_mode(False):
                self.cache[key] = build()
            print(f"[planner] built {key} in {time.perf_counter() - start:.1f} s", flush=True)
        return self.cache[key]

    def motion_planner(self, num_ik_seeds: int, num_trajopt_seeds: int, dt: float, buffer: int, joint_velocity):
        """The MotionPlanner _plan_env builds, cached by its settings."""
        return self._cached(
            ("planner", num_ik_seeds, num_trajopt_seeds, round(dt, 6), buffer, joint_velocity),
            lambda: MotionPlanner(
                MotionPlannerCfg.create(
                    robot=self._robot_cfg(joint_velocity),
                    self_collision_check=False,
                    use_cuda_graph=True,
                    num_ik_seeds=num_ik_seeds,
                    num_trajopt_seeds=num_trajopt_seeds,
                    interpolation_dt=dt,
                    interpolation_buffer_size=buffer,
                    device_cfg=DeviceCfg(),
                )
            ),
        )

    def retargeter(self, num_ik_seeds: int, dt: float):
        """The MotionRetargeter the rotate handler builds, cached by its settings."""
        return self._cached(
            ("retargeter", num_ik_seeds, round(dt, 6)),
            lambda: MotionRetargeter(
                MotionRetargeterCfg.create(
                    robot=self._robot_cfg(None),
                    tool_pose_criteria={
                        self.tool_frame: ToolPoseCriteria.track_position_and_orientation(
                            xyz=[1.0, 1.0, 1.0], rpy=[1.0, 1.0, 1.0], non_terminal_scale=1.0
                        )
                    },
                    num_envs=1,
                    use_mpc=False,
                    self_collision_check=False,
                    scene_model=None,
                    load_collision_spheres=False,
                    optimization_dt=dt,
                    num_seeds_global=num_ik_seeds,
                    num_seeds_local=1,
                    position_tolerance=0.002,
                    orientation_tolerance=0.02,
                    device_cfg=DeviceCfg(),
                )
            ),
        )

    def _joint_state(self, joints: torch.Tensor) -> JointState:
        """cuRobo joint state for (T, 7) positions in fr3_joint1..7 order."""
        return JointState.from_position(joints.contiguous(), joint_names=self.joint_names)

    def fk(self, joints: torch.Tensor, kinematics=None) -> torch.Tensor:
        """(T, 7) joints -> (T, 7) fr3_hand poses, xyzw."""
        kinematics = kinematics or self.retargeter(4, 0.02).kinematics
        poses = kinematics.compute_kinematics(self._joint_state(joints)).tool_poses.get_link_pose(self.tool_frame)
        return torch.cat((poses.position, xyzw(poses.quaternion)), dim=-1)

    def _tensor(self, values, width: int, name: str) -> torch.Tensor:
        """Request field as float rows of `width` on the GPU, or ValueError."""
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor[None]
        if tensor.shape[-1] != width or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite rows of {width} numbers")
        return tensor

    def info(self) -> dict:
        """Joint names, frames and robot files, for GET /info."""
        return {
            "joint_names": self.joint_names,
            "tool_frame": self.tool_frame,
            "base_link": self.base_link,
            "robot_yaml": str(self.robot_yaml),
            "urdf": str(self.urdf),
        }

    def fk_request(self, request: dict) -> dict:
        """fr3_hand poses for the request's `joints` rows, for POST /fk."""
        return {"tool_poses": self.fk(self._tensor(request["joints"], 7, "joints")).tolist()}

    def plan_pose(self, request: dict) -> dict:
        """Joint trajectory from `start` to the fr3_hand pose `goal`, interpolated at `dt`."""
        start = self._tensor(request["start"], 7, "start")
        goal = self._tensor(request["goal"], 7, "goal")
        dt = float(request.get("dt", 0.02))
        attempts = int(request.get("max_plan_attempts", 1))
        joint_velocity = request.get("max_joint_velocity")
        joint_velocity = tuple(float(v) for v in joint_velocity) if joint_velocity else None
        planner = self.motion_planner(
            int(request.get("num_ik_seeds", 4)),
            int(request.get("num_trajopt_seeds", 1)),
            dt,
            int(request.get("interpolation_buffer_size", 128)),
            joint_velocity if request.get("plan_within_joint_velocity") else None,
        )
        solve_start = time.perf_counter()
        with torch.inference_mode(False), torch.enable_grad():
            current = self._joint_state(start.clone())
            target = GoalToolPose.from_poses(
                {
                    planner.tool_frames[0]: Pose(
                        position=goal[:, :3].clone(),
                        quaternion=wxyz(goal[:, 3:]).contiguous().clone(),
                    )
                },
                ordered_tool_frames=planner.tool_frames,
                num_goalset=1,
            )
            result = planner.plan_pose(
                target,
                current,
                max_attempts=attempts,
                enable_graph_attempt=99,
            )
        if result is None or not bool(result.success.all().item()):
            return {
                "success": False,
                "message": f"no IK solution or trajectory within joint limits after {attempts} attempt(s) "
                "(goal unreachable, or start too close to a limit)",
                "solve_s": time.perf_counter() - solve_start,
            }
        trajectory = result.get_interpolated_plan().position
        if trajectory.ndim == 4:
            trajectory = trajectory[:, 0]
        joints = trajectory[0].detach()
        if joint_velocity:
            joints = limit_joint_speed(joints, torch.as_tensor(joint_velocity, device=joints.device) * dt)
        return {
            "success": True,
            "joints": joints.tolist(),
            "tool_poses": self.fk(joints, planner.kinematics).tolist(),
            "solve_s": time.perf_counter() - solve_start,
        }

    def plan_arc(self, request: dict) -> dict:
        """Joints that track the fr3_hand `poses` in order, seeded from `start` (pose 0 is the start)."""
        start = self._tensor(request["start"], 7, "start")
        poses = self._tensor(request["poses"], 7, "poses")
        dt = float(request.get("dt", 0.02))
        max_joint_step = float(request.get("max_joint_step", 0.15))
        retargeter = self.retargeter(int(request.get("num_ik_seeds", 4)), dt)
        solve_start = time.perf_counter()
        with torch.inference_mode(False), torch.enable_grad():
            current = self._joint_state(start.clone())
            targets = SequenceGoalToolPose(
                tool_frames=[self.tool_frame],
                position=poses[:, None, None, None, :3].contiguous().clone(),
                quaternion=wxyz(poses[:, 3:])[:, None, None, None, :].contiguous().clone(),
            )
            retargeter.reset()
            retargeter._prev_solution = current.position.clone()
            if current.velocity is not None:
                retargeter._prev_velocity = current.velocity.clone()
            solutions = [current.position.clone()]
            for index in range(1, targets.num_frames):
                solutions.append(
                    retargeter.solve_frame(targets.get_frame(index)).joint_state.reorder(self.joint_names).position
                )
        joints = torch.stack(solutions, dim=1)[0].detach()
        largest = float(torch.abs(joints[1:] - joints[:-1]).amax()) if joints.shape[0] > 1 else 0.0
        densified = largest > max_joint_step
        if densified and request.get("on_infeasible_arc", "densify") == "raise":
            return {
                "success": False,
                "message": f"arc needs a {largest:.3f} rad joint step",
            }
        max_error = float(request.get("max_arc_error", 0.005))
        max_angle = float(request.get("max_arc_angle", 0.05))
        reached = self.fk(joints, retargeter.kinematics)
        miss = torch.linalg.vector_norm(reached[:, :3] - poses[:, :3], dim=-1)
        dot = torch.abs((reached[:, 3:] * poses[:, 3:]).sum(-1))
        turn = 2.0 * torch.acos(torch.clamp(dot, max=1.0))
        worst = int(torch.argmax(torch.maximum(miss / max_error, turn / max_angle)))
        if float(miss[worst]) > max_error or float(turn[worst]) > max_angle:
            miss_mm, turn_deg = 1e3 * float(miss[worst]), math.degrees(float(turn[worst]))
            return {
                "success": False,
                "message": f"the retargeted joints miss arc waypoint {worst} of {poses.shape[0] - 1} by "
                f"{miss_mm:.1f} mm / {turn_deg:.1f} deg (limits {1e3 * max_error:.1f} mm / "
                f"{math.degrees(max_angle):.1f} deg), so the arm cannot track the arc from this grasp",
                "solve_s": time.perf_counter() - solve_start,
            }
        if densified:
            joints = densify(joints, max_joint_step)
        tool_poses = self.fk(joints, retargeter.kinematics)
        error = float(torch.linalg.vector_norm(tool_poses[-1, :3] - poses[-1, :3]))
        return {
            "success": True,
            "joints": joints.tolist(),
            "tool_poses": tool_poses.tolist(),
            "densified": densified,
            "largest_joint_step": largest,
            "final_position_error": error,
            "max_arc_error": float(miss.max()),
            "max_arc_angle": float(turn.max()),
            "solve_s": time.perf_counter() - solve_start,
        }


class Handler(BaseHTTPRequestHandler):
    """JSON in, JSON out; reaches the planner and its GPU thread through self.server."""

    server: PlannerServer

    def _reply(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        """Describe the loaded robot."""
        if self.path != "/info":
            self._reply(404, {"error": f"unknown path {self.path}"})
            return
        self._reply(200, self.server.planner.info())

    def do_POST(self) -> None:
        """Dispatch a planning request to the GPU thread."""
        route = self.server.routes.get(self.path)
        if route is None:
            self._reply(404, {"error": f"unknown path {self.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
            self._reply(200, self.server.run(route, request))
        except (KeyError, ValueError, TypeError) as exc:
            self._reply(400, {"error": f"{type(exc).__name__}: {exc}"})
        except Exception as exc:  # noqa: BLE001 - keep serving
            traceback.print_exc()
            self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt, *args) -> None:
        """One line per request on stdout."""
        print("[planner] " + fmt % args, flush=True)


class PlannerServer(ThreadingHTTPServer):
    """HTTP server that runs every planner call on one GPU thread, where cuRobo's CUDA graphs live."""

    def __init__(self, address: tuple[str, int], planner: CuroboPlanner):
        self.planner = planner
        self.gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
        self.routes = {"/plan/pose": planner.plan_pose, "/plan/arc": planner.plan_arc, "/fk": planner.fk_request}
        super().__init__(address, Handler)

    def run(self, function, request: dict, timeout: float = 120.0) -> dict:
        """function(request) on the GPU thread."""
        return self.gpu.submit(function, request).result(timeout=timeout)

    def server_close(self) -> None:
        """Close the socket and drop queued plans."""
        super().server_close()
        self.gpu.shutdown(wait=False, cancel_futures=True)


def main(argv=None) -> int:
    """Load the robot, warm the default solvers and serve."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isaaclab-hiveboard", type=Path, default=REPO, help="isaaclab-hiveboard checkout")
    parser.add_argument("--robot-yaml", type=Path, help="cuRobo robot yaml (default: its FR3 cuMotion yaml)")
    parser.add_argument("--urdf", type=Path, help="robot URDF (default: its FR3 cuMotion URDF)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dt", type=float, default=0.02, help="waypoint spacing to warm up for (1/executor rate)")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    args = parser.parse_args(argv)

    root = args.isaaclab_hiveboard.resolve()
    cumotion = root / "source/isaaclab_hiveboard/isaaclab_hiveboard/assets/franka/cumotion"
    planner = CuroboPlanner(
        root,
        (args.robot_yaml or cumotion / "fr3.yaml").resolve(),
        (args.urdf or cumotion / "fr3.urdf").resolve(),
    )
    with PlannerServer((args.host, args.port), planner) as server:
        if args.warmup:
            home = [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]
            pose = server.run(planner.fk_request, {"joints": [home]})["tool_poses"][0]
            server.run(planner.plan_pose, {"start": home, "goal": pose, "dt": args.dt})
            server.run(planner.plan_arc, {"start": home, "poses": [pose, pose], "dt": args.dt})
        robot = f"{planner.base_link} -> {planner.tool_frame} ({len(planner.joint_names)} joints)"
        print(f"[planner] {robot} on http://{args.host}:{args.port}", flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
