from __future__ import annotations

import math
import time
from dataclasses import MISSING, dataclass
from typing import Sequence, Type

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import BaseArticulation
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm
from isaaclab.managers.manager_term_cfg import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkers
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaaclab.utils.configclass import configclass


WRIST_BRANCH_NEAREST = 2.0
"""``wrist_branch`` value picking the IK branch nearest the current joints (see ``_plan_pose_on_branch``)."""

PHASES = ("approach", "engage", "grip", "actuate", "release", "retreat")
"""Phases of a manipulation sequence in their usual order (see ``skills.py``).

:attr:`SequentialPoseCommand.phase` reports them as indices into this tuple,
``-1`` for an untagged segment and ``len(PHASES)`` once the sequence is done.
"""


def _xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Reorder Isaac Lab (x, y, z, w) quaternions to cuRobo (w, x, y, z)."""
    return q[..., [3, 0, 1, 2]]


def _wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Reorder cuRobo (w, x, y, z) quaternions to Isaac Lab (x, y, z, w)."""
    return q[..., [1, 2, 3, 0]]


def _limit_joint_speed(
    tcp_pos: torch.Tensor, tcp_quat: torch.Tensor, joints: torch.Tensor, max_step: float | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Re-time one plan, one waypoint per step, so no joint moves more than ``max_step`` per step.

    ``max_step`` is one value for every joint or one per joint, ``(J,)``. Each interval between
    waypoints takes ``max(1, largest joint change / max_step)`` steps and the
    plan is resampled at one step per waypoint (linear, the quaternions normalized). Intervals within
    the limit keep their timing, faster ones slow to exactly it, and the path is unchanged.
    """
    step = ((joints[1:] - joints[:-1]).abs() / max_step).amax(dim=-1)
    if joints.shape[0] < 2 or bool((step <= 1.0).all()):
        return tcp_pos, tcp_quat, joints
    tau = torch.cat((step.new_zeros(1), step.clamp(min=1.0).cumsum(dim=0)))
    count = int(math.ceil(float(tau[-1]) - 1.0e-6)) + 1
    s = torch.arange(count, device=tau.device, dtype=tau.dtype).clamp(max=tau[-1])
    idx = torch.searchsorted(tau, s).clamp(1, len(tau) - 1)
    t0, t1 = tau[idx - 1], tau[idx]
    w = ((s - t0) / (t1 - t0)).clamp(0.0, 1.0)[:, None]
    q0, q1 = tcp_quat[idx - 1], tcp_quat[idx]
    # Same hemisphere before blending.
    q1 = torch.where((q0 * q1).sum(dim=-1, keepdim=True) < 0.0, -q1, q1)
    quat = torch.nn.functional.normalize(q0 * (1.0 - w) + q1 * w, dim=-1)
    pos = tcp_pos[idx - 1] * (1.0 - w) + tcp_pos[idx] * w
    q = joints[idx - 1] * (1.0 - w.to(joints.dtype)) + joints[idx] * w.to(joints.dtype)
    return pos, quat, q


def _pose_str(p: torch.Tensor, q: torch.Tensor) -> str:
    """Compact position + RPY (deg) rendering for plan audit logs."""
    rpy = torch.rad2deg(torch.stack(math_utils.euler_xyz_from_quat(q), dim=-1))
    return (
        f"pos={[round(float(v), 4) for v in p[0].detach().cpu().tolist()]} "
        f"rpy={[round(float(v), 1) for v in rpy[0].detach().cpu().tolist()]}"
    )


def _vec_str(t: torch.Tensor, prec: int = 4) -> str:
    vals = t.detach().reshape(-1).cpu().tolist()
    return "[" + ", ".join(f"{v:.{prec}f}" for v in vals) + "]"


def _joints_str(names: Sequence[str], q: torch.Tensor, prec: int = 4) -> str:
    vals = q.detach().reshape(-1).cpu().tolist()
    return "  ".join(f"{name}={value:.{prec}f}" for name, value in zip(names, vals))


class SequentialPoseCommand(CommandTerm):
    """A command term that executes a sequence of commands for a robot's end-effector."""

    cfg: "SequentialPoseCommandCfg"

    def __init__(self, cfg: "SequentialPoseCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # obtain the robot asset
        # -- robot
        self._asset: BaseArticulation = env.scene[cfg.asset_name]
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if not body_ids:
            raise ValueError(f"Body with name '{self.cfg.body_name}' not found in asset '{self.cfg.asset_name}'.")
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]

        # -- command sequence
        self._current_command_idx = torch.zeros(self._env.num_envs, device=self._env.device, dtype=torch.long)
        self._command_handlers = [cmd.class_type(cmd, self) for cmd in self.cfg.commands]

        # -- cuRobo solver cache: planner/retargeter construction (kinematics,
        # IK/trajopt solvers, PRM) costs seconds, while solves are sub-second
        # and warm-start from the previous call. Keyed by build config, shared
        # by all handlers so each solver is built once per process lifetime.
        self._curobo_solver_cache: dict = {}

        # -- stuck-segment diagnostics (watchdog + transition log)
        self._seg_time = torch.zeros(self._env.num_envs, device=self._env.device)
        self._seg_warned = torch.zeros(self._env.num_envs, device=self._env.device, dtype=torch.bool)
        self._prev_command_idx = torch.zeros(self._env.num_envs, device=self._env.device, dtype=torch.long)

        # -- build default command, quat can't be 0s
        self._command = torch.zeros((self._env.num_envs, 8), device=self._env.device, dtype=torch.float32)
        self._command[:, 0] = 1  # Close gripper
        self._command[:, 7] = 1.0  # xyzw identity (0,0,0,1)

        self._offset_pos, self._offset_rot = None, None
        self.set_body_offset(self.cfg.body_offset)

        self._initialize_joint_output()

        # -- optional valve task state
        self.valve_task_goal = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
        self.valve_joint_start = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.valve_joint_des = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.valve_rotate_angle_rad = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._valve_asset: BaseArticulation | None = None
        self._valve_joint_idx: int | None = None
        self._initialize_valve_task()

        # -- phase of each segment, plus one entry for the finished sequence
        for cmd in self.cfg.commands:
            if cmd.phase is not None and cmd.phase not in PHASES:
                raise ValueError(f"Unknown phase {cmd.phase!r}; expected one of {PHASES}")
        self._segment_phase = torch.tensor(
            [PHASES.index(cmd.phase) if cmd.phase is not None else -1 for cmd in self.cfg.commands] + [len(PHASES)],
            device=self.device,
            dtype=torch.long,
        )
        # -- optional mechanism goal ending each segment
        self._segment_until = [
            _MechanismGoal(cmd.until, self) if cmd.until is not None else None for cmd in self.cfg.commands
        ]

        # -- optional revolute/prismatic screw coupling
        self._screw_asset: BaseArticulation | None = None
        self._screw_revolute_idx: int | None = None
        self._screw_prismatic_idx: int | None = None
        self._screw_prev_angle = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._screw_axial_target = torch.zeros_like(self._screw_prev_angle)
        self._screw_revolute_target = torch.zeros_like(self._screw_prev_angle)
        self._initialize_screw_coupling()

        if self.cfg.debug_vis:
            self._target_pos_b = torch.zeros((self._env.num_envs, 3), device=self._env.device, dtype=torch.float32)
            self._target_quat_b = torch.zeros((self._env.num_envs, 4), device=self._env.device, dtype=torch.float32)
            self._target_quat_b[:, 3] = 1.0  # xyzw identity (0,0,0,1)

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired command.

        Default (TCP pose, 8-D):
        - gripper status (positive open, negative close)
        - target end-effector position in the base frame
        - target end-effector orientation (xyzw) in the base frame

        With ``output_joint_positions`` (action positions, 1 + n joints):
        - cuRobo joint waypoints in ``robot_joint_names`` order
        - gripper status (positive open, negative close)
        """
        if self._joint_command is not None:
            return self._joint_command
        return self._command

    @property
    def phase(self) -> torch.Tensor:
        """Phase of each env's active segment, ``(num_envs,)``.

        An index into :data:`PHASES`; ``-1`` for an untagged segment and
        ``len(PHASES)`` once the sequence is done.
        """
        return self._segment_phase[self._current_command_idx]

    def _resample_command(self, env_ids: Sequence[int] | slice | None | torch.Tensor = None):
        """Resets the command sequence for the specified environments."""
        from isaaclab_hiveboard.utils.frame_sensors import refresh_frame_sensors

        if isinstance(env_ids, slice) or env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._reset_screw_coupling(env_ids)
        self._sample_valve_task(env_ids)
        refresh_frame_sensors(self._env)
        self._current_command_idx[env_ids] = 0
        self._seg_time[env_ids] = 0.0
        self._seg_warned[env_ids] = False
        self._prev_command_idx[env_ids] = 0
        for handler in self._command_handlers:
            handler.reset(env_ids)
        if self._joint_command is not None:
            self._write_joint_command(self._asset.data.joint_pos.torch[:, self._ik_joint_ids])
        # Isaac Lab's reset() observes the command *before* the first
        # command_manager.compute(). Without this, play.py would step the
        # uninitialized buffer (TCP at the origin) and pull the arm off the
        # lever on the first decimation.
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        env_mask[env_ids_t] = True
        for i, handler in enumerate(self._command_handlers):
            active = env_mask & (self._current_command_idx == i)
            if torch.any(active):
                self._command[active] = handler.update(active)
                if self.cfg.debug_vis:
                    target_pos, target_quat = handler.get_target_in_base_frame(torch.where(active)[0])
                    self._target_pos_b[active] = target_pos
                    self._target_quat_b[active] = target_quat
        self._sync_joint_command()

    def _initialize_joint_output(self) -> None:
        """Publish cuRobo joint waypoints as the command when requested."""
        self._joint_command = None
        self._ik_joint_ids = None
        if not self.cfg.output_joint_positions:
            return
        self._ik_joint_ids = self._resolve_ik_joint_ids()
        self._joint_command = torch.zeros(
            self.num_envs, len(self._ik_joint_ids) + 1, device=self.device, dtype=torch.float32
        )
        self._write_joint_command(self._asset.data.joint_pos.torch[:, self._ik_joint_ids])

    def _resolve_ik_joint_ids(self) -> list[int]:
        names = self.cfg.ik_joint_names
        if not names:
            for cmd in self.cfg.commands:
                names = getattr(cmd, "robot_joint_names", None)
                if names:
                    break
        if not names:
            raise ValueError(
                "output_joint_positions requires ik_joint_names or a cuRobo command with robot_joint_names"
            )
        joint_ids, joint_names = self._asset.find_joints(list(names), preserve_order=True)
        if len(joint_ids) != len(names):
            raise ValueError(f"Expected {len(names)} IK joints {list(names)}, found {len(joint_ids)}: {joint_names}")
        return list(joint_ids)

    def _write_joint_command(self, arm_pos: torch.Tensor) -> None:
        self._joint_command[:, :-1] = arm_pos
        self._joint_command[:, -1] = self._command[:, 0]

    def _sync_joint_command(self) -> None:
        """Copy cuRobo q_des into the published command; otherwise hold the last joints."""
        if self._joint_command is None:
            return
        active, q_des = self.get_curobo_joint_targets()
        if q_des.shape[1] != 0 and torch.any(active):
            self._joint_command[active, :-1] = q_des[active]
        self._joint_command[:, -1] = self._command[:, 0]

    def _initialize_valve_task(self) -> None:
        """Resolve and validate the optional valve articulation and task limits."""
        if self.cfg.valve_asset_name is None:
            return
        if self.cfg.valve_asset_name not in self._env.scene.keys():
            raise ValueError(f"Valve asset '{self.cfg.valve_asset_name}' not found in scene.")
        if not 0.0 <= self.cfg.open_task_prob <= 1.0:
            raise ValueError("open_task_prob must be in [0, 1]")
        valve_span = abs(self.cfg.valve_joint_open - self.cfg.valve_joint_closed)
        if valve_span <= 0.0:
            raise ValueError("valve_joint_open and valve_joint_closed must differ")
        if not 0.0 <= self.cfg.valve_min_delta_rad <= valve_span:
            raise ValueError("valve_min_delta_rad must be within the valve range")

        self._valve_asset = self._env.scene[self.cfg.valve_asset_name]
        joint_ids, _ = self._valve_asset.find_joints(self.cfg.valve_joint_name)
        if len(joint_ids) != 1:
            raise ValueError(
                f"Expected exactly one valve joint named '{self.cfg.valve_joint_name}' "
                f"in '{self.cfg.valve_asset_name}', found {len(joint_ids)}."
            )
        self._valve_joint_idx = joint_ids[0]

    def _sample_valve_task(self, env_ids: torch.Tensor) -> None:
        """Choose a feasible open/close endpoint from the reset valve state.

        The reset event owns valve-state sampling because it also computes the
        matching Spot arm pose. This command term reads that state, samples the
        task direction, and flips infeasible directions near an endpoint so the
        requested motion always satisfies ``valve_min_delta_rad``.
        """
        if self._valve_asset is None or self._valve_joint_idx is None:
            return

        q_start = self._valve_asset.data.joint_pos.torch[env_ids, self._valve_joint_idx]
        q_open = torch.full_like(q_start, self.cfg.valve_joint_open)
        q_closed = torch.full_like(q_start, self.cfg.valve_joint_closed)
        min_delta = float(self.cfg.valve_min_delta_rad)
        can_open = torch.abs(q_open - q_start) >= min_delta
        can_close = torch.abs(q_closed - q_start) >= min_delta
        if torch.any(~(can_open | can_close)):
            bad_q = q_start[~(can_open | can_close)].detach().cpu().tolist()
            raise RuntimeError(
                f"Reset valve states leave no endpoint satisfying valve_min_delta_rad={min_delta}: {bad_q}"
            )

        open_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.open_task_prob
        open_mask = torch.where(can_open & ~can_close, True, open_mask)
        open_mask = torch.where(can_close & ~can_open, False, open_mask)
        goal = torch.where(open_mask, torch.ones_like(q_start), -torch.ones_like(q_start))
        q_des = torch.where(open_mask, q_open, q_closed)

        self.valve_task_goal[env_ids] = goal
        self.valve_joint_start[env_ids] = q_start
        self.valve_joint_des[env_ids] = q_des
        self.recompute_valve_rotate_angle(env_ids)

    def _initialize_screw_coupling(self) -> None:
        """Resolve the optional revolute-to-prismatic screw mechanism."""
        coupling = self.cfg.screw_coupling
        if coupling is None:
            return
        if coupling.asset_name not in self._env.scene.keys():
            raise ValueError(f"Screw asset '{coupling.asset_name}' not found in the scene.")
        if coupling.pitch_m_per_revolution == 0.0:
            raise ValueError("Screw pitch must be non-zero")
        if coupling.lower_limit >= coupling.upper_limit:
            raise ValueError("Screw lower_limit must be less than upper_limit")
        if coupling.velocity_epsilon <= 0.0:
            raise ValueError("Screw velocity_epsilon must be positive")

        self._screw_asset = self._env.scene[coupling.asset_name]
        revolute_ids, _ = self._screw_asset.find_joints(coupling.revolute_joint_name)
        prismatic_ids, _ = self._screw_asset.find_joints(coupling.prismatic_joint_name)
        if len(revolute_ids) != 1 or len(prismatic_ids) != 1:
            raise ValueError("Screw coupling requires exactly one revolute and one prismatic joint")
        self._screw_revolute_idx = revolute_ids[0]
        self._screw_prismatic_idx = prismatic_ids[0]

    def _reset_screw_coupling(self, env_ids: torch.Tensor) -> None:
        """Synchronize coupling state with the articulation after an episode reset."""
        if self._screw_asset is None or self._screw_revolute_idx is None or self._screw_prismatic_idx is None:
            return
        self._screw_prev_angle[env_ids] = self._screw_asset.data.joint_pos.torch[env_ids, self._screw_revolute_idx]
        self._screw_revolute_target[env_ids] = self._screw_prev_angle[env_ids]
        self._screw_axial_target[env_ids] = self._screw_asset.data.joint_pos.torch[env_ids, self._screw_prismatic_idx]
        self._write_screw_targets(env_ids)

    def _write_screw_targets(self, env_ids: torch.Tensor) -> None:
        """Apply the coupled linear target and USD-equivalent resistance torque."""
        if (
            self._screw_asset is None
            or self._screw_revolute_idx is None
            or self._screw_prismatic_idx is None
            or self.cfg.screw_coupling is None
        ):
            return
        coupling = self.cfg.screw_coupling
        velocity = self._screw_asset.data.joint_vel.torch[env_ids, self._screw_revolute_idx]
        speed = torch.abs(velocity)
        coulomb = torch.where(
            speed > coupling.velocity_epsilon,
            -coupling.coulomb_friction * torch.sign(velocity),
            torch.zeros_like(velocity),
        )
        stiction = torch.where(
            speed < coupling.velocity_epsilon,
            -coupling.stiction * velocity / coupling.velocity_epsilon,
            torch.zeros_like(velocity),
        )
        friction = -coupling.viscous_friction * velocity + coulomb + stiction

        distance_to_limit = torch.minimum(
            self._screw_axial_target[env_ids] - coupling.lower_limit,
            coupling.upper_limit - self._screw_axial_target[env_ids],
        )
        end_damping = torch.where(
            distance_to_limit < coupling.end_stop_activation_distance,
            coupling.end_stop_base_damping
            + (coupling.end_stop_scale / (distance_to_limit.clamp_min(0.0) + 1.0e-3)) ** coupling.end_stop_power,
            torch.zeros_like(distance_to_limit),
        )
        resistance = friction - end_damping * velocity

        self._screw_asset.set_joint_position_target(
            self._screw_axial_target[env_ids, None],
            joint_ids=[self._screw_prismatic_idx],
            env_ids=env_ids,
        )
        if coupling.command_joint_angle_scale != 0.0:
            # Drive the revolute with a position target so the bulb turns with
            # the commanded screw. Do not write_joint_state_to_sim: a kinematic
            # teleport while the gripper is in contact explodes PhysX.
            self._screw_asset.set_joint_position_target(
                self._screw_revolute_target[env_ids, None],
                joint_ids=[self._screw_revolute_idx],
                env_ids=env_ids,
            )
        else:
            self._screw_asset.set_joint_effort_target(
                resistance[:, None],
                joint_ids=[self._screw_revolute_idx],
                env_ids=env_ids,
            )

    def _update_screw_coupling(self) -> None:
        """Convert measured revolute travel into axial travel at the screw pitch."""
        if self._screw_asset is None or self._screw_revolute_idx is None or self.cfg.screw_coupling is None:
            return
        coupling = self.cfg.screw_coupling
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        angle = self._screw_asset.data.joint_pos.torch[:, self._screw_revolute_idx]
        raw_delta = angle - self._screw_prev_angle
        angle_delta = torch.atan2(torch.sin(raw_delta), torch.cos(raw_delta))
        self._screw_axial_target += angle_delta * coupling.pitch_m_per_revolution / (2.0 * math.pi)
        self._screw_axial_target.clamp_(min=coupling.lower_limit, max=coupling.upper_limit)
        self._screw_prev_angle.copy_(angle)
        self._write_screw_targets(env_ids)

    def recompute_valve_rotate_angle(self, env_ids: torch.Tensor) -> None:
        """Update the EE arc angle from the valve's remaining joint error."""
        if self._valve_asset is None or self._valve_joint_idx is None:
            return
        q_current = self._valve_asset.data.joint_pos.torch[env_ids, self._valve_joint_idx]
        self.valve_rotate_angle_rad[env_ids] = (self.valve_joint_des[env_ids] - q_current) * float(
            self.cfg.valve_ee_joint_angle_scale
        )

    def _handler_label(self, idx: int) -> str:
        if idx >= len(self._command_handlers):
            return "done"
        return type(self._command_handlers[idx]).__name__

    def _dump_stall(self, env_index: int) -> None:
        """Print everything needed to see why segment ``env_index`` is stuck."""
        idx = int(self._current_command_idx[env_index].item())
        handler = self._command_handlers[idx]
        print(
            f"[STALL] env={env_index} stuck on seg {idx}/{len(self._command_handlers)} "
            f"({type(handler).__name__}) for >={float(self.cfg.stall_timeout_s):.1f}s "
            f"(seg_time={float(self._seg_time[env_index].item()):.1f}s)",
            flush=True,
        )
        env_ids = torch.tensor([env_index], device=self.device, dtype=torch.long)
        try:
            target_pos_b, target_quat_b = handler.get_target_in_base_frame(env_ids)
            ee_pos_b, ee_quat_b = self._get_ee_in_base_frame(env_ids)
            pos_err = float(torch.linalg.vector_norm(ee_pos_b[0] - target_pos_b[0]).item())
            ori_err = float(math.degrees(math_utils.quat_error_magnitude(ee_quat_b, target_quat_b)[0].item()))
            print(
                f"[STALL] target_pos_b={target_pos_b[0].detach().cpu().tolist()} "
                f"ee_pos_b={ee_pos_b[0].detach().cpu().tolist()} pos_err_m={pos_err:.4f} "
                f"ori_err_deg={ori_err:.2f} gripper_cmd={float(self._command[env_index, 0].item()):+.1f}",
                flush=True,
            )
        except Exception as err:  # Gripper holds have no Cartesian target
            print(f"[STALL] (no Cartesian target readout: {err})", flush=True)
        if isinstance(handler, _CuroboPlannedGoToFrameHandler):
            if handler._fallback[env_index]:
                print("[STALL] curobo: plan failed, running fallback direct servo", flush=True)
            elif handler._densified[env_index]:
                print("[STALL] curobo: following a densified fallback arc", flush=True)
            elif not handler._planned[env_index]:
                print("[STALL] curobo: waypoints not planned yet", flush=True)
            else:
                wi = int(handler._waypoint_index[env_ids].item())
                total = int(handler._waypoint_count[env_ids].item())
                print(f"[STALL] curobo: waypoint {wi}/{total}", flush=True)
        try:
            q = self._asset.data.joint_pos[env_index].detach().cpu().tolist()
            print(f"[STALL] joint_pos={[round(v, 3) for v in q]}", flush=True)
        except Exception as err:
            print(f"[STALL] (no joint readout: {err})", flush=True)
        if self._valve_asset is not None and self._valve_joint_idx is not None:
            qv = float(self._valve_asset.data.joint_pos[env_index, self._valve_joint_idx].item())
            qd = float(self.valve_joint_des[env_index].item())
            print(f"[STALL] valve_q={qv:.4f} valve_des={qd:.4f}", flush=True)

    def _update_command(self):
        """Updates the command based on the current state of the command sequence."""
        step_dt = float(getattr(self._env, "step_dt", 0.02))
        changed = self._current_command_idx != self._prev_command_idx
        if torch.any(changed):
            if self.cfg.log_transitions:
                for e in torch.where(changed)[0].tolist():
                    prev = int(self._prev_command_idx[e].item())
                    new = int(self._current_command_idx[e].item())
                    print(
                        f"[SEQ] env={e} seg {prev} ({self._handler_label(prev)}) done "
                        f"in {float(self._seg_time[e].item()):.2f}s "
                        f"-> seg {new} ({self._handler_label(new)})",
                        flush=True,
                    )
            self._seg_time[changed] = 0.0
            self._seg_warned[changed] = False
            self._prev_command_idx[changed] = self._current_command_idx[changed]
        self._seg_time += step_dt
        if float(self.cfg.stall_timeout_s) > 0.0:
            stalled = (
                (self._seg_time >= float(self.cfg.stall_timeout_s))
                & ~self._seg_warned
                & (self._current_command_idx < len(self._command_handlers))
            )
            if torch.any(stalled):
                for e in torch.where(stalled)[0].tolist():
                    self._dump_stall(e)
                self._seg_warned[stalled] = True
        # Update the command from the current handler
        for i, handler in enumerate(self._command_handlers):
            env_mask = self._current_command_idx == i

            if not torch.any(env_mask):
                continue

            # check if the current command is done
            env_ids = torch.where(env_mask)[0]
            are_done = handler.is_done(env_ids)
            if self._segment_until[i] is not None:
                # Checked at any point of the segment: a cuRobo segment's own
                # check only runs once its plan has run out.
                are_done = are_done | self._segment_until[i].holds(env_ids)
            done_env_ids = env_ids[are_done]

            if len(done_env_ids) > 0:
                self._current_command_idx[done_env_ids] = self._current_command_idx[done_env_ids] + 1
                if i < (len(self._command_handlers) - 1):
                    self._command_handlers[i + 1].reset(done_env_ids)

            # update the command for the current handler
            self._command[env_mask] = handler.update(env_mask)

            if self.cfg.debug_vis:
                target_pos_b, target_quat_b = handler.get_target_in_base_frame(torch.where(env_mask)[0])
                self._target_pos_b[env_mask] = target_pos_b
                self._target_quat_b[env_mask] = target_quat_b

        # Apply after handlers so the coupling sees this step's revolute target.
        self._update_screw_coupling()
        self._sync_joint_command()

    def is_done(self) -> torch.Tensor:
        """Check if all commands were finished."""
        return self._current_command_idx == len(self._command_handlers)

    def get_curobo_joint_targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return direct joint targets for environments executing a cuRobo term.

        Used when ``output_joint_positions`` is set.
        Inactive or unplanned segments leave the last joints unchanged.
        """
        targets = torch.zeros(self.num_envs, 0, device=self.device, dtype=self._command.dtype)
        active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        for command_idx, handler in enumerate(self._command_handlers):
            if not isinstance(handler, _CuroboPlannedGoToFrameHandler):
                continue
            if handler._joint_target is None:
                continue
            if targets.shape[1] == 0:
                targets = torch.zeros(
                    self.num_envs,
                    handler._joint_target.shape[1],
                    device=self.device,
                    dtype=handler._joint_target.dtype,
                )
            elif targets.shape[1] != handler._joint_target.shape[1]:
                raise RuntimeError("All cuRobo command terms must command the same joint count.")
            # Unplanned and servo-fallback envs keep their last joints.
            env_mask = (self._current_command_idx == command_idx) & handler.joint_target_mask()
            targets[env_mask] = handler._joint_target[env_mask]
            active |= env_mask
        return active, targets

    def expert_fallback(self) -> torch.Tensor:
        """Per-env flag: the active segment is running a cuRobo fallback.

        Recorded next to every demonstration step, so datasets say which
        expert labels came from a plan cuRobo could not solve as specified.
        """
        flags = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        for command_idx, handler in enumerate(self._command_handlers):
            if isinstance(handler, _CuroboPlannedGoToFrameHandler):
                flags |= (self._current_command_idx == command_idx) & handler.fallback_mask()
        return flags

    def _update_metrics(self):
        """This command term does not have any metrics to update."""
        pass

    def get_curobo_solver(self, key: tuple, make):
        """Return a cached cuRobo solver, building it once per config.

        Args:
            key: Hashable build config identifying the solver.
            make: Zero-argument factory called on cache miss. The caller must
                invoke it inside ``torch.inference_mode(False)`` (and the
                cuRobo warp guard), like a fresh construction.

        Returns:
            The cached solver and the wall-clock build time in seconds
            (0.0 on a cache hit).
        """
        solver = self._curobo_solver_cache.get(key)
        if solver is not None:
            return solver, 0.0
        start = time.perf_counter()
        solver = make()
        build_s = time.perf_counter() - start
        self._curobo_solver_cache[key] = solver
        return solver, build_s

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)

            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self._asset.is_initialized:
            return

        asset_pos = self._asset.data.root_pos_w.torch
        asset_quat = self._asset.data.root_quat_w.torch

        # goal end-effector pose
        target_pos_w, target_quat_w = math_utils.combine_frame_transforms(
            asset_pos,
            asset_quat,
            self._target_pos_b,
            self._target_quat_b,
        )
        self.goal_pose_visualizer.visualize(target_pos_w, target_quat_w)

        # current end-effector pose
        ee_pos_b, ee_quat_b = self._get_ee_in_world_frame(slice(None))
        self.current_pose_visualizer.visualize(ee_pos_b, ee_quat_b)

    def set_body_offset(self, offset) -> None:
        """Replace the flange→TCP offset used by every handler and cuRobo plan.

        The command editor calls this to keep a live-calibrated offset in sync
        with the term; without it, cuRobo would keep planning against the
        offset the term was constructed with.
        """
        self.cfg.body_offset = offset
        # A term without an authored offset treats the flange itself as the TCP.
        if offset is None:
            self._offset_pos, self._offset_rot = None, None
            return
        self._offset_pos = torch.tensor(offset.pos, device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor(offset.rot, device=self.device).repeat(self.num_envs, 1)

    def _body_pose_in_base(self, env_ids: torch.Tensor | slice):
        """Commanded body (``arm_link_wr1``) pose in the robot base frame."""
        return math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w.torch[env_ids],
            self._asset.data.root_quat_w.torch[env_ids],
            self._asset.data.body_pos_w.torch[env_ids, self._body_idx],
            self._asset.data.body_quat_w.torch[env_ids, self._body_idx],
        )

    def _body_pose_to_tcp_pose(self, pos: torch.Tensor, quat: torch.Tensor, env_ids: torch.Tensor | slice):
        """Compose the flange→TCP offset onto a body pose."""
        if self._offset_pos is None or self._offset_rot is None:
            return pos, quat
        return math_utils.combine_frame_transforms(
            pos, quat, self._offset_pos[env_ids], self._offset_rot[env_ids]
        )

    def _tcp_pose_to_body_pose(self, pos: torch.Tensor, quat: torch.Tensor, env_ids: torch.Tensor | slice):
        """Map a TCP pose back to the flange body with the inverse offset."""
        if self._offset_pos is None or self._offset_rot is None:
            return pos, quat
        offset_pos = self._offset_pos[env_ids]
        offset_rot = self._offset_rot[env_ids]
        if offset_pos.shape[0] != pos.shape[0]:
            offset_pos = offset_pos.expand(pos.shape[0], -1)
            offset_rot = offset_rot.expand(pos.shape[0], -1)
        body_from_tcp_quat = math_utils.quat_inv(offset_rot)
        body_from_tcp_pos = -math_utils.quat_apply(body_from_tcp_quat, offset_pos)
        return math_utils.combine_frame_transforms(pos, quat, body_from_tcp_pos, body_from_tcp_quat)

    def _get_ee_in_base_frame(self, env_ids: torch.Tensor | slice):
        """TCP pose in the robot base frame."""
        return self._body_pose_to_tcp_pose(*self._body_pose_in_base(env_ids), env_ids)

    def _get_ee_in_world_frame(self, env_ids: torch.Tensor | slice):
        """TCP pose in the world frame."""
        return self._body_pose_to_tcp_pose(
            self._asset.data.body_pos_w.torch[env_ids, self._body_idx],
            self._asset.data.body_quat_w.torch[env_ids, self._body_idx],
            env_ids,
        )


class _BaseCmdHandler:
    """Base class for command handlers."""

    def __init__(self, cfg: BaseCmd, command_term: SequentialPoseCommand):
        self.cfg = cfg
        self._command_term = command_term
        self._asset = command_term._asset
        self._device = command_term.device
        self._num_envs = command_term.num_envs
        self._dt = float(command_term._env.step_dt)
        # Per-environment variations of the authored command, e.g. for a bank
        # of diverse demonstrations: a factor on ``velocity`` and
        # ``angular_velocity``, and a goal offset composed after
        # ``target_offset_pos``/``target_offset_rot`` (xyzw). Set them before
        # the command plans; the defaults change nothing.
        self.speed_scale = torch.ones(self._num_envs, device=self._device)
        self.env_offset_pos = torch.zeros(self._num_envs, 3, device=self._device)
        self.env_offset_rot = torch.zeros(self._num_envs, 4, device=self._device)
        self.env_offset_rot[:, 3] = 1.0

    def reset(self, env_ids: torch.Tensor):
        """Reset the handler for the given environment IDs."""
        raise NotImplementedError

    def update(self, env_mask: torch.Tensor) -> torch.Tensor:
        """Update the command for the given environment IDs."""
        raise NotImplementedError

    def is_done(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Check if the command is done for the given environment IDs."""
        raise NotImplementedError

    def get_target_in_base_frame(self, env_ids: torch.Tensor):
        """Get the target pose in the base frame for the given environment IDs."""
        raise NotImplementedError

    def _pack_command(self, gripper_open: bool, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        command = torch.zeros(pos.shape[0], 8, device=self._device)
        command[:, 0] = 1.0 if gripper_open else -1.0
        command[:, 1:4] = pos
        command[:, 4:8] = quat
        return command

    def _apply_target_offset(self, pos: torch.Tensor, quat: torch.Tensor, env_ids: torch.Tensor | None = None):
        """Compose the authored goal/pivot offset in its reference frame, then ``env_ids``' own offsets."""
        pos, quat = math_utils.combine_frame_transforms(
            pos, quat,
            pos.new_tensor(self.cfg.target_offset_pos).expand_as(pos),
            quat.new_tensor(self.cfg.target_offset_rot).expand_as(quat),
        )
        if env_ids is None:
            return pos, quat
        return math_utils.combine_frame_transforms(pos, quat, self.env_offset_pos[env_ids], self.env_offset_rot[env_ids])

    def _step_pos_towards(
        self,
        current: torch.Tensor,
        target: torch.Tensor,
        velocity: float | torch.Tensor,
    ) -> torch.Tensor:
        """Advance ``current`` toward ``target`` by at most ``velocity * dt``."""
        delta = target - current
        dist = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        scale = torch.clamp(velocity * self._dt / dist.clamp(min=1.0e-8), max=1.0)
        return current + delta * scale

    def _step_quat_towards(
        self, current: torch.Tensor, target: torch.Tensor, angular_velocity: float | torch.Tensor
    ) -> torch.Tensor:
        """Advance ``current`` toward ``target`` by at most ``angular_velocity * dt``."""
        err = math_utils.quat_box_minus(target, current)
        angle = torch.linalg.vector_norm(err, dim=-1, keepdim=True)
        scale = torch.clamp(angular_velocity * self._dt / angle.clamp(min=1.0e-8), max=1.0)
        return math_utils.quat_box_plus(current, err * scale)


class _GoToFrameHandler(_BaseCmdHandler):
    """Handles the GoToFrame command."""

    def __init__(
        self,
        cfg: GoToFrameCfg,
        command_term: SequentialPoseCommand,
    ):
        super().__init__(cfg, command_term)
        self.cfg: GoToFrameCfg
        if self.cfg.velocity <= 0.0:
            raise ValueError("GoToFrameCfg.velocity must be positive")
        if self.cfg.angular_velocity <= 0.0:
            raise ValueError("GoToFrameCfg.angular_velocity must be positive")
        if self.cfg.distance_threshold < 0.0:
            raise ValueError("GoToFrameCfg.distance_threshold must be non-negative")
        if self.cfg.orientation_threshold_deg < 0.0:
            raise ValueError("GoToFrameCfg.orientation_threshold_deg must be non-negative")

        if cfg.target_position_env is None:
            self._frame = command_term._env.scene[cfg.frame_name]
            self._frame_idx = self._frame.data.target_frame_names.index(cfg.target_frame_name)
        self.command_pos_b = torch.zeros(self._num_envs, 3, device=self._device)
        self.command_quat_b = torch.zeros(self._num_envs, 4, device=self._device)
        self.command_quat_b[:, 3] = 1.0
        self._held_quat_b = torch.zeros(self._num_envs, 4, device=self._device)
        self._held_quat_b[:, 3] = 1.0
        self._upward_flip = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._ori_threshold_rad = math.radians(self.cfg.orientation_threshold_deg)

    def reset(self, env_ids: torch.Tensor):
        ee_pos_b, ee_quat_b = self._command_term._get_ee_in_base_frame(env_ids)
        self.command_pos_b[env_ids] = ee_pos_b
        self.command_quat_b[env_ids] = ee_quat_b
        self._held_quat_b[env_ids] = ee_quat_b
        self._upward_flip[env_ids] = False
        if (
            self.cfg.canonicalize_upward
            and not self.cfg.hold_current_orientation
            and self.cfg.target_position_env is None
        ):
            # Pick the symmetric grasp once per segment. Re-evaluating "up"
            # every step flips the goal by 180 degrees when contact nudges a
            # horizontal handle across +/-90 degrees, preventing completion.
            _, target_quat_b = self.get_target_in_base_frame(env_ids)
            axis = target_quat_b.new_tensor(self.cfg.upright_axis).expand(len(env_ids), -1)
            self._upward_flip[env_ids] = math_utils.quat_apply(target_quat_b, axis)[:, 2] < 0.0

    def update(self, env_mask: torch.Tensor) -> torch.Tensor:
        env_ids = torch.where(env_mask)[0]
        target_pos_b, target_quat_b = self.get_target_in_base_frame(env_ids)

        scale = self.speed_scale[env_ids, None]
        self.command_pos_b[env_ids] = self._step_pos_towards(
            self.command_pos_b[env_ids], target_pos_b, self.cfg.velocity * scale
        )
        self.command_quat_b[env_ids] = self._step_quat_towards(
            self.command_quat_b[env_ids], target_quat_b, self.cfg.angular_velocity * scale
        )
        return self._pack_command(
            self.cfg.gripper_open,
            self.command_pos_b[env_ids],
            self.command_quat_b[env_ids],
        )

    def is_done(self, env_ids: torch.Tensor) -> torch.Tensor:
        ee_pos_b, ee_quat_b = self._command_term._get_ee_in_base_frame(env_ids)
        target_pos_b, target_quat_b = self.get_target_in_base_frame(env_ids)
        pos_err = torch.linalg.vector_norm(ee_pos_b - target_pos_b, dim=-1)
        ori_err = math_utils.quat_error_magnitude(ee_quat_b, target_quat_b)
        done = (pos_err <= self.cfg.distance_threshold) & (ori_err <= self._ori_threshold_rad)
        threshold = getattr(self.cfg, "valve_done_threshold_rad", None)
        term = self._command_term
        if (
            threshold is not None
            and float(threshold) > 0.0
            and term._valve_asset is not None
            and term._valve_joint_idx is not None
        ):
            # Contact-rich arc: the TCP can sit centimeters off while the
            # valve is already turned. Advance on the valve angle instead of
            # hanging on Cartesian thresholds.
            valve_q = term._valve_asset.data.joint_pos.torch[env_ids, term._valve_joint_idx]
            done = done | (torch.abs(valve_q - term.valve_joint_des[env_ids]) <= float(threshold))
        done_when_joint = getattr(self.cfg, "done_when_joint", None)
        if done_when_joint is not None:
            # Contact goals (a press, a push) stop the TCP short of its target
            # by a robot-specific finger length; advance on the mechanism.
            asset_name, joint_name, low, high = done_when_joint
            asset = term._env.scene[asset_name]
            joint_ids, _ = asset.find_joints(joint_name)
            joint_q = asset.data.joint_pos.torch[env_ids, joint_ids[0]]
            done = done | ((joint_q >= float(low)) & (joint_q <= float(high)))
        return done

    def get_target_in_base_frame(self, env_ids: torch.Tensor):
        # Target pose in base frame
        if self.cfg.target_position_env is not None:
            target_pos_w = self._command_term._env.scene.env_origins[env_ids] + torch.tensor(
                self.cfg.target_position_env, device=self._device
            )
            target_quat_w = None
            if self.cfg.target_orientation_env is not None:
                target_quat_w = torch.tensor(self.cfg.target_orientation_env, device=self._device).expand(
                    len(env_ids), -1
                )
            target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
                self._asset.data.root_pos_w.torch[env_ids],
                self._asset.data.root_quat_w.torch[env_ids],
                target_pos_w,
                target_quat_w,
            )
            return target_pos_b, (self._held_quat_b[env_ids] if target_quat_w is None else target_quat_b)
        target_pos_w = self._frame.data.target_pos_w.torch[env_ids, self._frame_idx]
        target_quat_w = self._frame.data.target_quat_w.torch[env_ids, self._frame_idx]

        target_pos_w, target_quat_w = self._apply_target_offset(target_pos_w, target_quat_w, env_ids)

        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w.torch[env_ids],
            self._asset.data.root_quat_w.torch[env_ids],
            target_pos_w,
            target_quat_w,
        )
        if self.cfg.canonicalize_upward:
            flip_x = target_quat_b.new_tensor([1.0, 0.0, 0.0, 0.0]).expand_as(target_quat_b)
            target_quat_b = torch.where(
                self._upward_flip[env_ids, None], math_utils.quat_mul(target_quat_b, flip_x), target_quat_b
            )
        if self.cfg.hold_current_orientation:
            target_quat_b = self._held_quat_b[env_ids]

        return target_pos_b, target_quat_b


class _CuroboPlannedGoToFrameHandler(_GoToFrameHandler):
    """Follow cuRobo joint-plan waypoints for a constrained Cartesian move."""

    def __init__(self, cfg: "CuroboPlannedGoToFrameCfg", command_term: SequentialPoseCommand):
        super().__init__(cfg, command_term)
        self.cfg: CuroboPlannedGoToFrameCfg
        self._init_plan_state()

    def _init_plan_state(self) -> None:
        """Allocate the per-env plan bookkeeping.

        Environments reset and finish segments independently, so every env
        owns its plan. Waypoint buffers are ``(num_envs, capacity, ...)``,
        grown to the longest plan so far; a shorter plan is padded with its
        final waypoint and ``_waypoint_count`` marks where it really ends.
        """
        self._waypoint_pos_b = None
        self._waypoint_quat_b = None
        self._joint_waypoints = None
        self._joint_target = None
        self._waypoint_index = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._waypoint_count = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._planned = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._fallback = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        # Envs following a plan that was patched to get past an infeasible
        # solve (see CuroboPlannedRotateFrameCfg.on_infeasible_arc). Unlike
        # ``_fallback`` the patched plan is still published as joint targets.
        self._densified = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        # Chains (see CuroboPlannedGoToFrameCfg.chain_with_next): envs whose plan
        # a chain leader installed, which the activation reset must keep, and
        # envs whose segment ends on waypoint exhaustion instead of settling.
        self._chain_owned = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._chain_exhaust = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._chain_rejected = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._chain_members_cache: list | None = None
        # Per-env IK branch of the trajopt goal (see CuroboPlannedGoToFrameCfg.wrist_flip_joints):
        # ``wrist_branch`` 0 keeps the IK solver's choice, -1/+1 asks for the
        # wrist pitch joint's sign at the goal, WRIST_BRANCH_NEAREST for the
        # branch nearest the current joints; ``forearm_winding`` +1/-1 is the way the first roll
        # joint turns by pi when flipped, 0 whichever lands nearer its current
        # position.
        self.wrist_branch = torch.zeros(self._num_envs, device=self._device)
        self.forearm_winding = torch.zeros(self._num_envs, device=self._device)

    def _clear_plan_state(self, env_ids: torch.Tensor) -> None:
        self._waypoint_index[env_ids] = 0
        self._waypoint_count[env_ids] = 0
        self._planned[env_ids] = False
        self._fallback[env_ids] = False
        self._densified[env_ids] = False
        self._chain_owned[env_ids] = False
        self._chain_exhaust[env_ids] = False
        self._chain_rejected[env_ids] = False

    def _chain_members(self) -> list:
        """The handlers of the chain this one leads, itself first; empty unless it leads one."""
        if self._chain_members_cache is None:
            handlers = self._command_term._command_handlers
            # A standalone handler (the command editor's preview) is not in
            # the term's sequence, so it leads no chain.
            i = next((k for k, h in enumerate(handlers) if h is self), None)
            members = []
            leads = getattr(self.cfg, "chain_with_next", False)
            if leads and i is not None and not (i > 0 and getattr(handlers[i - 1].cfg, "chain_with_next", False)):
                members = [self]
                j = i
                while getattr(handlers[j].cfg, "chain_with_next", False):
                    j += 1
                    if j >= len(handlers) or not isinstance(handlers[j], _CuroboPlannedGoToFrameHandler):
                        raise ValueError(
                            f"Command {j - 1} sets chain_with_next, but the next command is not a cuRobo "
                            "go-to or rotate command"
                        )
                    members.append(handlers[j])
            self._chain_members_cache = members
        return self._chain_members_cache

    def _reserve_plan_capacity(self, num_waypoints: int, num_joints: int, dtype: torch.dtype) -> None:
        """Make the waypoint buffers hold at least ``num_waypoints`` per env."""
        # Buffers may be written from reset (outside inference mode) and from
        # compute (inside it); inference tensors would reject the former.
        with torch.inference_mode(False):
            if self._waypoint_pos_b is None:
                self._waypoint_pos_b = torch.zeros(self._num_envs, num_waypoints, 3, device=self._device)
                self._waypoint_quat_b = torch.zeros(self._num_envs, num_waypoints, 4, device=self._device)
                self._waypoint_quat_b[..., 3] = 1.0
                self._joint_waypoints = torch.zeros(
                    self._num_envs, num_waypoints, num_joints, device=self._device, dtype=dtype
                )
                self._joint_target = torch.zeros(self._num_envs, num_joints, device=self._device, dtype=dtype)
                return
            pad = num_waypoints - self._waypoint_pos_b.shape[1]
            if pad <= 0:
                return
            # Each row is already padded with its own final waypoint, so the
            # last column is every env's hold pose.
            self._waypoint_pos_b = torch.cat(
                [self._waypoint_pos_b, self._waypoint_pos_b[:, -1:].expand(-1, pad, -1)], dim=1
            )
            self._waypoint_quat_b = torch.cat(
                [self._waypoint_quat_b, self._waypoint_quat_b[:, -1:].expand(-1, pad, -1)], dim=1
            )
            self._joint_waypoints = torch.cat(
                [self._joint_waypoints, self._joint_waypoints[:, -1:].expand(-1, pad, -1)], dim=1
            )

    def _planner_joint_velocity(self) -> dict[str, float] | None:
        """``cfg.max_joint_velocity`` by joint name for the trajectory planners to plan within; None unless
        ``cfg.plan_within_joint_velocity`` is set."""
        max_joint_velocity = getattr(self.cfg, "max_joint_velocity", None)
        if max_joint_velocity is None or not getattr(self.cfg, "plan_within_joint_velocity", False):
            return None
        names = list(self.cfg.robot_joint_names)
        if isinstance(max_joint_velocity, (int, float)):
            return dict.fromkeys(names, float(max_joint_velocity))
        if len(max_joint_velocity) != len(names):
            raise ValueError(
                f"max_joint_velocity has {len(max_joint_velocity)} values for {len(names)} robot_joint_names"
            )
        return {name: float(v) for name, v in zip(names, max_joint_velocity)}

    def _planner_robot_cfg(self, robot_cfg: dict) -> dict:
        """``robot_cfg`` with cuRobo's joint speed limits lowered to :meth:`_planner_joint_velocity`.

        Trajectory optimization then plans and times within them; :meth:`_store_plan` still re-times
        whatever exceeds them (cuRobo clamps its time step, which can leave a long move too fast).
        """
        from isaaclab_hiveboard.mdp.curobo_robot_cfg import limit_joint_velocity

        limits = self._planner_joint_velocity()
        return robot_cfg if limits is None else limit_joint_velocity(robot_cfg, limits)

    def _store_plan(
        self, env_ids: torch.Tensor, tcp_pos: torch.Tensor, tcp_quat: torch.Tensor, joints: torch.Tensor
    ) -> None:
        """Install one env's plan: ``(T, 3)`` TCP positions, ``(T, 4)`` quats, ``(T, J)`` joints."""
        max_joint_velocity = getattr(self.cfg, "max_joint_velocity", None)
        if max_joint_velocity is not None:
            max_step = torch.as_tensor(max_joint_velocity, device=joints.device, dtype=joints.dtype) * self._dt
            tcp_pos, tcp_quat, joints = _limit_joint_speed(tcp_pos, tcp_quat, joints, max_step)
        n = tcp_pos.shape[0]
        self._reserve_plan_capacity(n, joints.shape[-1], joints.dtype)
        self._waypoint_pos_b[env_ids, :n] = tcp_pos
        self._waypoint_pos_b[env_ids, n:] = tcp_pos[-1]
        self._waypoint_quat_b[env_ids, :n] = tcp_quat
        self._waypoint_quat_b[env_ids, n:] = tcp_quat[-1]
        self._joint_waypoints[env_ids, :n] = joints
        self._joint_waypoints[env_ids, n:] = joints[-1]
        self._waypoint_count[env_ids] = n
        self._waypoint_index[env_ids] = 0
        self._planned[env_ids] = True

    def _follow_plan(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance planned envs one waypoint; return their TCP pose and set their joint targets."""
        # CommandManager advances a completed handler and still invokes its
        # update once in that same tick.  Use the final waypoint for that
        # harmless trailing update while retaining the one-past-end completion
        # sentinel in ``_waypoint_index``.
        index = torch.minimum(self._waypoint_index[env_ids], self._waypoint_count[env_ids] - 1)
        pos = self._waypoint_pos_b[env_ids, index]
        quat = self._waypoint_quat_b[env_ids, index]
        self._joint_target[env_ids] = self._joint_waypoints[env_ids, index]
        # Keep one past the final waypoint as a completion sentinel.  Clamping
        # at ``last`` would mark the handler done before ever commanding the
        # final planned joint target.
        self._waypoint_index[env_ids] = index + 1
        return pos, quat

    def joint_target_mask(self) -> torch.Tensor:
        """Envs whose ``_joint_target`` holds a live cuRobo plan."""
        return self._planned & ~self._fallback

    def fallback_mask(self) -> torch.Tensor:
        """Envs whose current segment is not running the plan cuRobo solved.

        Either the solve failed and the handler servos directly (which, with
        ``output_joint_positions``, holds the last joints), or the plan was
        densified past a joint discontinuity, or a chain this command leads
        could not be planned as one motion.
        """
        return self._fallback | self._densified | self._chain_rejected

    def reset(self, env_ids: torch.Tensor):
        # A leader resets before its followers, at episode reset and on its own
        # activation, so the chain it planned last episode is released here.
        for follower in self._chain_members()[1:]:
            follower._chain_owned[env_ids] = False
        fresh = env_ids[~self._chain_owned[env_ids]]
        if len(fresh) > 0:
            super().reset(fresh)
            self._clear_plan_state(fresh)

    def _plan(self, env_ids: torch.Tensor) -> None:
        """Plan every env in ``env_ids``.

        By default one at a time: envs enter a segment at different ticks, so
        each gets its own solve from its own measured state. With
        ``cfg.plan_batch_size > 1`` the envs entering on the same tick are
        solved together in padded batches of that size (``BatchMotionPlanner``
        / a batched ``MotionRetargeter``), with the same targets and joint
        seeding as the single-env path.
        """
        members = self._chain_members()
        if members:
            env_ids = self._plan_chain(env_ids, members)
            if len(env_ids) == 0:
                return
        batch = int(getattr(self.cfg, "plan_batch_size", 1))
        if batch > 1:
            for start in range(0, len(env_ids), batch):
                self._plan_batch(env_ids[start : start + batch])
            return
        for i in range(len(env_ids)):
            self._plan_env(env_ids[i : i + 1])

    def _plan_chain(self, env_ids: torch.Tensor, members: list) -> torch.Tensor:
        """Plan ``members`` as one motion for ``env_ids``; return the envs whose chain was rejected.

        Rejected envs plan their segments one by one but are flagged as an
        expert fallback (see :meth:`fallback_mask`) for the whole leading
        segment, so datasets can drop them.
        """
        size = max(1, int(self.cfg.plan_batch_size))
        rejected = []
        for start in range(0, len(env_ids), size):
            chunk = env_ids[start : start + size]
            try:
                rejected.append(self._plan_chain_batch(chunk, members))
            except Exception as err:
                print(
                    f"[WARN] cuRobo chain plan threw ({type(err).__name__}: {err}); "
                    f"{len(chunk)} envs plan their segments one by one (flagged as fallback)",
                    flush=True,
                )
                rejected.append(chunk)
        rejected = torch.cat(rejected) if rejected else env_ids[:0]
        self._chain_rejected[rejected] = True
        return rejected

    def _chain_legs(
        self, env_ids: torch.Tensor, members: list, pos: torch.Tensor, quat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dense TCP poses of a chain's Cartesian legs at the control rate, from ``pos``/``quat``.

        Go-to legs are straight lines (orientation slerped) at the command's
        ``velocity``/``angular_velocity``; a rotate leg is the command's arc at
        its ``angular_velocity``, placed at the previous leg's end pose.

        Returns:
            ``(pos, quat, counts)``: ``(B, L, 3)`` / ``(B, L, 4)`` poses in the base
            frame (xyzw), excluding the start pose, each env's frames packed first
            and padded with its final pose; ``(B, K)`` frames per leg.
        """
        dt = self._dt
        num = len(env_ids)
        rows = torch.arange(num, device=self._device)
        legs_pos, legs_quat, counts = [], [], []
        for member in members:
            if isinstance(member, _CuroboPlannedRotateFrameHandler):
                _RotateFrameHandler.reset(member, env_ids)
                member._setup_arc(env_ids, pos, quat)
                angle = member.angle_rad_tensor[env_ids]
                angular_velocity = member.cfg.angular_velocity * member.speed_scale[env_ids]
                count = torch.ceil(angle.abs() / (angular_velocity * dt)).clamp(min=1).long()
            else:
                _GoToFrameHandler.reset(member, env_ids)
                target_pos, target_quat = member.get_target_in_base_frame(env_ids)
                scale = member.speed_scale[env_ids]
                duration = torch.maximum(
                    torch.linalg.vector_norm(target_pos - pos, dim=-1) / (member.cfg.velocity * scale),
                    math_utils.quat_error_magnitude(target_quat, quat) / (member.cfg.angular_velocity * scale),
                )
                count = torch.ceil(duration / dt).clamp(min=1).long()
            horizon = int(count.max())
            frac = (torch.arange(1, horizon + 1, device=self._device)[None] / count[:, None]).clamp(max=1.0)
            flat_frac = frac.reshape(-1)
            ids = rows.repeat_interleave(horizon)
            if isinstance(member, _CuroboPlannedRotateFrameHandler):
                flat_ids = env_ids[ids]
                flat_angle = angle[ids] * flat_frac
                axis = member.rot_axis_b[flat_ids]
                leg_pos = (
                    member.axis_pos_b[flat_ids]
                    + member.axial_vec[flat_ids]
                    + member._rodrigues_rotate(member.radius_vec[flat_ids], axis, flat_angle)
                )
                leg_quat = math_utils.quat_mul(
                    math_utils.quat_from_angle_axis(flat_angle, axis), member.initial_quat_b[flat_ids]
                )
            else:
                leg_pos = pos[ids] + (target_pos - pos)[ids] * flat_frac[:, None]
                leg_quat = math_utils.quat_box_plus(
                    quat[ids], math_utils.quat_box_minus(target_quat, quat)[ids] * flat_frac[:, None]
                )
            leg_pos, leg_quat = leg_pos.view(num, horizon, 3), leg_quat.view(num, horizon, 4)
            legs_pos.append(leg_pos)
            legs_quat.append(leg_quat)
            counts.append(count)
            pos, quat = leg_pos[rows, count - 1], leg_quat[rows, count - 1]
        counts = torch.stack(counts, dim=1)
        return self._pack_legs(legs_pos, counts), self._pack_legs(legs_quat, counts), counts

    def _pack_legs(self, legs: list[torch.Tensor], counts: torch.Tensor) -> torch.Tensor:
        """Concatenate ``(B, H_k, D)`` legs keeping each env's first ``counts[:, k]`` frames.

        Each env's frames come first; its tail repeats its final frame.
        """
        valid = torch.cat(
            [torch.arange(leg.shape[1], device=self._device)[None] < c[:, None] for leg, c in zip(legs, counts.T)],
            dim=1,
        )
        order = torch.sort((~valid).int(), dim=1, stable=True).indices
        seq = torch.gather(torch.cat(legs, dim=1), 1, order[..., None].expand(-1, -1, legs[0].shape[-1]))
        total = counts.sum(dim=1)
        rows = torch.arange(len(total), device=self._device)
        tail = torch.arange(seq.shape[1], device=self._device)[None] >= total[:, None]
        return torch.where(tail[..., None], seq[rows, total - 1][:, None], seq)

    @staticmethod
    def _blend_in_time(x: torch.Tensor, half_width: int) -> torch.Tensor:
        """Filter ``(B, T, D)`` samples with a triangular kernel of ``2 * half_width + 1`` frames.

        The ends are padded with ``2 * half_width`` copies of the first and last
        sample, so the output (``T + 2 * half_width`` frames) starts and ends at
        rest exactly on them, and input frame ``i`` is centred on output frame
        ``i + half_width``. Velocity steps become ramps and corners blends.
        """
        h = half_width
        if h <= 0:
            return x
        frames = x.shape[1]
        x = torch.cat((x[:, :1].expand(-1, 2 * h, -1), x, x[:, -1:].expand(-1, 2 * h, -1)), dim=1)
        weights = [float(h + 1 - abs(j)) for j in range(-h, h + 1)]
        norm = sum(weights)
        return sum(w / norm * x[:, k : k + frames + 2 * h] for k, w in enumerate(weights))

    def _retime_reach(
        self, planner, trajectory: torch.Tensor, last: torch.Tensor, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resample trajopt joint paths so the TCP moves at the command's speeds throughout.

        cuRobo's timing starts and ends at rest; only its geometric path is
        kept, re-timed so each step advances by ``max(travel / velocity,
        rotation / angular_velocity)`` = ``dt``. The blend adds the ramps.

        Returns:
            ``(joints, counts)``: ``(B, R, J)`` joints excluding the start, padded
            with the final joints, and ``(B,)`` frames per env.
        """
        from curobo.types import JointState

        num, frames, dof = trajectory.shape
        _, joint_names = self._asset.find_joints(self.cfg.robot_joint_names, preserve_order=True)
        flange = planner.compute_kinematics(
            JointState.from_position(trajectory.reshape(-1, dof).contiguous(), joint_names=joint_names)
        ).tool_poses.get_link_pose(planner.tool_frames[0])
        offset_pos = self._command_term._offset_pos[0].expand(num * frames, -1)
        offset_rot = self._command_term._offset_rot[0].expand(num * frames, -1)
        tcp_pos, tcp_quat = math_utils.combine_frame_transforms(
            flange.position.reshape(-1, 3), _wxyz_to_xyzw(flange.quaternion.reshape(-1, 4)), offset_pos, offset_rot
        )
        tcp_pos, tcp_quat = tcp_pos.view(num, frames, 3), tcp_quat.view(num, frames, 4)
        scale = self.speed_scale[env_ids, None]
        step = torch.maximum(
            torch.linalg.vector_norm(tcp_pos[:, 1:] - tcp_pos[:, :-1], dim=-1) / (self.cfg.velocity * scale),
            math_utils.quat_error_magnitude(tcp_quat[:, 1:], tcp_quat[:, :-1]) / (self.cfg.angular_velocity * scale),
        )
        step = step * (torch.arange(1, frames, device=self._device)[None] <= last[:, None])
        tau = torch.cat((torch.zeros(num, 1, device=self._device), step.cumsum(dim=1)), dim=1)
        counts = torch.ceil(tau[:, -1] / self._dt).clamp(min=1).long()
        horizon = int(counts.max())
        frac = (torch.arange(1, horizon + 1, device=self._device)[None] / counts[:, None]).clamp(max=1.0)
        s = frac * tau[:, -1:]
        idx = torch.searchsorted(tau.contiguous(), s.contiguous()).clamp(1, frames - 1)
        t0, t1 = torch.gather(tau, 1, idx - 1), torch.gather(tau, 1, idx)
        w = ((s - t0) / (t1 - t0).clamp(min=1.0e-9)).clamp(0.0, 1.0)[..., None]
        q0 = torch.gather(trajectory, 1, (idx - 1)[..., None].expand(-1, -1, dof))
        q1 = torch.gather(trajectory, 1, idx[..., None].expand(-1, -1, dof))
        return q0 * (1.0 - w) + q1 * w, counts

    def _chain_retargeter(self, size: int):
        """The cached batched ``MotionRetargeter``, shared with the valve arc.

        Call inside ``curobo_compatible_warp``.
        """
        from curobo.motion_retargeter import MotionRetargeterCfg
        from curobo.types import DeviceCfg, ToolPoseCriteria

        from isaaclab_hiveboard.assets import ASSET_DIR
        from isaaclab_hiveboard.mdp.curobo_robot_cfg import load_curobo_robot_cfg

        robot_cfg = load_curobo_robot_cfg(
            self.cfg.robot_curobo_yaml or f"{ASSET_DIR}/franka/cumotion/fr3.yaml",
            self.cfg.robot_urdf or f"{ASSET_DIR}/franka/cumotion/fr3.urdf",
        )
        tool_frame = robot_cfg["robot_cfg"]["kinematics"]["tool_frames"][0]
        cache_key = ("retargeter", self.cfg.robot_curobo_yaml, self.cfg.robot_urdf, self.cfg.num_ik_seeds, size)

        def _make_retargeter():
            with torch.inference_mode(False):
                return _batched_retargeter_class()(
                    MotionRetargeterCfg.create(
                        robot=robot_cfg,
                        tool_pose_criteria={
                            tool_frame: ToolPoseCriteria.track_position_and_orientation(
                                xyz=[1.0, 1.0, 1.0], rpy=[1.0, 1.0, 1.0], non_terminal_scale=1.0
                            )
                        },
                        num_envs=size,
                        use_mpc=False,
                        self_collision_check=False,
                        scene_model=None,
                        load_collision_spheres=False,
                        optimization_dt=self._dt,
                        num_seeds_global=self.cfg.num_ik_seeds,
                        num_seeds_local=1,
                        position_tolerance=0.002,
                        orientation_tolerance=0.02,
                        device_cfg=DeviceCfg(),
                    )
                )

        retargeter, _ = self._command_term.get_curobo_solver(cache_key, _make_retargeter)
        return retargeter

    def _plan_chain_batch(self, env_ids: torch.Tensor, members: list) -> torch.Tensor:
        """Plan one batch of chains; return the envs whose chain was rejected.

        The free-space reach (this command) is a cuRobo trajopt plan, whose
        joint-space path handles the wrist singularity the arm starts next to,
        re-timed not to stop at the target. The later legs are Cartesian (see
        :meth:`_chain_legs`), placed at the planned target and retargeted to
        joints in one batched ``MotionRetargeter`` pass warm-started from the
        reach's end. The joints of the whole chain are then blended in time
        (:meth:`_blend_in_time`). A chain is rejected if trajopt fails, the
        retargeted TCP strays more than ``chain_max_pose_error_m`` from its
        path, or a blended joint moves more than ``chain_max_joint_step``.
        """
        from isaaclab_hiveboard.mdp.curobo_warp import curobo_compatible_warp

        term = self._command_term
        if term._offset_pos is None or term._offset_rot is None:
            raise ValueError("A cuRobo chain requires pose_command.body_offset")
        with curobo_compatible_warp():
            from curobo.motion_retargeter import SequenceGoalToolPose
            from curobo.types import JointState

            planner, build_s, size = self._batch_planner()
            retargeter = self._chain_retargeter(size)
            tool_frame = planner.tool_frames[0]
            solve_start = time.perf_counter()
            n = len(env_ids)
            pad = torch.cat((torch.arange(n, device=self._device), torch.full((size - n,), n - 1, device=self._device)))
            pad_env_ids = env_ids[pad]
            rows = torch.arange(size, device=self._device)
            joint_ids, joint_names = self._asset.find_joints(self.cfg.robot_joint_names, preserve_order=True)
            q = self._asset.data.joint_pos.torch[pad_env_ids][:, joint_ids]

            # Reach: trajopt to this command's target, re-timed.
            _GoToFrameHandler.reset(self, pad_env_ids)
            target_pos, target_quat = self.get_target_in_base_frame(pad_env_ids)
            flange_pos, flange_quat = term._tcp_pose_to_body_pose(target_pos, target_quat, pad_env_ids)
            result, base_pos, base_quat = self._solve_batch_trajopt(planner, pad_env_ids, q, flange_pos, flange_quat)
            if result is None:
                return env_ids
            solved = result.success.view(size, -1).any(dim=-1)
            trajectory = result.interpolated_trajectory.position.view(size, -1, len(joint_names))
            last = result.interpolated_last_tstep.view(size, -1)[:, 0]
            reach, reach_counts = self._retime_reach(planner, trajectory, last, pad_env_ids)

            # Cartesian legs from the planned target, retargeted from the reach's end.
            leg_pos, leg_quat, leg_counts = self._chain_legs(pad_env_ids, members[1:], target_pos, target_quat)
            frames = leg_pos.shape[1]
            flat_ids = pad_env_ids.repeat_interleave(frames)
            leg_flange_pos, leg_flange_quat = term._tcp_pose_to_body_pose(
                leg_pos.reshape(-1, 3), leg_quat.reshape(-1, 4), flat_ids
            )
            goal_pos, goal_quat = math_utils.subtract_frame_transforms(
                base_pos.repeat_interleave(frames, 0),
                base_quat.repeat_interleave(frames, 0),
                leg_flange_pos,
                leg_flange_quat,
            )
            goal_pos = goal_pos.view(size, frames, 3)
            goal_quat = _xyzw_to_wxyz(goal_quat).view(size, frames, 4)
            with torch.inference_mode(False), torch.enable_grad():
                # SequenceGoalToolPose is (frames, envs, links, goalset, 3/4).
                targets = SequenceGoalToolPose(
                    tool_frames=[tool_frame],
                    position=goal_pos.transpose(0, 1)[:, :, None, None, :].contiguous().clone(),
                    quaternion=goal_quat.transpose(0, 1)[:, :, None, None, :].contiguous().clone(),
                )
                retargeter.reset()
                retargeter._prev_solution = reach[rows, reach_counts - 1].clone()
                leg_joints = torch.stack(
                    [
                        retargeter.solve_frame(targets.get_frame(i)).joint_state.reorder(joint_names).position
                        for i in range(frames)
                    ],
                    dim=1,
                )
                reached = retargeter.kinematics.compute_kinematics(
                    JointState.from_position(
                        leg_joints.reshape(-1, len(joint_names)).contiguous(), joint_names=joint_names
                    )
                ).tool_poses.get_link_pose(tool_frame)
            pose_err = torch.linalg.vector_norm(reached.position.view(size, frames, 3) - goal_pos, dim=-1)
            pose_err = pose_err * (torch.arange(frames, device=self._device)[None] < leg_counts.sum(dim=1)[:, None])

            # The whole chain's joints from the current ones, blended in time.
            # leg_joints is already packed, so it counts as one piece here.
            pieces = torch.stack((reach_counts, leg_counts.sum(dim=1)), dim=1)
            joints = torch.cat((q[:, None], self._pack_legs([reach, leg_joints], pieces)), dim=1)
            counts = torch.cat((reach_counts[:, None], leg_counts), dim=1)
            half_width = max(0, int(round(self.cfg.chain_blend_s / (2.0 * self._dt))))
            joints = self._blend_in_time(joints, half_width)
            joint_step = (joints[:, 1:] - joints[:, :-1]).abs().amax(dim=-1)
            solve_s = time.perf_counter() - solve_start

            # Leg k ends on input frame ends[:, k], i.e. output frame ends + half_width.
            ends = counts.cumsum(dim=1) + half_width
            lengths = counts.sum(dim=1) + 1 + 2 * half_width
            rejected = []
            worst_err, worst_step, grasp_dev = 0.0, 0.0, []
            for k in range(n):
                env = env_ids[k : k + 1]
                length = int(lengths[k])
                err, step = float(pose_err[k].max()), float(joint_step[k, : length - 1].max())
                worst_err, worst_step = max(worst_err, err), max(worst_step, step)
                if not bool(solved[k]) or err > self.cfg.chain_max_pose_error_m or step > self.cfg.chain_max_joint_step:
                    rejected.append(env)
                    continue
                tcp_pos, tcp_quat = self._joints_to_tcp_b(
                    planner, tool_frame, joints[k, :length], base_pos[k], base_quat[k]
                )
                # Output frame 0 is the current pose; the leader starts on frame 1.
                bounds = [1] + [int(e) + 1 for e in ends[k, :-1]] + [length]
                for m, member in enumerate(members):
                    seg = slice(bounds[m], bounds[m + 1])
                    member._store_plan(env, tcp_pos[seg], tcp_quat[seg], joints[k, seg])
                    member._chain_exhaust[env] = m < len(members) - 1
                    member._chain_owned[env] = m > 0
                # How far the blend leaves the TCP from the last leg's start
                # (the grasp) when that segment takes over.
                grasp = int(ends[k, -2]) - half_width - int(reach_counts[k]) - 1
                if grasp >= 0:
                    grasp_dev.append(float(torch.linalg.vector_norm(tcp_pos[bounds[-2]] - leg_pos[k, grasp])))
            grasp_mm = f"{1000 * sum(grasp_dev) / len(grasp_dev):.1f}" if grasp_dev else "n/a"
            print(
                f"[INFO] cuRobo chain of {len(members)} segments: {n - len(rejected)}/{n} envs planned "
                f"({int((~solved[:n]).sum())} trajopt failures), {joints.shape[1]} frames max, worst IK error "
                f"{worst_err * 1000:.1f} mm, worst joint step {worst_step:.3f} rad, mean TCP offset at the "
                f"last handover {grasp_mm} mm (batch={size} build={build_s:.2f}s solve={solve_s:.2f}s)",
                flush=True,
            )
            return torch.cat(rejected) if rejected else env_ids[:0]

    def _curobo_base_in_isaac_base(self, kinematics_state, tool_frame: str, env_ids: torch.Tensor):
        """Per-env cuRobo URDF base pose in the Isaac base frame (xyzw), as in :meth:`_plan_env`."""
        curobo_flange = kinematics_state.tool_poses.get_link_pose(tool_frame)
        isaac_flange_pos_b, isaac_flange_quat_b = math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w.torch[env_ids],
            self._asset.data.root_quat_w.torch[env_ids],
            self._asset.data.body_pos_w.torch[env_ids, self._command_term._body_idx],
            self._asset.data.body_quat_w.torch[env_ids, self._command_term._body_idx],
        )
        flange_quat_inv_c = math_utils.quat_inv(_wxyz_to_xyzw(curobo_flange.quaternion))
        flange_pos_inv_c = -math_utils.quat_apply(flange_quat_inv_c, curobo_flange.position)
        return math_utils.combine_frame_transforms(
            isaac_flange_pos_b, isaac_flange_quat_b, flange_pos_inv_c, flange_quat_inv_c
        )

    def _batch_planner(self):
        """The cached ``BatchMotionPlanner`` for this command's settings: ``(planner, build_s, size)``.

        Call inside ``curobo_compatible_warp``.
        """
        from curobo.batch_motion_planner import BatchMotionPlanner, MotionPlannerCfg
        from curobo.types import DeviceCfg

        from isaaclab_hiveboard.assets import ASSET_DIR
        from isaaclab_hiveboard.mdp.curobo_robot_cfg import load_curobo_robot_cfg

        robot_cfg = self._planner_robot_cfg(
            load_curobo_robot_cfg(
                self.cfg.robot_curobo_yaml or f"{ASSET_DIR}/franka/cumotion/fr3.yaml",
                self.cfg.robot_urdf or f"{ASSET_DIR}/franka/cumotion/fr3.urdf",
            )
        )
        size = max(1, int(self.cfg.plan_batch_size))
        joint_velocity = self._planner_joint_velocity()
        cache_key = (
            "batch_planner",
            self.cfg.robot_curobo_yaml,
            self.cfg.robot_urdf,
            self.cfg.num_ik_seeds,
            self.cfg.num_trajopt_seeds,
            self.cfg.interpolation_buffer_size,
            size,
            None if joint_velocity is None else tuple(joint_velocity.items()),
        )

        def _make_planner() -> BatchMotionPlanner:
            with torch.inference_mode(False):
                return BatchMotionPlanner(
                    MotionPlannerCfg.create(
                        robot=robot_cfg,
                        self_collision_check=False,
                        use_cuda_graph=True,
                        num_ik_seeds=self.cfg.num_ik_seeds,
                        num_trajopt_seeds=self.cfg.num_trajopt_seeds,
                        interpolation_dt=self._dt,
                        interpolation_buffer_size=self.cfg.interpolation_buffer_size,
                        device_cfg=DeviceCfg(),
                        max_batch_size=size,
                    )
                )

        planner, build_s = self._command_term.get_curobo_solver(cache_key, _make_planner)
        return planner, build_s, size

    def _solve_batch_trajopt(
        self,
        planner,
        pad_env_ids: torch.Tensor,
        q: torch.Tensor,
        flange_pos_b: torch.Tensor,
        flange_quat_b: torch.Tensor,
    ):
        """One padded batch of trajopt solves from joints ``q`` to flange poses (base frame, xyzw).

        Call inside ``curobo_compatible_warp``.

        Returns:
            ``(result, base_pos, base_quat)``: the solver result (``None`` if it
            returned none) and each problem's cuRobo base pose in the Isaac base frame.
        """
        from curobo.types import GoalToolPose, JointState, Pose

        tool_frame = planner.tool_frames[0]
        _, joint_names = self._asset.find_joints(self.cfg.robot_joint_names, preserve_order=True)
        current = JointState.from_position(q, joint_names=joint_names)
        base_pos, base_quat = self._curobo_base_in_isaac_base(
            planner.compute_kinematics(current), tool_frame, pad_env_ids
        )
        flange_pos_c, flange_quat_c = math_utils.subtract_frame_transforms(
            base_pos, base_quat, flange_pos_b, flange_quat_b
        )
        with torch.inference_mode(False), torch.enable_grad():
            current = JointState.from_position(q.clone(), joint_names=joint_names)
            goal = GoalToolPose.from_poses(
                {tool_frame: Pose(position=flange_pos_c.clone(), quaternion=_xyzw_to_wxyz(flange_quat_c.clone()))},
                ordered_tool_frames=planner.tool_frames,
                num_goalset=1,
            )
            if getattr(self.cfg, "wrist_flip_joints", None) and bool((self.wrist_branch[pad_env_ids] != 0).any()):
                result = self._plan_pose_on_branch(planner, goal, current, pad_env_ids)
            else:
                result = planner.plan_pose(
                    goal, current, max_attempts=self.cfg.max_plan_attempts, enable_graph_attempt=99
                )
        return result, base_pos, base_quat

    def _plan_pose_on_branch(self, planner, goal, current, pad_env_ids: torch.Tensor):
        """``planner.plan_pose`` (one attempt) with each env's IK goal moved to its :attr:`wrist_branch`.

        With a (roll, pitch, roll) wrist whose axes meet, turning the first
        roll by pi, negating the pitch and turning the last roll by pi leaves
        the flange pose unchanged, so the IK goal can be flipped exactly. A
        flip that would leave the joint limits (minus ``wrist_flip_margin``) is
        not applied; the env keeps the solver's branch. With
        :data:`WRIST_BRANCH_NEAREST` the goal is whichever of the two is nearer
        the current joints, so the branch follows from the arm's state.
        Call with grad enabled.
        """
        num_seeds = planner.trajopt_solver.config.num_seeds
        ik = planner.ik_solver.solve_pose(goal, return_seeds=num_seeds, current_state=current)
        if torch.count_nonzero(ik.success) == 0:
            return None
        solution = ik.solution.clone()  # (B, seeds, dof)
        names = list(planner.joint_names)
        roll1, pitch, roll2 = (names.index(n) for n in self.cfg.wrist_flip_joints)
        limits = planner.kinematics.get_joint_limits().position
        margin = float(self.cfg.wrist_flip_margin)
        low, high = limits[0] + margin, limits[1] - margin
        q_now = current.position[:, None, :].expand_as(solution)

        want = self.wrist_branch[pad_env_ids][:, None].expand(-1, solution.shape[1])
        sign = torch.where(solution[..., pitch] >= 0.0, 1.0, -1.0)
        flip = (want != 0) & (want != WRIST_BRANCH_NEAREST) & (sign != want)

        winding = self.forearm_winding[pad_env_ids][:, None].expand_as(want)
        up, down = solution[..., roll1] + math.pi, solution[..., roll1] - math.pi
        nearer_up = (up - q_now[..., roll1]).abs() <= (down - q_now[..., roll1]).abs()
        prefer_up = torch.where(winding == 0, nearer_up, winding > 0)
        up_ok = (up >= low[roll1]) & (up <= high[roll1])
        down_ok = (down >= low[roll1]) & (down <= high[roll1])
        new_roll1 = torch.where((prefer_up & up_ok) | ~down_ok, up, down)
        roll1_ok = up_ok | down_ok
        # The last roll's range spans 2 pi at most, so one of the two fits.
        up2, down2 = solution[..., roll2] + math.pi, solution[..., roll2] - math.pi
        up2_ok = (up2 >= low[roll2]) & (up2 <= high[roll2])
        down2_ok = (down2 >= low[roll2]) & (down2 <= high[roll2])
        new_roll2 = torch.where(down2_ok, down2, up2)
        flipped = solution.clone()
        flipped[..., roll1], flipped[..., pitch], flipped[..., roll2] = new_roll1, -solution[..., pitch], new_roll2
        nearer = (flipped - q_now).norm(dim=-1) < (solution - q_now).norm(dim=-1)
        flip = (flip | ((want == WRIST_BRANCH_NEAREST) & nearer)) & roll1_ok & (up2_ok | down2_ok)

        solution[..., roll1] = torch.where(flip, new_roll1, solution[..., roll1])
        solution[..., pitch] = torch.where(flip, -solution[..., pitch], solution[..., pitch])
        solution[..., roll2] = torch.where(flip, new_roll2, solution[..., roll2])
        result = planner.trajopt_solver.solve_pose(
            goal, current, seed_config=solution, use_implicit_goal=True, finetune_attempts=1
        )
        return result

    def _joints_to_tcp_b(self, kinematics, tool_frame: str, joints: torch.Tensor, base_pos, base_quat):
        """TCP poses (Isaac base frame, xyzw) of ``(T, J)`` joints for one env with cuRobo base ``base_pos/quat``."""
        from curobo.types import JointState

        _, joint_names = self._asset.find_joints(self.cfg.robot_joint_names, preserve_order=True)
        poses = kinematics.compute_kinematics(
            JointState.from_position(joints.contiguous(), joint_names=joint_names)
        ).tool_poses.get_link_pose(tool_frame)
        tcp_pos, tcp_quat = math_utils.combine_frame_transforms(
            poses.position,
            _wxyz_to_xyzw(poses.quaternion),
            self._command_term._offset_pos[0].expand_as(poses.position),
            self._command_term._offset_rot[0].expand_as(poses.quaternion),
        )
        return math_utils.combine_frame_transforms(
            base_pos.expand_as(tcp_pos), base_quat.expand_as(tcp_quat), tcp_pos, tcp_quat
        )

    def _plan_batch(self, env_ids: torch.Tensor) -> None:
        """Plan up to ``cfg.plan_batch_size`` envs in one ``BatchMotionPlanner`` solve.

        Mirrors :meth:`_plan_env` (trivial-move shortcut, base calibration,
        fallback on failure) without its per-env audit prints. The batch is
        padded to the planner's fixed size by repeating the last problem.
        """
        from isaaclab_hiveboard.mdp.curobo_warp import curobo_compatible_warp

        if self._command_term._offset_pos is None or self._command_term._offset_rot is None:
            raise ValueError("CuroboPlannedGoToFrameCfg requires pose_command.body_offset")
        with curobo_compatible_warp():
            planner, build_s, size = self._batch_planner()
            solve_start = time.perf_counter()
            tool_frame = planner.tool_frames[0]
            joint_ids, joint_names = self._asset.find_joints(self.cfg.robot_joint_names, preserve_order=True)

            target_pos_b, target_quat_b = self.get_target_in_base_frame(env_ids)
            flange_pos_b, flange_quat_b = self._command_term._tcp_pose_to_body_pose(
                target_pos_b, target_quat_b, env_ids
            )
            ee_pos_b, ee_quat_b = self._command_term._get_ee_in_base_frame(env_ids)
            q = self._asset.data.joint_pos.torch[env_ids][:, joint_ids]
            near = torch.linalg.vector_norm(ee_pos_b - target_pos_b, dim=-1) <= float(self.cfg.distance_threshold)
            aligned = math_utils.quat_error_magnitude(ee_quat_b, target_quat_b) <= math.radians(
                self.cfg.orientation_threshold_deg
            )
            trivial = near & aligned
            for k in torch.where(trivial)[0].tolist():
                self._store_plan(env_ids[k : k + 1], ee_pos_b[k : k + 1], ee_quat_b[k : k + 1], q[k : k + 1])
            rows = torch.where(~trivial)[0]
            if len(rows) == 0:
                return
            # Pad to the planner's fixed batch with copies of the last problem.
            pad = torch.cat((rows, rows[-1:].expand(size - len(rows))))
            pad_env_ids = env_ids[pad]

            try:
                result, base_pos, base_quat = self._solve_batch_trajopt(
                    planner, pad_env_ids, q[pad], flange_pos_b[pad], flange_quat_b[pad]
                )
            except Exception as err:
                print(
                    f"[WARN] cuRobo batch plan threw ({type(err).__name__}: {err}); "
                    f"{len(rows)} envs fall back to direct servo",
                    flush=True,
                )
                self._fallback[env_ids[rows]] = True
                return
            if result is None:
                self._fallback[env_ids[rows]] = True
                return
            success = result.success.view(size, -1).any(dim=-1)
            trajectory = result.interpolated_trajectory.position.view(size, -1, len(joint_names))
            last = result.interpolated_last_tstep.view(size, -1)[:, 0]
            solve_s = time.perf_counter() - solve_start
            planned = 0
            for k in range(len(rows)):
                env = env_ids[rows[k] : rows[k] + 1]
                if not bool(success[k]):
                    self._fallback[env] = True
                    continue
                joints = trajectory[k, : int(last[k]) + 1]
                tcp_pos, tcp_quat = self._joints_to_tcp_b(planner, tool_frame, joints, base_pos[k], base_quat[k])
                self._store_plan(env, tcp_pos, tcp_quat, joints)
                planned += 1
            print(
                f"[INFO] cuRobo batch plan {self.cfg.target_frame_name or self.cfg.target_position_env}: "
                f"{planned}/{len(rows)} solved (batch={size} build={build_s:.2f}s solve={solve_s:.2f}s)",
                flush=True,
            )

    def _plan_env(self, env_ids: torch.Tensor) -> None:
        """Build a bounded cuRobo plan for one env and convert it to TCP pose waypoints."""
        from isaaclab_hiveboard.assets import ASSET_DIR
        from isaaclab_hiveboard.mdp.curobo_robot_cfg import load_curobo_robot_cfg
        from isaaclab_hiveboard.mdp.curobo_warp import curobo_compatible_warp

        with curobo_compatible_warp():
            from curobo.types import DeviceCfg, JointState

            robot_cfg = load_curobo_robot_cfg(
                self.cfg.robot_curobo_yaml or f"{ASSET_DIR}/franka/cumotion/fr3.yaml",
                self.cfg.robot_urdf or f"{ASSET_DIR}/franka/cumotion/fr3.urdf",
            )
            robot_cfg = self._planner_robot_cfg(robot_cfg)
            joint_velocity = self._planner_joint_velocity()
            from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
            from curobo.types import GoalToolPose, Pose

            # Reuse one planner per build config (see _curobo_solver_cache):
            # construction costs seconds, solves are sub-second and warm-start
            # from the previous call. The factory must run in inference_mode
            # (False), otherwise cuRobo's reusable goal buffers become
            # inference tensors that TrajOpt cannot update in place.
            cache_key = (
                "planner",
                self.cfg.robot_curobo_yaml,
                self.cfg.robot_urdf,
                self.cfg.num_ik_seeds,
                self.cfg.num_trajopt_seeds,
                self.cfg.interpolation_buffer_size,
                None if joint_velocity is None else tuple(joint_velocity.items()),
            )

            def _make_planner() -> MotionPlanner:
                with torch.inference_mode(False):
                    return MotionPlanner(
                        MotionPlannerCfg.create(
                            robot=robot_cfg,
                            self_collision_check=False,
                            use_cuda_graph=True,
                            num_ik_seeds=self.cfg.num_ik_seeds,
                            num_trajopt_seeds=self.cfg.num_trajopt_seeds,
                            interpolation_dt=self._dt,
                            interpolation_buffer_size=self.cfg.interpolation_buffer_size,
                            device_cfg=DeviceCfg(),
                        )
                    )

            planner, build_s = self._command_term.get_curobo_solver(cache_key, _make_planner)
            solve_start = time.perf_counter()
            if self._command_term._offset_pos is None or self._command_term._offset_rot is None:
                raise ValueError("CuroboPlannedGoToFrameCfg requires pose_command.body_offset")
            try:
                target_pos_b, target_quat_b = self.get_target_in_base_frame(env_ids)
                flange_pos_b, flange_quat_b = self._command_term._tcp_pose_to_body_pose(
                    target_pos_b, target_quat_b, env_ids
                )
                # Skip the solver when already there (e.g. a home-to-home leg):
                # publish the current pose/joints as a single waypoint instead
                # of burning seconds of trajopt on a no-op.
                rep = env_ids[0:1]
                ee_pos_b, ee_quat_b = self._command_term._get_ee_in_base_frame(rep)
                if float(torch.linalg.vector_norm(ee_pos_b[0] - target_pos_b[0]).item()) <= float(
                    self.cfg.distance_threshold
                ) and float(
                    math_utils.quat_error_magnitude(ee_quat_b, target_quat_b)[0].item()
                ) <= math.radians(
                    self.cfg.orientation_threshold_deg
                ):
                    rep_jids, _ = self._asset.find_joints(self.cfg.robot_joint_names, preserve_order=True)
                    rep_q = self._asset.data.joint_pos.torch[rep][:, rep_jids]
                    self._publish_plan(env_ids, ee_pos_b, ee_quat_b, rep_q.unsqueeze(0), origin="trivial")
                    return
                joint_ids, joint_names = self._asset.find_joints(self.cfg.robot_joint_names, preserve_order=True)
                current = JointState.from_position(
                    self._asset.data.joint_pos.torch[env_ids][:, joint_ids],
                    joint_names=joint_names,
                )
                # The Isaac articulation root and the FR3 URDF root are not
                # the same frame.  Calibrate the constant transform from the
                # current shared joint configuration, then express the Isaac
                # target flange pose in cuRobo's URDF base frame.
                curobo_flange = planner.compute_kinematics(current).tool_poses.get_link_pose(planner.tool_frames[0])
                isaac_flange_pos_b, isaac_flange_quat_b = math_utils.subtract_frame_transforms(
                    self._asset.data.root_pos_w.torch[env_ids],
                    self._asset.data.root_quat_w.torch[env_ids],
                    self._asset.data.body_pos_w.torch[env_ids, self._command_term._body_idx],
                    self._asset.data.body_quat_w.torch[env_ids, self._command_term._body_idx],
                )
                # Planning audit (first active env): measured start vs
                # commanded goal, flange and TCP, all in the base frame. An
                # inverted axis shows up as a negated component or ~180 deg
                # RPY jump here, before cuRobo is even involved.
                e0 = int(env_ids[0].item())
                start_tcp_b, start_tcp_q = math_utils.combine_frame_transforms(
                    isaac_flange_pos_b[0:1],
                    isaac_flange_quat_b[0:1],
                    self._command_term._offset_pos[env_ids[0:1]],
                    self._command_term._offset_rot[env_ids[0:1]],
                )

                try:
                    audit_seg = self._command_term._command_handlers.index(self)
                except ValueError:
                    audit_seg = "?"
                print(
                    f"[INFO] plan seg {audit_seg} env={e0} start EE-flange_b {_pose_str(isaac_flange_pos_b[0:1], isaac_flange_quat_b[0:1])}",
                    flush=True,
                )
                print(
                    f"[INFO] plan seg {audit_seg} env={e0} goal  EE-flange_b {_pose_str(flange_pos_b[0:1], flange_quat_b[0:1])}",
                    flush=True,
                )
                print(
                    f"[INFO] plan seg {audit_seg} env={e0} start EE-TCP_b    {_pose_str(start_tcp_b, start_tcp_q)}",
                    flush=True,
                )
                print(
                    f"[INFO] plan seg {audit_seg} env={e0} goal  EE-TCP_b    {_pose_str(target_pos_b[0:1], target_quat_b[0:1])}",
                    flush=True,
                )
                # cuRobo quaternions are (w, x, y, z); math_utils is (x, y, z, w).
                curobo_flange_quat_b = _wxyz_to_xyzw(curobo_flange.quaternion)
                flange_quat_inv_c = math_utils.quat_inv(curobo_flange_quat_b)
                flange_pos_inv_c = -math_utils.quat_apply(flange_quat_inv_c, curobo_flange.position)
                curobo_base_pos_b, curobo_base_quat_b = math_utils.combine_frame_transforms(
                    isaac_flange_pos_b,
                    isaac_flange_quat_b,
                    flange_pos_inv_c,
                    flange_quat_inv_c,
                )
                flange_pos_c, flange_quat_c = math_utils.subtract_frame_transforms(
                    curobo_base_pos_b,
                    curobo_base_quat_b,
                    flange_pos_b,
                    flange_quat_b,
                )
                # Isaac Lab evaluates command terms under ``inference_mode``.
                # cuRobo's seed IK intentionally differentiates its pose cost;
                # ``enable_grad`` alone cannot override inference mode.  Clone
                # the input tensors in this context too: otherwise cuRobo
                # adopts an inference-mode goal tensor as its mutable cache.
                with torch.inference_mode(False), torch.enable_grad():
                    current = JointState.from_position(current.position.clone(), joint_names=joint_names)
                    goal = GoalToolPose.from_poses(
                        {
                            planner.tool_frames[0]: Pose(
                                position=flange_pos_c.clone(),
                                quaternion=_xyzw_to_wxyz(flange_quat_c.clone()),
                            )
                        },
                        ordered_tool_frames=planner.tool_frames,
                        num_goalset=1,
                    )
                    # NOTE: enable_graph_attempt=99 effectively disables graph
                    # seeding (attempts only run 0..max_plan_attempts-1, which
                    # never reach 99). Pure IK + trajopt is enough here and far
                    # cheaper than PRM queries; lower it only if you also want
                    # graph seeds on later attempts.
                    result = planner.plan_pose(
                        goal,
                        current,
                        max_attempts=self.cfg.max_plan_attempts,
                        enable_graph_attempt=99,
                    )
                if result is None or not bool(result.success.all().item()):
                    try:
                        seg = self._command_term._command_handlers.index(self)
                    except ValueError:
                        seg = "?"
                    print(
                        f"[WARN] cuRobo plan failed on seg {seg}; falling back to direct servo for "
                        f"{self.cfg.target_frame_name or self.cfg.target_position_env}. "
                        f"start_q={current.position.detach().cpu().tolist()} "
                        f"isaac_flange={isaac_flange_pos_b.detach().cpu().tolist()} "
                        f"curobo_flange={curobo_flange.position.detach().cpu().tolist()} "
                        f"curobo_base={curobo_base_pos_b.detach().cpu().tolist()} "
                        f"flange_pos_c={flange_pos_c.detach().cpu().tolist()} "
                        f"flange_quat_c={flange_quat_c.detach().cpu().tolist()}"
                    )
                    # Degrade to the parent velocity-limited servo for this
                    # segment instead of killing the episode. Servo state was
                    # seeded from the current EE pose at reset, and
                    # _joint_target stays None so the action term does not
                    # apply stale joint overrides.
                    self._fallback[env_ids] = True
                    return
                trajectory = result.get_interpolated_plan().position
                if trajectory.ndim == 4:
                    trajectory = trajectory[:, 0]
                poses = planner.compute_kinematics(
                    JointState.from_position(trajectory[0], joint_names=joint_names)
                ).tool_poses.get_link_pose(planner.tool_frames[0])
                tcp_pos, tcp_quat = math_utils.combine_frame_transforms(
                    poses.position,
                    _wxyz_to_xyzw(poses.quaternion),
                    self._command_term._offset_pos[0].expand_as(poses.position),
                    self._command_term._offset_rot[0].expand_as(poses.quaternion),
                )
                # Convert visual TCP waypoints back to Isaac's base frame;
                # direct joint execution below is the authoritative motion.
                tcp_pos, tcp_quat = math_utils.combine_frame_transforms(
                    curobo_base_pos_b[0].expand_as(tcp_pos),
                    curobo_base_quat_b[0].expand_as(tcp_quat),
                    tcp_pos,
                    tcp_quat,
                )
                solve_s = time.perf_counter() - solve_start
                self._publish_plan(
                    env_ids, tcp_pos, tcp_quat, trajectory, origin="planned", build_s=build_s, solve_s=solve_s
                )
            except Exception as err:
                # A throwing solver must not kill the episode (the original
                # cuRobo crash mode): degrade to direct servo like a failed
                # plan. Misconfiguration still raises above, outside this try.
                try:
                    seg = self._command_term._command_handlers.index(self)
                except ValueError:
                    seg = "?"
                print(
                    f"[WARN] cuRobo plan threw on seg {seg} "
                    f"({type(err).__name__}: {err}); falling back to direct servo",
                    flush=True,
                )
                self._fallback[env_ids] = True
                return

    def _publish_plan(
        self,
        env_ids: torch.Tensor,
        tcp_pos: torch.Tensor,
        tcp_quat: torch.Tensor,
        trajectory: torch.Tensor,
        origin: str,
        build_s: float = 0.0,
        solve_s: float = 0.0,
    ) -> None:
        """Store one env's waypoints/joints and emit plan diagnostics (shared by both planners).

        ``trajectory`` is the solver's single-problem ``(1, T, J)`` joint plan.
        """
        env_index = int(env_ids[0].item())
        if getattr(self.cfg, "valve_done_threshold_rad", None) is not None:
            self._validate_arc_radius(tcp_pos, env_index)
        self._store_plan(env_ids, tcp_pos, tcp_quat, trajectory[0])
        print(
            f"[INFO] cuRobo env={env_index} {origin} {tcp_pos.shape[0]} TCP waypoints for "
            f"{self.cfg.target_frame_name or self.cfg.target_position_env} "
            f"(build={build_s:.2f}s solve={solve_s:.2f}s)."
        )
        # Orientation audit: planned TCP RPY (deg, XYZ order) against
        # the straight Cartesian slerp at the same fractions. A joint
        # -space plan can land perfectly while twisting through a
        # different route mid-path; that divergence shows up here.
        n_wp = tcp_quat.shape[0]
        print(
            f"[INFO] plan ori path (planned-rpy vs slerp-rpy deg, N={n_wp}):",
            flush=True,
        )
        q0, q1 = tcp_quat[0], tcp_quat[-1]
        for f in (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0):
            qp = tcp_quat[min(int(round(f * (n_wp - 1))), n_wp - 1)]
            qs = math_utils.quat_slerp(q0, q1, float(f)).reshape(1, 4)
            rp = torch.rad2deg(torch.stack(math_utils.euler_xyz_from_quat(qp.unsqueeze(0)), dim=-1))[0]
            rs = torch.rad2deg(torch.stack(math_utils.euler_xyz_from_quat(qs), dim=-1))[0]
            print(
                f"[INFO]   f={f:.2f} plan={[round(float(v), 1) for v in rp.tolist()]} "
                f"slerp={[round(float(v), 1) for v in rs.tolist()]}",
                flush=True,
            )

    def _validate_arc_radius(self, tcp_pos_b: torch.Tensor, env_index: int, wander_tol_m: float = 0.01) -> None:
        """Check the planned TCP path against its own endpoint chord.

        A plan between two bead targets should not wander beyond what the
        straight chord between its endpoints implies: compare the range of
        ``|TCP - pivot|`` along the plan vs along the chord. Warns when the
        plan bows out (bad arc / wrong pivot), not when the beads themselves
        imply varying radius. Display only; never fails the plan.
        """
        term = self._command_term
        valve = term._valve_asset
        if valve is None or tcp_pos_b.shape[0] < 2:
            return
        if not hasattr(self, "_valve_pivot_idx"):
            # Anchor = joint anchor (a fixed point ON the rotation axis).
            # Per Ball_Valve.urdf, RevoluteJoint's origin is identity in its
            # parent valvula_esfera, so that body's origin IS the anchor.
            self._valve_pivot_idx = None
            self._valve_pivot_name = "valve-root"
            for candidate in ("valvula_esfera", "alavanca_pivot"):
                try:
                    ids, _ = valve.find_bodies(candidate)
                except Exception:
                    ids = []
                if ids:
                    self._valve_pivot_idx = int(ids[0])
                    self._valve_pivot_name = candidate
                    break
        if self._valve_pivot_idx is None:
            pivot_w = valve.data.root_pos_w.torch[env_index]
        else:
            pivot_w = valve.data.body_pos_w.torch[env_index, self._valve_pivot_idx]
        pivot_name = self._valve_pivot_name
        n = tcp_pos_b.shape[0]
        root_pos = self._asset.data.root_pos_w.torch[env_index : env_index + 1].expand(n, -1)
        root_quat = self._asset.data.root_quat_w.torch[env_index : env_index + 1].expand(n, -1)
        identity = torch.zeros(n, 4, device=self._device)
        identity[:, 3] = 1.0
        tcp_w, _ = math_utils.combine_frame_transforms(root_pos, root_quat, tcp_pos_b, identity)

        def _range(t: torch.Tensor) -> float:
            d = torch.linalg.vector_norm(t - pivot_w.unsqueeze(0), dim=-1)
            return float(d.max().item() - d.min().item()), float(d.mean().item())

        plan_range, plan_mean = _range(tcp_w)
        f = torch.linspace(0.0, 1.0, n, device=self._device).unsqueeze(-1)
        chord = tcp_w[0:1] * (1.0 - f) + tcp_w[-1:] * f
        chord_range, _ = _range(chord)
        plan_mm, chord_mm = plan_range * 1000.0, chord_range * 1000.0
        try:
            seg = self._command_term._command_handlers.index(self)
        except ValueError:
            seg = "?"
        print(
            f"[INFO] plan seg {seg} arc radius vs {pivot_name} "
            f"@ {[round(float(v), 4) for v in pivot_w.detach().cpu().tolist()]}: "
            f"plan range={plan_mm:.1f}mm chord range={chord_mm:.1f}mm mean={plan_mean:.4f}m (N={n})",
            flush=True,
        )
        if plan_mm > chord_mm + wander_tol_m * 1000.0:
            print(
                f"[WARN] plan seg {seg} wanders {plan_mm - chord_mm:.1f}mm beyond its "
                f"endpoint chord: path not circular about the valve pivot",
                flush=True,
            )

    def update(self, env_mask: torch.Tensor) -> torch.Tensor:
        env_ids = torch.where(env_mask)[0]
        # CommandManager resets every sequence handler at episode reset. Plan
        # here instead, once this term actually becomes active after the arm
        # has reached the lower lever pose and closed the gripper.
        unplanned = env_ids[~self._planned[env_ids] & ~self._fallback[env_ids]]
        if len(unplanned) > 0:
            self._plan(unplanned)
        command = torch.zeros(len(env_ids), 8, device=self._device)
        fallback = self._fallback[env_ids]
        if torch.any(fallback):
            fallback_mask = torch.zeros_like(env_mask)
            fallback_mask[env_ids[fallback]] = True
            command[fallback] = super().update(fallback_mask)
        if not torch.all(fallback):
            pos, quat = self._follow_plan(env_ids[~fallback])
            command[~fallback] = self._pack_command(self.cfg.gripper_open, pos, quat)
        return command

    def is_done(self, env_ids: torch.Tensor) -> torch.Tensor:
        done = torch.zeros(len(env_ids), device=self._device, dtype=torch.bool)
        fallback = self._fallback[env_ids]
        if torch.any(fallback):
            done[fallback] = super().is_done(env_ids[fallback])
        planned = self._planned[env_ids] & ~fallback
        exhausted = planned & (self._waypoint_index[env_ids] >= self._waypoint_count[env_ids])
        # Mid-chain segments hand over on the last waypoint: the next segment's
        # plan already continues from it, so settling here would be a pause.
        chained = exhausted & self._chain_exhaust[env_ids]
        done[chained] = True
        exhausted &= ~chained
        if torch.any(exhausted):
            # Waypoint exhaustion alone is not convergence: the final waypoint is
            # first commanded the same tick the index exhausts, so advancing here
            # would chain the next plan from lagged joints. update() keeps
            # publishing the final waypoint/joints, so waiting for the parent
            # distance/orientation check lets the arm settle onto the plan end.
            # A plan that ends out of tolerance trips the stall watchdog instead.
            done[exhausted] = super().is_done(env_ids[exhausted])
        return done


class _GripperHandler(_BaseCmdHandler):
    """Handles the GripperCommand."""

    def __init__(self, cfg: GripperCommand, command_term: SequentialPoseCommand):
        super().__init__(cfg, command_term)
        self.cfg: GripperCommand
        if self.cfg.duration_s <= 0.0:
            raise ValueError("GripperCommand.duration_s must be positive")
        self._elapsed_s = torch.zeros(self._num_envs, device=self._device)

    def reset(self, env_ids: torch.Tensor):
        self._elapsed_s[env_ids] = 0.0

    def update(self, env_mask: torch.Tensor) -> torch.Tensor:
        self._elapsed_s[env_mask] += self._dt
        # Hold the last pose while changing only the gripper command.
        last_command = self._command_term._command[env_mask].clone()
        last_command[:, 0] = 1.0 if self.cfg.open_gripper else -1.0
        return last_command

    def is_done(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self._elapsed_s[env_ids] >= self.cfg.duration_s

    def get_target_in_base_frame(self, env_ids: torch.Tensor):
        return (
            self._command_term._command[env_ids, 1:4],
            self._command_term._command[env_ids, 4:8],
        )


class _RotateFrameHandler(_BaseCmdHandler):
    """Handles the RotateFrame command."""

    def __init__(
        self,
        cfg: RotateFrameCfg,
        command_term: SequentialPoseCommand,
    ):
        super().__init__(cfg, command_term)
        self.cfg: RotateFrameCfg
        if self.cfg.angular_velocity <= 0.0:
            raise ValueError("RotateFrameCfg.angular_velocity must be positive")
        if self.cfg.angle_threshold_deg < 0.0:
            raise ValueError("RotateFrameCfg.angle_threshold_deg must be non-negative")

        self._frame: FrameTransformer = command_term._env.scene[cfg.frame_name]
        self._frame_idx = self._frame.data.target_frame_names.index(cfg.target_frame_name)
        self.initial_quat_b = torch.zeros(self._num_envs, 4, device=self._device)
        self.axis_pos_b = torch.zeros(self._num_envs, 3, device=self._device)
        self.axis_quat_b = torch.zeros(self._num_envs, 4, device=self._device)
        self._progress_abs = torch.zeros(self._num_envs, device=self._device)
        self.angle_rad_tensor = torch.deg2rad(
            torch.tensor(self.cfg.angle_deg, device=self._device, dtype=torch.float32)
        ).repeat(self._num_envs)
        self.final_quat_b = torch.zeros(self._num_envs, 4, device=self._device)
        self.rot_axis_b = torch.zeros(self._num_envs, 3, device=self._device)
        self.radius_vec = torch.zeros(self._num_envs, 3, device=self._device)
        self.axial_vec = torch.zeros(self._num_envs, 3, device=self._device)
        self._angle_threshold_rad = math.radians(self.cfg.angle_threshold_deg)
        # Per-environment extra arc [rad] in the direction of the turn, e.g. to
        # press a loaded lever onto its end stop despite the grip's compliance.
        self.angle_extra_rad = torch.zeros(self._num_envs, device=self._device)

    def reset(self, env_ids: torch.Tensor):
        self._progress_abs[env_ids] = 0.0

        if self._command_term._valve_asset is not None and self.cfg.use_valve_angle:
            self._command_term.recompute_valve_rotate_angle(env_ids)
            self.angle_rad_tensor[env_ids] = self._command_term.valve_rotate_angle_rad[env_ids]
        else:
            self.angle_rad_tensor[env_ids] = math.radians(self.cfg.angle_deg)
        self.angle_rad_tensor[env_ids] += torch.copysign(self.angle_extra_rad[env_ids], self.angle_rad_tensor[env_ids])
        self._clamp_ee_rotation(env_ids)
        self._setup_arc(env_ids, *self._command_term._get_ee_in_base_frame(env_ids))

    def _setup_arc(self, env_ids: torch.Tensor, ee_pos_b: torch.Tensor, ee_quat_b: torch.Tensor) -> None:
        """Place the arc for a TCP starting at ``ee_pos_b``/``ee_quat_b`` (base frame, xyzw)."""
        self.initial_quat_b[env_ids] = ee_quat_b

        self.axis_pos_b[env_ids], axis_quat_b = self._get_rotation_axis_pose_b(env_ids)
        if self.cfg.rotate_in_place:
            # Keep the measured TCP fixed; the reference still supplies the axis.
            self.axis_pos_b[env_ids] = ee_pos_b

        if self._command_term.cfg.debug_vis:
            print(
                "Initial Pose in relation to axis: ",
                ee_pos_b - self.axis_pos_b[env_ids],
            )
            print("Initial pose: ", ee_pos_b)
            print("Axis position: ", self.axis_pos_b[env_ids])
            print("Axis quat: ", axis_quat_b)

        # The rotation vector is explicit in the axis frame. Keeping it in the
        # command config prevents a grasp-frame rotation from silently changing
        # the mechanical joint axis.
        axis_in_frame = torch.tensor(self.cfg.axis, device=self._device, dtype=torch.float32).repeat(len(env_ids), 1)
        if torch.any(torch.linalg.vector_norm(axis_in_frame, dim=-1) < 1.0e-6):
            raise ValueError("RotateFrameCfg.axis must be non-zero")

        # Rotation axis expressed in the robot base frame.
        rot_axis_b = math_utils.quat_apply(axis_quat_b, axis_in_frame)
        self.rot_axis_b[env_ids] = rot_axis_b / torch.linalg.vector_norm(rot_axis_b, dim=-1, keepdim=True)
        if self._command_term.cfg.debug_vis:
            print("Rotation axis: ", self.rot_axis_b[env_ids])

        # Split EE-to-hub into in-plane radius (orbits) and axial offset (held).
        # Dropping the axial part would snap the TCP onto the hub plane on the
        # first step, so the hand slides along the stem instead of spinning.
        radius_vec = ee_pos_b - self.axis_pos_b[env_ids]
        self.radius_vec[env_ids] = self._get_ortogonal_vector(radius_vec, self.rot_axis_b[env_ids])
        self.axial_vec[env_ids] = radius_vec - self.radius_vec[env_ids]
        if self._command_term.cfg.debug_vis:
            print("Radius vector: ", self.radius_vec[env_ids])
            print("Axial offset: ", self.axial_vec[env_ids])

        # Get final motion poses
        angle = self.angle_rad_tensor[env_ids]
        total_rotation = math_utils.quat_from_angle_axis(angle, self.rot_axis_b[env_ids])
        self.final_quat_b[env_ids] = math_utils.quat_mul(total_rotation, self.initial_quat_b[env_ids])

    def get_target_in_base_frame(self, env_ids: torch.Tensor):
        # Get final motion poses
        angle = self.angle_rad_tensor[env_ids]

        v_rot = self._rodrigues_rotate(self.radius_vec[env_ids], self.rot_axis_b[env_ids], angle)
        final_pose_b = self.axis_pos_b[env_ids] + self.axial_vec[env_ids] + v_rot
        final_pose_b += self._screw_offset(self.rot_axis_b[env_ids], angle)

        total_rotation = math_utils.quat_from_angle_axis(angle, self.rot_axis_b[env_ids])
        final_quat = math_utils.quat_mul(total_rotation, self.initial_quat_b[env_ids])

        return final_pose_b, final_quat

    def _screw_offset(self, axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        pitch = float(getattr(self.cfg, "screw_pitch_m_per_revolution", 0.0))
        return axis * (angle * pitch / (2.0 * math.pi))[:, None]

    def _rodrigues_rotate(self, v: torch.Tensor, axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return (
            v * torch.cos(theta)[:, None]
            + torch.cross(axis, v, dim=-1) * torch.sin(theta)[:, None]
            + axis * torch.sum(axis * v, dim=-1)[:, None] * (1 - torch.cos(theta))[:, None]
        )

    def _get_ortogonal_vector(self, vector: torch.Tensor, ortogonal_to: torch.Tensor) -> torch.Tensor:
        """
        Given a vector V and K, we can decompose on V = V || k  + V |_ k

        This function returns the V |_ k
        """
        return vector - torch.sum(vector * ortogonal_to, dim=-1, keepdim=True) * ortogonal_to

    def update(self, env_mask: torch.Tensor) -> torch.Tensor:
        env_ids = torch.where(env_mask)[0]
        abs_angle = torch.abs(self.angle_rad_tensor[env_ids])
        self._progress_abs[env_ids] = torch.clamp(
            self._progress_abs[env_ids] + self.cfg.angular_velocity * self.speed_scale[env_ids] * self._dt,
            max=abs_angle,
        )
        angle = torch.copysign(self._progress_abs[env_ids], self.angle_rad_tensor[env_ids])

        # -- position
        # rotate radius vector using Rodrigues' rotation formula
        v_rot = self._rodrigues_rotate(self.radius_vec[env_ids], self.rot_axis_b[env_ids], angle)
        target_pos_b = self.axis_pos_b[env_ids] + self.axial_vec[env_ids] + v_rot
        target_pos_b += self._screw_offset(self.rot_axis_b[env_ids], angle)

        # Same signed angle-axis as the position orbit. Slerping to the
        # endpoint via quat_box_minus is ambiguous at ±180 deg and can spin
        # the gripper the opposite way from the TCP, so the wrist offset
        # walks through the hub instead of riding the grasp circle.
        delta_q = math_utils.quat_from_angle_axis(angle, self.rot_axis_b[env_ids])
        interp_quat_b = math_utils.quat_mul(delta_q, self.initial_quat_b[env_ids])

        return self._pack_command(self.cfg.gripper_open, target_pos_b, interp_quat_b)

    def is_done(self, env_ids: torch.Tensor) -> torch.Tensor:
        remaining = torch.abs(self.angle_rad_tensor[env_ids]) - self._progress_abs[env_ids]
        return remaining <= self._angle_threshold_rad

    def _get_rotation_axis_pose_b(self, env_ids: torch.Tensor):
        # Target pose in base frame
        target_pos_w = self._frame.data.target_pos_w.torch[env_ids, self._frame_idx]
        target_quat_w = self._frame.data.target_quat_w.torch[env_ids, self._frame_idx]

        target_pos_w, target_quat_w = self._apply_target_offset(target_pos_w, target_quat_w, env_ids)

        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w.torch[env_ids],
            self._asset.data.root_quat_w.torch[env_ids],
            target_pos_w,
            target_quat_w,
        )
        if self.cfg.axis_position_override_b is not None:
            for axis, value in enumerate(self.cfg.axis_position_override_b):
                if value is not None:
                    target_pos_b[:, axis] = float(value)

        return target_pos_b, target_quat_b

    def _clamp_ee_rotation(self, env_ids: torch.Tensor) -> None:
        limit_deg = float(getattr(self.cfg, "max_ee_rotation_deg", 0.0) or 0.0)
        if limit_deg <= 0.0:
            return
        max_rad = math.radians(limit_deg)
        self.angle_rad_tensor[env_ids] = self.angle_rad_tensor[env_ids].clamp(-max_rad, max_rad)


class _CuroboPlannedRotateFrameHandler(_CuroboPlannedGoToFrameHandler, _RotateFrameHandler):
    """Solve the complete valve arc as ordered cuRobo IK joint waypoints."""

    def __init__(self, cfg: "CuroboPlannedRotateFrameCfg", command_term: SequentialPoseCommand):
        # Initialize the mechanical rotation geometry, then the cuRobo state.
        _RotateFrameHandler.__init__(self, cfg, command_term)
        self.cfg: CuroboPlannedRotateFrameCfg
        self._init_plan_state()
        # Env whose arc the infeasible-arc dump describes.
        self._diag_env = 0

    def reset(self, env_ids: torch.Tensor):
        # Keep an arc a chain leader already planned from the planned grasp.
        fresh = env_ids[~self._chain_owned[env_ids]]
        if len(fresh) > 0:
            _RotateFrameHandler.reset(self, fresh)
            self._clear_plan_state(fresh)

    def get_target_in_base_frame(self, env_ids: torch.Tensor):
        return _RotateFrameHandler.get_target_in_base_frame(self, env_ids)

    def update(self, env_mask: torch.Tensor) -> torch.Tensor:
        """Follow the retargeted joint arc. Do not fall back to Cartesian servo."""
        env_ids = torch.where(env_mask)[0]
        unplanned = env_ids[~self._planned[env_ids]]
        if len(unplanned) > 0:
            self._plan(unplanned)
        pos, quat = self._follow_plan(env_ids)
        return self._pack_command(self.cfg.gripper_open, pos, quat)

    def is_done(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self._planned[env_ids] & (self._waypoint_index[env_ids] >= self._waypoint_count[env_ids])

    def _plan_env(self, env_ids: torch.Tensor) -> None:
        """Retarget one env's complete Cartesian arc from its current grasp state.

        An infeasible arc is handled per ``cfg.on_infeasible_arc``. There is no
        Cartesian-servo fallback: that path used ``_GoToFrameHandler`` state
        this handler never initializes, and with ``output_joint_positions`` a
        servo fallback publishes no joints at all, so the arm would freeze.
        """
        self._diag_env = int(env_ids[0].item())

        from isaaclab_hiveboard.assets import ASSET_DIR
        from isaaclab_hiveboard.mdp.curobo_robot_cfg import load_curobo_robot_cfg
        from isaaclab_hiveboard.mdp.curobo_warp import curobo_compatible_warp

        with curobo_compatible_warp():
            from curobo.motion_retargeter import (
                MotionRetargeter,
                MotionRetargeterCfg,
                SequenceGoalToolPose,
            )
            from curobo.types import DeviceCfg, JointState, ToolPoseCriteria

            robot_cfg = load_curobo_robot_cfg(
                self.cfg.robot_curobo_yaml or f"{ASSET_DIR}/franka/cumotion/fr3.yaml",
                self.cfg.robot_urdf or f"{ASSET_DIR}/franka/cumotion/fr3.urdf",
            )
            tool_frame = robot_cfg["robot_cfg"]["kinematics"]["tool_frames"][0]
            cache_key = (
                "retargeter",
                self.cfg.robot_curobo_yaml,
                self.cfg.robot_urdf,
                self.cfg.num_ik_seeds,
            )

            def _make_retargeter() -> MotionRetargeter:
                with torch.inference_mode(False):
                    return MotionRetargeter(
                        MotionRetargeterCfg.create(
                            robot=robot_cfg,
                            tool_pose_criteria={
                                tool_frame: ToolPoseCriteria.track_position_and_orientation(
                                    xyz=[1.0, 1.0, 1.0],
                                    rpy=[1.0, 1.0, 1.0],
                                    non_terminal_scale=1.0,
                                )
                            },
                            num_envs=1,
                            use_mpc=False,
                            self_collision_check=False,
                            scene_model=None,
                            load_collision_spheres=False,
                            optimization_dt=self._dt,
                            num_seeds_global=self.cfg.num_ik_seeds,
                            num_seeds_local=1,
                            position_tolerance=0.002,
                            orientation_tolerance=0.02,
                            device_cfg=DeviceCfg(),
                        )
                    )

            retargeter, _ = self._command_term.get_curobo_solver(cache_key, _make_retargeter)

            if self._command_term._offset_pos is None or self._command_term._offset_rot is None:
                raise ValueError("CuroboPlannedRotateFrameCfg requires pose_command.body_offset")
            final_angle = self.angle_rad_tensor[env_ids][0]
            step_angle = self.cfg.angular_velocity * float(self.speed_scale[env_ids][0]) * self._dt
            num_steps = max(1, int(math.ceil(abs(float(final_angle.item())) / step_angle)))
            num_waypoints = num_steps + 1
            angles = torch.linspace(
                0.0,
                float(final_angle.item()),
                num_waypoints,
                device=self._device,
                dtype=torch.float32,
            )
            axis_b = self.rot_axis_b[env_ids].expand(num_waypoints, -1)
            radius_b = self.radius_vec[env_ids].expand(num_waypoints, -1)
            axis_pos_b = self.axis_pos_b[env_ids].expand(num_waypoints, -1)
            tcp_pos_b = (
                axis_pos_b
                + self.axial_vec[env_ids].expand(num_waypoints, -1)
                + self._rodrigues_rotate(radius_b, axis_b, angles)
                + self._screw_offset(axis_b, angles)
            )
            delta_quat_b = math_utils.quat_from_angle_axis(angles, axis_b)
            tcp_quat_b = math_utils.quat_mul(
                delta_quat_b,
                self.initial_quat_b[env_ids].expand(num_waypoints, -1),
            )

            flange_pos_b, flange_quat_b = self._command_term._tcp_pose_to_body_pose(
                tcp_pos_b, tcp_quat_b, env_ids
            )

            joint_ids, joint_names = self._asset.find_joints(self.cfg.robot_joint_names, preserve_order=True)
            current = JointState.from_position(
                self._asset.data.joint_pos.torch[env_ids][:, joint_ids],
                joint_names=joint_names,
            )

            curobo_flange = retargeter.kinematics.compute_kinematics(current).tool_poses.get_link_pose(tool_frame)
            isaac_flange_pos_b, isaac_flange_quat_b = math_utils.subtract_frame_transforms(
                self._asset.data.root_pos_w.torch[env_ids],
                self._asset.data.root_quat_w.torch[env_ids],
                self._asset.data.body_pos_w.torch[env_ids, self._command_term._body_idx],
                self._asset.data.body_quat_w.torch[env_ids, self._command_term._body_idx],
            )
            # cuRobo quaternions are (w, x, y, z); math_utils is (x, y, z, w).
            flange_quat_inv_c = math_utils.quat_inv(_wxyz_to_xyzw(curobo_flange.quaternion))
            flange_pos_inv_c = -math_utils.quat_apply(flange_quat_inv_c, curobo_flange.position)
            curobo_base_pos_b, curobo_base_quat_b = math_utils.combine_frame_transforms(
                isaac_flange_pos_b,
                isaac_flange_quat_b,
                flange_pos_inv_c,
                flange_quat_inv_c,
            )
            flange_pos_c, flange_quat_c = math_utils.subtract_frame_transforms(
                curobo_base_pos_b.expand(num_waypoints, -1),
                curobo_base_quat_b.expand(num_waypoints, -1),
                flange_pos_b,
                flange_quat_b,
            )

            with torch.inference_mode(False), torch.enable_grad():
                current = JointState.from_position(current.position.clone(), joint_names=joint_names)
                arc_targets = SequenceGoalToolPose(
                    tool_frames=[tool_frame],
                    position=flange_pos_c[:, None, None, None, :].clone(),
                    quaternion=_xyzw_to_wxyz(flange_quat_c[:, None, None, None, :].clone()),
                )
                # Do not call solve_sequence(): it reset()s and global-IKs
                # frame 0, which can pick a different arm configuration than
                # the measured grasp. Seed local IK from the grasp and keep
                # waypoint 0 as those joints (angle 0 is the current TCP).
                retargeter.reset()
                retargeter._prev_solution = current.position.clone()
                if current.velocity is not None:
                    retargeter._prev_velocity = current.velocity.clone()
                frame_solutions = [current.position.clone()]
                for index in range(1, arc_targets.num_frames):
                    frame_solutions.append(
                        retargeter.solve_frame(arc_targets.get_frame(index))
                        .joint_state.reorder(joint_names)
                        .position
                    )
                joint_waypoints = torch.stack(frame_solutions, dim=1)

            infeasible = self._check_joint_discontinuity(
                joint_waypoints,
                joint_names,
                tcp_pos_b=tcp_pos_b,
                tcp_quat_b=tcp_quat_b,
                flange_pos_b=flange_pos_b,
                flange_quat_b=flange_quat_b,
                angles=angles,
            )
            joints = joint_waypoints[0]
            if infeasible:
                tcp_pos_b, tcp_quat_b, joints = self._densify_arc(tcp_pos_b, tcp_quat_b, joints)
                self._densified[env_ids] = True
                print(
                    f"[WARN] cuRobo env={self._diag_env} valve arc densified to {joints.shape[0]} waypoints "
                    f"so no joint moves more than {self.cfg.max_joint_step:.3f} rad per step; "
                    "segment flagged as expert fallback",
                    flush=True,
                )

            self._store_plan(env_ids, tcp_pos_b, tcp_quat_b, joints)
            print(
                f"[INFO] cuRobo env={self._diag_env} retargeted {num_waypoints} ordered waypoints for the valve arc."
            )

    def _plan_batch(self, env_ids: torch.Tensor) -> None:
        """Retarget up to ``cfg.plan_batch_size`` envs' arcs in one batched solve.

        Same arc, seeding and infeasible-arc handling as :meth:`_plan_env`.
        Arcs with fewer waypoints (a part-open valve turns less) are padded by
        holding their final pose; each env stores only its own waypoints.
        """
        from isaaclab_hiveboard.assets import ASSET_DIR
        from isaaclab_hiveboard.mdp.curobo_robot_cfg import load_curobo_robot_cfg
        from isaaclab_hiveboard.mdp.curobo_warp import curobo_compatible_warp

        if self._command_term._offset_pos is None or self._command_term._offset_rot is None:
            raise ValueError("CuroboPlannedRotateFrameCfg requires pose_command.body_offset")
        with curobo_compatible_warp():
            from curobo.motion_retargeter import MotionRetargeter, MotionRetargeterCfg, SequenceGoalToolPose
            from curobo.types import DeviceCfg, JointState, ToolPoseCriteria

            robot_cfg = load_curobo_robot_cfg(
                self.cfg.robot_curobo_yaml or f"{ASSET_DIR}/franka/cumotion/fr3.yaml",
                self.cfg.robot_urdf or f"{ASSET_DIR}/franka/cumotion/fr3.urdf",
            )
            tool_frame = robot_cfg["robot_cfg"]["kinematics"]["tool_frames"][0]
            size = int(self.cfg.plan_batch_size)
            cache_key = ("retargeter", self.cfg.robot_curobo_yaml, self.cfg.robot_urdf, self.cfg.num_ik_seeds, size)

            def _make_retargeter() -> MotionRetargeter:
                with torch.inference_mode(False):
                    return _batched_retargeter_class()(
                        MotionRetargeterCfg.create(
                            robot=robot_cfg,
                            tool_pose_criteria={
                                tool_frame: ToolPoseCriteria.track_position_and_orientation(
                                    xyz=[1.0, 1.0, 1.0], rpy=[1.0, 1.0, 1.0], non_terminal_scale=1.0
                                )
                            },
                            num_envs=size,
                            use_mpc=False,
                            self_collision_check=False,
                            scene_model=None,
                            load_collision_spheres=False,
                            optimization_dt=self._dt,
                            num_seeds_global=self.cfg.num_ik_seeds,
                            num_seeds_local=1,
                            position_tolerance=0.002,
                            orientation_tolerance=0.02,
                            device_cfg=DeviceCfg(),
                        )
                    )

            retargeter, build_s = self._command_term.get_curobo_solver(cache_key, _make_retargeter)
            solve_start = time.perf_counter()
            n = len(env_ids)
            pad = torch.cat((torch.arange(n, device=self._device), torch.full((size - n,), n - 1, device=self._device)))
            pad_env_ids = env_ids[pad]

            # Per-env arc: same angle step as _plan_env, padded to the longest.
            final_angle = self.angle_rad_tensor[pad_env_ids]
            step_angle = self.cfg.angular_velocity * self.speed_scale[pad_env_ids] * self._dt
            num_waypoints = (torch.ceil(final_angle.abs() / step_angle).clamp(min=1) + 1).long()
            horizon = int(num_waypoints.max())
            frac = torch.arange(horizon, device=self._device, dtype=torch.float32)[None] / (
                (num_waypoints - 1).clamp(min=1)[:, None].float()
            )
            angles = final_angle[:, None] * frac.clamp(max=1.0)  # (size, horizon)
            flat_ids = pad_env_ids.repeat_interleave(horizon)
            flat_angles = angles.reshape(-1)
            axis_b = self.rot_axis_b[flat_ids]
            tcp_pos_b = (
                self.axis_pos_b[flat_ids]
                + self.axial_vec[flat_ids]
                + self._rodrigues_rotate(self.radius_vec[flat_ids], axis_b, flat_angles)
                + self._screw_offset(axis_b, flat_angles)
            )
            tcp_quat_b = math_utils.quat_mul(
                math_utils.quat_from_angle_axis(flat_angles, axis_b), self.initial_quat_b[flat_ids]
            )
            flange_pos_b, flange_quat_b = self._command_term._tcp_pose_to_body_pose(tcp_pos_b, tcp_quat_b, flat_ids)

            joint_ids, joint_names = self._asset.find_joints(self.cfg.robot_joint_names, preserve_order=True)
            q = self._asset.data.joint_pos.torch[pad_env_ids][:, joint_ids]
            base_pos, base_quat = self._curobo_base_in_isaac_base(
                retargeter.kinematics.compute_kinematics(JointState.from_position(q, joint_names=joint_names)),
                tool_frame,
                pad_env_ids,
            )
            flange_pos_c, flange_quat_c = math_utils.subtract_frame_transforms(
                base_pos.repeat_interleave(horizon, 0),
                base_quat.repeat_interleave(horizon, 0),
                flange_pos_b,
                flange_quat_b,
            )
            # SequenceGoalToolPose is (frames, envs, links, goalset, 3/4).
            seq_pos = flange_pos_c.view(size, horizon, 3).transpose(0, 1)
            seq_quat = _xyzw_to_wxyz(flange_quat_c).view(size, horizon, 4).transpose(0, 1)

            with torch.inference_mode(False), torch.enable_grad():
                arc_targets = SequenceGoalToolPose(
                    tool_frames=[tool_frame],
                    position=seq_pos[:, :, None, None, :].contiguous().clone(),
                    quaternion=seq_quat[:, :, None, None, :].contiguous().clone(),
                )
                # Seed local IK from each env's grasp joints (see _plan_env).
                retargeter.reset()
                retargeter._prev_solution = q.clone()
                frame_solutions = [q.clone()]
                for index in range(1, horizon):
                    frame_solutions.append(
                        retargeter.solve_frame(arc_targets.get_frame(index)).joint_state.reorder(joint_names).position
                    )
                joint_waypoints = torch.stack(frame_solutions, dim=1)  # (size, horizon, J)

            solve_s = time.perf_counter() - solve_start
            tcp_pos_b = tcp_pos_b.view(size, horizon, 3)
            tcp_quat_b = tcp_quat_b.view(size, horizon, 4)
            densified = 0
            for k in range(n):
                env = env_ids[k : k + 1]
                self._diag_env = int(env.item())
                count = int(num_waypoints[k])
                joints, pos, quat = joint_waypoints[k, :count], tcp_pos_b[k, :count], tcp_quat_b[k, :count]
                if self._check_joint_discontinuity(
                    joints[None], joint_names, tcp_pos_b=pos, tcp_quat_b=quat, angles=angles[k, :count]
                ):
                    pos, quat, joints = self._densify_arc(pos, quat, joints)
                    self._densified[env] = True
                    densified += 1
                self._store_plan(env, pos, quat, joints)
            print(
                f"[INFO] cuRobo batch arc: {n} envs, {horizon} waypoints max, {densified} densified "
                f"(batch={size} build={build_s:.2f}s solve={solve_s:.2f}s)",
                flush=True,
            )

    def _densify_arc(
        self, tcp_pos_b: torch.Tensor, tcp_quat_b: torch.Tensor, joints: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subdivide an arc so no joint moves more than ``max_joint_step`` per waypoint.

        The joints are interpolated linearly across each oversized step, so the
        arm crosses an IK branch change gradually instead of in one tick. The
        TCP leaves the ideal arc while it does; the TCP waypoints (used only for
        visualization and the pose command) hold the segment's end pose.

        Args:
            tcp_pos_b: ``(T, 3)`` TCP positions.
            tcp_quat_b: ``(T, 4)`` TCP orientations (xyzw).
            joints: ``(T, J)`` joint waypoints.

        Returns:
            The densified ``(tcp_pos_b, tcp_quat_b, joints)``.
        """
        steps = torch.abs(joints[1:] - joints[:-1]).amax(dim=-1)
        pieces = torch.clamp(torch.ceil(steps / self.cfg.max_joint_step), min=1).long().tolist()
        out_pos, out_quat, out_joints = [tcp_pos_b[:1]], [tcp_quat_b[:1]], [joints[:1]]
        for i, n in enumerate(pieces):
            frac = torch.arange(1, n + 1, device=joints.device, dtype=joints.dtype)[:, None] / n
            out_joints.append(joints[i] + frac * (joints[i + 1] - joints[i]))
            out_pos.append(tcp_pos_b[i + 1].expand(n, -1))
            out_quat.append(tcp_quat_b[i + 1].expand(n, -1))
        return torch.cat(out_pos), torch.cat(out_quat), torch.cat(out_joints)

    def _check_joint_discontinuity(
        self,
        joint_waypoints: torch.Tensor,
        joint_names: list[str],
        *,
        tcp_pos_b: torch.Tensor | None = None,
        tcp_quat_b: torch.Tensor | None = None,
        flange_pos_b: torch.Tensor | None = None,
        flange_quat_b: torch.Tensor | None = None,
        angles: torch.Tensor | None = None,
        context: int = 2,
    ) -> bool:
        """Report whether the arc exceeds ``max_joint_step``, dumping why if it does.

        Raises:
            RuntimeError: If the arc is infeasible and ``on_infeasible_arc`` is ``"raise"``.
        """
        if joint_waypoints.shape[1] < 2:
            return False
        deltas = torch.abs(joint_waypoints[:, 1:] - joint_waypoints[:, :-1])
        max_joint_step = torch.max(deltas)
        if max_joint_step <= self.cfg.max_joint_step:
            return False
        abs_d = deltas[0]
        n_joints = abs_d.shape[-1]
        flat = int(torch.argmax(abs_d).item())
        waypoint_i = flat // n_joints
        joint_i = flat % n_joints
        header = (
            "cuRobo valve arc is infeasible: "
            f"max joint step={float(max_joint_step.item()):.3f} rad > "
            f"max_joint_step={self.cfg.max_joint_step:.3f} rad "
            f"(waypoint {waypoint_i}->{waypoint_i + 1}, joint {joint_names[joint_i]})"
        )
        dump = self._format_infeasible_arc_dump(
            header=header,
            joint_waypoints=joint_waypoints,
            joint_names=joint_names,
            waypoint_i=waypoint_i,
            joint_i=joint_i,
            abs_d=abs_d,
            tcp_pos_b=tcp_pos_b,
            tcp_quat_b=tcp_quat_b,
            flange_pos_b=flange_pos_b,
            flange_quat_b=flange_quat_b,
            angles=angles,
            context=context,
        )
        print(dump, flush=True)
        if self.cfg.on_infeasible_arc == "raise":
            raise RuntimeError(dump)
        return True

    def _format_infeasible_arc_dump(
        self,
        *,
        header: str,
        joint_waypoints: torch.Tensor,
        joint_names: list[str],
        waypoint_i: int,
        joint_i: int,
        abs_d: torch.Tensor,
        tcp_pos_b: torch.Tensor | None,
        tcp_quat_b: torch.Tensor | None,
        flange_pos_b: torch.Tensor | None,
        flange_quat_b: torch.Tensor | None,
        angles: torch.Tensor | None,
        context: int,
    ) -> str:
        q_seed = joint_waypoints[0, 0]
        lines = [
            header,
            "--- robot state (grasp seed / waypoint 0) ---",
            f"  {_joints_str(joint_names, q_seed)}",
        ]
        live_q = self._measured_arm_q(joint_names)
        if live_q is not None:
            lines.append(f"  measured: {_joints_str(joint_names, live_q)}")
        limit_line = self._joint_limit_line(joint_names, q_seed)
        if limit_line:
            lines.append(limit_line)

        final_angle = float(self.angle_rad_tensor[self._diag_env].item()) if hasattr(self, "angle_rad_tensor") else float("nan")
        lines.extend(
            [
                "--- requested arc ---",
                (
                    f"  angle={math.degrees(final_angle):.2f} deg "
                    f"({final_angle:.4f} rad)  cfg.angle_deg={getattr(self.cfg, 'angle_deg', float('nan'))}  "
                    f"max_ee_rotation_deg={getattr(self.cfg, 'max_ee_rotation_deg', 0.0)}"
                ),
            ]
        )
        if hasattr(self, "rot_axis_b"):
            radius = float(torch.linalg.vector_norm(self.radius_vec[self._diag_env]).item())
            axial = float(torch.linalg.vector_norm(self.axial_vec[self._diag_env]).item())
            step_angle = float(self.cfg.angular_velocity) * self._dt
            lines.extend(
                [
                    f"  hub_b={_vec_str(self.axis_pos_b[self._diag_env])}  axis_b={_vec_str(self.rot_axis_b[self._diag_env])}",
                    f"  radius={radius:.4f} m  axial_offset={axial:.4f} m",
                    (
                        f"  n_waypoints={joint_waypoints.shape[1]}  dt={self._dt:.4f} s  "
                        f"ang_vel={self.cfg.angular_velocity:.3f} rad/s  step_angle={step_angle:.4f} rad"
                    ),
                ]
            )

        per_joint_max = abs_d.max(dim=0).values
        per_joint_at = abs_d.argmax(dim=0)
        max_bits = []
        for name, mag, at in zip(joint_names, per_joint_max, per_joint_at):
            mark = " <<" if name == joint_names[joint_i] else ""
            max_bits.append(f"{name}={float(mag):.3f}@{int(at)}->{int(at) + 1}{mark}")
        lines.extend(
            [
                "--- retargeted joints: per-joint max |Δq| (rad) ---",
                "  " + "  ".join(max_bits),
            ]
        )

        over = (abs_d > self.cfg.max_joint_step).nonzero(as_tuple=False)
        over_jumps = [(int(wp), int(j)) for wp, j in over[:8]]
        if over_jumps:
            lines.append(
                f"--- all joints exceeding max_joint_step={self.cfg.max_joint_step:.3f} rad ---"
            )
            for wp, j in over_jumps:
                mark = " << worst" if (wp, j) == (waypoint_i, joint_i) else ""
                lines.append(
                    f"  {joint_names[j]} {float(abs_d[wp, j]):.3f} rad at {wp}->{wp + 1}{mark}"
                )

        for wp, j in over_jumps or [(waypoint_i, joint_i)]:
            lines.extend(
                self._format_jump_block(
                    joint_waypoints=joint_waypoints,
                    joint_names=joint_names,
                    waypoint_i=wp,
                    joint_i=j,
                    tcp_pos_b=tcp_pos_b,
                    tcp_quat_b=tcp_quat_b,
                    flange_pos_b=flange_pos_b,
                    flange_quat_b=flange_quat_b,
                )
            )

        n_wp = joint_waypoints.shape[1]
        window_wps = [wp for wp, _ in over_jumps] or [waypoint_i]
        lo = max(0, min(window_wps) - context)
        hi = min(n_wp - 1, max(window_wps) + 1 + context)
        jump_at = {wp + 1: (wp, j) for wp, j in (over_jumps or [(waypoint_i, joint_i)])}
        lines.append(
            f"--- plan around discontinuities [{lo}..{hi}] ---"
        )
        for index in range(lo, hi + 1):
            bits = [f"  [{index}]"]
            if angles is not None:
                bits.append(f"ang={float(angles[index].item()):+.4f} rad")
            if tcp_pos_b is not None and tcp_quat_b is not None:
                bits.append("tcp " + _pose_str(tcp_pos_b[index : index + 1], tcp_quat_b[index : index + 1]))
            if flange_pos_b is not None and flange_quat_b is not None:
                bits.append(
                    "flange " + _pose_str(flange_pos_b[index : index + 1], flange_quat_b[index : index + 1])
                )
            bits.append("q " + _joints_str(joint_names, joint_waypoints[0, index]))
            limit_at = self._joint_limit_line(joint_names, joint_waypoints[0, index])
            if limit_at and index in jump_at:
                bits.append(limit_at.strip())
            lines.append("  ".join(bits))
            if index in jump_at:
                src, j = jump_at[index]
                lines.append(f"       << jump from {src} on {joint_names[j]}")
        return "\n".join(lines)

    def _format_jump_block(
        self,
        *,
        joint_waypoints: torch.Tensor,
        joint_names: list[str],
        waypoint_i: int,
        joint_i: int,
        tcp_pos_b: torch.Tensor | None,
        tcp_quat_b: torch.Tensor | None,
        flange_pos_b: torch.Tensor | None,
        flange_quat_b: torch.Tensor | None,
    ) -> list[str]:
        q_a = joint_waypoints[0, waypoint_i]
        q_b = joint_waypoints[0, waypoint_i + 1]
        signed = q_b - q_a
        lines = [f"--- jump {waypoint_i}->{waypoint_i + 1} on {joint_names[joint_i]}  Δq ---"]
        jump_bits = []
        for name, dq in zip(joint_names, signed.detach().cpu().tolist()):
            mark = " <<" if name == joint_names[joint_i] else ""
            jump_bits.append(f"{name}={dq:+.4f}{mark}")
        lines.append("  " + "  ".join(jump_bits))
        before = self._joint_limit_line(joint_names, q_a)
        after = self._joint_limit_line(joint_names, q_b)
        if before:
            lines.append(f"  before {before.strip()}")
        if after:
            lines.append(f"  after  {after.strip()}")
        pose_step = self._pose_step_line("tcp", tcp_pos_b, tcp_quat_b, waypoint_i)
        if pose_step:
            lines.append(pose_step)
        pose_step = self._pose_step_line("flange", flange_pos_b, flange_quat_b, waypoint_i)
        if pose_step:
            lines.append(pose_step)
        return lines

    @staticmethod
    def _pose_step_line(
        label: str,
        pos: torch.Tensor | None,
        quat: torch.Tensor | None,
        waypoint_i: int,
    ) -> str | None:
        if pos is None or quat is None:
            return None
        dpos = float(torch.linalg.vector_norm(pos[waypoint_i + 1] - pos[waypoint_i]).item())
        dori = float(
            torch.linalg.vector_norm(
                math_utils.quat_box_minus(
                    quat[waypoint_i + 1 : waypoint_i + 2],
                    quat[waypoint_i : waypoint_i + 1],
                )
            ).item()
        )
        return f"  {label} step: Δpos={dpos:.4f} m  Δori={dori:.4f} rad"

    def _measured_arm_q(self, joint_names: list[str]) -> torch.Tensor | None:
        try:
            joint_ids, _ = self._asset.find_joints(joint_names, preserve_order=True)
            q = self._asset.data.joint_pos
            if hasattr(q, "torch"):
                q = q.torch
            return q[self._diag_env, joint_ids]
        except (AttributeError, TypeError, IndexError, ValueError):
            return None

    def _joint_limit_line(self, joint_names: list[str], q: torch.Tensor) -> str | None:
        try:
            joint_ids, _ = self._asset.find_joints(joint_names, preserve_order=True)
            limits = self._asset.data.joint_pos_limits
            if hasattr(limits, "torch"):
                limits = limits.torch
            lo_hi = limits[self._diag_env, joint_ids]
        except (AttributeError, TypeError, IndexError, ValueError):
            return None
        bits = []
        for name, value, bound in zip(joint_names, q.detach().cpu().tolist(), lo_hi.detach().cpu().tolist()):
            lo, hi = float(bound[0]), float(bound[1])
            bits.append(f"{name}={value:.4f} in [{lo:.4f}, {hi:.4f}] to_lo={value - lo:.4f} to_hi={hi - value:.4f}")
        return "  limits: " + "  ".join(bits)


def _batched_retargeter_class():
    """``MotionRetargeter`` whose IK solvers accept ``num_envs`` problems per call.

    cuRobo's retargeter takes ``num_envs`` but builds its global and local IK
    solvers with the default ``max_batch_size=1``, so any batch above one is
    rejected by ``IKSolver.solve_pose``. This rebuilds both solvers with the
    same settings plus ``max_batch_size=num_envs``. Import lazily: cuRobo must
    be imported inside ``curobo_compatible_warp``.
    """
    from curobo._src.solver.solver_ik import IKSolver
    from curobo._src.solver.solver_ik_cfg import IKSolverCfg
    from curobo.motion_retargeter import MotionRetargeter

    class _BatchedMotionRetargeter(MotionRetargeter):
        def _build_global_ik_solver(self) -> IKSolver:
            cfg = self._config
            solver = IKSolver(
                IKSolverCfg.create(
                    robot=cfg.robot,
                    num_seeds=cfg.num_seeds_global,
                    position_tolerance=cfg.position_tolerance,
                    orientation_tolerance=cfg.orientation_tolerance,
                    self_collision_check=cfg.self_collision_check,
                    scene_model=cfg.scene_model,
                    optimization_dt=None,
                    override_iters_for_multi_link_ik=None,
                    override_optimizer_num_iters={"lbfgs": cfg.global_ik_num_iters},
                    optimizer_configs=cfg.ik_optimizer_configs,
                    device_cfg=cfg.device_cfg,
                    load_collision_spheres=cfg.load_collision_spheres,
                    optimizer_collision_activation_distance=cfg.collision_activation_distance,
                    max_batch_size=cfg.num_envs,
                )
            )
            solver.update_tool_pose_criteria(self._tool_pose_criteria)
            return solver

        def _build_local_ik_solver(self) -> IKSolver:
            cfg = self._config
            solver = IKSolver(
                IKSolverCfg.create(
                    robot=cfg.robot,
                    optimizer_configs=cfg.ik_optimizer_configs,
                    num_seeds=cfg.num_seeds_local,
                    position_tolerance=cfg.position_tolerance,
                    orientation_tolerance=cfg.orientation_tolerance,
                    self_collision_check=cfg.self_collision_check,
                    scene_model=cfg.scene_model,
                    optimization_dt=cfg.optimization_dt,
                    override_iters_for_multi_link_ik=None,
                    override_optimizer_num_iters={"lbfgs": cfg.local_ik_num_iters},
                    device_cfg=cfg.device_cfg,
                    load_collision_spheres=cfg.load_collision_spheres,
                    velocity_regularization_weight=cfg.velocity_regularization_weight,
                    acceleration_regularization_weight=cfg.acceleration_regularization_weight,
                    optimizer_collision_activation_distance=cfg.collision_activation_distance,
                    max_batch_size=cfg.num_envs,
                )
            )
            solver.update_tool_pose_criteria(self._tool_pose_criteria)
            solver.use_lm_seed = False
            solver.exit_early = False
            return solver

    return _BatchedMotionRetargeter


@dataclass
class _ScrewMimicEntry:
    """A registered coupling plus the joint positions its constraint is anchored to.

    ``revolute_pos_at_reset``/``prismatic_pos_at_reset`` must be the same
    values :class:`~isaaclab.assets.ArticulationCfg`'s ``init_state.joint_pos``
    will write on every episode reset — *not* whatever the USD authored as
    each joint's own default coordinate, which can (and for the lamp, does)
    differ. See :func:`register_screw_joint_mimic`.
    """

    coupling: "ScrewJointCouplingCfg"
    revolute_pos_at_reset: float
    prismatic_pos_at_reset: float


def _find_world_joint(builder, world: int, joint_name: str) -> int | None:
    """Find the single joint named ``joint_name`` that belongs to ``world``.

    ``builder.joint_label`` holds each joint's full USD prim path, and after
    replication every world's joints carry that world's own path (rewritten
    by ``_rename_builder_labels``), so matching on the trailing path segment
    disambiguates joints that share a short name across assets. Returns
    ``None`` if no joint (or more than one) matches, so the caller can warn
    and skip instead of guessing.
    """
    suffix = "/" + joint_name
    match: int | None = None
    for idx, (label, joint_world) in enumerate(zip(builder.joint_label, builder.joint_world)):
        if joint_world != world:
            continue
        if label == joint_name or label.endswith(suffix):
            if match is not None:
                return None
            match = idx
    return match


def _add_registered_screw_mimics_to_builder(_payload=None) -> None:
    """``PhysicsEvent.MODEL_INIT`` callback: add one native mimic constraint per env.

    Runs after replication, so the shared Newton builder (the simulation's
    ``NewtonBuilderCfg`` backend) already holds every
    env's joints in one flat builder (see :func:`register_screw_joint_mimic`
    for why this can't run earlier), and resolves per-env leader/follower
    joint indices from it. ``current_world`` is a read-only property normally driven by
    ``begin_world``/``end_world``, and those can only *open a new* world, not
    re-enter one that replication already closed — so this reaches past the
    property (``builder._current_world``) to satisfy ``add_constraint_mimic``'s
    same-world check for each already-built env in turn.

    The outer loop over ``registry`` before the inner loop over ``world``
    means multiple registered couplings would *not* end up contiguous per
    world in ``constraint_mimic_world`` (all of coupling A's envs, then all
    of coupling B's). The MuJoCo/MJWarp solver's mimic-constraint replication
    assumes exactly that contiguity (``mimic_per_world = constraint_mimic_
    count // world_count``, templated from ``world == 0``), so two
    simultaneously-registered couplings would need the loop nesting swapped.
    Harmless today since the lamp is the only registrant.
    """
    from isaaclab.sim import SimulationContext
    from isaaclab_newton.physics import NewtonBuilderCfg, NewtonManager

    registry = getattr(NewtonManager, "_screw_mimic_registry", None)
    if not registry:
        return
    sim = SimulationContext.instance()
    builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
    num_worlds = builder.world_count
    for entry in registry:
        coupling = entry.coupling
        coef1 = coupling.pitch_m_per_revolution / (2.0 * math.pi)
        coef0 = entry.prismatic_pos_at_reset - coef1 * entry.revolute_pos_at_reset
        for world in range(num_worlds):
            revolute_idx = _find_world_joint(builder, world, coupling.revolute_joint_name)
            prismatic_idx = _find_world_joint(builder, world, coupling.prismatic_joint_name)
            if revolute_idx is None or prismatic_idx is None:
                print(
                    f"[WARN] Screw mimic: could not uniquely resolve "
                    f"'{coupling.revolute_joint_name}'/'{coupling.prismatic_joint_name}' for "
                    f"asset '{coupling.asset_name}' in world {world}; skipping native constraint there.",
                    flush=True,
                )
                continue
            prev_world = builder._current_world
            builder._current_world = world
            try:
                builder.add_constraint_mimic(
                    joint0=prismatic_idx,
                    joint1=revolute_idx,
                    coef0=coef0,
                    coef1=coef1,
                    enabled=True,
                    label=f"{coupling.asset_name}_screw_mimic",
                )
            finally:
                builder._current_world = prev_world

    # add_constraint_mimic only appends, but ModelBuilder.finalize() requires
    # constraint_mimic_world to be globally non-decreasing — and other mimic
    # joints can already be sitting in that array in world order before this
    # callback ever runs. For example Franka's URDF-imported gripper mimic
    # (fr3_finger_joint2 mimicking fr3_finger_joint1) loads fine through the
    # normal USD import path — unlike the lamp's screw ActionGraph, it isn't
    # affected by the apiSchemas-drop bug described on ScrewJointCouplingCfg
    # — and lands as one constraint per world, already in order. Appending
    # this coupling's own per-world entries after that leaves the array as
    # [0,1,2,3, 0,1,2,3]: correct within each block, but not globally
    # non-decreasing. A stable sort on world fixes that regardless of what
    # else contributed constraints.
    order = sorted(range(len(builder.constraint_mimic_world)), key=lambda i: builder.constraint_mimic_world[i])
    if order != list(range(len(order))):
        builder.constraint_mimic_joint0 = [builder.constraint_mimic_joint0[i] for i in order]
        builder.constraint_mimic_joint1 = [builder.constraint_mimic_joint1[i] for i in order]
        builder.constraint_mimic_coef0 = [builder.constraint_mimic_coef0[i] for i in order]
        builder.constraint_mimic_coef1 = [builder.constraint_mimic_coef1[i] for i in order]
        builder.constraint_mimic_enabled = [builder.constraint_mimic_enabled[i] for i in order]
        builder.constraint_mimic_label = [builder.constraint_mimic_label[i] for i in order]
        builder.constraint_mimic_world = [builder.constraint_mimic_world[i] for i in order]


def _install_screw_mimic_hook() -> None:
    """Idempotently register :func:`_add_registered_screw_mimics_to_builder`."""
    from isaaclab.physics import PhysicsEvent
    from isaaclab_newton.physics import NewtonManager

    if not hasattr(NewtonManager, "_screw_mimic_registry"):
        NewtonManager._screw_mimic_registry = []
    if not getattr(NewtonManager, "_screw_mimic_hook_installed", False):
        NewtonManager._screw_mimic_hook_installed = True
        NewtonManager.register_callback(
            _add_registered_screw_mimics_to_builder,
            PhysicsEvent.MODEL_INIT,
            name="screw_joint_mimic",
        )


def register_screw_joint_mimic(
    coupling: "ScrewJointCouplingCfg",
    *,
    revolute_pos_at_reset: float,
    prismatic_pos_at_reset: float,
) -> None:
    """Register ``coupling`` for a native Newton mimic constraint at ``MODEL_INIT``.

    ``SequentialPoseCommand.__init__`` runs too late for this: by the time a
    ``CommandTerm`` is constructed, ``ManagerBasedEnv.__init__`` has already
    called ``sim.reset()``, which fires ``MODEL_INIT`` and finalizes the
    model. Call this instead from an env cfg's ``__post_init__`` (the last
    point that both sees fully-resolved cfg values — after subclasses like
    Franka's have applied their overrides — and still runs before the scene
    and sim are built). See :class:`ScrewJointCouplingCfg` for the full
    story of why this needs a builder-level hook instead of the USD-authored
    ``<mimic>`` schema.

    Args:
        coupling: The screw coupling to add a native constraint for.
        revolute_pos_at_reset: The revolute joint position [rad] that
            ``ArticulationCfg.init_state.joint_pos`` writes on every episode
            reset. Pass ``self.scene.<asset_name>.init_state.joint_pos`` from
            the calling ``__post_init__`` — *not* a value read back from the
            builder or the live sim, since the USD's own build-time default
            coordinate can differ from what IsaacLab resets to (it does, for
            the lamp: 0.0 either way, but only by coincidence — see
            ``prismatic_pos_at_reset``). The constraint is a fixed affine
            relationship anchored at this pair, so it must match reset state
            exactly or every reset starts the episode with the constraint
            already violated.
        prismatic_pos_at_reset: As above, for the prismatic joint. For the
            lamp this is ``0.024`` (its unscrewed reset position) — *not* ``0.0``, which is what
            the USD itself authors as the joint's build-time default.
    """
    _install_screw_mimic_hook()
    from isaaclab_newton.physics import NewtonManager

    for existing in NewtonManager._screw_mimic_registry:
        if (
            existing.coupling.revolute_joint_name == coupling.revolute_joint_name
            and existing.coupling.prismatic_joint_name == coupling.prismatic_joint_name
            and existing.coupling.asset_name == coupling.asset_name
        ):
            return
    NewtonManager._screw_mimic_registry.append(
        _ScrewMimicEntry(
            coupling=coupling,
            revolute_pos_at_reset=revolute_pos_at_reset,
            prismatic_pos_at_reset=prismatic_pos_at_reset,
        )
    )


@configclass
class ScrewJointCouplingCfg:
    """Environment-side equivalent of the lamp USD's screw ActionGraph.

    A solver-native alternative exists (Newton's ``add_constraint_mimic`` /
    URDF ``<mimic>``, enforcing ``prismatic = coef0 + coef1 * revolute`` as a
    real constraint instead of this per-step position target). Authoring it
    as USD ``<mimic>`` metadata does not currently work through this
    project's USD spawn path: Isaac Lab flattens the composed stage before
    Newton's importer parses it, and that flatten drops the ``apiSchemas``
    list metadata (``NewtonJointAPI``/``NewtonMimicAPI``) even though the
    underlying ``newton:mimic*`` attributes survive. Newton's importer gates
    its whole mimic pass on ``prim.HasAPI("NewtonMimicAPI")``, so the
    constraint silently never loads (``model.constraint_mimic_count`` stays
    0) — confirmed by instrumenting ``newton/_src/utils/import_usd.py``
    directly. Installing the ``newton-usd-schemas`` PyPI package does not
    fix it: it registers against the venv's own ``pxr``, not whatever USD
    library Isaac Lab's stage composition actually runs through here.

    :func:`register_screw_joint_mimic` takes the other viable path instead:
    bypassing USD entirely and adding the constraint straight to the Newton
    ``ModelBuilder`` in Python, via a
    ``NewtonManager.register_callback(..., PhysicsEvent.MODEL_INIT)`` hook
    (the same mechanism ``isaaclab_contrib``'s deformable objects use to
    register cloth/soft-body meshes into the builder — see
    :func:`register_screw_joint_mimic`). This project's env cloning goes
    through ``newton_physics_replicate`` (``isaaclab_newton.cloner``), which
    runs well before ``MODEL_INIT`` and leaves the shared Newton builder
    already fully replicated and label-renamed by the time the callback
    fires — unlike the single-call standalone lead-screw example
    (``newton/examples/basic/example_basic_mimic_joint.py``), the hook
    therefore resolves ``RevoluteJoint``/``PrismaticJoint`` indices and
    reopens each env's ``current_world`` (see
    :func:`_add_registered_screw_mimics_to_builder`) to add one
    ``add_constraint_mimic`` call per environment.

    This constraint is a fixed affine relationship, computed once from the
    ``revolute_pos_at_reset``/``prismatic_pos_at_reset`` the caller passes to
    :func:`register_screw_joint_mimic` — it does not re-anchor itself the way
    :meth:`SequentialPoseCommand._reset_screw_coupling` re-anchors the
    per-step position target from whatever is measured at reset. Those two
    values must be exactly what ``ArticulationCfg.init_state.joint_pos``
    writes on every episode reset, *not* the joint's own USD-authored
    build-time default: for the lamp these differ (build-time prismatic
    coordinate is ``0.0``; the reset value used throughout this task is
    ``LAMP_UNSCREWED_POSITION`` = ``0.024``), and anchoring to the wrong one
    means every reset starts the episode with the constraint already
    violated by that gap. Fixed-anchor re-anchoring like this is fine only
    because this task's resets are deterministic (``reset_scene_to_default``,
    no joint randomization on this pair); a task that randomizes the
    revolute/prismatic start pose independently would need a different
    anchoring scheme entirely. The per-step position target and
    friction/end-stop resistance model below keep running unchanged
    alongside the native constraint — it is an additive, solver-level
    backstop, not a replacement, so existing behavior is preserved even if
    the constraint fails to resolve for some env (see the warning printed
    by :func:`_add_registered_screw_mimics_to_builder`).
    """

    asset_name: str = MISSING  # type: ignore
    revolute_joint_name: str = "RevoluteJoint"
    prismatic_joint_name: str = "PrismaticJoint"
    pitch_m_per_revolution: float = 0.006
    command_joint_angle_scale: float = 0.0
    lower_limit: float = 0.0
    upper_limit: float = 0.024
    viscous_friction: float = 2.0
    coulomb_friction: float = 1.0
    stiction: float = 3.0
    velocity_epsilon: float = 0.01
    end_stop_base_damping: float = 10.0
    end_stop_scale: float = 0.02
    end_stop_power: float = 2.5
    end_stop_activation_distance: float = 0.005


@configclass
class SequentialPoseCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    @configclass
    class OffsetCfg:
        """The offset pose from parent frame to child frame.

        On many robots, end-effector frames are fictitious frames that do not have a corresponding
        rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body.
        For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the
        "fr3_hand" (or similar) frame.
        """

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        """Quaternion rotation ``(x, y, z, w)`` w.r.t. the parent frame. Defaults to identity."""

    class_type: type = SequentialPoseCommand

    asset_name: str = MISSING  # type: ignore
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING  # type: ignore
    """Name of the end-effector body used for pose commands."""

    body_offset: OffsetCfg | None = None
    """Offset of target frame w.r.t. to the body frame. Defaults to None, in which case no offset is applied."""

    commands: Sequence[BaseCmd] = MISSING  # type: ignore
    """The sequence of commands to execute."""

    screw_coupling: ScrewJointCouplingCfg | None = None
    """Optional measured revolute-to-prismatic screw coupling."""

    valve_asset_name: str | None = None
    """Valve articulation to use for per-episode open/close tasks."""
    valve_joint_name: str = "RevoluteJoint"
    """Valve joint controlled by the task."""
    open_task_prob: float = 0.5
    """Probability of selecting the open endpoint when both directions are feasible."""
    valve_joint_closed: float = 0.0
    """Joint position representing the closed endpoint [rad]."""
    valve_joint_open: float = -math.pi / 2
    """Joint position representing the open endpoint [rad]."""
    valve_min_delta_rad: float = math.radians(20.0)
    """Minimum required distance between the sampled start and desired endpoint."""
    valve_ee_joint_angle_scale: float = 1.0
    """Scale from remaining valve error to the commanded EE arc angle."""

    output_joint_positions: bool = False
    """If True, ``command`` is cuRobo joint waypoints plus the gripper bit.

    Handlers still track a TCP pose internally. Between plans the last
    joints are held. Matches a ``JointPositionAction`` arm term plus gripper.
    """
    ik_joint_names: Sequence[str] | None = None
    """Arm joints for ``output_joint_positions``. Defaults to the cuRobo command's names."""

    debug_vis: bool = False

    log_transitions: bool = False
    """Log segment start/finish with per-segment durations."""

    stall_timeout_s: float = 0.0
    """Warn + dump diagnostics when a segment exceeds this duration [s]. 0 disables."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(  # type: ignore
        prim_path="/Visuals/Command/pose_goal"
    )

    """The configuration for the goal pose visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(  # type: ignore
        prim_path="/Visuals/Command/current_pose"
    )

    """The configuration for the current pose visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # type: ignore
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # type: ignore


@configclass
class MechanismGoalCfg:
    """A mechanism state that ends a segment: a valve turned open, a button pressed in.

    Every condition that is set must hold. Unset conditions are ignored, and at
    least one must be set.
    """

    asset_name: str | None = None
    """Articulation holding the joint. None: the command's ``valve_asset_name``."""
    joint_name: str | None = None
    """Joint to check. None: the command's ``valve_joint_name``."""
    low: float | None = None
    """The joint is at or above this [rad or m]."""
    high: float | None = None
    """The joint is at or below this [rad or m]."""
    goal_tolerance: float | None = None
    """The joint is within this of the goal the command sampled for the episode
    (``valve_joint_des``, open or closed) [rad or m]. Only for the command's own mechanism."""
    settle_speed: float | None = None
    """The joint moves slower than this [rad/s or m/s]: at rest, e.g. on its end stop."""


class _MechanismGoal:
    """Evaluates a :class:`MechanismGoalCfg` against the scene."""

    def __init__(self, cfg: MechanismGoalCfg, command_term: SequentialPoseCommand):
        self.cfg = cfg
        self._term = command_term
        if all(getattr(cfg, name) is None for name in ("low", "high", "goal_tolerance", "settle_speed")):
            raise ValueError("MechanismGoalCfg needs at least one of low, high, goal_tolerance, settle_speed")
        asset_name = cfg.asset_name or command_term.cfg.valve_asset_name
        joint_name = cfg.joint_name or command_term.cfg.valve_joint_name
        if asset_name is None:
            raise ValueError("MechanismGoalCfg needs asset_name: the command has no valve_asset_name")
        own = (asset_name, joint_name) == (command_term.cfg.valve_asset_name, command_term.cfg.valve_joint_name)
        if cfg.goal_tolerance is not None and not own:
            raise ValueError("MechanismGoalCfg.goal_tolerance only applies to the command's own valve joint")
        self._asset: BaseArticulation = command_term._env.scene[asset_name]
        joint_ids, _ = self._asset.find_joints(joint_name)
        if len(joint_ids) != 1:
            raise ValueError(f"Expected one joint '{joint_name}' in '{asset_name}', found {len(joint_ids)}")
        self._joint = joint_ids[0]

    def holds(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Whether the goal holds for ``env_ids``."""
        q = self._asset.data.joint_pos.torch[env_ids, self._joint]
        ok = torch.ones_like(q, dtype=torch.bool)
        if self.cfg.low is not None:
            ok &= q >= self.cfg.low
        if self.cfg.high is not None:
            ok &= q <= self.cfg.high
        if self.cfg.goal_tolerance is not None:
            ok &= (q - self._term.valve_joint_des[env_ids]).abs() <= self.cfg.goal_tolerance
        if self.cfg.settle_speed is not None:
            ok &= self._asset.data.joint_vel.torch[env_ids, self._joint].abs() <= self.cfg.settle_speed
        return ok


@configclass
class BaseCmd:
    """Base configuration for frame pose command generators."""

    class_type: Type[_BaseCmdHandler] = MISSING  # type: ignore
    phase: str | None = None
    """Phase of the sequence this segment belongs to, one of :data:`PHASES` (see ``skills.py``). None: untagged."""
    until: MechanismGoalCfg | None = None
    """Also end the segment as soon as this mechanism goal holds, at any point of the segment. None: no goal."""


@configclass
class GoToFrameCfg(BaseCmd):
    class_type = _GoToFrameHandler

    frame_name: str = MISSING  # type: ignore
    """Name of the frame used for pose commands."""
    target_frame_name: str = MISSING  # type: ignore
    """Index of the frame used for pose commands."""
    target_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Goal translation in the selected target frame [m]; ignored for fixed env goals."""
    target_offset_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Goal rotation (xyzw) in the selected target frame; ignored for fixed env goals."""
    target_position_env: tuple[float, float, float] | None = None
    """Optional fixed position relative to the environment origin.

    When set, bypass the frame sensor and retain the orientation at reset.
    Set both frame names to empty strings for this mode.
    """
    target_orientation_env: tuple[float, float, float, float] | None = None
    """Fixed target quaternion (xyzw), used with target_position_env.

    None retains the orientation at reset. Environment axes match world axes.
    """
    gripper_open: bool = False
    """Status of the gripper during the command."""
    velocity: float = 0.2
    """Linear speed of the commanded pose toward the target frame [m/s]."""
    angular_velocity: float = 1.0
    """Angular speed of the commanded orientation toward the target [rad/s]."""
    distance_threshold: float = 0.05
    """End-effector distance to the target frame at which the command is done [m]."""
    orientation_threshold_deg: float = 10.0
    """End-effector orientation error to the target at which the command is done [deg]."""
    valve_done_threshold_rad: float | None = None
    """Also complete once the valve is within this of its desired angle [rad].

    For contact-rich segments (e.g. the valve arc) the TCP can hold centimeters
    off-target while the task is physically done. None disables.
    """
    done_when_joint: tuple[str, str, float, float] | None = None
    """Also complete once ``(asset_name, joint_name, low, high)`` has the joint in ``[low, high]``.

    For pushes and presses whose stop leaves the TCP short of the target by a
    robot-specific amount. None disables.
    """
    upright_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    canonicalize_upward: bool = True
    """If True, choose the upright TCP +X half-turn once at the segment start.

    Keep that choice while following the moving reference so crossing a
    horizontal pose does not abruptly reverse the orientation target.

    The selected ``upright_axis`` in TCP coordinates must point toward base +Z.
    Disable this for targets whose authored orientation must be used as-is.
    """
    hold_current_orientation: bool = False
    """Translate to the target while retaining the orientation captured at reset."""


@configclass
class CuroboPlannedGoToFrameCfg(GoToFrameCfg):
    """A cuRobo-planned variant of :class:`GoToFrameCfg`."""

    class_type = _CuroboPlannedGoToFrameHandler
    robot_joint_names: list[str] = MISSING  # type: ignore
    robot_curobo_yaml: str | None = None
    robot_urdf: str | None = None
    num_ik_seeds: int = 4
    num_trajopt_seeds: int = 1
    max_plan_attempts: int = 1
    interpolation_buffer_size: int = 128
    plan_batch_size: int = 1
    """Envs solved per cuRobo call. ``1`` plans each env alone; larger values batch
    the envs that enter this segment on the same tick (padded to this size)."""
    chain_with_next: bool = False
    """Plan this segment and the next as one motion, without stopping between them.

    Consecutive cuRobo commands flagged this way, plus the command after the
    last flagged one, form a chain. The first command of the chain builds the
    whole Cartesian path at once - straight lines to each go-to target, then a
    valve arc placed at the planned (not measured) grasp - smooths it in time
    so the speed changes and corners blend, retargets it to joints in one pass,
    and hands each command its slice. Chained segments end when their slice
    runs out instead of waiting for the arm to settle on the target. An env
    whose chain cannot be retargeted plans its segments one by one as usual.
    """
    chain_blend_s: float = 0.4
    """Width of the triangular time kernel that blends a chain's legs [s] (chain leader only)."""
    chain_max_joint_step: float = 0.15
    """Largest joint change between chain waypoints before the chain is rejected [rad] (leader only)."""
    chain_max_pose_error_m: float = 0.01
    """Largest TCP position error of the retargeted chain before it is rejected [m] (leader only)."""
    wrist_flip_joints: tuple[str, str, str] | None = None
    """The arm's (roll, pitch, roll) wrist joints, whose axes meet, enabling a per-env IK branch.

    Flipping them (first roll and last roll by pi, pitch negated) leaves the
    flange pose unchanged; the handler's ``wrist_branch``/``forearm_winding``
    then pick the trajopt goal's branch per env. Batched plans only."""
    wrist_flip_margin: float = 0.05
    """Distance from the joint limits a flipped goal keeps [rad]; closer flips are not applied."""
    max_joint_velocity: float | tuple[float, ...] | None = None
    """Largest joint speed of the plan [rad/s], one for every joint or one per ``robot_joint_names``
    entry; None keeps the planner's timing.

    Faster stretches of the plan are slowed to this speed, on its path (see :func:`_limit_joint_speed`).
    Applies to every plan this command stores, chain slices included."""
    plan_within_joint_velocity: bool = False
    """Also lower cuRobo's joint speed limits to :attr:`max_joint_velocity` (in a copy of the URDF, see
    :func:`~isaaclab_hiveboard.mdp.curobo_robot_cfg.limit_joint_velocity`), so trajectory optimization plans
    and times within it, keeping its acceleration and jerk limits. Only the cuRobo trajectory planners use
    it; chained reaches and arcs are still re-timed. On a 128-trajectory ANYmal M30Thread bank at 1/2 rad/s
    the approach took 0.3 s longer than with re-timing alone, with the same success."""


@configclass
class GripperCommand(BaseCmd):
    class_type = _GripperHandler

    open_gripper: bool = True
    """Whether to open or close the gripper."""
    duration_s: float = 0.2
    """Hold time after changing the gripper command [s]."""


@configclass
class RotateFrameCfg(BaseCmd):
    class_type = _RotateFrameHandler

    frame_name: str = MISSING  # type: ignore
    """Name of the frame used to rotate around."""
    target_frame_name: str = "rotate_frame"
    """Name of the frame used to rotate around."""
    target_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Rotation-center translation in the selected reference frame [m]."""
    target_offset_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Rotation-axis frame orientation (xyzw) relative to the selected reference."""
    axis: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    """Rotation axis expressed in ``target_frame_name`` coordinates."""
    angle_deg: float = -90.0
    """Angle in degrees to rotate around :attr:`axis`."""
    use_valve_angle: bool = True
    """Use the remaining task valve angle when available; False uses angle_deg."""
    angular_velocity: float = 0.3
    """Angular speed of the commanded arc [rad/s]."""
    gripper_open: bool = False
    """Status of the gripper during the command."""
    angle_threshold_deg: float = 5.0
    """Remaining commanded arc at which the command is done [deg]."""
    max_ee_rotation_deg: float = 0.0
    """Clamp |arc| to this many degrees. ``0`` leaves the commanded angle unchanged."""
    axis_position_override_b: tuple[float | None, float | None, float | None] | None = None
    """Optional per-axis rotation-center overrides in the robot base frame."""
    rotate_in_place: bool = False
    """Rotate about the measured TCP, using the reference frame only for the axis."""


@configclass
class CuroboPlannedRotateFrameCfg(RotateFrameCfg):
    """A cuRobo-retargeted arc variant of :class:`RotateFrameCfg`."""

    class_type = _CuroboPlannedRotateFrameHandler
    screw_pitch_m_per_revolution: float = 0.0
    """Signed axial travel per positive revolution [m]; zero gives a circular arc."""
    max_joint_velocity: float | tuple[float, ...] | None = None
    """Largest joint speed of the plan [rad/s], one for every joint or one per ``robot_joint_names``
    entry; None keeps the planner's timing.

    Faster stretches of the plan are slowed to this speed, on its path (see :func:`_limit_joint_speed`).
    A slowed arc turns the valve slower than ``angular_velocity``."""
    robot_joint_names: list[str] = MISSING  # type: ignore
    robot_curobo_yaml: str | None = None
    robot_urdf: str | None = None
    num_ik_seeds: int = 4
    interpolation_buffer_size: int = 128
    plan_batch_size: int = 1
    """Envs retargeted per cuRobo call; see :attr:`CuroboPlannedGoToFrameCfg.plan_batch_size`."""
    max_joint_step: float = 0.15
    """Maximum accepted change of any joint between arc waypoints [rad]."""
    on_infeasible_arc: str = "densify"
    """What to do when the retargeted arc exceeds :attr:`max_joint_step`.

    ``"densify"`` prints the infeasible-arc dump, subdivides the arc so every
    step is within the limit, and flags the segment as an expert fallback
    (see :meth:`SequentialPoseCommand.expert_fallback`), so DAgger can still
    label the off-nominal grasps a learner drifts into. ``"raise"`` stops with
    the dump instead, which is the stricter choice while tuning a sequence.
    """
