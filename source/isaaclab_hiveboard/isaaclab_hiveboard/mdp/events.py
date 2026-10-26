from collections.abc import Sequence

import numpy as np
import torch
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg


def apply_articulation_gravcomp(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gravcomp: float = 1.0,
) -> None:
    """Set Newton ``mjc:gravcomp`` on every body of an articulation.

    Isaac Lab ``modify_rigid_body_properties`` is ``@apply_nested`` and stops at
    the first ``RigidBodyAPI`` prim (Spot's welded torso), so spawn-time
    ``gravcomp=1`` never reaches the arm. This writes ``model.mujoco.gravcomp``
    for all robot links and notifies MJWarp.
    """
    del env_ids
    try:
        from isaaclab_newton.physics import NewtonManager as SimulationManager
        from newton import ModelFlags
    except ImportError as err:
        print(f"[WARN] apply_articulation_gravcomp: Newton not available ({err}).")
        return

    import warp as wp

    asset = env.scene[asset_cfg.name]
    model = SimulationManager.get_model()
    mujoco = getattr(model, "mujoco", None)
    gc = None if mujoco is None else getattr(mujoco, "gravcomp", None)
    if gc is None:
        print("[WARN] apply_articulation_gravcomp: model.mujoco.gravcomp is missing.")
        return

    labels = [str(label) for label in getattr(model, "body_label", [])]
    if not labels:
        print("[WARN] apply_articulation_gravcomp: model.body_label is empty.")
        return
    view_labels = {str(label) for label in getattr(asset.root_view, "link_labels", [])}
    name_set = {str(name) for name in asset.body_names}
    values = gc.numpy().copy()
    matched: list[tuple[int, str, float]] = []
    for index, label in enumerate(labels):
        leaf = label.rsplit("/", 1)[-1]
        if label not in view_labels and leaf not in name_set:
            continue
        matched.append((index, label, float(values[index])))
        values[index] = float(gravcomp)
    if not matched:
        print(
            "[WARN] apply_articulation_gravcomp: no bodies matched "
            f"{asset_cfg.name} names={sorted(name_set)} "
            f"model_bodies={labels[:24]}",
            flush=True,
        )
        return
    wp.copy(gc, wp.array(values.astype(np.float32), dtype=wp.float32, device=gc.device))
    SimulationManager.add_model_change(ModelFlags.BODY_INERTIAL_PROPERTIES)
    solver = getattr(SimulationManager, "_solver", None)
    if solver is not None:
        with wp.ScopedDevice(gc.device):
            solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)
    preview = ", ".join(f"{label}={old:.2f}->{gravcomp:g}" for _, label, old in matched[:16])
    if len(matched) > 16:
        preview += f", … ({len(matched)} bodies)"
    print(f"[INFO] gravcomp={gravcomp:g} on {len(matched)} {asset_cfg.name} bodies [{preview}]", flush=True)


def set_contact_stiffness(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | torch.Tensor,
    shape_regex: str,
    ke: float,
    kd: float,
    solimp: tuple[float, float, float] | None = None,
) -> None:
    """Stiffen Newton/MJWarp contacts on shapes whose label matches ``shape_regex``.

    MJWarp gets each geom's solref from ``convert_solref(ke, kd)``:
    timeconst = 2 / kd, dampratio = kd / 2 / sqrt(ke). solref is mass-normalized,
    so on light bodies (the 5 g small-valve stem) the default (0.02, 1) yields
    only a few hundred N/m and a small squeeze sinks millimetres in. Keep
    timeconst >= 2 x the physics substep, or MuJoCo clamps it.

    ``solimp`` sets (dmin, dmax, width [m]). Write these to the Newton model, not
    to ``mjw_model.geom_solref``: the solver re-syncs geoms from the model. Run
    after any material randomization, which re-syncs shapes too.
    """
    del env_ids
    import re

    try:
        from isaaclab_newton.physics import NewtonManager as SimulationManager
        from newton import ModelFlags
    except ImportError as err:
        print(f"[WARN] set_contact_stiffness: Newton not available ({err}).")
        return

    model = SimulationManager.get_model()
    pattern = re.compile(shape_regex)
    sel = [i for i, label in enumerate(model.shape_label) if pattern.search(str(label))]
    if not sel:
        print(f"[WARN] set_contact_stiffness: no shape matched {shape_regex!r}.", flush=True)
        return
    ke_np = model.shape_material_ke.numpy()
    kd_np = model.shape_material_kd.numpy()
    ke_np[sel] = ke
    kd_np[sel] = kd
    model.shape_material_ke.assign(ke_np)
    model.shape_material_kd.assign(kd_np)
    if solimp is not None:
        attr = getattr(getattr(model, "mujoco", None), "geom_solimp", None)
        if attr is None:
            print("[WARN] set_contact_stiffness: model.mujoco.geom_solimp is missing; solimp unchanged.")
        else:
            values = attr.numpy()
            values[sel, :3] = solimp
            attr.assign(values)
    SimulationManager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
    solver = getattr(SimulationManager, "_solver", None)
    if solver is not None:
        solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)
    print(
        f"[INFO] contact ke={ke:g} kd={kd:g} (solref {2.0 / kd:.4f} s) solimp={solimp} "
        f"on {len(sel)} shapes matching {shape_regex!r}",
        flush=True,
    )


def set_joint_limit_damping(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | torch.Tensor,
    asset_cfg: SceneEntityCfg,
    kd: float,
) -> None:
    """Set the Newton joint-limit damping ``kd`` [N·m·s/rad] of ``asset_cfg``'s joints.

    MJWarp scales Newton's force-space limit ``(ke, kd)`` by the joint's inverse
    inertia, so on a light joint the default ``kd`` 10 gives a damping ratio
    well below one and the end stop throws the joint back. Isaac Lab has no
    property for it; this writes ``model.joint_limit_kd``, which the solver
    keeps when later events change the joint's armature or friction.
    """
    del env_ids
    import re

    try:
        from isaaclab_newton.physics import NewtonManager as SimulationManager
        from newton import ModelFlags
    except ImportError as err:
        print(f"[WARN] set_joint_limit_damping: Newton not available ({err}).")
        return

    asset = env.scene[asset_cfg.name]
    names = asset.find_joints(asset_cfg.joint_names)[1]
    model = SimulationManager.get_model()
    pattern = re.compile(rf"{asset.cfg.prim_path}/.*/({'|'.join(map(re.escape, names))})$")
    sel = [i for i, label in enumerate(model.joint_label) if pattern.match(str(label))]
    if not sel:
        print(f"[WARN] set_joint_limit_damping: no joint matched {pattern.pattern!r}.", flush=True)
        return
    dofs = model.joint_qd_start.numpy()[sel]
    values = model.joint_limit_kd.numpy()
    values[dofs] = kd
    model.joint_limit_kd.assign(values)
    SimulationManager.add_model_change(ModelFlags.JOINT_DOF_PROPERTIES)
    solver = getattr(SimulationManager, "_solver", None)
    if solver is not None:
        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
    print(f"[INFO] joint limit kd={kd:g} on {len(sel)} {asset_cfg.name} joints {names}", flush=True)
