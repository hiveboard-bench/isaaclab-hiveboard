# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regenerate kitless Newton USD assets via urdf-usd-converter (UUC).

The Newton tasks load preconverted USD (gitignored ``**/*.usd*``). This script
is the record of how those files are produced.

Spot: UUC conversion of ``spot_with_arm.urdf`` into ``spot/usd/uuc/``.
Valve: UUC conversion into ``ball_valve/usd/uuc/`` plus a CoACD overlay
(``Ball_Valve_uuc_newton.usda``). A single UUC convexHull wraps the lever in
~3x phantom volume; Newton hardcodes CoACD threshold=0.5, so parts are baked
here (``coacd`` + ``trimesh``, generation-time only).
Circuit breaker: UUC conversion of the HiveBoard
``Circuit_Breaker_Assembly.urdf`` plus the same CoACD overlay.
High-torque / small gate valves: UUC conversion of the HiveBoard URDFs
plus the same CoACD overlay.
Lamp: UUC conversion of the HiveBoard lamp URDF, including its primitive
bulb colliders. The environment implements the revolute/prismatic coupling.
Button: UUC conversion of the HiveBoard hidden-button URDF (lid hinge and
button slide are already articulated upstream).
Drawer / key: UUC conversion of the re-authored URDFs in ``hiveboard/drawer``
and ``hiveboard/key`` (the upstream ones have no joints). The drawer is split
into a kinematic housing and a free box; cuts and shafts stay URDF primitives,
so there is no CoACD overlay: a hull of the slotted plate would fill the exit.
The key is likewise split: the lock (housing + turning plug) is a fixed
articulation, the key a free body that the robot carries welded to its hand.
Threads / peg / shock absorber: UUC conversion of the re-authored URDFs in
``hiveboard/{m8_thread,m30_thread,peg_insertion,shock_absorber}``, plus a
CoACD overlay for the shock-absorber spring.

UUC needs a Python that can ``import urdf_usd_converter`` (isolated venv;
the project venv does not ship it). Overlay rewrite is kitless.

Usage::

    uv run python scripts/generate_newton_usd.py --verify-only
    uv run python scripts/generate_newton_usd.py --skip-conversion
    uv run python scripts/generate_newton_usd.py --uuc-python /path/to/uuc-venv/bin/python
    uv run python scripts/generate_newton_usd.py --strip-obj
    # or: just strip-obj-unused-verts
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXT_ASSETS = REPO_ROOT / "source/isaaclab_hiveboard/isaaclab_hiveboard/assets"

MESH_COLLISION_APIS = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]

SPOT_URDF = EXT_ASSETS / "spot/spot_with_arm.urdf"
SPOT_UUC_DIR = EXT_ASSETS / "spot/usd/uuc"
VALVE_URDF = EXT_ASSETS / "hiveboard/ball_valve/Ball_Valve.urdf"
VALVE_USD_DIR = EXT_ASSETS / "hiveboard/ball_valve/usd"
VALVE_UUC_DIR = VALVE_USD_DIR / "uuc"
VALVE_OVERLAY = VALVE_USD_DIR / "Ball_Valve_uuc_newton.usda"

# (link path under /Ball_Valve, collision mesh name, coacd threshold)
VALVE_TARGETS: tuple[tuple[str, str, float], ...] = (
    ("Geometry/valvula_esfera", "Mesh_1", 0.1),
    ("Geometry/valvula_esfera/alavanca_pivot", "Mesh_1_1", 0.05),
)

BREAKER_URDF = (
    REPO_ROOT / "dependencies/HiveBoard/Simulation/Circuit Breaker/Circuit_Breaker_Assembly.urdf"
)
BREAKER_USD_DIR = EXT_ASSETS / "hiveboard/circuit_breaker/usd"
BREAKER_UUC_DIR = BREAKER_USD_DIR / "uuc"
BREAKER_OVERLAY = BREAKER_USD_DIR / "Circuit_Breaker_uuc_newton.usda"
BREAKER_TARGETS: tuple[tuple[str, str, float], ...] = (
    ("Geometry/World", "tn__corpo1_dC_Corpo1_Mesh_1", 0.1),
    ("Geometry/World/lever_pivot", "tn__Alavanca1_zH_Corpo1_Mesh_1", 0.05),
)

HT_URDF = (
    REPO_ROOT / "dependencies/HiveBoard/Simulation/Valves/Gate Valve/High Torque Valve/High_Torque_Valve.urdf"
)
HT_USD_DIR = EXT_ASSETS / "hiveboard/high_torque_valve/usd"
HT_UUC_DIR = HT_USD_DIR / "uuc"
HT_OVERLAY = HT_USD_DIR / "High_Torque_Valve_uuc_newton.usda"
HT_TARGETS: tuple[tuple[str, str, float], ...] = (
    ("Geometry/World/base", "Corpo1__2__1", 0.1),
    ("Geometry/World/nut", "Corpo1__1__1", 0.05),
)

SM_URDF = REPO_ROOT / "dependencies/HiveBoard/Simulation/Valves/Gate Valve/Small Valve/Small_Valve.urdf"
SM_USD_DIR = EXT_ASSETS / "hiveboard/small_valve/usd"
SM_UUC_DIR = SM_USD_DIR / "uuc"
SM_OVERLAY = SM_USD_DIR / "Small_Valve_uuc_newton.usda"
SM_TARGETS: tuple[tuple[str, str, float], ...] = (
    ("Geometry/valvula_gaveta", "Mesh_1_1", 0.2),
    ("Geometry/valvula_gaveta/pivot_registro", "Cylinder_1", 0.1),
    ("Geometry/valvula_gaveta/pivot_registro/eixo_trans", "Mesh_22_1", 0.2),
)

DEFAULT_UUC_PYTHON = str(REPO_ROOT / "dependencies/urdf-usd-converter/.venv/bin/python")

LAMP_URDF = REPO_ROOT / "dependencies/HiveBoard/Simulation/Lamp/Lamp_Assembly.urdf"
LAMP_UUC_DIR = EXT_ASSETS / "hiveboard/lamp/usd/uuc"

# Hidden push button: the upstream URDF is already articulated (lid hinge +
# button slide) and its per-link convex hulls do not cover the button face,
# so a plain UUC conversion is enough.
BUTTON_URDF = REPO_ROOT / "dependencies/HiveBoard/Simulation/Button/Button_Assembly.urdf"
BUTTON_UUC_DIR = EXT_ASSETS / "hiveboard/button/usd/uuc"

# Drawer and key: the upstream URDFs are a single rigid link, so these convert
# the re-authored URDFs committed next to their USD output instead.
# The drawer is two rigid bodies (housing, free box). Cut collision is boxes.
DRAWER_URDF = EXT_ASSETS / "hiveboard/drawer/Drawer_Assembly.urdf"
DRAWER_USD_DIR = EXT_ASSETS / "hiveboard/drawer/usd"
DRAWER_HOUSING_UUC_DIR = DRAWER_USD_DIR / "uuc_housing"
DRAWER_BOX_UUC_DIR = DRAWER_USD_DIR / "uuc_box"
DRAWER_HOUSING_OVERLAY = DRAWER_USD_DIR / "Drawer_Housing_uuc_newton.usda"
DRAWER_BOX_OVERLAY = DRAWER_USD_DIR / "Drawer_Box_uuc_newton.usda"
LOCK_URDF = EXT_ASSETS / "hiveboard/key/Lock_Assembly.urdf"
LOCK_UUC_DIR = EXT_ASSETS / "hiveboard/key/usd/uuc_lock"
KEY_URDF = EXT_ASSETS / "hiveboard/key/Key.urdf"
KEY_UUC_DIR = EXT_ASSETS / "hiveboard/key/usd/uuc_key"

# Threads, peg and shock absorber: re-authored URDFs (upstream welds the peg
# and pin, and collides with base hulls that cone over the thread). The
# spring's single hull would swallow the pin head, so it gets CoACD parts.
M8_URDF = EXT_ASSETS / "hiveboard/m8_thread/M8_Assy.urdf"
M8_UUC_DIR = EXT_ASSETS / "hiveboard/m8_thread/usd/uuc"
M30_URDF = EXT_ASSETS / "hiveboard/m30_thread/M30.urdf"
M30_UUC_DIR = EXT_ASSETS / "hiveboard/m30_thread/usd/uuc"
PEG_URDF = EXT_ASSETS / "hiveboard/peg_insertion/Peg_Insertion.urdf"
PEG_UUC_DIR = EXT_ASSETS / "hiveboard/peg_insertion/usd/uuc"
SHOCK_URDF = EXT_ASSETS / "hiveboard/shock_absorber/Shock_Absorber_Assembly.urdf"
SHOCK_USD_DIR = EXT_ASSETS / "hiveboard/shock_absorber/usd"
SHOCK_UUC_DIR = SHOCK_USD_DIR / "uuc"
SHOCK_OVERLAY = SHOCK_USD_DIR / "Shock_Absorber_Assembly_uuc_newton.usda"
# 0.1 gives ~40 parts; 0.05 (74 parts) segfaults the ANYmal scene's Newton import.
SHOCK_TARGETS: tuple[tuple[str, str, float], ...] = (("Geometry/board", "Corpo3", 0.1),)

# (label, UUC output, default prim, [(joint name, schema)]) checked by verify().
ARTICULATED_UUC_ASSETS = (
    (
        "button",
        BUTTON_UUC_DIR / "Button_Assembly.usda",
        "Button_Assembly",
        (("RevoluteJoint", "PhysicsRevoluteJoint"), ("PrismaticJoint", "PhysicsPrismaticJoint")),
    ),
    ("lock", LOCK_UUC_DIR / "Lock_Assembly.usda", "Lock_Assembly", (("RevoluteJoint", "PhysicsRevoluteJoint"),)),
    *(
        (label, usda, prim, (("RevoluteJoint", "PhysicsRevoluteJoint"), ("PrismaticJoint", "PhysicsPrismaticJoint")))
        for label, usda, prim in (
            ("m8_thread", M8_UUC_DIR / "M8_Assy.usda", "M8_Assy"),
            ("m30_thread", M30_UUC_DIR / "M30.usda", "M30"),
            ("peg_insertion", PEG_UUC_DIR / "Peg_Insertion.usda", "Peg_Insertion"),
        )
    ),
    ("shock_absorber", SHOCK_OVERLAY, "Shock_Absorber_Assembly", (("PrismaticJoint", "PhysicsPrismaticJoint"),)),
)

SPOT_MESH_DIR = EXT_ASSETS / "spot/meshes"


def urdf_link_collision_meshes(urdf: Path) -> dict[str, list[str]]:
    """Map URDF link name -> authored collision mesh filenames."""
    import xml.etree.ElementTree as ET

    out: dict[str, list[str]] = {}
    for link in ET.parse(str(urdf)).getroot().iter("link"):
        files = [
            mesh.get("filename")
            for col in link.iter("collision")
            for mesh in col.findall("geometry/mesh")
            if mesh.get("filename")
        ]
        if files:
            out[link.get("name")] = files
    return out


def resolve_mesh_file(filename: str, urdf: Path) -> Path:
    """Resolve a URDF mesh filename (relative / package:// / absolute)."""
    if "://" in filename:
        filename = filename.split("://", 1)[1].split("/", 1)[1]
    path = Path(filename)
    if path.is_absolute():
        return path
    return urdf.parent / path


def decompose_mesh(mesh_file: Path, threshold: float) -> list[tuple[object, object]]:
    """Split a mesh into convex parts with CoACD. Returns (verts, faces) pairs."""
    try:
        import coacd  # noqa: PLC0415
        import trimesh  # noqa: PLC0415
    except ImportError as err:
        raise RuntimeError(
            "Baked convex decomposition needs the `coacd` and `trimesh` packages "
            "(generation-time only, e.g. `uv pip install coacd trimesh`)."
        ) from err
    import numpy as np  # noqa: PLC0415

    loaded = trimesh.load(str(mesh_file), process=True, force="mesh")
    verts = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int32)
    return coacd.run_coacd(coacd.Mesh(verts, faces), threshold=threshold)


def _fmt_floats(values: object) -> str:
    import numpy as np  # noqa: PLC0415

    flat = np.asarray(values, dtype=np.float64).ravel()
    return ", ".join(f"{v:.9g}" for v in flat)


def _fmt_points(verts: object) -> str:
    import numpy as np  # noqa: PLC0415

    return ", ".join(f"({_fmt_floats(v)})" for v in np.asarray(verts, dtype=np.float64))


def _fmt_ints(values: object) -> str:
    import numpy as np  # noqa: PLC0415

    return ", ".join(str(int(v)) for v in np.asarray(values).ravel())


def _parse_obj_index(token: str, count: int) -> int:
    index = int(token)
    if index < 0:
        index = count + index + 1
    return index


def strip_unused_obj_topology(path: Path) -> bool:
    """Drop face-unreferenced v/vt/vn and OBJ polylines (`l`).

    usdex ``definePolyMesh`` rejects meshes with points no face references.
    Some Spot OBJs keep CAD leftover verts only used by ``l`` polylines.
    Returns True if the file changed.
    """
    raw = path.read_text()
    newline = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.splitlines()
    positions, uvs, normals = [], [], []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("v "):
            positions.append(line)
        elif stripped.startswith("vt "):
            uvs.append(line)
        elif stripped.startswith("vn "):
            normals.append(line)
    counts = [len(positions), len(uvs), len(normals)]
    used: list[set[int]] = [set(), set(), set()]
    for line in lines:
        stripped = line.lstrip()
        if not stripped.startswith("f "):
            continue
        for token in stripped.split()[1:]:
            parts = token.split("/")
            for kind, part in enumerate(parts):
                if part:
                    used[kind].add(_parse_obj_index(part, counts[kind]))
    maps: list[dict[int, int]] = []
    kept: list[list[str]] = []
    for kind, src in enumerate((positions, uvs, normals)):
        mapping: dict[int, int] = {}
        keep: list[str] = []
        next_index = 1
        for old, record in enumerate(src, start=1):
            if old in used[kind]:
                mapping[old] = next_index
                keep.append(record)
                next_index += 1
        maps.append(mapping)
        kept.append(keep)

    def remap_corner(corner: str) -> str:
        parts = corner.split("/")
        out = []
        for kind, part in enumerate(parts):
            if part == "":
                out.append("")
            else:
                out.append(str(maps[kind][_parse_obj_index(part, counts[kind])]))
        return "/".join(out)

    out: list[str] = []
    emitted = [False, False, False]
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("l "):
            continue
        if stripped.startswith("v "):
            if not emitted[0]:
                out.extend(kept[0])
                emitted[0] = True
            continue
        if stripped.startswith("vt "):
            if not emitted[1]:
                out.extend(kept[1])
                emitted[1] = True
            continue
        if stripped.startswith("vn "):
            if not emitted[2]:
                out.extend(kept[2])
                emitted[2] = True
            continue
        if stripped.startswith("f "):
            prefix = line[: len(line) - len(line.lstrip())]
            corners = stripped.split()[1:]
            out.append(f"{prefix}f {' '.join(remap_corner(c) for c in corners)}")
            continue
        out.append(line)
    text = newline.join(out) + newline
    if text == raw:
        return False
    path.write_text(text)
    return True


def spot_obj_files() -> list[Path]:
    """All committed Spot OBJ meshes (visual + collision)."""
    return sorted(SPOT_MESH_DIR.rglob("*.obj"))


def strip_spot_obj_topology() -> int:
    """Strip unused vertices from every Spot OBJ. Returns how many files changed."""
    changed = 0
    for obj in spot_obj_files():
        if strip_unused_obj_topology(obj):
            print(f"[OBJ] stripped unused topology: {obj.relative_to(EXT_ASSETS)}")
            changed += 1
        else:
            print(f"[OBJ] already clean: {obj.relative_to(EXT_ASSETS)}")
    return changed


def run_uuc_conversion(urdf: Path, out_dir: Path, uuc_python: str) -> None:
    """Run urdf-usd-converter into ``out_dir`` (wipes it first)."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    cmd = [uuc_python, "-m", "urdf_usd_converter", str(urdf), str(out_dir)]
    print(f"[UUC] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _op_lines(prim) -> list[str]:
    """Render a prim's ordered xformOps as USDA attribute lines."""
    from pxr import UsdGeom  # noqa: PLC0415

    lines = []
    order = []
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        name = op.GetOpName()
        order.append(name)
        attr = op.GetAttr()
        value = op.Get()
        type_name = str(attr.GetTypeName())
        if "quat" in type_name:
            value = (value.GetReal(), *value.GetImaginary())
        lines.append(f"                {type_name} {name} = ({_fmt_floats(value)})")
    lines.append(
        f"                uniform token[] xformOpOrder = [{', '.join(f'\"{n}\"' for n in order)}]"
    )
    return lines


def render_coacd_overlay(
    *,
    urdf: Path,
    uuc_dir: Path,
    usda_name: str,
    default_prim: str,
    targets: tuple[tuple[str, str, float], ...],
) -> str:
    """Disable UUC single hulls and add baked CoACD parts in the mesh xform."""
    from pxr import Usd  # noqa: PLC0415

    base = uuc_dir / usda_name
    stage = Usd.Stage.Open(str(base))
    if not stage:
        raise RuntimeError(f"pxr could not open UUC base: {base}")
    link_meshes = urdf_link_collision_meshes(urdf)
    api_list = ", ".join(f'"{api}"' for api in MESH_COLLISION_APIS)
    lines = [
        "#usda 1.0",
        "(",
        f'    defaultPrim = "{default_prim}"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        "# Overlay over the urdf-usd-converter base (usd/uuc/). UUC already places",
        "# collision schemas on the meshes; this swaps the single-hull colliders",
        "# for baked CoACD parts (a lone convexHull wraps thin levers in ~3x phantom",
        "# volume).",
        f'def Xform "{default_prim}" (',
        f"    prepend references = @./uuc/{usda_name}@</{default_prim}>",
        ")",
        "{",
    ]
    tree: dict = {}
    contents: dict[tuple[str, ...], list[str]] = {}
    for link_path, mesh_name, threshold in targets:
        link = link_path.split("/")[-1]
        mesh_path = f"/{default_prim}/{link_path}/{mesh_name}"
        prim = stage.GetPrimAtPath(mesh_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"UUC base target missing: {mesh_path}")
        mesh_files = link_meshes.get(link, [])
        if len(mesh_files) != 1:
            raise RuntimeError(f"Expected 1 URDF collision mesh for link '{link}', got {mesh_files}")
        parts = decompose_mesh(resolve_mesh_file(mesh_files[0], urdf), threshold)
        print(f"[DECOMP] {link}/{mesh_name}: {len(parts)} parts (threshold={threshold})")
        elems = tuple(link_path.split("/"))
        node = tree
        for elem in elems:
            node = node.setdefault(elem, {})
        body: list[str] = [
            f'over "{mesh_name}"',
            "{",
            "    bool physics:collisionEnabled = false",
            "}",
            f'def Xform "decomp_{link}"',
            "{",
            *_op_lines(prim),
        ]
        for part_index, (verts, faces) in enumerate(parts):
            body += [
                f'    def Mesh "part_{part_index}" (',
                f"        prepend apiSchemas = [{api_list}]",
                "    )",
                "    {",
                f"        point3f[] points = [{_fmt_points(verts)}]",
                f"        int[] faceVertexCounts = [{_fmt_ints([3] * len(faces))}]",
                f"        int[] faceVertexIndices = [{_fmt_ints(faces)}]",
                '        uniform token physics:approximation = "convexHull"',
                "    }",
            ]
        body.append("}")
        contents[elems] = body

    def emit(node: dict, path: tuple[str, ...], indent: str) -> None:
        for elem, children in node.items():
            lines.append(f'{indent}over "{elem}"')
            lines.append(f"{indent}{{")
            for content_line in contents.get((*path, elem), []):
                lines.append(f"{indent}    {content_line}")
            emit(children, (*path, elem), indent + "    ")
            lines.append(f"{indent}}}")

    emit(tree, (), "    ")
    lines.append("}")
    return "\n".join(lines) + "\n"


def render_valve_overlay() -> str:
    return render_coacd_overlay(
        urdf=VALVE_URDF,
        uuc_dir=VALVE_UUC_DIR,
        usda_name="Ball_Valve.usda",
        default_prim="Ball_Valve",
        targets=VALVE_TARGETS,
    )


def render_breaker_overlay() -> str:
    return render_coacd_overlay(
        urdf=BREAKER_URDF,
        uuc_dir=BREAKER_UUC_DIR,
        usda_name="Circuit_Breaker_Assembly.usda",
        default_prim="Circuit_Breaker_Assembly",
        targets=BREAKER_TARGETS,
    )


def render_ht_overlay() -> str:
    return render_coacd_overlay(
        urdf=HT_URDF,
        uuc_dir=HT_UUC_DIR,
        usda_name="High_Torque_Valve.usda",
        default_prim="High_Torque_Valve",
        targets=HT_TARGETS,
    )


DRAWER_SOURCE_URDF = REPO_ROOT / "dependencies/HiveBoard/Simulation/Drawer/Drawer_Assembly.urdf"
DRAWER_GUIDES = {
    "left_Guide_Cube": "cut_left_ceiling",
    "left_Guide_Cube_01": "cut_left_upper",
    "left_Guide_Cube_02": "cut_left_floor",
    "left_Guide_Cube_03": "cut_left_lip",
    "left_Guide_Cube_04": "cut_left_rear",
    "Cylinder": "cut_left_round",
    "right_Guide_Cube": "cut_right_ceiling",
    "right_Guide_Cube_01": "cut_right_upper",
    "right_Guide_Cube_02": "cut_right_floor",
    "right_Guide_Cube_03": "cut_right_lip",
    "right_Guide_Cube_04": "cut_right_rear",
    "right_Guide_Cylinder_01": "cut_right_round",
    "tn__Gaveta1_aE_Corpo1_Cylinder_01": "shaft_left",
    "Cylinder_02": "shaft_right",
}


def _drawer_source_primitives() -> list:
    import xml.etree.ElementTree as ET

    shapes = []
    for item in ET.parse(DRAWER_SOURCE_URDF).getroot().iter("collision"):
        path = DRAWER_SOURCE_URDF.parent / item.find("geometry/mesh").attrib["filename"]
        name = DRAWER_GUIDES.get(path.stem)
        if name is None:
            continue
        points = [
            tuple(float(value) * 0.001 for value in line.split()[1:4])
            for line in path.read_text().splitlines()
            if line.startswith("v ")
        ]
        low = tuple(min(point[i] for point in points) for i in range(3))
        high = tuple(max(point[i] for point in points) for i in range(3))
        assert all(abs(a + b) / 2 <= 1e-7 for a, b in zip(low, high)), path
        col = ET.Element("collision", name=name)
        ET.SubElement(col, "origin", dict(item.find("origin").attrib))
        geom = ET.SubElement(col, "geometry")
        if "Cylinder" in path.stem:
            radius = f"{(high[0] - low[0]) / 2:.10f}"
            ET.SubElement(geom, "cylinder", radius=radius, length=f"{high[2] - low[2]:.10f}")
        else:
            size = " ".join(f"{b - a:.10f}" for a, b in zip(low, high))
            ET.SubElement(geom, "box", size=size)
        shapes.append(col)
    assert len(shapes) == len(DRAWER_GUIDES)
    return shapes


def sync_drawer_collision() -> None:
    import xml.etree.ElementTree as ET

    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(DRAWER_URDF, parser=parser)
    links = {link.attrib["name"]: link for link in tree.getroot().findall("link")}
    for link in links.values():
        for col in list(link.findall("collision")):
            if col.attrib.get("name", "").startswith(("cut_", "shaft_")):
                link.remove(col)
    for shape in _drawer_source_primitives():
        links["base" if shape.attrib["name"].startswith("cut_") else "drawer"].append(shape)
    drawer = links["drawer"]
    left = next(s for s in drawer.findall("collision") if s.attrib["name"] == "shaft_left")
    x, _, z = left.find("origin").attrib["xyz"].split()
    drawer.find("inertial/origin").set("xyz", f"{x} 0 {z}")
    ET.indent(tree, space="  ")
    tree.write(DRAWER_URDF, encoding="utf-8", xml_declaration=True)


def _split_drawer_urdfs(directory: Path) -> tuple[Path, Path]:
    """Write one-link URDFs so each Newton body is free of the other.

    The authored assembly keeps both links and a floating joint. A single USD
    nests the box under the housing articulation, and a rigid-object spawn
    rejects that. Mesh paths stay valid because these files sit next to the
    source URDF. Callers delete them after conversion.
    """
    import copy
    import xml.etree.ElementTree as ET

    root = ET.parse(str(DRAWER_URDF)).getroot()

    def write(name: str, link: str) -> Path:
        robot = ET.Element("robot", {"name": name})
        for child in root:
            if child.tag == "link" and child.attrib.get("name") == link:
                robot.append(copy.deepcopy(child))
        path = directory / f"{name}.urdf"
        tree = ET.ElementTree(robot)
        ET.indent(tree, space="  ")
        tree.write(path, encoding="utf-8", xml_declaration=True)
        return path

    return write("Drawer_Housing", "base"), write("Drawer_Box", "drawer")


def _anchor_housing(payload: Path) -> None:
    """Mark the welded housing as an articulation root so the case stays fixed."""
    text = payload.read_text(encoding="utf-8")
    needle = 'prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "NewtonMassAPI"]'
    replacement = (
        'prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsArticulationRootAPI", '
        '"NewtonArticulationRootAPI", "PhysicsMassAPI", "NewtonMassAPI"]'
    )
    if "PhysicsArticulationRootAPI" not in text:
        if needle not in text:
            raise RuntimeError(f"{payload} has no base rigid-body schema to anchor")
        text = text.replace(needle, replacement, 1)
        payload.write_text(text, encoding="utf-8")
    if 'PhysicsFixedJoint "root_joint"' not in text:
        raise RuntimeError(f"{payload} is missing the root weld")


def _disable_visual_mesh_collision(geometry_usda: Path) -> None:
    """Visual meshes stay visible. Only the named primitive cuts and shafts collide."""
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    stage = Usd.Stage.Open(str(geometry_usda))
    if stage is None:
        raise RuntimeError(f"could not open {geometry_usda}")
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        collision = UsdPhysics.CollisionAPI.Apply(prim)
        collision.CreateCollisionEnabledAttr(False)
    stage.GetRootLayer().Save()


def _strip_world_fixed_joint(payload: Path) -> None:
    """Drop UUC's root weld so the link is a free rigid body.

    The converter welds every non-ghost root link to the default prim. That
    weld is a joint, and a fixed-base rigid object does not initialize. The
    housing is held still by the kinematic flag instead.
    """
    from pxr import Usd  # noqa: PLC0415

    stage = Usd.Stage.Open(str(payload))
    if stage is None:
        raise RuntimeError(f"could not open {payload}")
    paths = [
        prim.GetPath()
        for prim in stage.Traverse()
        if prim.GetName() == "root_joint" and prim.GetTypeName() == "PhysicsFixedJoint"
    ]
    for path in paths:
        stage.RemovePrim(path)
    stage.GetRootLayer().Save()


def render_rigid_overlay(
    default_prim: str, reference: str, anchor_body: str | None = None, soft_grasp_body: str | None = None
) -> str:
    """Reference one UUC layer. Do not hull the drawer's slotted plate.

    anchor_body: weld a collision-free child body to this link. Newton's USD
    importer leaves a lone body on a world FixedJoint as an orphan joint with
    no articulation, so Articulation() on it finds nothing. A second body makes
    UsdPhysics report a real fixed-base articulation.
    """
    head = [
        "#usda 1.0",
        "(",
        f'    defaultPrim = "{default_prim}"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        "# urdf-usd-converter output. Shafts and side cuts are URDF primitives;",
        "# do not convex-hull the slotted plate over them.",
    ]
    if anchor_body is None:
        body = ["{", "}"]
        if soft_grasp_body is not None:
            # The rubber pads grip a surface patch. Enable torsional friction
            # on the tab so that the drawer can transmit a lift without
            # freely pivoting about the two point-contact normals. Newton's
            # default torsional friction length is 5 mm, matching this tab.
            body = [
                "{",
                '    over "Geometry"',
                "    {",
                f'        over "{soft_grasp_body}"',
                "        {",
                '            over "face"',
                "            {",
                "                int mjc:condim = 4",
                "            }",
                "        }",
                "    }",
                "}",
            ]
    else:
        head += [
            "#",
            "# anchor: a near-massless, collision-free child welded to the base so",
            "# Newton builds a fixed-base articulation (see render_rigid_overlay).",
        ]
        body = [
            "{",
            '    over "Geometry"',
            "    {",
            '        def Xform "anchor" (',
            '            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]',
            "        )",
            "        {",
            "            float3 physics:diagonalInertia = (1e-7, 1e-7, 1e-7)",
            "            float physics:mass = 0.001",
            "        }",
            "    }",
            "",
            '    over "Physics"',
            "    {",
            '        def PhysicsFixedJoint "anchor_joint"',
            "        {",
            f"            rel physics:body0 = </{default_prim}/Geometry/{anchor_body}>",
            f"            rel physics:body1 = </{default_prim}/Geometry/anchor>",
            "        }",
            "    }",
            "}",
        ]
    return "\n".join(
        (
            *head,
            f'def Xform "{default_prim}" (',
            f"    prepend references = @{reference}@</{default_prim}>",
            ")",
            *body,
            "",
        )
    )


def render_sm_overlay() -> str:
    return render_coacd_overlay(
        urdf=SM_URDF,
        uuc_dir=SM_UUC_DIR,
        usda_name="Small_Valve.usda",
        default_prim="Small_Valve",
        targets=SM_TARGETS,
    )


def apply_drawer_contact_materials(path: Path, body_name: str) -> None:
    """Keep sliding guide friction separate from the handle's pad friction."""
    from pxr import Usd, UsdPhysics, UsdShade

    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim().GetPath()
    guide = UsdShade.Material.Define(stage, root.AppendPath("Materials/Guide"))
    material = UsdPhysics.MaterialAPI.Apply(guide.GetPrim())
    material.CreateStaticFrictionAttr(0.1)
    material.CreateDynamicFrictionAttr(0.08)
    material.CreateRestitutionAttr(0.0)
    pad = UsdShade.Material.Define(stage, root.AppendPath("Materials/Handle"))
    material = UsdPhysics.MaterialAPI.Apply(pad.GetPrim())
    material.CreateStaticFrictionAttr(0.8)
    material.CreateDynamicFrictionAttr(0.8)
    material.CreateRestitutionAttr(0.0)
    body = stage.GetPrimAtPath(root.AppendPath(f"Geometry/{body_name}"))
    for prim in body.GetChildren():
        if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                pad if prim.GetName() == "face" else guide, materialPurpose="physics"
            )
    stage.GetRootLayer().Save()


def render_shock_overlay() -> str:
    return render_coacd_overlay(
        urdf=SHOCK_URDF,
        uuc_dir=SHOCK_UUC_DIR,
        usda_name="Shock_Absorber_Assembly.usda",
        default_prim="Shock_Absorber_Assembly",
        targets=SHOCK_TARGETS,
    )


def verify() -> list[str]:
    """Check generated UUC USD resolves. Empty list = ok."""
    from pxr import Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

    problems = []
    spot = SPOT_UUC_DIR / "spot_with_arm.usda"
    if not spot.exists():
        problems.append(f"missing Spot UUC: {spot}")
    else:
        stage = Usd.Stage.Open(str(spot))
        if not stage:
            problems.append(f"pxr could not open {spot}")
        else:
            for name, path in (
                ("sh0", "/spot_with_arm/Geometry/body/arm_link_sh0/arm_link_sh0"),
                (
                    "wr0",
                    "/spot_with_arm/Geometry/body/arm_link_sh0/arm_link_sh1/"
                    "arm_link_el0/arm_link_el1/arm_link_wr0/arm_link_wr0",
                ),
            ):
                prim = stage.GetPrimAtPath(path)
                if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
                    problems.append(f"Spot UUC missing {name} visual mesh: {path}")
    if not VALVE_OVERLAY.exists():
        problems.append(f"missing valve overlay: {VALVE_OVERLAY}")
    else:
        stage = Usd.Stage.Open(str(VALVE_OVERLAY))
        if not stage:
            problems.append(f"pxr could not open {VALVE_OVERLAY}")
        else:
            lever = stage.GetPrimAtPath(
                "/Ball_Valve/Geometry/valvula_esfera/alavanca_pivot/decomp_alavanca_pivot/part_0"
            )
            if not lever or not lever.IsValid():
                problems.append("valve overlay missing CoACD lever part_0")
    if not BREAKER_OVERLAY.exists():
        problems.append(f"missing circuit breaker overlay: {BREAKER_OVERLAY}")
    else:
        stage = Usd.Stage.Open(str(BREAKER_OVERLAY))
        if not stage:
            problems.append(f"pxr could not open {BREAKER_OVERLAY}")
        else:
            lever = stage.GetPrimAtPath(
                "/Circuit_Breaker_Assembly/Geometry/World/lever_pivot/decomp_lever_pivot/part_0"
            )
            if not lever or not lever.IsValid():
                problems.append("circuit breaker overlay missing CoACD lever part_0")
    if not HT_OVERLAY.exists():
        problems.append(f"missing high-torque valve overlay: {HT_OVERLAY}")
    else:
        stage = Usd.Stage.Open(str(HT_OVERLAY))
        if not stage:
            problems.append(f"pxr could not open {HT_OVERLAY}")
        else:
            nut = stage.GetPrimAtPath("/High_Torque_Valve/Geometry/World/nut/decomp_nut/part_0")
            if not nut or not nut.IsValid():
                problems.append("high-torque overlay missing CoACD nut part_0")
    if not SM_OVERLAY.exists():
        problems.append(f"missing small valve overlay: {SM_OVERLAY}")
    else:
        stage = Usd.Stage.Open(str(SM_OVERLAY))
        if not stage:
            problems.append(f"pxr could not open {SM_OVERLAY}")
        else:
            stem = stage.GetPrimAtPath(
                "/Small_Valve/Geometry/valvula_gaveta/pivot_registro/eixo_trans/decomp_eixo_trans/part_0"
            )
            if not stem or not stem.IsValid():
                problems.append("small valve overlay missing CoACD eixo_trans part_0")
    lamp = LAMP_UUC_DIR / "Lamp_Assembly.usda"
    if not lamp.exists():
        problems.append(f"missing lamp UUC: {lamp}")
    else:
        stage = Usd.Stage.Open(str(lamp))
        if not stage:
            problems.append(f"pxr could not open {lamp}")
        else:
            bulb = stage.GetPrimAtPath("/Lamp_Assembly/Geometry/World/rotation_pivot/lamp_pivot")
            if not bulb or not bulb.HasAPI(UsdPhysics.RigidBodyAPI):
                problems.append("lamp UUC missing lamp_pivot rigid body")
            for name, joint_type in (
                ("RevoluteJoint", UsdPhysics.RevoluteJoint),
                ("PrismaticJoint", UsdPhysics.PrismaticJoint),
            ):
                joint = stage.GetPrimAtPath(f"/Lamp_Assembly/Physics/{name}")
                if not joint or not joint.IsA(joint_type):
                    problems.append(f"lamp UUC missing {name}")
    for label, overlay, default_prim in (
        ("drawer housing", DRAWER_HOUSING_OVERLAY, "Drawer_Housing"),
        ("drawer box", DRAWER_BOX_OVERLAY, "Drawer_Box"),
        ("key", KEY_UUC_DIR / "Key.usda", "Key"),
    ):
        if not overlay.exists():
            problems.append(f"missing {label}: {overlay}")
        else:
            stage = Usd.Stage.Open(str(overlay))
            if not stage:
                problems.append(f"pxr could not open {overlay}")
            else:
                bodies = [
                    prim.GetPath().pathString
                    for prim in stage.Traverse()
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI)
                ]
                # The housing carries a welded anchor body (render_rigid_overlay).
                movers = [b for b in bodies if not b.endswith("/anchor")]
                if len(movers) != 1:
                    problems.append(f"{label} should be one rigid body, found {bodies}")
                for prim in stage.Traverse():
                    if prim.IsA(UsdPhysics.PrismaticJoint) or prim.GetTypeName() == "PhysicsPrismaticJoint":
                        problems.append(f"{label} still has prismatic joint {prim.GetPath()}")
                root = stage.GetPrimAtPath(f"/{default_prim}")
                if not root or not root.IsValid():
                    problems.append(f"{label} missing default prim {default_prim}")
    for label, usda, default_prim, joints in ARTICULATED_UUC_ASSETS:
        if not usda.exists():
            problems.append(f"missing {label} UUC: {usda}")
            continue
        stage = Usd.Stage.Open(str(usda))
        if not stage:
            problems.append(f"pxr could not open {usda}")
            continue
        for name, type_name in joints:
            joint = stage.GetPrimAtPath(f"/{default_prim}/Physics/{name}")
            if not joint or joint.GetTypeName() != type_name:
                problems.append(f"{label} UUC missing {name}")
    return problems


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--assets",
        choices=[
            "all", "spot", "ball_valve", "circuit_breaker", "high_torque_valve", "small_valve", "lamp",
            "button", "drawer", "key", "m8_thread", "m30_thread", "peg_insertion", "shock_absorber",
        ],
        default="all",
        help="Which asset to process (default: all).",
    )
    parser.add_argument(
        "--uuc-python",
        default=DEFAULT_UUC_PYTHON,
        help="Python with urdf_usd_converter importable.",
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip UUC; only rewrite the valve CoACD overlay / verify.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not write anything; check generated USD with pxr.",
    )
    parser.add_argument(
        "--strip-obj",
        action="store_true",
        help="Only drop unused OBJ vertices/polylines on Spot meshes so UUC "
        "usdex definePolyMesh will accept them. Does not convert USD.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    want_spot = args.assets in ("all", "spot")
    want_valve = args.assets in ("all", "ball_valve")
    want_breaker = args.assets in ("all", "circuit_breaker")
    want_ht = args.assets in ("all", "high_torque_valve")
    want_sm = args.assets in ("all", "small_valve")
    want_lamp = args.assets in ("all", "lamp")
    want_button = args.assets in ("all", "button")
    want_drawer = args.assets in ("all", "drawer")
    want_key = args.assets in ("all", "key")
    want_shock = args.assets in ("all", "shock_absorber")
    uuc_only = [
        (label, urdf, out_dir)
        for label, urdf, out_dir in (
            ("m8_thread", M8_URDF, M8_UUC_DIR),
            ("m30_thread", M30_URDF, M30_UUC_DIR),
            ("peg_insertion", PEG_URDF, PEG_UUC_DIR),
            ("shock_absorber", SHOCK_URDF, SHOCK_UUC_DIR),
        )
        if args.assets in ("all", label)
    ]

    if args.verify_only:
        problems = verify()
        if problems:
            print("[FAIL]")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print("[OK] Spot UUC + HiveBoard overlays resolve")
        return 0

    if args.strip_obj:
        n = strip_spot_obj_topology()
        print(f"[OBJ] {n} file(s) changed")
        return 0

    if not args.skip_conversion:
        if want_lamp:
            run_uuc_conversion(LAMP_URDF, LAMP_UUC_DIR, args.uuc_python)
            from generate_honeycomb_collisions import generate as generate_honeycomb_collisions

            generate_honeycomb_collisions()
            print(f"[UUC] lamp: {LAMP_UUC_DIR / 'Lamp_Assembly.usda'}")
        if want_button:
            run_uuc_conversion(BUTTON_URDF, BUTTON_UUC_DIR, args.uuc_python)
            print(f"[UUC] button: {BUTTON_UUC_DIR / 'Button_Assembly.usda'}")
        if want_drawer:
            sync_drawer_collision()
            housing_urdf, box_urdf = _split_drawer_urdfs(DRAWER_URDF.parent)
            try:
                run_uuc_conversion(housing_urdf, DRAWER_HOUSING_UUC_DIR, args.uuc_python)
                run_uuc_conversion(box_urdf, DRAWER_BOX_UUC_DIR, args.uuc_python)
            finally:
                housing_urdf.unlink(missing_ok=True)
                box_urdf.unlink(missing_ok=True)
            _strip_world_fixed_joint(DRAWER_BOX_UUC_DIR / "Payload" / "Physics.usda")
            _anchor_housing(DRAWER_HOUSING_UUC_DIR / "Payload" / "Physics.usda")
            _disable_visual_mesh_collision(DRAWER_HOUSING_UUC_DIR / "Payload" / "Geometry.usda")
            _disable_visual_mesh_collision(DRAWER_BOX_UUC_DIR / "Payload" / "Geometry.usda")
            print(f"[UUC] drawer housing: {DRAWER_HOUSING_UUC_DIR / 'Drawer_Housing.usda'}")
            print(f"[UUC] drawer box: {DRAWER_BOX_UUC_DIR / 'Drawer_Box.usda'}")
        if want_key:
            run_uuc_conversion(LOCK_URDF, LOCK_UUC_DIR, args.uuc_python)
            run_uuc_conversion(KEY_URDF, KEY_UUC_DIR, args.uuc_python)
            # The robot spawn welds the key to its hand; UUC's world weld would fight that.
            _strip_world_fixed_joint(KEY_UUC_DIR / "Payload" / "Physics.usda")
            _disable_visual_mesh_collision(LOCK_UUC_DIR / "Payload" / "Geometry.usda")
            _disable_visual_mesh_collision(KEY_UUC_DIR / "Payload" / "Geometry.usda")
            print(f"[UUC] lock: {LOCK_UUC_DIR / 'Lock_Assembly.usda'}")
            print(f"[UUC] key: {KEY_UUC_DIR / 'Key.usda'}")
        for label, urdf, out_dir in uuc_only:
            run_uuc_conversion(urdf, out_dir, args.uuc_python)
            print(f"[UUC] {label}: {out_dir / urdf.with_suffix('.usda').name}")
        if want_spot:
            strip_spot_obj_topology()
            run_uuc_conversion(SPOT_URDF, SPOT_UUC_DIR, args.uuc_python)
            print(f"[UUC] spot: {SPOT_UUC_DIR / 'spot_with_arm.usda'}")
        if want_valve:
            run_uuc_conversion(VALVE_URDF, VALVE_UUC_DIR, args.uuc_python)
            print(f"[UUC] ball_valve: {VALVE_UUC_DIR / 'Ball_Valve.usda'}")
        if want_breaker:
            run_uuc_conversion(BREAKER_URDF, BREAKER_UUC_DIR, args.uuc_python)
            print(f"[UUC] circuit_breaker: {BREAKER_UUC_DIR / 'Circuit_Breaker_Assembly.usda'}")
        if want_ht:
            run_uuc_conversion(HT_URDF, HT_UUC_DIR, args.uuc_python)
            print(f"[UUC] high_torque_valve: {HT_UUC_DIR / 'High_Torque_Valve.usda'}")
        if want_sm:
            run_uuc_conversion(SM_URDF, SM_UUC_DIR, args.uuc_python)
            print(f"[UUC] small_valve: {SM_UUC_DIR / 'Small_Valve.usda'}")

    if want_valve:
        text = render_valve_overlay()
        VALVE_OVERLAY.write_text(text, encoding="utf-8")
        print(f"[OVERLAY] wrote {VALVE_OVERLAY} ({len(text) // 1024} KiB)")
    if want_breaker:
        text = render_breaker_overlay()
        BREAKER_OVERLAY.parent.mkdir(parents=True, exist_ok=True)
        BREAKER_OVERLAY.write_text(text, encoding="utf-8")
        print(f"[OVERLAY] wrote {BREAKER_OVERLAY} ({len(text) // 1024} KiB)")
    if want_ht:
        text = render_ht_overlay()
        HT_OVERLAY.parent.mkdir(parents=True, exist_ok=True)
        HT_OVERLAY.write_text(text, encoding="utf-8")
        print(f"[OVERLAY] wrote {HT_OVERLAY} ({len(text) // 1024} KiB)")
    if want_sm:
        text = render_sm_overlay()
        SM_OVERLAY.parent.mkdir(parents=True, exist_ok=True)
        SM_OVERLAY.write_text(text, encoding="utf-8")
        print(f"[OVERLAY] wrote {SM_OVERLAY} ({len(text) // 1024} KiB)")

    if want_drawer:
        housing_text = render_rigid_overlay("Drawer_Housing", "./uuc_housing/Drawer_Housing.usda", anchor_body="base")
        box_text = render_rigid_overlay("Drawer_Box", "./uuc_box/Drawer_Box.usda", soft_grasp_body="drawer")
        DRAWER_HOUSING_OVERLAY.write_text(housing_text, encoding="utf-8")
        DRAWER_BOX_OVERLAY.write_text(box_text, encoding="utf-8")
        apply_drawer_contact_materials(DRAWER_HOUSING_OVERLAY, "base")
        apply_drawer_contact_materials(DRAWER_BOX_OVERLAY, "drawer")
        print(f"[OVERLAY] wrote {DRAWER_HOUSING_OVERLAY}")
        print(f"[OVERLAY] wrote {DRAWER_BOX_OVERLAY}")
    if want_shock:
        text = render_shock_overlay()
        SHOCK_OVERLAY.write_text(text, encoding="utf-8")
        print(f"[OVERLAY] wrote {SHOCK_OVERLAY} ({len(text) // 1024} KiB)")

    problems = verify()
    for problem in problems:
        print(f"[WARN] {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
