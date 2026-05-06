"""Decimate collision-active meshes for MJX compatibility.

What this does
--------------
For each hand (v2 left, v2 right):

1. Parse the body.xml to find geoms that are collision-active (no
   ``contype="0"`` override).
2. For each unique mesh those geoms reference, locate its STL via the
   ``orcahand_{left,right}.mjcf`` asset block, compute a 16-vertex farthest-point
   convex hull, and save it under ``assets/{left,right}_collision/<name>.stl``.
3. Rewrite ``orcahand_{left,right}.mjcf`` to declare a parallel ``X_coll`` mesh
   for every collision-active mesh.
4. Rewrite ``orcahand_{left,right}_body.xml`` so each collision-active geom
   becomes visual-only (``contype="0" conaffinity="0"``) and gains a sibling
   collision geom referencing the ``X_coll`` mesh (``group="3"`` to hide from
   the default render groups).

The original STL files are not touched — visual meshes keep full fidelity.

Usage
-----
    conda run -n orca python scripts/decimate_collision_meshes.py
"""
from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_V2 = REPO_ROOT / "src" / "orca_sim" / "models" / "v2"

HANDS = [
    {
        "side": "left",
        "mjcf": MODELS_V2 / "mjcf" / "orcahand_left.mjcf",
        "body": MODELS_V2 / "mjcf" / "orcahand_left_body.xml",
        "asset_dir": MODELS_V2 / "assets" / "left",
        "coll_dir": MODELS_V2 / "assets" / "left_collision",
        "rel_coll_path": "../assets/left_collision",
    },
    {
        "side": "right",
        "mjcf": MODELS_V2 / "mjcf" / "orcahand_right.mjcf",
        "body": MODELS_V2 / "mjcf" / "orcahand_right_body.xml",
        "asset_dir": MODELS_V2 / "assets" / "right",
        "coll_dir": MODELS_V2 / "assets" / "right_collision",
        "rel_coll_path": "../assets/right_collision",
    },
]

TARGET_VERTS = 16
COLL_SUFFIX = "_coll"


def aggressive_convex_hull(stl_path: Path, target_verts: int = TARGET_VERTS) -> trimesh.Trimesh:
    m = trimesh.load(stl_path, force="mesh")
    hull = m.convex_hull
    pts = np.asarray(hull.vertices)
    if len(pts) <= target_verts:
        return hull

    # Farthest-point sampling: deterministic seed, reproducible runs.
    rng = np.random.default_rng(0)
    sel = [int(rng.integers(len(pts)))]
    dists = np.linalg.norm(pts - pts[sel[0]], axis=1)
    while len(sel) < target_verts:
        i = int(np.argmax(dists))
        sel.append(i)
        dists = np.minimum(dists, np.linalg.norm(pts - pts[i], axis=1))

    sub = pts[sel]
    ch = ConvexHull(sub)
    final = trimesh.Trimesh(vertices=sub, faces=ch.simplices, process=True)
    # Re-hull once more in case process() merged degenerate faces.
    return final.convex_hull


def find_mesh_assets(mjcf_path: Path) -> dict[str, Path]:
    """Return mapping from mesh-name → absolute STL path declared in the mjcf."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    out: dict[str, Path] = {}
    for mesh in root.iter("mesh"):
        name = mesh.get("name")
        f = mesh.get("file")
        if name and f:
            out[name] = (mjcf_path.parent / f).resolve()
    return out


def find_collision_active_meshes(body_path: Path) -> list[ET.Element]:
    """Return the geom elements that are collision-active.

    A geom is collision-active when it (a) references a mesh and (b) does NOT
    have an explicit ``contype="0"`` attribute. The mjcf's ``<default>`` block
    sets ``contype=1 conaffinity=1`` for unclassed geoms, so any geom without
    an override participates in collision.
    """
    tree = ET.parse(body_path)
    root = tree.getroot()
    out = []
    for g in root.iter("geom"):
        if g.get("mesh") is None:
            continue
        if g.get("contype") == "0":
            continue
        out.append(g)
    return out


def decimate_for_hand(spec: dict) -> set[str]:
    """Generate decimated _coll.stl files. Returns the set of mesh names that need _coll variants."""
    body = spec["body"]
    mjcf = spec["mjcf"]
    coll_dir: Path = spec["coll_dir"]

    asset_map = find_mesh_assets(mjcf)
    coll_geoms = find_collision_active_meshes(body)
    needed_meshes: set[str] = {g.get("mesh") for g in coll_geoms}

    coll_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{spec['side']}] {len(coll_geoms)} collision-active geoms, "
          f"{len(needed_meshes)} unique meshes to decimate")

    for mesh_name in sorted(needed_meshes):
        stl_in = asset_map.get(mesh_name)
        if stl_in is None or not stl_in.exists():
            raise FileNotFoundError(
                f"Mesh '{mesh_name}' (from {body.name}) not declared / file missing"
            )
        # Output filename: same as input but in the collision directory.
        stl_out = coll_dir / stl_in.name
        hull = aggressive_convex_hull(stl_in, TARGET_VERTS)
        hull.export(stl_out)
        print(f"  {mesh_name}: V={len(hull.vertices)} F={len(hull.faces)} → {stl_out.relative_to(REPO_ROOT)}")

    return needed_meshes


def patch_mjcf(spec: dict, needed_meshes: set[str]) -> None:
    """Add ``<mesh name="X_coll" file="..."/>`` declarations for each needed mesh."""
    mjcf_path = spec["mjcf"]
    rel_coll = spec["rel_coll_path"]
    asset_map = find_mesh_assets(mjcf_path)

    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    asset_block = root.find("asset")
    if asset_block is None:
        raise RuntimeError(f"{mjcf_path} has no <asset> block")

    existing_names = {m.get("name") for m in asset_block.iter("mesh")}
    added = 0
    for name in sorted(needed_meshes):
        coll_name = name + COLL_SUFFIX
        if coll_name in existing_names:
            continue
        orig_path = asset_map[name]
        coll_filename = orig_path.name  # same basename, lives in coll_dir
        new = ET.SubElement(asset_block, "mesh")
        new.set("name", coll_name)
        new.set("file", f"{rel_coll}/{coll_filename}")
        added += 1

    if added:
        ET.indent(tree, space="\t")
        tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)
        # ET strips trailing newline; add one for cleaner diffs.
        with open(mjcf_path, "a") as f:
            f.write("\n")
        print(f"  patched {mjcf_path.name}: added {added} _coll mesh declarations")
    else:
        print(f"  {mjcf_path.name}: already has _coll declarations, no changes")


def patch_body_xml(spec: dict) -> None:
    """For each collision-active geom: disable its collision and add a sibling _coll geom."""
    body_path = spec["body"]
    tree = ET.parse(body_path)
    root = tree.getroot()

    # Walk parents to safely insert siblings (ET doesn't expose parents).
    parent_map = {child: parent for parent in root.iter() for child in parent}

    coll_geoms = []
    for g in root.iter("geom"):
        if g.get("mesh") is None:
            continue
        if g.get("contype") == "0":
            continue
        # Skip if this geom is already a _coll proxy (idempotent).
        if g.get("mesh", "").endswith(COLL_SUFFIX):
            continue
        coll_geoms.append(g)

    for g in coll_geoms:
        mesh_name = g.get("mesh")
        # Disable collision on the original (visual) geom.
        g.set("contype", "0")
        g.set("conaffinity", "0")

        # Build a sibling collision-only geom with the SAME pos/quat.
        sibling = ET.Element("geom")
        sibling.set("mesh", mesh_name + COLL_SUFFIX)
        sibling.set("group", "3")  # hidden from default render groups (0-2).
        for attr in ("pos", "quat"):
            val = g.get(attr)
            if val is not None:
                sibling.set(attr, val)

        parent = parent_map[g]
        idx = list(parent).index(g)
        parent.insert(idx + 1, sibling)

    ET.indent(tree, space="\t")
    tree.write(body_path, encoding="utf-8", xml_declaration=True)
    with open(body_path, "a") as f:
        f.write("\n")
    print(f"  patched {body_path.name}: split {len(coll_geoms)} visual+collision geoms")


def main() -> None:
    for spec in HANDS:
        print(f"\n=== {spec['side']} hand ===")
        needed = decimate_for_hand(spec)
        patch_mjcf(spec, needed)
        patch_body_xml(spec)

    print("\nDone.")


if __name__ == "__main__":
    main()
