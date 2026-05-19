from __future__ import annotations

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


CAMERA_SPECS = {
    "openarm_body_link0": {
        "name": "chest_table_camera",
        "pos": "0.08 0 0.72",
        "mode": "targetbody",
        "target": "table",
        "fovy": "75",
    },
    "orcahand_left_ForeArmStructure-Model_e18f2368": {
        "name": "left_wrist_camera",
        "pos": "-0.01 0.085 -0.04",
        "mode": "targetbody",
        "target": "left_wrist_camera_target",
        "fovy": "85",
    },
    "orcahand_right_ForeArmStructure-Model_e18f2368": {
        "name": "right_wrist_camera",
        "pos": "-0.01 0.085 -0.04",
        "mode": "targetbody",
        "target": "right_wrist_camera_target",
        "fovy": "85",
    },
}

CAMERA_TARGET_SPECS = {
    "orcahand_left_ForeArmStructure-Model_e18f2368": {
        "name": "left_wrist_camera_target",
        "pos": "-0.01 0.175 -0.075",
        "site": {
            "name": "left_wrist_camera_target_site",
            "type": "sphere",
            "pos": "0 0 0",
            "size": "0.006",
            "rgba": "1 0.85 0.05 1",
            "group": "3",
        },
    },
    "orcahand_right_ForeArmStructure-Model_e18f2368": {
        "name": "right_wrist_camera_target",
        "pos": "-0.01 0.175 -0.075",
        "site": {
            "name": "right_wrist_camera_target_site",
            "type": "sphere",
            "pos": "0 0 0",
            "size": "0.006",
            "rgba": "1 0.85 0.05 1",
            "group": "3",
        },
    },
}

CAMERA_SITE_SPECS = {
    "chest_table_camera": {
        "name": "chest_table_camera_site",
        "type": "sphere",
        "pos": CAMERA_SPECS["openarm_body_link0"]["pos"],
        "size": "0.015",
        "rgba": "0.1 1 0.1 1",
        "group": "3",
    },
    "left_wrist_camera": {
        "name": "left_wrist_camera_site",
        "type": "sphere",
        "pos": CAMERA_SPECS["orcahand_left_ForeArmStructure-Model_e18f2368"]["pos"],
        "size": "0.01",
        "rgba": "1 0.1 0.1 1",
        "group": "3",
    },
    "right_wrist_camera": {
        "name": "right_wrist_camera_site",
        "type": "sphere",
        "pos": CAMERA_SPECS["orcahand_right_ForeArmStructure-Model_e18f2368"]["pos"],
        "size": "0.01",
        "rgba": "0.1 0.3 1 1",
        "group": "3",
    },
}


def camera_names() -> tuple[str, ...]:
    return tuple(spec["name"] for spec in CAMERA_SPECS.values())


def build_orcaarm_camera_mjcf(
    output_path: str | Path | None = None,
    *,
    debug_sites: bool = False,
) -> Path:
    import orca_arm

    source_path = Path(orca_arm.MJCF_PATH).resolve()
    if output_path is None:
        output_path = (
            Path(tempfile.gettempdir())
            / "orca_sim"
            / f"{source_path.stem}_with_cameras.xml"
        )
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(source_path)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        mesh_file = mesh.get("file")
        if mesh_file is None:
            continue
        mesh_path = Path(mesh_file)
        if not mesh_path.is_absolute():
            mesh.set("file", str((source_path.parent / mesh_path).resolve()))

    for body_name, camera_spec in CAMERA_SPECS.items():
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            raise RuntimeError(
                f"Unable to mount camera on {body_name!r}; body is missing in {source_path}."
            )
        for existing in body.findall("camera"):
            if existing.get("name") == camera_spec["name"]:
                body.remove(existing)
        site_spec = CAMERA_SITE_SPECS[camera_spec["name"]]
        for existing in body.findall("site"):
            if existing.get("name") == site_spec["name"]:
                body.remove(existing)
        target_spec = CAMERA_TARGET_SPECS.get(body_name)
        if target_spec is not None:
            for existing in body.findall("body"):
                if existing.get("name") == target_spec["name"]:
                    body.remove(existing)
            target_body = ET.SubElement(
                body,
                "body",
                {"name": target_spec["name"], "pos": target_spec["pos"]},
            )
            if debug_sites:
                ET.SubElement(target_body, "site", target_spec["site"])
        ET.SubElement(body, "camera", camera_spec)
        if debug_sites:
            ET.SubElement(body, "site", site_spec)

    tree.write(output_path, encoding="unicode")
    return output_path
