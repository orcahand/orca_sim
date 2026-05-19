from __future__ import annotations

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


MOUNT_POS = "-0.05 0 0.08"
ASSET_PREFIX = "orcapanda_"
WRIST_CAMERA_BODY = "orcahand_right_ForeArmStructure-Model_e18f2368"
WRIST_CAMERA_SPEC = {
    "name": "orcapanda_wrist_camera",
    "pos": "-0.01 0.085 -0.04",
    "mode": "targetbody",
    "target": "orcapanda_wrist_camera_target",
    "fovy": "85",
}
WRIST_CAMERA_TARGET_SPEC = {
    "name": "orcapanda_wrist_camera_target",
    "pos": "-0.01 0.175 -0.075",
}


def build_orcapanda_mjcf(output_path: str | Path | None = None) -> Path:
    source_path = _resolve_orcapanda_mjcf_path()
    if output_path is None:
        output_path = (
            Path(tempfile.gettempdir())
            / "orca_sim"
            / f"{source_path.stem}_namespaced.xml"
        )
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(source_path)
    root = tree.getroot()
    _strip_global_elements(root)
    _namespace_default_classes(root)
    _namespace_assets(root, source_path)
    _mount_base(root)
    _add_wrist_camera(root)

    tree.write(output_path, encoding="unicode")
    return output_path


def _resolve_orcapanda_mjcf_path() -> Path:
    try:
        import orca_arm
    except ModuleNotFoundError:
        package_path = None
    else:
        package_path = getattr(orca_arm, "ORCAPANDA_MJCF_PATH", None)
        if package_path is not None:
            source_path = Path(package_path).resolve()
            if source_path.exists():
                return source_path

    repo_path = (
        Path(__file__).resolve().parents[3].parent
        / "orca_arm"
        / "orca_arm"
        / "orcapanda.xml"
    )
    if repo_path.exists():
        return repo_path

    raise RuntimeError(
        "Could not find orcapanda.xml. Reinstall the local orca_arm package with "
        "`python -m pip install -e ../orca_arm`, or update the fallback path in "
        "this builder."
    )


def _strip_global_elements(root: ET.Element) -> None:
    for tag in ("compiler", "option", "keyframe"):
        for element in root.findall(tag):
            root.remove(element)


def _namespace_default_classes(root: ET.Element) -> None:
    class_names = {
        element.get("class")
        for element in root.findall(".//default")
        if element.get("class") is not None
    }
    class_name_map = {
        class_name: f"{ASSET_PREFIX}{class_name}" for class_name in class_names
    }
    for element in root.iter():
        class_name = element.get("class")
        if class_name in class_name_map:
            element.set("class", class_name_map[class_name])
        childclass_name = element.get("childclass")
        if childclass_name in class_name_map:
            element.set("childclass", class_name_map[childclass_name])


def _namespace_assets(root: ET.Element, source_path: Path) -> None:
    material_name_map: dict[str, str] = {}
    for material in root.findall(".//material"):
        name = material.get("name")
        if name is None:
            continue
        namespaced_name = f"{ASSET_PREFIX}{name}"
        material_name_map[name] = namespaced_name
        material.set("name", namespaced_name)

    mesh_name_map: dict[str, str] = {}
    for mesh in root.findall(".//mesh"):
        mesh_file = mesh.get("file")
        if mesh_file is None:
            continue

        mesh_path = Path(mesh_file)
        if not mesh_path.is_absolute():
            mesh.set("file", str((source_path.parent / mesh_path).resolve()))
        if mesh.get("scale") is None:
            mesh.set("scale", "1 1 1")

        name = mesh.get("name")
        if name is None:
            name = Path(mesh_file).stem
        namespaced_name = f"{ASSET_PREFIX}{name}"
        mesh_name_map[name] = namespaced_name
        mesh.set("name", namespaced_name)

    for element in root.iter():
        material = element.get("material")
        if material in material_name_map:
            element.set("material", material_name_map[material])

        mesh = element.get("mesh")
        if mesh in mesh_name_map:
            element.set("mesh", mesh_name_map[mesh])


def _mount_base(root: ET.Element) -> None:
    base = root.find(".//body[@name='panda_link0']")
    if base is None:
        raise RuntimeError("Unable to mount OrcaPanda; body 'panda_link0' is missing.")
    base.set("pos", MOUNT_POS)


def _add_wrist_camera(root: ET.Element) -> None:
    body = root.find(f".//body[@name='{WRIST_CAMERA_BODY}']")
    if body is None:
        raise RuntimeError(
            f"Unable to mount OrcaPanda wrist camera; body {WRIST_CAMERA_BODY!r} is missing."
        )

    for existing in body.findall("camera"):
        if existing.get("name") == WRIST_CAMERA_SPEC["name"]:
            body.remove(existing)
    for existing in body.findall("body"):
        if existing.get("name") == WRIST_CAMERA_TARGET_SPEC["name"]:
            body.remove(existing)

    ET.SubElement(body, "body", WRIST_CAMERA_TARGET_SPEC)
    ET.SubElement(body, "camera", WRIST_CAMERA_SPEC)
