from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from importlib.resources import as_file, files

from orca_core.hand_config import OrcaHandConfig


@lru_cache
def canonical_single_hand_joint_ids() -> tuple[str, ...]:
    config_ref = files("orca_core").joinpath("models/orcahand_v1_right/config.yaml")
    with as_file(config_ref) as config_path:
        config = OrcaHandConfig.from_config_path(str(config_path))
    return tuple(config.joint_ids)


V2_LOCAL_SCENE_JOINT_BY_CANONICAL: dict[str, str] = {
    # NOTE: v2 uses a different thumb naming scheme in the MJCF
    "thumb_mcp": "t-cmc",
    "thumb_abd": "t-abd",
    "thumb_pip": "t-mcp",
    "thumb_dip": "t-pip",
    "index_abd": "i-abd",
    "index_mcp": "i-mcp",
    "index_pip": "i-pip",
    "middle_abd": "m-abd",
    "middle_mcp": "m-mcp",
    "middle_pip": "m-pip",
    "ring_abd": "r-abd",
    "ring_mcp": "r-mcp",
    "ring_pip": "r-pip",
    "pinky_abd": "p-abd",
    "pinky_mcp": "p-mcp",
    "pinky_pip": "p-pip",
    "wrist": "wrist",
}


def default_joint_name_to_scene_joint_name(
    *,
    scene_file: str,
    version: str,
) -> tuple[str | None, dict[str, str]]:
    canonical = canonical_single_hand_joint_ids()
    if version == "v1":
        local_mapping = {joint: joint for joint in canonical}
    elif version == "v2":
        local_mapping = V2_LOCAL_SCENE_JOINT_BY_CANONICAL
    else:
        raise FileNotFoundError(f"Unsupported embodiment version for joint mapping: {version}")

    if "combined" in scene_file:
        mapping: dict[str, str] = {}
        for side in ("left", "right"):
            for joint in canonical:
                mapping[f"{side}_{joint}"] = f"{side}_{local_mapping[joint]}"
        return None, mapping

    if "left" in scene_file:
        return (
            "left",
            {joint: f"left_{local_mapping[joint]}" for joint in canonical},
        )

    if "right" in scene_file:
        return (
            "right",
            {joint: f"right_{local_mapping[joint]}" for joint in canonical},
        )

    raise ValueError(
        "Unable to infer a default hand-joint mapping from scene_file. "
        "Provide joint_name_to_scene_joint_name explicitly."
    )


def infer_hand_type_from_joint_names(joint_names: list[str]) -> str | None:
    prefixes = {
        joint_name.split("_", maxsplit=1)[0]
        for joint_name in joint_names
        if "_" in joint_name
    }
    if prefixes == {"left"}:
        return "left"
    if prefixes == {"right"}:
        return "right"
    return None


def resolve_joint_mapping(
    *,
    scene_file: str,
    version: str,
    joint_name_to_scene_joint_name: Mapping[str, str] | None,
    hand_type: str | None,
) -> tuple[str | None, dict[str, str]]:
    if joint_name_to_scene_joint_name is None:
        return default_joint_name_to_scene_joint_name(
            scene_file=scene_file,
            version=version,
        )

    resolved_mapping = dict(joint_name_to_scene_joint_name)
    resolved_type = hand_type or infer_hand_type_from_joint_names(list(resolved_mapping))
    return resolved_type, resolved_mapping
