from pathlib import Path

import mujoco
import pytest

from orca_sim.versions import (
    LATEST_VERSION,
    PACKAGE_ROOT,
    latest_version,
    list_versions,
    resolve_scene_path,
    resolve_version,
)


def test_version_discovery_defaults_to_latest() -> None:
    versions = list_versions()

    assert versions is not None, "No versions found"
    assert LATEST_VERSION in versions, "LATEST_VERSION not found in versions"
    assert latest_version() == LATEST_VERSION, "latest_version() did not return LATEST_VERSION"
    assert resolve_version() == LATEST_VERSION, "resolve_version(None) did not return LATEST_VERSION"


@pytest.mark.parametrize(
    "scene_file",
    [
        "scene_left.xml",
        "scene_right.xml",
        "scene_combined.xml",
        "scene_left_extended.xml",
        "scene_right_extended.xml",
        "scene_combined_extended.xml",
    ],
)
def test_scene_paths_exist_for_each_scene(scene_file: str) -> None:
    scene_path = resolve_scene_path(scene_file)

    assert scene_path.is_absolute(), "Scene path is not absolute"
    assert scene_path.exists(), "Scene path does not exist"
    assert PACKAGE_ROOT in scene_path.parents, "PACKAGE_ROOT not in scene path parents"


def test_resolve_version_rejects_unknown_versions() -> None:
    with pytest.raises(FileNotFoundError, match="Unknown embodiment version"):
        resolve_version("does-not-exist")


@pytest.mark.parametrize(
    "scene_path",
    [
        "src/orca_sim/scenes/v1/scene_left.xml",
        "src/orca_sim/scenes/v1/scene_right.xml",
        "src/orca_sim/scenes/v1/scene_combined.xml",
        "src/orca_sim/scenes/v1/scene_left_extended.xml",
        "src/orca_sim/scenes/v1/scene_right_extended.xml",
        "src/orca_sim/scenes/v1/scene_combined_extended.xml",
        "src/orca_sim/scenes/v1/scene_right_cube_orientation.xml",
        "src/orca_sim/scenes/v2/scene_left.xml",
        "src/orca_sim/scenes/v2/scene_right.xml",
        "src/orca_sim/scenes/v2/scene_combined.xml",
        "src/orca_sim/scenes/v2/scene_right_cube_orientation.xml",
    ],
)
def test_scene_paths_load_directly(scene_path: str) -> None:
    mujoco.MjModel.from_xml_path(str(Path(scene_path)))
