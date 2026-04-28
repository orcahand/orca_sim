import os
from pathlib import Path
import shutil
import subprocess
import sys

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
    mujoco.MjModel.from_xml_path(str(Path(scene_path).resolve()))


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("CI"),
    reason="Packaging smoke test is slow; run it in CI or locally with CI=1.",
)
def test_built_wheel_installs_and_resets_right_hand(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    project_dir = tmp_path / "project"
    wheel_dir = tmp_path / "wheelhouse"
    shutil.copytree(
        root,
        project_dir,
        ignore=shutil.ignore_patterns(
            ".git",
            ".pytest_cache",
            "__pycache__",
            "*.pyc",
            "build",
            "dist",
            "*.egg-info",
        ),
    )
    wheel_dir.mkdir()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
        ],
        cwd=project_dir,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel_path = next(wheel_dir.glob("orca_sim-*.whl"))
    venv_dir = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
        cwd=project_dir,
        check=True,
        capture_output=True,
        text=True,
    )

    python_in_venv = venv_dir / "bin" / "python"
    subprocess.run(
        [str(python_in_venv), "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel_path)],
        cwd=project_dir,
        check=True,
        capture_output=True,
        text=True,
    )

    smoke_test = subprocess.run(
        [
            str(python_in_venv),
            "-c",
            (
                "from orca_sim import OrcaHandRight; "
                "env = OrcaHandRight(render_mode='rgb_array'); "
                "obs, info = env.reset(); "
                "print(obs.shape, type(info).__name__); "
                "env.close()"
            ),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "(34,)" in smoke_test.stdout
    assert "dict" in smoke_test.stdout
