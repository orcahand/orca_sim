import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import subprocess
import sys

import mujoco
import numpy as np
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
        "cube_stacking.xml",
        "orcaarm_cube_stacking.xml",
        "orcaarm_cube_stacking_cameras.xml",
        "orcapanda_cube_stacking.xml",
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


def test_unversioned_scene_resolves_from_scenes_root() -> None:
    scene_path = resolve_scene_path("cube_stacking.xml")

    assert scene_path == PACKAGE_ROOT / "scenes" / "cube_stacking.xml"


def test_unversioned_scene_can_resolve_with_version_hint() -> None:
    scene_path = resolve_scene_path("cube_stacking.xml", version="v2")

    assert scene_path == PACKAGE_ROOT / "scenes" / "cube_stacking.xml"


def test_orcaarm_cube_stacking_scene_references_orca_arm_mjcf_path() -> None:
    orca_arm = pytest.importorskip("orca_arm")
    scene_path = PACKAGE_ROOT / "scenes" / "orcaarm_cube_stacking.xml"
    include_file = ET.parse(scene_path).getroot().findall("include")[-1].get("file")

    assert (scene_path.parent / include_file).resolve() == Path(orca_arm.MJCF_PATH).resolve()


def test_orcaarm_cube_stacking_scene_loads_directly() -> None:
    pytest.importorskip("orca_arm")

    mujoco.MjModel.from_xml_path(
        str((PACKAGE_ROOT / "scenes" / "orcaarm_cube_stacking.xml").resolve())
    )


def test_orcaarm_cube_stacking_camera_scene_loads_directly() -> None:
    pytest.importorskip("orca_arm")

    model = mujoco.MjModel.from_xml_path(
        str((PACKAGE_ROOT / "scenes" / "orcaarm_cube_stacking_cameras.xml").resolve())
    )
    camera_names = {model.camera(camera_id).name for camera_id in range(model.ncam)}
    assert {
        "chest_table_camera",
        "left_wrist_camera",
        "right_wrist_camera",
    }.issubset(camera_names)


def test_orcapanda_cube_stacking_scene_loads_directly() -> None:
    pytest.importorskip("orca_arm")

    model = mujoco.MjModel.from_xml_path(
        str((PACKAGE_ROOT / "scenes" / "orcapanda_cube_stacking.xml").resolve())
    )

    assert model.nu == 24
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "panda_link0") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "orcapanda_home") >= 0
    assert (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "orcapanda_wrist_camera")
        >= 0
    )


def test_orcapanda_cube_stacking_home_keyframe_sets_ready_pose() -> None:
    pytest.importorskip("orca_arm")
    q_home = np.array(
        [-0.1, -1.6, -0.1, -3.0718, -0.15, 2.85, -1.4027],
        dtype=np.float64,
    )
    model = mujoco.MjModel.from_xml_path(
        str((PACKAGE_ROOT / "scenes" / "orcapanda_cube_stacking.xml").resolve())
    )
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "orcapanda_home")

    mujoco.mj_resetDataKeyframe(model, data, key_id)

    arm_joint_names = [f"panda_joint{i}" for i in range(1, 8)]
    arm_qpos = np.array(
        [
            data.qpos[int(model.jnt_qposadr[model.joint(joint_name).id])]
            for joint_name in arm_joint_names
        ],
        dtype=np.float64,
    )
    arm_ctrl = []
    for joint_name in arm_joint_names:
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            if model.joint(joint_id).name == joint_name:
                arm_ctrl.append(data.ctrl[actuator_id])
                break

    np.testing.assert_allclose(arm_qpos, q_home)
    np.testing.assert_allclose(np.asarray(arm_ctrl, dtype=np.float64), q_home)


def test_orcaarm_cube_stacking_home_keyframe_sets_arm_pose() -> None:
    pytest.importorskip("orca_arm")
    q_home = np.array(
        [
            0.0,
            0.004,
            0.0,
            1.520,
            1.570796,
            0.0,
            0.005,
            0.0,
            1.530,
            -1.570796,
        ],
        dtype=np.float64,
    )
    model = mujoco.MjModel.from_xml_path(
        str((PACKAGE_ROOT / "scenes" / "orcaarm_cube_stacking.xml").resolve())
    )
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "orcaarm_home")

    mujoco.mj_resetDataKeyframe(model, data, key_id)

    arm_joint_names = [
        *[f"openarm_left_joint{i}" for i in range(1, 6)],
        *[f"openarm_right_joint{i}" for i in range(1, 6)],
    ]
    arm_qpos = np.array(
        [
            data.qpos[int(model.jnt_qposadr[model.joint(joint_name).id])]
            for joint_name in arm_joint_names
        ],
        dtype=np.float64,
    )
    arm_ctrl = []
    arm_joint_ids = set()
    arm_actuator_ids = set()
    for joint_name in arm_joint_names:
        arm_joint_ids.add(model.joint(joint_name).id)
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            if model.joint(joint_id).name == joint_name:
                arm_ctrl.append(data.ctrl[actuator_id])
                arm_actuator_ids.add(actuator_id)
                break
    np.testing.assert_allclose(arm_qpos, q_home)
    np.testing.assert_allclose(np.asarray(arm_ctrl, dtype=np.float64), q_home)

    for joint_id in range(model.njnt):
        if joint_id in arm_joint_ids:
            continue
        joint_type = int(model.jnt_type[joint_id])
        qpos_width = 7 if joint_type == mujoco.mjtJoint.mjJNT_FREE else 1
        qpos_adr = int(model.jnt_qposadr[joint_id])
        np.testing.assert_allclose(
            data.qpos[qpos_adr : qpos_adr + qpos_width],
            model.qpos0[qpos_adr : qpos_adr + qpos_width],
        )
    for actuator_id in range(model.nu):
        if actuator_id not in arm_actuator_ids:
            assert data.ctrl[actuator_id] == pytest.approx(0.0)


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
        "src/orca_sim/scenes/cube_stacking.xml",
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
