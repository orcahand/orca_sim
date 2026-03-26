from __future__ import annotations

"""Integration tests for the runtime hand API stack.

These checks are intentionally broader than unit tests. They validate that the
package remains coherent end-to-end across:

- MuJoCo scene assets
- SimOrcaHandConfig metadata derivation
- SimOrcaHand's shared BaseHand contract
- Gym env wrappers built on top of SimOrcaHand
- Task env wrappers built on top of SimOrcaHand

The goal is to protect the public runtime architecture, not only the original
port from the vertically integrated implementation.
"""

from unittest.mock import patch

import mujoco
import numpy as np
import pytest
from orca_core.joint_position import OrcaJointPositions

from orca_sim import (
    OrcaHandRight,
    OrcaHandRightCubeOrientation,
    SimOrcaHand,
    SimOrcaHandConfig,
)
from orca_sim.joint_mapping import (
    canonical_single_hand_joint_ids,
    default_joint_name_to_scene_joint_name,
)
from orca_sim.versions import resolve_scene_path


def _joint_names_from_model(model: mujoco.MjModel) -> list[str]:
    return [
        model.joint(int(model.actuator_trnid[actuator_id, 0])).name
        for actuator_id in range(model.nu)
    ]


def _qpos_indices_from_model(model: mujoco.MjModel) -> list[int]:
    return [
        int(model.jnt_qposadr[int(model.actuator_trnid[actuator_id, 0])])
        for actuator_id in range(model.nu)
    ]


def _qvel_indices_from_model(model: mujoco.MjModel) -> list[int]:
    return [
        int(model.jnt_dofadr[int(model.actuator_trnid[actuator_id, 0])])
        for actuator_id in range(model.nu)
    ]


def _build_valid_qpos(model: mujoco.MjModel) -> np.ndarray:
    qpos = model.qpos0.copy()
    ctrl_mid = model.actuator_ctrlrange.mean(axis=1)
    for qpos_idx, value in zip(_qpos_indices_from_model(model), ctrl_mid, strict=True):
        qpos[qpos_idx] = value
    return qpos


def _raw_reset(
    scene_path,
    config: SimOrcaHandConfig,
    *,
    qpos: np.ndarray | None = None,
    qvel: np.ndarray | None = None,
    ctrl: np.ndarray | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    if qpos is not None:
        data.qpos[:] = np.asarray(qpos, dtype=np.float64)
    if qvel is not None:
        data.qvel[:] = np.asarray(qvel, dtype=np.float64)

    if ctrl is None:
        ctrl = np.array(
            [data.qpos[qpos_idx] for qpos_idx in config.actuator_qpos_indices],
            dtype=np.float32,
        )
    else:
        ctrl = np.asarray(ctrl, dtype=np.float32)
    ctrl = np.clip(ctrl, np.asarray(config.action_low), np.asarray(config.action_high))
    for actuator_id, value in zip(config.actuator_ids, ctrl, strict=True):
        data.ctrl[actuator_id] = float(value)

    mujoco.mj_forward(model, data)
    obs = np.concatenate([data.qpos.copy(), data.qvel.copy()])
    return model, data, obs


@pytest.mark.parametrize(
    ("scene_file", "version", "expected_type"),
    [
        ("scene_left.xml", "v1", "left"),
        ("scene_left.xml", "v2", "left"),
        ("scene_right.xml", "v1", "right"),
        ("scene_right.xml", "v2", "right"),
        ("scene_combined.xml", "v1", None),
        ("scene_combined.xml", "v2", None),
    ],
)
def test_sim_hand_config_matches_scene_metadata(
    scene_file: str, version: str, expected_type: str | None
) -> None:
    config = SimOrcaHandConfig.from_config_path(scene_file=scene_file, version=version)
    scene_path = resolve_scene_path(scene_file, version=version)
    model = mujoco.MjModel.from_xml_path(str(scene_path))

    resolved_type, expected_mapping = default_joint_name_to_scene_joint_name(
        scene_file=scene_file,
        version=version,
    )
    expected_joint_ids = list(expected_mapping)
    expected_scene_joint_names = list(expected_mapping.values())

    scene_joint_to_actuator_id = {}
    scene_joint_to_qpos_idx = {}
    scene_joint_to_qvel_idx = {}
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = model.joint(joint_id).name
        scene_joint_to_actuator_id[joint_name] = actuator_id
        scene_joint_to_qpos_idx[joint_name] = int(model.jnt_qposadr[joint_id])
        scene_joint_to_qvel_idx[joint_name] = int(model.jnt_dofadr[joint_id])

    assert config.scene_path == str(scene_path)
    assert config.scene_file == scene_file
    assert config.version == version
    assert resolved_type == expected_type
    assert config.type == expected_type
    assert config.joint_ids == expected_joint_ids
    assert list(config.scene_joint_names) == expected_scene_joint_names
    assert list(config.actuator_ids) == [
        scene_joint_to_actuator_id[joint_name] for joint_name in expected_scene_joint_names
    ]
    assert list(config.actuator_qpos_indices) == [
        scene_joint_to_qpos_idx[joint_name] for joint_name in expected_scene_joint_names
    ]
    assert list(config.actuator_qvel_indices) == [
        scene_joint_to_qvel_idx[joint_name] for joint_name in expected_scene_joint_names
    ]
    np.testing.assert_allclose(
        np.asarray(config.action_low),
        [model.actuator_ctrlrange[actuator_id, 0] for actuator_id in config.actuator_ids],
    )
    np.testing.assert_allclose(
        np.asarray(config.action_high),
        [model.actuator_ctrlrange[actuator_id, 1] for actuator_id in config.actuator_ids],
    )

    for joint_name, scene_joint_name in expected_mapping.items():
        qpos_idx = scene_joint_to_qpos_idx[scene_joint_name]
        assert config.neutral_position[joint_name] == pytest.approx(model.qpos0[qpos_idx])


def test_sim_hand_uses_orca_core_canonical_order_for_single_hand_configs() -> None:
    canonical_joint_ids = list(canonical_single_hand_joint_ids())

    right_config = SimOrcaHandConfig.from_config_path(scene_file="scene_right.xml", version="v1")
    left_config = SimOrcaHandConfig.from_config_path(scene_file="scene_left.xml", version="v2")
    combined_config = SimOrcaHandConfig.from_config_path(scene_file="scene_combined.xml", version="v2")

    assert right_config.joint_ids == canonical_joint_ids
    assert left_config.joint_ids == canonical_joint_ids
    assert combined_config.joint_ids == [
        *[f"left_{joint}" for joint in canonical_joint_ids],
        *[f"right_{joint}" for joint in canonical_joint_ids],
    ]


@pytest.mark.parametrize(
    ("scene_file", "version"),
    [
        ("scene_right.xml", "v1"),
        ("scene_right.xml", "v2"),
        ("scene_right_cube_orientation.xml", "v1"),
        ("scene_right_cube_orientation.xml", "v2"),
    ],
)
def test_sim_hand_reset_matches_raw_mujoco(scene_file: str, version: str) -> None:
    config = SimOrcaHandConfig.from_config_path(scene_file=scene_file, version=version)
    scene_path = resolve_scene_path(scene_file, version=version)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    qpos = _build_valid_qpos(model)
    qvel = np.linspace(-0.2, 0.2, model.nv, dtype=np.float64)
    ctrl = np.linspace(-10.0, 10.0, len(config.joint_ids), dtype=np.float32)

    _, raw_data, raw_obs = _raw_reset(scene_path, config, qpos=qpos, qvel=qvel, ctrl=ctrl)

    hand = SimOrcaHand(scene_file=scene_file, version=version)
    try:
        obs = hand.reset(qpos=qpos, qvel=qvel, ctrl=ctrl)

        np.testing.assert_allclose(obs, raw_obs)
        np.testing.assert_allclose(hand.data.qpos, raw_data.qpos)
        np.testing.assert_allclose(hand.data.qvel, raw_data.qvel)
        np.testing.assert_allclose(hand.data.ctrl, raw_data.ctrl)
    finally:
        hand.close()


@pytest.mark.parametrize(
    ("scene_file", "version"),
    [
        ("scene_right.xml", "v1"),
        ("scene_right.xml", "v2"),
        ("scene_right_cube_orientation.xml", "v1"),
        ("scene_right_cube_orientation.xml", "v2"),
    ],
)
def test_sim_hand_step_matches_raw_mujoco(scene_file: str, version: str) -> None:
    config = SimOrcaHandConfig.from_config_path(scene_file=scene_file, version=version)
    scene_path = resolve_scene_path(scene_file, version=version)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    qpos = _build_valid_qpos(model)
    qvel = np.linspace(-0.1, 0.1, model.nv, dtype=np.float64)
    action = np.linspace(-5.0, 5.0, len(config.joint_ids), dtype=np.float32)

    raw_model, raw_data, _ = _raw_reset(scene_path, config, qpos=qpos, qvel=qvel)
    clipped_action = np.clip(action, np.asarray(config.action_low), np.asarray(config.action_high))
    for actuator_id, value in zip(config.actuator_ids, clipped_action, strict=True):
        raw_data.ctrl[actuator_id] = float(value)
    mujoco.mj_step(raw_model, raw_data, nstep=5)
    raw_obs = np.concatenate([raw_data.qpos.copy(), raw_data.qvel.copy()])

    hand = SimOrcaHand(scene_file=scene_file, version=version)
    try:
        hand.reset(qpos=qpos, qvel=qvel)
        obs = hand.step(action)

        np.testing.assert_allclose(obs, raw_obs)
        np.testing.assert_allclose(hand.data.qpos, raw_data.qpos)
        np.testing.assert_allclose(hand.data.qvel, raw_data.qvel)
        np.testing.assert_allclose(hand.data.ctrl, raw_data.ctrl)
    finally:
        hand.close()


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_shared_base_hand_helpers_work_for_sim_hand(version: str) -> None:
    hand = SimOrcaHand(scene_file="scene_right.xml", version=version)
    try:
        hand.reset()

        target = np.linspace(-0.05, 0.05, len(hand.config.joint_ids), dtype=np.float64)
        hand.set_joint_positions(target)
        typed_joint_positions = hand.get_joint_position()

        assert isinstance(typed_joint_positions, OrcaJointPositions)
        np.testing.assert_allclose(
            typed_joint_positions.as_array(hand.config.joint_ids),
            target,
        )

        hand.set_zero_position()
        np.testing.assert_allclose(
            hand.get_joint_position().as_array(hand.config.joint_ids),
            np.zeros(len(hand.config.joint_ids), dtype=np.float64),
        )

        hand.set_neutral_position()
        np.testing.assert_allclose(
            hand.get_joint_position().as_array(hand.config.joint_ids),
            np.array(
                [hand.config.neutral_position[joint] for joint in hand.config.joint_ids],
                dtype=np.float64,
            ),
        )
    finally:
        hand.close()


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_base_env_delegates_reset_and_step_to_sim_hand(version: str) -> None:
    env = OrcaHandRight(version=version)
    try:
        with patch.object(env.hand, "reset", wraps=env.hand.reset) as hand_reset:
            env.reset()
            hand_reset.assert_called_once()

        with patch.object(env.hand, "step", wraps=env.hand.step) as hand_step:
            env.step(env.action_space.sample())
            hand_step.assert_called_once()
    finally:
        env.close()


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_task_env_delegates_reset_and_step_to_sim_hand(version: str) -> None:
    env = OrcaHandRightCubeOrientation(version=version)
    try:
        with (
            patch.object(env.hand, "reset", wraps=env.hand.reset) as hand_reset,
            patch.object(env.hand, "step", wraps=env.hand.step) as hand_step,
        ):
            env.reset(options={"settle_steps": 2})
            hand_reset.assert_called_once()
            assert hand_step.call_count == 2

        with patch.object(env.hand, "step", wraps=env.hand.step) as hand_step:
            env.step(env.action_space.sample())
            hand_step.assert_called_once()
    finally:
        env.close()
