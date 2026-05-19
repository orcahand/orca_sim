import numpy as np
import pytest

from orca_core.base_hand import BaseHand

from orca_sim import OrcaHandCombined, OrcaHandLeft, OrcaHandRight, SimOrcaHand


@pytest.mark.parametrize(
    ("env_cls", "obs_size", "action_size", "version"),
    [
        (OrcaHandLeft, 34, 17, "v1"),
        (OrcaHandLeft, 34, 17, "v2"),
        (OrcaHandRight, 34, 17, "v1"),
        (OrcaHandRight, 34, 17, "v2"),
        (OrcaHandCombined, 68, 34, "v1"),
        (OrcaHandCombined, 68, 34, "v2"),
    ],
)
def test_env_reset_and_step_smoke(
    env_cls, obs_size: int, action_size: int, version: str
) -> None:
    env = env_cls(version=version)
    try:
        obs, info = env.reset()

        assert obs.shape == (obs_size,)
        assert info == {}
        assert env.action_space.shape == (action_size,)

        next_obs, reward, terminated, truncated, next_info = env.step(
            env.action_space.sample()
        )

        assert next_obs.shape == (obs_size,), "Next observation shape is not correct"
        assert isinstance(reward, float), "Reward is not a float"
        assert isinstance(terminated, bool), "Terminated is not a bool"
        assert isinstance(truncated, bool), "Truncated is not a bool"
        assert isinstance(next_info, dict), "Next info is not a dict"
    finally:
        env.close()


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_envs_compose_shared_sim_hand_backend(version: str) -> None:
    env = OrcaHandRight(version=version)
    try:
        assert isinstance(env.hand, SimOrcaHand)
        assert isinstance(env.hand, BaseHand)
        assert env.hand.version == version
        assert env.hand.scene_path == env.scene_path
    finally:
        env.close()


def test_reset_accepts_explicit_qpos_and_qvel() -> None:
    env = OrcaHandRight()
    try:
        qpos = np.linspace(-0.1, 0.1, env.data.qpos.size)
        qvel = np.linspace(-0.2, 0.2, env.data.qvel.size)

        obs, _ = env.reset(options={"qpos": qpos, "qvel": qvel})

        np.testing.assert_allclose(env.data.qpos, qpos)
        np.testing.assert_allclose(env.data.qvel, qvel)
        np.testing.assert_allclose(obs[: env.data.qpos.size], qpos)
        np.testing.assert_allclose(obs[env.data.qpos.size :], qvel)
    finally:
        env.close()


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_sim_hand_clamps_joint_commands(version: str) -> None:
    hand = SimOrcaHand(scene_file="scene_right.xml", version=version)
    try:
        hand.reset()
        joint_name = hand.config.joint_ids[0]
        hand.set_joint_positions({joint_name: hand.action_high[0] + 10.0})
        joint_positions = hand.get_joint_position().as_dict()

        assert joint_positions[joint_name] == pytest.approx(hand.action_high[0])
    finally:
        hand.close()


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_sim_hand_preserves_unspecified_joint_commands(version: str) -> None:
    hand = SimOrcaHand(scene_file="scene_right.xml", version=version)
    try:
        hand.reset()
        first_joint, second_joint = hand.config.joint_ids[:2]

        hand.set_joint_positions(
            {
                first_joint: float(hand.action_low[0]),
                second_joint: float(hand.action_high[1]),
            }
        )
        hand.set_joint_positions({second_joint: 0.0})

        joint_positions = hand.get_joint_position().as_dict()
        assert joint_positions[first_joint] == pytest.approx(hand.action_low[0])
        assert joint_positions[second_joint] == pytest.approx(0.0)
    finally:
        hand.close()


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_sim_hand_steps_simulation_without_gym(version: str) -> None:
    hand = SimOrcaHand(scene_file="scene_right.xml", version=version)
    try:
        obs = hand.reset()
        next_obs = hand.step(hand.action_high + 10.0)  # overwrap all motors

        assert obs.shape == next_obs.shape == (hand.data.qpos.size + hand.data.qvel.size,)
        np.testing.assert_allclose(hand.data.ctrl[list(hand.config.actuator_ids)], hand.action_high)
    finally:
        hand.close()


def test_step_clips_actions_to_actuator_limits() -> None:
    env = OrcaHandLeft()
    try:
        env.reset()
        env.step(env.action_high + 10.0)
        # indexing of joint ids is specified by the hand config
        np.testing.assert_allclose(env.data.ctrl[list(env.hand.config.actuator_ids)], env.action_high)
    finally:
        env.close()


def test_invalid_render_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported render_mode"):
        OrcaHandRight(render_mode="wireframe")


def test_reset_rejects_wrong_qpos_shape() -> None:
    env = OrcaHandRight()
    try:
        wrong_shape = np.zeros(env.data.qpos.size + 1)
        with pytest.raises(ValueError, match="Expected qpos shape"):
            env.reset(options={"qpos": wrong_shape})
    finally:
        env.close()


def test_step_rejects_wrong_action_shape() -> None:
    env = OrcaHandRight()
    try:
        env.reset()
        wrong_shape = np.zeros(env.action_space.shape[0] + 1, dtype=np.float32)
        with pytest.raises(ValueError, match="Expected action shape"):
            env.step(wrong_shape)
    finally:
        env.close()


