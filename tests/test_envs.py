import numpy as np
import pytest

from orca_sim import OrcaHandCombined, OrcaHandLeft, OrcaHandRight


@pytest.mark.parametrize(
    ("env_cls", "obs_size", "action_size"),
    [
        (OrcaHandLeft, 34, 17),
        (OrcaHandRight, 34, 17),
        (OrcaHandCombined, 68, 34),
    ],
)
def test_env_reset_and_step_smoke(env_cls, obs_size: int, action_size: int) -> None:
    env = env_cls()
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


def test_step_clips_actions_to_actuator_limits() -> None:
    env = OrcaHandLeft()
    try:
        env.reset()
        env.step(env.action_high + 10.0)
        np.testing.assert_allclose(env.data.ctrl, env.action_high)
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
