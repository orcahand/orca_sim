import numpy as np
import pytest


pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")


@pytest.fixture(scope="module")
def jax_module():
    import jax

    return jax


def test_single_env_smoke(jax_module):
    from orca_sim.envs_mjx import OrcaHandLeftMjx

    env = OrcaHandLeftMjx()
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)

    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert obs2.shape == obs.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    env.close()


def test_single_env_invalid_action_shape():
    from orca_sim.envs_mjx import OrcaHandLeftMjx

    env = OrcaHandLeftMjx()
    env.reset()
    bad = np.zeros(env.action_space.shape[0] + 1, dtype=np.float32)
    with pytest.raises(ValueError):
        env.step(bad)
    env.close()


def test_vector_env_smoke(jax_module):
    from orca_sim.envs_mjx import OrcaHandMjxVectorEnv

    num_envs = 4
    env = OrcaHandMjxVectorEnv("scene_left.xml", num_envs=num_envs)
    obs, info = env.reset()
    assert obs.shape == (num_envs, env.single_observation_space.shape[0])

    actions = env.action_space.sample()
    assert actions.shape == (num_envs, env.single_action_space.shape[0])

    obs2, rewards, terminateds, truncateds, infos = env.step(actions)
    assert obs2.shape == obs.shape
    assert rewards.shape == (num_envs,)
    assert terminateds.shape == (num_envs,)
    assert truncateds.shape == (num_envs,)
    env.close()


def test_vector_env_invalid_action_shape():
    from orca_sim.envs_mjx import OrcaHandMjxVectorEnv

    env = OrcaHandMjxVectorEnv("scene_left.xml", num_envs=2)
    env.reset()
    bad = np.zeros((3, env.single_action_space.shape[0]), dtype=np.float32)
    with pytest.raises(ValueError):
        env.step(bad)
    env.close()
