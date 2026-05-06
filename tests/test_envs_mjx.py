import numpy as np
import pytest


pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")


@pytest.fixture(scope="module")
def jax_module():
    import jax

    return jax


def test_named_env_default_smoke(jax_module):
    from orca_sim.envs_mjx import OrcaHandLeftMjx

    env = OrcaHandLeftMjx()
    obs, info = env.reset()
    assert obs.shape == (1, env.single_observation_space.shape[0])
    assert isinstance(obs, jax_module.Array)
    assert isinstance(info, dict)

    actions = env.action_space.sample()
    assert actions.shape == (1, env.single_action_space.shape[0])

    obs2, reward, terminated, truncated, info2 = env.step(actions)
    assert obs2.shape == obs.shape
    assert reward.shape == (1,)
    assert terminated.shape == (1,)
    assert truncated.shape == (1,)
    assert isinstance(obs2, jax_module.Array)
    assert isinstance(reward, jax_module.Array)
    assert isinstance(terminated, jax_module.Array)
    assert isinstance(truncated, jax_module.Array)
    env.close()


def test_named_env_batched_smoke(jax_module):
    from orca_sim.envs_mjx import OrcaHandLeftMjx

    num_envs = 64
    env = OrcaHandLeftMjx(num_envs=num_envs)
    obs, info = env.reset()
    assert obs.shape == (num_envs, env.single_observation_space.shape[0])
    assert isinstance(obs, jax_module.Array)

    actions = env.action_space.sample()
    assert actions.shape == (num_envs, env.single_action_space.shape[0])

    obs2, reward, terminated, truncated, info2 = env.step(actions)
    assert obs2.shape == obs.shape
    assert reward.shape == (num_envs,)
    assert terminated.shape == (num_envs,)
    assert truncated.shape == (num_envs,)
    assert isinstance(reward, jax_module.Array)
    env.close()


def test_invalid_action_shape():
    from orca_sim.envs_mjx import OrcaHandLeftMjx

    env = OrcaHandLeftMjx(num_envs=2)
    env.reset()
    bad = np.zeros((3, env.single_action_space.shape[0]), dtype=np.float32)
    with pytest.raises(ValueError):
        env.step(bad)
    env.close()
