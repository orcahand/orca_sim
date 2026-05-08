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


def test_cube_orientation_mjx_smoke(jax_module):
    from orca_sim.task_envs_mjx import OrcaHandRightCubeOrientationMjx

    num_envs = 2
    env = OrcaHandRightCubeOrientationMjx(num_envs=num_envs)
    obs, info = env.reset()
    assert obs.shape == (num_envs, env.single_observation_space.shape[0])
    for key in (
        "cube_pos",
        "cube_quat",
        "cube_qvel",
        "red_face_world_normal",
        "red_face_up_alignment",
        "red_face_up_angle_rad",
        "is_success",
        "dropped",
        "elapsed_steps",
    ):
        assert key in info, f"missing info key: {key}"
    assert info["cube_pos"].shape == (num_envs, 3)
    assert info["cube_quat"].shape == (num_envs, 4)
    assert info["red_face_up_alignment"].shape == (num_envs,)

    for _ in range(5):
        actions = env.action_space.sample()
        obs2, reward, terminated, truncated, info2 = env.step(actions)
    assert obs2.shape == obs.shape
    assert reward.shape == (num_envs,)
    assert terminated.shape == (num_envs,)
    assert truncated.shape == (num_envs,)
    assert isinstance(reward, jax_module.Array)
    env.close()


def test_cube_orientation_mjx_reset_options(jax_module):
    from orca_sim.task_envs_mjx import OrcaHandRightCubeOrientationMjx

    num_envs = 3
    env = OrcaHandRightCubeOrientationMjx(num_envs=num_envs)

    # Per-env cube_quat override: each env starts with a different orientation.
    per_env_quat = np.tile(
        OrcaHandRightCubeOrientationMjx.RED_DOWN_QUAT, (num_envs, 1)
    )
    per_env_quat[1] = np.array([1.0, 0.0, 0.0, 0.0])  # identity → red face up
    obs, info = env.reset(options={"cube_quat": per_env_quat})
    alignments = np.asarray(info["red_face_up_alignment"])
    assert alignments[0] < 0.0  # red face down
    assert alignments[1] > 0.99  # red face up (identity quat)

    # Broadcast cube_quat: single (4,) is broadcast across all envs.
    obs, info = env.reset(
        options={"cube_quat": np.array([1.0, 0.0, 0.0, 0.0])}
    )
    alignments = np.asarray(info["red_face_up_alignment"])
    assert np.all(alignments > 0.99)
    env.close()
