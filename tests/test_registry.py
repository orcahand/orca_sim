import gymnasium as gym

from orca_sim import register_envs


def test_register_envs_is_idempotent_and_envs_can_be_made() -> None:
    register_envs()
    register_envs()

    for env_id, obs_shape, action_shape in [
        ("OrcaHandLeft-v1", (34,), (17,)),
        ("OrcaHandRight-v1", (34,), (17,)),
        ("OrcaHandCombined-v1", (68,), (34,)),
    ]:
        assert env_id in gym.registry

        env = gym.make(env_id)
        try:
            obs, info = env.reset()

            assert obs.shape == obs_shape
            assert env.action_space.shape == action_shape
            assert info == {}
        finally:
            env.close()
