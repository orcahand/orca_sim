import gymnasium as gym


def register_envs() -> None:
    registry = gym.registry
    specs = {
        "OrcaHandLeft-v1": "orca_sim.envs:OrcaHandLeft",
        "OrcaHandLeftExtended-v1": "orca_sim.envs:OrcaHandLeftExtended",
        "OrcaHandRight-v1": "orca_sim.envs:OrcaHandRight",
        "OrcaHandRightExtended-v1": "orca_sim.envs:OrcaHandRightExtended",
        "OrcaHandCombined-v1": "orca_sim.envs:OrcaHandCombined",
        "OrcaHandCombinedExtended-v1": "orca_sim.envs:OrcaHandCombinedExtended",
    }
    for env_id, entry_point in specs.items():
        if env_id not in registry:
            gym.register(id=env_id, entry_point=entry_point)
