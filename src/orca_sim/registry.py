import gymnasium as gym

from orca_sim.versions import list_versions, resolve_scene_path


def register_envs() -> None:
    registry = gym.registry
    specs = {
        "OrcaHandLeft": ("orca_sim.envs:OrcaHandLeft", "scene_left.xml"),
        "OrcaHandLeftExtended": (
            "orca_sim.envs:OrcaHandLeftExtended",
            "scene_left_extended.xml",
        ),
        "OrcaHandRight": ("orca_sim.envs:OrcaHandRight", "scene_right.xml"),
        "OrcaHandRightExtended": (
            "orca_sim.envs:OrcaHandRightExtended",
            "scene_right_extended.xml",
        ),
        "OrcaHandCombined": ("orca_sim.envs:OrcaHandCombined", "scene_combined.xml"),
        "OrcaHandCombinedExtended": (
            "orca_sim.envs:OrcaHandCombinedExtended",
            "scene_combined_extended.xml",
        ),
        "OrcaHandRightCubeOrientation": (
            "orca_sim.task_envs:OrcaHandRightCubeOrientation",
            "scene_right_cube_orientation.xml",
        ),
    }
    for version in list_versions():
        for env_name, (entry_point, scene_file) in specs.items():
            try:
                resolve_scene_path(scene_file, version=version)
            except FileNotFoundError:
                continue

            env_id = f"{env_name}-{version}"
            if env_id not in registry:
                gym.register(
                    id=env_id,
                    entry_point=entry_point,
                    kwargs={"version": version},
                )
