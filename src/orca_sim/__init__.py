from orca_sim.versions import (
    latest_version,
    list_versions,
)
from orca_sim.registry import register_envs

try:
    from orca_sim.envs import (
        OrcaHandCombined,
        OrcaHandCombinedExtended,
        OrcaHandLeft,
        OrcaHandLeftExtended,
        OrcaHandRight,
        OrcaHandRightExtended,
    )
    from orca_sim.hand import SimOrcaHand, SimOrcaHandConfig
    from orca_sim.task_envs import (
        CubeStackingTabletop,
        OrcaArmCubeStacking,
        OrcaHandRightCubeOrientation,
        OrcaPandaCubeStacking,
    )
except ModuleNotFoundError as exc:
    if exc.name != "mujoco":
        raise

    CubeStackingTabletop = None
    OrcaArmCubeStacking = None
    OrcaHandCombined = None
    OrcaHandCombinedExtended = None
    OrcaHandLeft = None
    OrcaHandLeftExtended = None
    OrcaHandRight = None
    OrcaHandRightCubeOrientation = None
    OrcaHandRightExtended = None
    OrcaPandaCubeStacking = None
    SimOrcaHand = None
    SimOrcaHandConfig = None

__all__ = [
    "CubeStackingTabletop",
    "OrcaArmCubeStacking",
    "OrcaHandCombined",
    "OrcaHandCombinedExtended",
    "OrcaHandLeft",
    "OrcaHandLeftExtended",
    "OrcaHandRight",
    "OrcaHandRightCubeOrientation",
    "OrcaHandRightExtended",
    "OrcaPandaCubeStacking",
    "SimOrcaHand",
    "SimOrcaHandConfig",
    "latest_version",
    "list_versions",
    "register_envs",
]
