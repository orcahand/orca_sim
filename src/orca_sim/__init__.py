from orca_sim.versions import (
    latest_version,
    list_versions,
)
from orca_sim.envs import (
    OrcaHandCombined,
    OrcaHandCombinedExtended,
    OrcaHandLeft,
    OrcaHandLeftExtended,
    OrcaHandRight,
    OrcaHandRightExtended,
)
from orca_sim.registry import register_envs
from orca_sim.task_envs import OrcaHandRightCubeOrientation

__all__ = [
    "OrcaHandCombined",
    "OrcaHandCombinedExtended",
    "OrcaHandLeft",
    "OrcaHandLeftExtended",
    "OrcaHandRight",
    "OrcaHandRightCubeOrientation",
    "OrcaHandRightExtended",
    "latest_version",
    "list_versions",
    "register_envs",
]

try:
    from orca_sim.envs_mjx import (
        BaseOrcaHandMjxEnv,
        OrcaHandCombinedExtendedMjx,
        OrcaHandCombinedMjx,
        OrcaHandLeftExtendedMjx,
        OrcaHandLeftMjx,
        OrcaHandRightExtendedMjx,
        OrcaHandRightMjx,
    )
    from orca_sim.task_envs_mjx import OrcaHandRightCubeOrientationMjx
except ImportError:
    pass
else:
    __all__ += [
        "BaseOrcaHandMjxEnv",
        "OrcaHandCombinedExtendedMjx",
        "OrcaHandCombinedMjx",
        "OrcaHandLeftExtendedMjx",
        "OrcaHandLeftMjx",
        "OrcaHandRightCubeOrientationMjx",
        "OrcaHandRightExtendedMjx",
        "OrcaHandRightMjx",
    ]
