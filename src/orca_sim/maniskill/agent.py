"""ManiSkill agent definition for the orca_arm bimanual robot.

Registers orca_arm under uid="orca_arm" with a single PD-joint-position
controller covering every movable joint in orca_arm's URDF. Picks the
right hand's wrist tower as the TCP for downstream task code.

Joint groups are derived at import time from ``orca_arm.URDF_PATH`` by
splitting movable joint names by prefix, so adding/removing joints
upstream is reflected here automatically.

Controller gains and force limits are read from ``orca_arm.MJCF_PATH`` so
the ManiSkill controller mirrors the MuJoCo position actuator defaults.
"""
import xml.etree.ElementTree as ET
from copy import deepcopy

import numpy as np
import orca_arm
import sapien

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils


_JOINT_PREFIXES = (
    "openarm_left_",
    "openarm_right_",
    "orcahand_left_",
    "orcahand_right_",
)


def _float_attr(element: ET.Element, attr: str, context: str) -> float:
    value = element.get(attr)
    if value is None:
        raise RuntimeError(f"Expected {context} to define {attr!r}.")
    return float(value)


def _symmetric_force_limit(position: ET.Element, context: str) -> float:
    value = position.get("forcerange")
    if value is None:
        raise RuntimeError(f"Expected {context} to define 'forcerange'.")

    low, high = (float(v) for v in value.split())
    if not np.isclose(abs(low), abs(high)):
        raise RuntimeError(
            f"Expected {context} forcerange to be symmetric, got {value!r}."
        )
    return abs(high)


def _pd_position_defaults(mjcf_path: str, default_class: str) -> dict[str, float]:
    """Read ManiSkill PD position controller values from MJCF defaults."""
    context = f"orca_arm MJCF default class {default_class!r}"
    default = ET.parse(mjcf_path).getroot().find(
        f".//default[@class='{default_class}']"
    )
    if default is None:
        raise RuntimeError(f"Expected {context} to exist.")

    position = default.find("position")
    if position is None:
        raise RuntimeError(f"Expected {context} to define a position actuator.")

    return {
        "stiffness": _float_attr(position, "kp", context),
        "damping": _float_attr(position, "kv", context),
        "force_limit": _symmetric_force_limit(position, context),
    }


def _group_movable_joints(urdf_path: str) -> dict[str, list[str]]:
    """Group movable joints in the URDF by name prefix, preserving URDF order.

    Order within each group = order of appearance in the URDF, which sets the
    layout of the controller's action vector. Fails loudly if a movable joint
    doesn't match any known prefix, so upstream additions can't silently drop
    out of the action space.
    """
    groups: dict[str, list[str]] = {p: [] for p in _JOINT_PREFIXES}
    for joint in ET.parse(urdf_path).getroot().findall("joint"):
        if joint.get("type") == "fixed":
            continue
        name = joint.get("name")
        for prefix in _JOINT_PREFIXES:
            if name.startswith(prefix):
                groups[prefix].append(name)
                break
        else:
            raise RuntimeError(
                f"orca_arm URDF joint {name!r} matches no known prefix "
                f"{_JOINT_PREFIXES}; update orca_sim agent groupings."
            )
    return groups


_GROUPS = _group_movable_joints(orca_arm.URDF_PATH)
LEFT_ARM_JOINTS = _GROUPS["openarm_left_"]
RIGHT_ARM_JOINTS = _GROUPS["openarm_right_"]
LEFT_HAND_JOINTS = _GROUPS["orcahand_left_"]
RIGHT_HAND_JOINTS = _GROUPS["orcahand_right_"]

ALL_ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
ALL_HAND_JOINTS = LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS
ALL_JOINTS = ALL_ARM_JOINTS + ALL_HAND_JOINTS

ARM_CONTROLLER_DEFAULTS = _pd_position_defaults(orca_arm.MJCF_PATH, "arm_joint")
HAND_CONTROLLER_DEFAULTS = _pd_position_defaults(orca_arm.MJCF_PATH, "hand_joint")


def _carpals_link(urdf_path: str, prefix: str) -> str:
    """The unique ``Carpals`` link for a given hand — the wrist/palm body."""
    matches = [
        link.get("name")
        for link in ET.parse(urdf_path).getroot().findall("link")
        if link.get("name", "").startswith(prefix) and "Carpals" in link.get("name", "")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one 'Carpals' link with prefix {prefix!r}, found {matches}."
        )
    return matches[0]


# Wrist link of the right hand — used as the TCP for tasks that need a
# single end-effector reference point.
RIGHT_TCP_LINK = _carpals_link(orca_arm.URDF_PATH, "orcahand_right_")
LEFT_TCP_LINK = _carpals_link(orca_arm.URDF_PATH, "orcahand_left_")

@register_agent()
class OrcaArm(BaseAgent):
    uid = "orca_arm"
    urdf_path = orca_arm.URDF_PATH

    # NOTE: need an update to the OrcaArm with updated STLs
    # to remove this disabling of self-collisions (see v0.0.0-release
    # notes of OrcaArm)
    disable_self_collisions = True

    arm_stiffness = ARM_CONTROLLER_DEFAULTS["stiffness"]
    arm_damping = ARM_CONTROLLER_DEFAULTS["damping"]
    arm_force_limit = ARM_CONTROLLER_DEFAULTS["force_limit"]

    hand_stiffness = HAND_CONTROLLER_DEFAULTS["stiffness"]
    hand_damping = HAND_CONTROLLER_DEFAULTS["damping"]
    hand_force_limit = HAND_CONTROLLER_DEFAULTS["force_limit"]

    keyframes = dict(
        rest=Keyframe(
            qpos=np.zeros(len(ALL_JOINTS)),
            pose=sapien.Pose(),
        )
    )

    @property
    def _controller_configs(self):
        arm = PDJointPosControllerConfig(
            ALL_ARM_JOINTS,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        hand = PDJointPosControllerConfig(
            ALL_HAND_JOINTS,
            lower=None,
            upper=None,
            stiffness=self.hand_stiffness,
            damping=self.hand_damping,
            force_limit=self.hand_force_limit,
            normalize_action=False,
        )
        configs = dict(
            pd_joint_pos=dict(arm=arm, hand=hand),
        )
        return deepcopy(configs)

    def _after_init(self):
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), RIGHT_TCP_LINK
        )

    @property
    def tcp_pose(self):
        return self.tcp.pose
