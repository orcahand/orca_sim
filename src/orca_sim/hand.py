from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from orca_core.base_hand import BaseHand
from orca_core.hand_config import BaseHandConfig
from orca_core.joint_position import OrcaJointPositions

from orca_sim.joint_mapping import resolve_joint_mapping
from orca_sim.versions import resolve_scene_path


@dataclass(frozen=True, kw_only=True)
class SimOrcaHandConfig(BaseHandConfig):
    scene_file: str
    scene_path: str
    version: str
    scene_joint_names: tuple[str, ...]
    actuator_ids: tuple[int, ...]
    actuator_qpos_indices: tuple[int, ...]
    actuator_qvel_indices: tuple[int, ...]
    action_low: tuple[float, ...]
    action_high: tuple[float, ...]

    @classmethod
    def from_config_path(
        cls,
        config_path: str | None = None,
        *,
        scene_file: str = "scene_right.xml",
        version: str | None = None,
        joint_name_to_scene_joint_name: Mapping[str, str] | None = None,
        hand_type: str | None = None,
        model_version: str | None = None,
        model_name: str | None = None,
    ) -> "SimOrcaHandConfig":
        del model_name
        if version is None:
            version = model_version

        if config_path is None:
            scene_path = resolve_scene_path(scene_file, version=version)
        else:
            scene_path = Path(config_path).expanduser().resolve()
            if not scene_path.exists():
                raise FileNotFoundError(f"Scene file not found: {scene_path}")
            scene_file = scene_path.name

        model = mujoco.MjModel.from_xml_path(str(scene_path))

        resolved_hand_type, resolved_mapping = resolve_joint_mapping(
            scene_file=scene_file,
            version=scene_path.parent.name,
            joint_name_to_scene_joint_name=joint_name_to_scene_joint_name,
            hand_type=hand_type,
        )

        actuator_metadata_by_scene_joint: dict[str, tuple[int, int, int, float, float]] = {}
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            joint_name = model.joint(joint_id).name
            if joint_name in actuator_metadata_by_scene_joint:
                raise ValueError(
                    f"Scene joint {joint_name!r} is driven by multiple actuators, "
                    "which SimOrcaHandConfig does not currently support."
                )

            qpos_idx = int(model.jnt_qposadr[joint_id])
            qvel_idx = int(model.jnt_dofadr[joint_id])
            low, high = model.actuator_ctrlrange[actuator_id]
            actuator_metadata_by_scene_joint[joint_name] = (
                actuator_id,
                qpos_idx,
                qvel_idx,
                float(low),
                float(high),
            )

        joint_ids: list[str] = []
        scene_joint_names: list[str] = []
        actuator_ids: list[int] = []
        joint_roms_dict: dict[str, list[float]] = {}
        neutral_position: dict[str, float] = {}
        actuator_qpos_indices: list[int] = []
        actuator_qvel_indices: list[int] = []
        action_low: list[float] = []
        action_high: list[float] = []

        for joint_name, scene_joint_name in resolved_mapping.items():
            try:
                actuator_id, qpos_idx, qvel_idx, low, high = actuator_metadata_by_scene_joint[
                    scene_joint_name
                ]
            except KeyError as exc:
                raise ValueError(
                    f"Scene joint {scene_joint_name!r} mapped from canonical joint "
                    f"{joint_name!r} is not actuator-controlled in {scene_path}."
                ) from exc

            joint_ids.append(joint_name)
            scene_joint_names.append(scene_joint_name)
            actuator_ids.append(actuator_id)
            joint_roms_dict[joint_name] = [low, high]
            neutral_position[joint_name] = float(model.qpos0[qpos_idx])
            actuator_qpos_indices.append(qpos_idx)
            actuator_qvel_indices.append(qvel_idx)
            action_low.append(low)
            action_high.append(high)

        return cls(
            config_path=str(scene_path),
            type=resolved_hand_type,
            joint_ids=joint_ids,
            joint_roms_dict=joint_roms_dict,
            neutral_position=neutral_position,
            scene_file=scene_file,
            scene_path=str(scene_path),
            version=scene_path.parent.name,
            scene_joint_names=tuple(scene_joint_names),
            actuator_ids=tuple(actuator_ids),
            actuator_qpos_indices=tuple(actuator_qpos_indices),
            actuator_qvel_indices=tuple(actuator_qvel_indices),
            action_low=tuple(action_low),
            action_high=tuple(action_high),
        )

    def validate(self) -> None:
        super().validate()

        if not self.scene_file:
            raise ValueError("scene_file must be provided for a simulated hand.")
        if not self.scene_path:
            raise ValueError("scene_path must be provided for a simulated hand.")
        if not self.version:
            raise ValueError("version must be provided for a simulated hand.")

        expected_len = len(self.joint_ids)
        if len(self.scene_joint_names) != expected_len:
            raise ValueError("Each canonical joint must map to exactly one scene joint.")
        if len(self.actuator_ids) != expected_len:
            raise ValueError("Each simulated joint must have an actuator id.")
        if len(self.actuator_qpos_indices) != expected_len:
            raise ValueError("Each simulated joint must have a qpos index.")
        if len(self.actuator_qvel_indices) != expected_len:
            raise ValueError("Each simulated joint must have a qvel index.")
        if len(self.action_low) != expected_len:
            raise ValueError("Each simulated joint must have a lower control bound.")
        if len(self.action_high) != expected_len:
            raise ValueError("Each simulated joint must have an upper control bound.")

    def __post_init__(self) -> None:
        self.validate()

    @property
    def joint_name_to_scene_joint_name(self) -> dict[str, str]:
        return dict(zip(self.joint_ids, self.scene_joint_names, strict=True))

    @property
    def joint_name_to_actuator_id(self) -> dict[str, int]:
        return dict(zip(self.joint_ids, self.actuator_ids, strict=True))

    @property
    def joint_name_to_qpos_idx(self) -> dict[str, int]:
        return dict(zip(self.joint_ids, self.actuator_qpos_indices, strict=True))

    @property
    def joint_name_to_qvel_idx(self) -> dict[str, int]:
        return dict(zip(self.joint_ids, self.actuator_qvel_indices, strict=True))


class SimOrcaHand(BaseHand):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    config_cls = SimOrcaHandConfig

    def __init__(
        self,
        scene_file: str = "scene_right.xml",
        version: str | None = None,
        frame_skip: int = 5,
        render_mode: str | None = None,
        config_path: str | None = None,
        config: SimOrcaHandConfig | None = None,
    ) -> None:
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.frame_skip = frame_skip
        self.render_mode = render_mode

        super().__init__(
            config_path=config_path,
            config=config,
            scene_file=scene_file,
            version=version,
        )

        self.scene_path = Path(self.config.scene_path)
        self.version = self.config.version
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)

        self.action_low = np.asarray(self.config.action_low, dtype=np.float32)
        self.action_high = np.asarray(self.config.action_high, dtype=np.float32)

        self._renderer: mujoco.Renderer | None = None
        self._viewer: Any | None = None

    def _get_joint_positions(self) -> OrcaJointPositions:
        return OrcaJointPositions.from_dict(
            {
                joint_name: float(self.data.qpos[self.config.joint_name_to_qpos_idx[joint_name]])
                for joint_name in self.config.joint_ids
            }
        )

    def _set_joint_positions(self, joint_pos: OrcaJointPositions) -> bool:
        current_joint_positions = self._get_joint_positions().as_dict()
        for joint_name, value in joint_pos:
            if joint_name not in current_joint_positions:
                raise ValueError(f"Unknown canonical joint name: {joint_name}")
            current_joint_positions[joint_name] = float(value)

        for joint_name, value in current_joint_positions.items():
            self.data.qpos[self.config.joint_name_to_qpos_idx[joint_name]] = value
            # Setting a target joint configuration is a teleport-like state edit,
            # so we explicitly zero the corresponding joint velocity.
            self.data.qvel[self.config.joint_name_to_qvel_idx[joint_name]] = 0.0

        ctrl = np.array(
            [current_joint_positions[joint_name] for joint_name in self.config.joint_ids],
            dtype=np.float32,
        )
        self.set_control(ctrl)

        # step simulation forward based on current simulator's state
        mujoco.mj_forward(self.model, self.data)

        if self.render_mode == "human":
            self.render()

        return True

    def observe(self) -> np.ndarray:
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    def _validate_qpos(self, qpos: np.ndarray) -> np.ndarray:
        resolved_qpos = np.asarray(qpos, dtype=np.float64)
        if resolved_qpos.shape != self.data.qpos.shape:
            raise ValueError(
                f"Expected qpos shape {self.data.qpos.shape}, got {resolved_qpos.shape}"
            )
        return resolved_qpos

    def _validate_qvel(self, qvel: np.ndarray) -> np.ndarray:
        resolved_qvel = np.asarray(qvel, dtype=np.float64)
        if resolved_qvel.shape != self.data.qvel.shape:
            raise ValueError(
                f"Expected qvel shape {self.data.qvel.shape}, got {resolved_qvel.shape}"
            )
        return resolved_qvel

    def _validate_ctrl(self, ctrl: np.ndarray) -> np.ndarray:
        resolved_ctrl = np.asarray(ctrl, dtype=np.float32)
        expected_shape = (len(self.config.joint_ids),)
        if resolved_ctrl.shape != expected_shape:
            raise ValueError(f"Expected action shape {expected_shape}, got {resolved_ctrl.shape}")
        return np.clip(resolved_ctrl, self.action_low, self.action_high)

    def set_state(
        self,
        *,
        qpos: np.ndarray | None = None,
        qvel: np.ndarray | None = None,
        ctrl: np.ndarray | None = None,
        forward: bool = True,
    ) -> np.ndarray:
        if qpos is not None:
            self.data.qpos[:] = self._validate_qpos(qpos)

        if qvel is not None:
            self.data.qvel[:] = self._validate_qvel(qvel)

        if ctrl is not None:
            self.set_control(ctrl)

        if forward:
            self.forward()

        return self.observe()

    def set_control(self, ctrl: np.ndarray) -> np.ndarray:
        resolved_ctrl = self._validate_ctrl(ctrl)
        for actuator_id, value in zip(
            self.config.actuator_ids,
            resolved_ctrl,
            strict=True,
        ):
            self.data.ctrl[actuator_id] = float(value)
        return self.data.ctrl[list(self.config.actuator_ids)].copy()

    def forward(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        if self.render_mode == "human":
            self.render()
        
        return self.observe()

    def reset(
        self,
        *,
        qpos: np.ndarray | None = None,
        qvel: np.ndarray | None = None,
        ctrl: np.ndarray | None = None,
    ) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        default_ctrl = self._get_joint_positions().as_array(self.config.joint_ids).astype(
            np.float32
        )
        return self.set_state(
            qpos=qpos,
            qvel=qvel,
            ctrl=default_ctrl if ctrl is None else ctrl,
            forward=True,
        )

    def step(self, action: np.ndarray | None = None, *, nstep: int | None = None) -> np.ndarray:
        if action is not None:
            self.set_control(action)  # mujoco simulation is stateful - this updates the state for stepping

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip if nstep is None else nstep)

        if self.render_mode == "human":
            self.render()

        return self.observe()

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer

                try:
                    self._viewer = viewer.launch_passive(self.model, self.data)
                except RuntimeError as exc:
                    if sys.platform == "darwin" and "mjpython" in str(exc):
                        raise RuntimeError(
                            "On macOS, MuJoCo human rendering must be launched with "
                            "`mjpython`, not plain `python3`."
                        ) from exc
                    raise
                mujoco.mjv_defaultFreeCamera(self.model, self._viewer.cam)
            self._viewer.sync()
            return None

        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
