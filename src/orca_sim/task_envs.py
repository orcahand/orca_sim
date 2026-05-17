from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from orca_sim.envs import BaseOrcaHandEnv
from orca_sim.builders.orcaarm_camera_mjcf import (
    camera_names as default_orcaarm_camera_names,
)
from orca_sim.versions import SCENES_ROOT, resolve_scene_path


class CubeStackingTabletop(gym.Env[np.ndarray, np.ndarray]):
    """Robot-free MuJoCo tabletop scene with two randomly placed stacking cubes."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    CUBE_NAMES = ("red_cube", "blue_cube")
    CUBE_JOINT_NAMES = ("red_cube_freejoint", "blue_cube_freejoint")
    TARGET_BODY_NAME = "stack_target"
    DEFAULT_WORKSPACE_BOUNDS = np.array(
        [[0.38, 0.58], [-0.22, 0.22]],
        dtype=np.float64,
    )

    def __init__(
        self,
        render_mode: str | None = None,
        version: str | None = None,
        *,
        scene_file: str = "cube_stacking.xml",
        frame_skip: int = 5,
        workspace_bounds: np.ndarray | list[list[float]] | tuple[tuple[float, float], tuple[float, float]] = DEFAULT_WORKSPACE_BOUNDS,
        min_cube_spacing: float = 0.09,
        target_bounds: np.ndarray | list[list[float]] | tuple[tuple[float, float], tuple[float, float]] | None = None,
        min_target_cube_spacing: float = 0.09,
        randomize_yaw: bool = True,
        render_camera: str = "topdown",
        top_cube_name: str = "red_cube",
        base_cube_name: str = "blue_cube",
        cube_size: float = 0.05,
        stack_xy_tolerance: float = 0.025,
        stack_height_tolerance: float = 0.012,
        target_xy_tolerance: float = 0.025,
        settle_velocity_tolerance: float = 0.05,
    ) -> None:
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        if top_cube_name not in self.CUBE_NAMES:
            raise ValueError(f"Unknown top_cube_name: {top_cube_name!r}")
        if base_cube_name not in self.CUBE_NAMES:
            raise ValueError(f"Unknown base_cube_name: {base_cube_name!r}")
        if top_cube_name == base_cube_name:
            raise ValueError("top_cube_name and base_cube_name must differ.")

        super().__init__()
        self.scene_file = scene_file
        self.scene_path = resolve_scene_path(scene_file, version=version)
        self.version = None if self.scene_path.parent == SCENES_ROOT else self.scene_path.parent.name
        self.frame_skip = int(frame_skip)
        self.render_mode = render_mode
        self.workspace_bounds = self._validate_workspace_bounds(workspace_bounds)
        self.min_cube_spacing = float(min_cube_spacing)
        self.target_bounds = self._validate_workspace_bounds(
            self.workspace_bounds if target_bounds is None else target_bounds
        )
        self.min_target_cube_spacing = float(min_target_cube_spacing)
        self.randomize_yaw = bool(randomize_yaw)
        self.render_camera = render_camera
        self.top_cube_name = top_cube_name
        self.base_cube_name = base_cube_name
        self.cube_size = float(cube_size)
        self.stack_xy_tolerance = float(stack_xy_tolerance)
        self.stack_height_tolerance = float(stack_height_tolerance)
        self.target_xy_tolerance = float(target_xy_tolerance)
        self.settle_velocity_tolerance = float(settle_velocity_tolerance)

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self._renderer: mujoco.Renderer | None = None
        self._viewer: Any | None = None

        self._cube_qpos_adrs = {
            cube_name: int(self.model.jnt_qposadr[self.model.joint(joint_name).id])
            for cube_name, joint_name in zip(
                self.CUBE_NAMES,
                self.CUBE_JOINT_NAMES,
                strict=True,
            )
        }
        self._cube_qvel_adrs = {
            cube_name: int(self.model.jnt_dofadr[self.model.joint(joint_name).id])
            for cube_name, joint_name in zip(
                self.CUBE_NAMES,
                self.CUBE_JOINT_NAMES,
                strict=True,
            )
        }
        self._default_cube_qpos = {
            cube_name: self.model.qpos0[qpos_adr : qpos_adr + 7].copy()
            for cube_name, qpos_adr in self._cube_qpos_adrs.items()
        }
        self._target_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.TARGET_BODY_NAME,
        )
        if self._target_body_id < 0:
            raise ValueError(
                f"Scene {self.scene_path} is missing target body {self.TARGET_BODY_NAME!r}."
            )
        self._target_mocap_id = int(self.model.body_mocapid[self._target_body_id])
        if self._target_mocap_id < 0:
            raise ValueError(
                f"Target body {self.TARGET_BODY_NAME!r} must be a mocap body."
            )
        self._default_target_pos = self.model.body_pos[self._target_body_id].copy()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(0,), dtype=np.float32)
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float64,
        )

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    def _get_info(self) -> dict[str, Any]:
        cube_pos, cube_quat, cube_qvel = self._cube_state()
        success_info = self._stack_success_info(cube_pos, cube_qvel)
        return {
            "cube_pos": cube_pos,
            "cube_quat": cube_quat,
            "cube_qvel": cube_qvel,
            "target_pos": self._target_pos().copy(),
            "top_cube": self.top_cube_name,
            "base_cube": self.base_cube_name,
            **success_info,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = {} if options is None else dict(options)
        mujoco.mj_resetData(self.model, self.data)

        if "qpos" in options:
            qpos = np.asarray(options["qpos"], dtype=np.float64)
            if qpos.shape != self.data.qpos.shape:
                raise ValueError(
                    f"Expected qpos shape {self.data.qpos.shape}, got {qpos.shape}"
                )
            self.data.qpos[:] = qpos
            cube_positions = self._cube_positions_from_qpos(self.data.qpos)
        else:
            qpos = self.model.qpos0.copy()
            cube_positions = self._reset_cubes_from_options(options, qpos=qpos)
            self.data.qpos[:] = qpos

        self._reset_target_from_options(options, cube_positions)

        if "qvel" in options:
            qvel = np.asarray(options["qvel"], dtype=np.float64)
            if qvel.shape != self.data.qvel.shape:
                raise ValueError(
                    f"Expected qvel shape {self.data.qvel.shape}, got {qvel.shape}"
                )
            self.data.qvel[:] = qvel
        else:
            self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        settle_steps = int(options.get("settle_steps", 0))
        if settle_steps > 0:
            mujoco.mj_step(self.model, self.data, nstep=settle_steps)

        return self._get_obs(), self._get_info()

    def step(
        self,
        action: np.ndarray | None,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if action is not None:
            resolved_action = np.asarray(action, dtype=np.float32)
            if resolved_action.shape != self.action_space.shape:
                raise ValueError(
                    f"Expected action shape {self.action_space.shape}, got {resolved_action.shape}"
                )

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        info = self._get_info()
        return self._get_obs(), float(info["is_success"]), bool(info["is_success"]), False, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model)
            self._renderer.update_scene(self.data, camera=self.render_camera)
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

    def _cube_state(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        cube_pos = {
            cube_name: self._cube_qpos(cube_name)[:3].copy()
            for cube_name in self.CUBE_NAMES
        }
        cube_quat = {
            cube_name: self._cube_qpos(cube_name)[3:7].copy()
            for cube_name in self.CUBE_NAMES
        }
        cube_qvel = {
            cube_name: self.data.qvel[qvel_adr : qvel_adr + 6].copy()
            for cube_name, qvel_adr in self._cube_qvel_adrs.items()
        }
        return cube_pos, cube_quat, cube_qvel

    def _reset_cubes_from_options(
        self,
        options: Mapping[str, Any],
        *,
        qpos: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        cube_positions = options.get("cube_positions")
        if cube_positions is None:
            cube_positions = self._sample_cube_positions()
        cube_quats = options.get("cube_quats", {})
        resolved_positions: dict[str, np.ndarray] = {}

        target_qpos = self.data.qpos if qpos is None else qpos
        for cube_name in self.CUBE_NAMES:
            cube_qpos = self._default_cube_qpos[cube_name].copy()
            if cube_name in cube_positions:
                cube_pos = np.asarray(cube_positions[cube_name], dtype=np.float64)
                if cube_pos.shape != (3,):
                    raise ValueError(
                        f"Expected {cube_name} position shape (3,), got {cube_pos.shape}"
                    )
                cube_qpos[:3] = cube_pos
            if cube_name in cube_quats:
                cube_qpos[3:7] = self._normalize_quat(
                    np.asarray(cube_quats[cube_name], dtype=np.float64)
                )
            elif self.randomize_yaw:
                cube_qpos[3:7] = self._sample_yaw_quat()

            qpos_adr = self._cube_qpos_adrs[cube_name]
            target_qpos[qpos_adr : qpos_adr + 7] = cube_qpos
            resolved_positions[cube_name] = cube_qpos[:3].copy()

        return resolved_positions

    def _reset_target_from_options(
        self,
        options: Mapping[str, Any],
        cube_positions: Mapping[str, np.ndarray],
    ) -> None:
        target_position = options.get("target_position")
        if target_position is None:
            target_position = self._sample_target_position(cube_positions)
        target_pos = np.asarray(target_position, dtype=np.float64)
        if target_pos.shape != (3,):
            raise ValueError(f"Expected target_position shape (3,), got {target_pos.shape}")
        self.data.mocap_pos[self._target_mocap_id] = target_pos
        self.data.mocap_quat[self._target_mocap_id] = np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )

    def _sample_cube_positions(self) -> dict[str, np.ndarray]:
        sampled: list[np.ndarray] = []
        z_by_cube = {
            cube_name: self._default_cube_qpos[cube_name][2]
            for cube_name in self.CUBE_NAMES
        }
        for cube_name in self.CUBE_NAMES:
            for _ in range(200):
                xy = self.np_random.uniform(
                    low=self.workspace_bounds[:, 0],
                    high=self.workspace_bounds[:, 1],
                )
                candidate = np.array([xy[0], xy[1], z_by_cube[cube_name]], dtype=np.float64)
                if all(
                    np.linalg.norm(candidate[:2] - existing[:2]) >= self.min_cube_spacing
                    for existing in sampled
                ):
                    sampled.append(candidate)
                    break
            else:
                raise RuntimeError(
                    "Unable to sample non-overlapping cube positions in the tabletop workspace."
                )
        return dict(zip(self.CUBE_NAMES, sampled, strict=True))

    def _sample_target_position(
        self,
        cube_positions: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        cube_xy = [np.asarray(pos, dtype=np.float64)[:2] for pos in cube_positions.values()]
        for _ in range(200):
            xy = self.np_random.uniform(
                low=self.target_bounds[:, 0],
                high=self.target_bounds[:, 1],
            )
            if all(
                np.linalg.norm(xy - existing_xy) >= self.min_target_cube_spacing
                for existing_xy in cube_xy
            ):
                return np.array(
                    [xy[0], xy[1], self._default_target_pos[2]],
                    dtype=np.float64,
                )
        raise RuntimeError(
            "Unable to sample a target position away from cubes in the tabletop workspace."
        )

    def _cube_positions_from_qpos(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
        return {
            cube_name: qpos[qpos_adr : qpos_adr + 3].copy()
            for cube_name, qpos_adr in self._cube_qpos_adrs.items()
        }

    def _target_pos(self) -> np.ndarray:
        return self.data.mocap_pos[self._target_mocap_id]

    def _stack_success_info(
        self,
        cube_pos: Mapping[str, np.ndarray],
        cube_qvel: Mapping[str, np.ndarray],
    ) -> dict[str, bool]:
        top = cube_pos[self.top_cube_name]
        base = cube_pos[self.base_cube_name]
        top_vel = cube_qvel[self.top_cube_name][:3]
        base_vel = cube_qvel[self.base_cube_name][:3]
        target = self._target_pos()

        xy_close = np.linalg.norm(top[:2] - base[:2]) < self.stack_xy_tolerance
        height_ok = abs((top[2] - base[2]) - self.cube_size) < self.stack_height_tolerance
        target_xy_close = np.linalg.norm(base[:2] - target[:2]) < self.target_xy_tolerance
        settled = (
            np.linalg.norm(top_vel) < self.settle_velocity_tolerance
            and np.linalg.norm(base_vel) < self.settle_velocity_tolerance
        )
        is_success = bool(xy_close and height_ok and target_xy_close and settled)

        return {
            "xy_close": bool(xy_close),
            "height_ok": bool(height_ok),
            "target_xy_close": bool(target_xy_close),
            "settled": bool(settled),
            "is_success": is_success,
        }

    def _sample_yaw_quat(self) -> np.ndarray:
        yaw = float(self.np_random.uniform(low=-np.pi, high=np.pi))
        return np.array(
            [np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)],
            dtype=np.float64,
        )

    def _cube_qpos(self, cube_name: str) -> np.ndarray:
        qpos_adr = self._cube_qpos_adrs[cube_name]
        return self.data.qpos[qpos_adr : qpos_adr + 7]

    @staticmethod
    def _normalize_quat(quat: np.ndarray) -> np.ndarray:
        if quat.shape != (4,):
            raise ValueError(f"Expected quaternion shape (4,), got {quat.shape}")
        norm = np.linalg.norm(quat)
        if norm <= 0:
            raise ValueError("Quaternion must have non-zero norm.")
        return quat / norm

    @staticmethod
    def _validate_workspace_bounds(
        workspace_bounds: np.ndarray | list[list[float]] | tuple[tuple[float, float], tuple[float, float]],
    ) -> np.ndarray:
        bounds = np.asarray(workspace_bounds, dtype=np.float64)
        if bounds.shape != (2, 2):
            raise ValueError(f"Expected workspace_bounds shape (2, 2), got {bounds.shape}")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Each workspace lower bound must be below its upper bound.")
        return bounds


class OrcaArmCubeStacking(CubeStackingTabletop):
    """OrcaArm cube stacking task built directly on the composed MuJoCo scene."""

    DEFAULT_KEYFRAME = "orcaarm_home"

    def __init__(
        self,
        render_mode: str | None = None,
        version: str | None = None,
        *,
        scene_file: str = "orcaarm_cube_stacking_cameras.xml",
        frame_skip: int = 5,
        actuator_names: Sequence[str] | None = None,
        camera_names: Sequence[str] | None = default_orcaarm_camera_names(),
        camera_width: int = 128,
        camera_height: int = 128,
        workspace_bounds: np.ndarray | list[list[float]] | tuple[tuple[float, float], tuple[float, float]] = CubeStackingTabletop.DEFAULT_WORKSPACE_BOUNDS,
        min_cube_spacing: float = 0.09,
        target_bounds: np.ndarray | list[list[float]] | tuple[tuple[float, float], tuple[float, float]] | None = None,
        min_target_cube_spacing: float = 0.09,
        randomize_yaw: bool = True,
        render_camera: str = "topdown",
        home_keyframe: str = DEFAULT_KEYFRAME,
        top_cube_name: str = "red_cube",
        base_cube_name: str = "blue_cube",
        cube_size: float = 0.05,
        stack_xy_tolerance: float = 0.025,
        stack_height_tolerance: float = 0.012,
        target_xy_tolerance: float = 0.025,
        settle_velocity_tolerance: float = 0.05,
        max_episode_steps: int = 200,
    ) -> None:
        self.home_keyframe = home_keyframe
        self.max_episode_steps = int(max_episode_steps)
        self.camera_names = tuple(camera_names or ())
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self._elapsed_steps = 0

        super().__init__(
            render_mode=render_mode,
            version=version,
            scene_file=scene_file,
            frame_skip=frame_skip,
            workspace_bounds=workspace_bounds,
            min_cube_spacing=min_cube_spacing,
            target_bounds=target_bounds,
            min_target_cube_spacing=min_target_cube_spacing,
            randomize_yaw=randomize_yaw,
            render_camera=render_camera,
            top_cube_name=top_cube_name,
            base_cube_name=base_cube_name,
            cube_size=cube_size,
            stack_xy_tolerance=stack_xy_tolerance,
            stack_height_tolerance=stack_height_tolerance,
            target_xy_tolerance=target_xy_tolerance,
            settle_velocity_tolerance=settle_velocity_tolerance,
        )

        self._home_keyframe_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_KEY,
            self.home_keyframe,
        )
        if self._home_keyframe_id < 0:
            raise ValueError(
                f"Scene {self.scene_path} is missing keyframe {self.home_keyframe!r}."
            )

        self.actuator_names = self._resolve_actuator_names(actuator_names)
        self.actuator_ids = tuple(self.model.actuator(name).id for name in self.actuator_names)
        action_low = self.model.actuator_ctrlrange[list(self.actuator_ids), 0].astype(np.float32)
        action_high = self.model.actuator_ctrlrange[list(self.actuator_ids), 1].astype(np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        self._camera_renderer: mujoco.Renderer | None = None

        for camera_name in self.camera_names:
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name) < 0:
                raise ValueError(
                    f"Scene {self.scene_path} is missing camera {camera_name!r}."
                )

    def _resolve_actuator_names(
        self,
        actuator_names: Sequence[str] | None,
    ) -> tuple[str, ...]:
        if actuator_names is None:
            return tuple(self.model.actuator(actuator_id).name for actuator_id in range(self.model.nu))

        resolved_names = tuple(actuator_names)
        missing = [
            actuator_name
            for actuator_name in resolved_names
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name) < 0
        ]
        if missing:
            raise ValueError(f"Unknown actuator name(s): {missing}")
        return resolved_names

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        gym.Env.reset(self, seed=seed)
        self._elapsed_steps = 0
        options = {} if options is None else dict(options)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_keyframe_id)

        if "qpos" in options:
            qpos = np.asarray(options["qpos"], dtype=np.float64)
            if qpos.shape != self.data.qpos.shape:
                raise ValueError(
                    f"Expected qpos shape {self.data.qpos.shape}, got {qpos.shape}"
                )
            self.data.qpos[:] = qpos
            cube_positions = self._cube_positions_from_qpos(self.data.qpos)
        else:
            cube_positions = self._reset_cubes_from_options(options)

        self._reset_target_from_options(options, cube_positions)

        if "qvel" in options:
            qvel = np.asarray(options["qvel"], dtype=np.float64)
            if qvel.shape != self.data.qvel.shape:
                raise ValueError(
                    f"Expected qvel shape {self.data.qvel.shape}, got {qvel.shape}"
                )
            self.data.qvel[:] = qvel
        else:
            self.data.qvel[:] = 0.0
            cube_qvels = options.get("cube_qvels", {})
            for cube_name, cube_qvel in cube_qvels.items():
                qvel = np.asarray(cube_qvel, dtype=np.float64)
                if qvel.shape != (6,):
                    raise ValueError(
                        f"Expected {cube_name} qvel shape (6,), got {qvel.shape}"
                    )
                qvel_adr = self._cube_qvel_adrs[cube_name]
                self.data.qvel[qvel_adr : qvel_adr + 6] = qvel

        if "ctrl" in options:
            ctrl = np.asarray(options["ctrl"], dtype=np.float64)
            if ctrl.shape != self.data.ctrl.shape:
                raise ValueError(
                    f"Expected ctrl shape {self.data.ctrl.shape}, got {ctrl.shape}"
                )
            self.data.ctrl[:] = ctrl

        mujoco.mj_forward(self.model, self.data)
        settle_steps = int(options.get("settle_steps", 0))
        if settle_steps > 0:
            mujoco.mj_step(self.model, self.data, nstep=settle_steps)

        return self._get_obs(), self._get_info()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        resolved_action = np.asarray(action, dtype=np.float32)
        if resolved_action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {resolved_action.shape}"
            )
        clipped_action = np.clip(resolved_action, self.action_space.low, self.action_space.high)
        self.data.ctrl[list(self.actuator_ids)] = clipped_action

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._elapsed_steps += 1

        obs = self._get_obs()
        info = self._get_info()
        reward = float(info["is_success"])
        terminated = bool(info["is_success"])
        truncated = self._elapsed_steps >= self.max_episode_steps
        return obs, reward, terminated, truncated, info

    def _get_info(self) -> dict[str, Any]:
        info = super()._get_info()
        info["elapsed_steps"] = self._elapsed_steps
        return info

    def render_camera_observations(self) -> dict[str, np.ndarray]:
        if not self.camera_names:
            return {}
        if self._camera_renderer is None:
            self._camera_renderer = mujoco.Renderer(
                self.model,
                width=self.camera_width,
                height=self.camera_height,
            )
        images: dict[str, np.ndarray] = {}
        for camera_name in self.camera_names:
            self._camera_renderer.update_scene(self.data, camera=camera_name)
            images[camera_name] = self._camera_renderer.render()
        return images

    def close(self) -> None:
        if self._camera_renderer is not None:
            self._camera_renderer.close()
            self._camera_renderer = None
        super().close()


class OrcaPandaCubeStacking(OrcaArmCubeStacking):
    """Single-arm OrcaPanda cube stacking task in the shared tabletop scene."""

    DEFAULT_KEYFRAME = "orcapanda_home"
    DEFAULT_CAMERA_NAMES = ("orcapanda_overview", "topdown", "angled")

    def __init__(
        self,
        render_mode: str | None = None,
        version: str | None = None,
        *,
        scene_file: str = "orcapanda_cube_stacking.xml",
        frame_skip: int = 5,
        actuator_names: Sequence[str] | None = None,
        camera_names: Sequence[str] | None = DEFAULT_CAMERA_NAMES,
        camera_width: int = 128,
        camera_height: int = 128,
        workspace_bounds: np.ndarray | list[list[float]] | tuple[tuple[float, float], tuple[float, float]] = CubeStackingTabletop.DEFAULT_WORKSPACE_BOUNDS,
        min_cube_spacing: float = 0.09,
        target_bounds: np.ndarray | list[list[float]] | tuple[tuple[float, float], tuple[float, float]] | None = None,
        min_target_cube_spacing: float = 0.09,
        randomize_yaw: bool = True,
        render_camera: str = "orcapanda_overview",
        home_keyframe: str = DEFAULT_KEYFRAME,
        top_cube_name: str = "red_cube",
        base_cube_name: str = "blue_cube",
        cube_size: float = 0.05,
        stack_xy_tolerance: float = 0.025,
        stack_height_tolerance: float = 0.012,
        target_xy_tolerance: float = 0.025,
        settle_velocity_tolerance: float = 0.05,
        max_episode_steps: int = 200,
    ) -> None:
        super().__init__(
            render_mode=render_mode,
            version=version,
            scene_file=scene_file,
            frame_skip=frame_skip,
            actuator_names=actuator_names,
            camera_names=camera_names,
            camera_width=camera_width,
            camera_height=camera_height,
            workspace_bounds=workspace_bounds,
            min_cube_spacing=min_cube_spacing,
            target_bounds=target_bounds,
            min_target_cube_spacing=min_target_cube_spacing,
            randomize_yaw=randomize_yaw,
            render_camera=render_camera,
            home_keyframe=home_keyframe,
            top_cube_name=top_cube_name,
            base_cube_name=base_cube_name,
            cube_size=cube_size,
            stack_xy_tolerance=stack_xy_tolerance,
            stack_height_tolerance=stack_height_tolerance,
            target_xy_tolerance=target_xy_tolerance,
            settle_velocity_tolerance=settle_velocity_tolerance,
            max_episode_steps=max_episode_steps,
        )


class OrcaHandRightCubeOrientation(BaseOrcaHandEnv):
    """Sample in-hand cube reorientation task with a single red target face."""

    DEFAULT_INITIAL_RED_FACE = "down"
    DEFAULT_CUBE_POS_XY_JITTER = np.array([0.0, 0.0], dtype=np.float64)
    RED_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    RED_FACE_LOCAL_NORMAL = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def __init__(
        self,
        render_mode: str | None = None,
        version: str | None = None,
        *,
        scene_file: str = "scene_right_cube_orientation.xml",
        cube_joint_name: str = "cube_freejoint",
        cube_body_name: str = "task_cube",
        hand_pose_by_joint: Mapping[str, float] | None = None,
        initial_red_face: str = DEFAULT_INITIAL_RED_FACE,
        cube_pos_xy_jitter: float | tuple[float, float] = 0.0,
        max_episode_steps: int = 200,
        success_tolerance_rad: float = np.deg2rad(15.0),
        drop_height: float = 0.05,
    ) -> None:
        self.scene_file = scene_file
        self.cube_joint_name = cube_joint_name
        self.cube_body_name = cube_body_name
        self._requested_hand_pose_by_joint = (
            None if hand_pose_by_joint is None else dict(hand_pose_by_joint)
        )
        self.initial_red_face = self._validate_initial_red_face(initial_red_face)
        self.cube_pos_xy_jitter = self._normalize_xy_jitter(cube_pos_xy_jitter)
        self.max_episode_steps = max_episode_steps
        self.success_tolerance_rad = float(success_tolerance_rad)
        self.drop_height = float(drop_height)
        self._elapsed_steps = 0

        super().__init__(
            scene_file,
            version=version,
            frame_skip=5,
            render_mode=render_mode,
        )

        self._cube_joint_id = self.model.joint(self.cube_joint_name).id
        self._cube_qpos_adr = int(self.model.jnt_qposadr[self._cube_joint_id])
        self._cube_qvel_adr = int(self.model.jnt_dofadr[self._cube_joint_id])
        self._cube_body_id = self.model.body(self.cube_body_name).id

        self._actuator_qpos_indices = self._resolve_actuator_qpos_indices()
        self._default_cube_pos = self.model.qpos0[
            self._cube_qpos_adr : self._cube_qpos_adr + 3
        ].copy()
        self._default_cube_quat = self._normalize_quat(
            self.model.qpos0[self._cube_qpos_adr + 3 : self._cube_qpos_adr + 7].copy()
        )
        if self._requested_hand_pose_by_joint is None:
            self._default_hand_qpos = self.model.qpos0[: self._cube_qpos_adr].copy()
            self._hand_pose_by_joint = self._extract_hand_pose_by_joint(
                self._default_hand_qpos
            )
        else:
            self._hand_pose_by_joint = dict(self._requested_hand_pose_by_joint)
            self._default_hand_qpos = self._build_hand_qpos(self._hand_pose_by_joint)

        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float64,
        )

    def _resolve_actuator_qpos_indices(self) -> np.ndarray:
        indices = np.empty(self.model.nu, dtype=np.int32)
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(
                    f"Actuator {self.model.actuator(actuator_id).name!r} is attached to a "
                    "non-hinge joint, which this sample environment does not support."
                )
            indices[actuator_id] = int(self.model.jnt_qposadr[joint_id])
        return indices

    def _build_hand_qpos(self, pose_by_joint: Mapping[str, float]) -> np.ndarray:
        hand_qpos = self.model.qpos0[: self._cube_qpos_adr].copy()
        for joint_name, joint_value in pose_by_joint.items():
            joint_id = self.model.joint(joint_name).id
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            if qpos_adr >= self._cube_qpos_adr:
                raise ValueError(
                    f"Joint '{joint_name}' does not belong to the hand qpos slice."
                )
            hand_qpos[qpos_adr] = float(joint_value)
        return hand_qpos

    def _extract_hand_pose_by_joint(self, hand_qpos: np.ndarray) -> dict[str, float]:
        pose_by_joint: dict[str, float] = {}
        for actuator_id, qpos_adr in enumerate(self._actuator_qpos_indices):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            pose_by_joint[self.model.joint(joint_id).name] = float(hand_qpos[qpos_adr])
        return pose_by_joint

    def _resolve_default_cube_pos(self, jitter_xy: np.ndarray) -> np.ndarray:
        cube_pos = self._default_cube_pos.copy()
        if np.any(jitter_xy):
            cube_pos[:2] += self.np_random.uniform(low=-jitter_xy, high=jitter_xy)
        return cube_pos

    def nominal_reset_options(self) -> dict[str, Any]:
        return {
            "hand_pose_by_joint": dict(self._hand_pose_by_joint),
            "cube_pos": self._default_cube_pos.copy(),
            "cube_quat": self._default_cube_quat.copy(),
            "settle_steps": 0,
        }

    def sample_randomized_reset_options(
        self,
        *,
        seed: int | None = None,
        initial_red_face: str = "random",
        cube_pos_xy_jitter: float | tuple[float, float] | list[float] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        jitter_xy = (
            self.cube_pos_xy_jitter.copy()
            if cube_pos_xy_jitter is None
            else self._normalize_xy_jitter(cube_pos_xy_jitter)
        )
        cube_pos = self._default_cube_pos.copy()
        if np.any(jitter_xy):
            cube_pos[:2] += rng.uniform(low=-jitter_xy, high=jitter_xy)

        initial_red_face = self._validate_initial_red_face(initial_red_face)
        if initial_red_face == "down":
            cube_quat = self._default_cube_quat.copy()
        else:
            cube_quat = self._sample_random_nonsolved_quaternion(rng)

        return {
            "hand_pose_by_joint": dict(self._hand_pose_by_joint),
            "cube_pos": cube_pos,
            "cube_quat": cube_quat,
            "settle_steps": 0,
        }

    def _resolve_initial_cube_quat(self, options: dict[str, Any]) -> np.ndarray:
        if "cube_quat" in options:
            return self._normalize_quat(np.asarray(options["cube_quat"], dtype=np.float64))

        initial_red_face = self._validate_initial_red_face(
            options.get("initial_red_face", self.initial_red_face)
        )
        if initial_red_face == "down":
            return self._default_cube_quat.copy()
        return self._sample_random_nonsolved_quaternion(self.np_random)

    def _compose_ctrl_from_qpos(self, qpos: np.ndarray | None = None) -> np.ndarray:
        source_qpos = self.data.qpos if qpos is None else np.asarray(qpos, dtype=np.float64)
        ctrl = np.zeros(len(self.hand.config.joint_ids), dtype=np.float32)
        for ctrl_idx, qpos_idx in enumerate(self.hand.config.actuator_qpos_indices):
            ctrl[ctrl_idx] = float(
                np.clip(
                    source_qpos[qpos_idx],
                    self.action_low[ctrl_idx],
                    self.action_high[ctrl_idx],
                )
            )
        return ctrl

    def _cube_quat(self) -> np.ndarray:
        return self.data.qpos[self._cube_qpos_adr + 3 : self._cube_qpos_adr + 7].copy()

    def _cube_pos(self) -> np.ndarray:
        return self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3].copy()

    def _cube_qvel(self) -> np.ndarray:
        return self.data.qvel[self._cube_qvel_adr : self._cube_qvel_adr + 6].copy()

    def _cube_red_face_world_normal(self) -> np.ndarray:
        quat = self._normalize_quat(self._cube_quat())
        w, x, y, z = quat
        return np.array(
            [
                2.0 * (x * z + y * w),
                2.0 * (y * z - x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
            dtype=np.float64,
        )

    def _red_face_up_alignment(self) -> float:
        return float(np.dot(self._cube_red_face_world_normal(), self.WORLD_UP))

    def _red_face_up_angle_rad(self) -> float:
        alignment = np.clip(self._red_face_up_alignment(), -1.0, 1.0)
        return float(np.arccos(alignment))

    def _goal_reached(self) -> bool:
        return bool(self._red_face_up_alignment() >= np.cos(self.success_tolerance_rad))

    def _cube_dropped(self) -> bool:
        return bool(self.data.xpos[self._cube_body_id, 2] < self.drop_height)

    def _get_obs(self) -> np.ndarray:
        base_obs = super()._get_obs()
        if not hasattr(self, "_cube_qpos_adr"):
            return base_obs
        return np.concatenate(
            [
                base_obs,
                self._cube_red_face_world_normal(),
                np.array([self._red_face_up_alignment()], dtype=np.float64),
            ]
        )

    def _get_reward(self) -> float:
        alignment_reward = 0.5 * (self._red_face_up_alignment() + 1.0)
        lift_bonus = np.clip(self.data.xpos[self._cube_body_id, 2] - 0.12, 0.0, 0.12) / 0.12
        drop_penalty = 1.0 if self._cube_dropped() else 0.0
        return float(alignment_reward + 0.10 * lift_bonus - drop_penalty)

    def _get_terminated(self) -> bool:
        return self._goal_reached() or self._cube_dropped()

    def _get_truncated(self) -> bool:
        return self._elapsed_steps >= self.max_episode_steps

    def _get_info(self) -> dict[str, Any]:
        return {
            "cube_pos": self._cube_pos(),
            "cube_quat": self._cube_quat(),
            "cube_qvel": self._cube_qvel(),
            "red_face_world_normal": self._cube_red_face_world_normal(),
            "red_face_up_alignment": self._red_face_up_alignment(),
            "red_face_up_angle_rad": self._red_face_up_angle_rad(),
            "is_success": self._goal_reached(),
            "dropped": self._cube_dropped(),
            "elapsed_steps": self._elapsed_steps,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        gym.Env.reset(self, seed=seed)
        self._elapsed_steps = 0

        options = {} if options is None else dict(options)
        full_qpos = options.get("qpos")
        full_qvel = options.get("qvel")

        if full_qpos is not None:
            qpos = np.asarray(full_qpos, dtype=np.float64)
            if qpos.shape != self.data.qpos.shape:
                raise ValueError(
                    f"Expected qpos shape {self.data.qpos.shape}, got {qpos.shape}"
                )
            self.data.qpos[:] = qpos
        else:
            hand_qpos = self._default_hand_qpos.copy()
            if "hand_pose_by_joint" in options:
                hand_qpos = self._build_hand_qpos(options["hand_pose_by_joint"])
            if "hand_qpos" in options:
                hand_qpos = np.asarray(options["hand_qpos"], dtype=np.float64)
                if hand_qpos.shape != (self._cube_qpos_adr,):
                    raise ValueError(
                        f"Expected hand_qpos shape {(self._cube_qpos_adr,)}, got {hand_qpos.shape}"
                    )

            if "cube_pos" in options:
                cube_pos = np.asarray(options["cube_pos"], dtype=np.float64)
            else:
                jitter_xy = self.cube_pos_xy_jitter.copy()
                if "cube_pos_xy_jitter" in options:
                    jitter_xy = self._normalize_xy_jitter(options["cube_pos_xy_jitter"])
                cube_pos = self._resolve_default_cube_pos(jitter_xy)
            if cube_pos.shape != (3,):
                raise ValueError(f"Expected cube_pos shape (3,), got {cube_pos.shape}")

            cube_quat = self._resolve_initial_cube_quat(options)

            qpos = self.model.qpos0.copy()
            qpos[: self._cube_qpos_adr] = hand_qpos
            qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3] = cube_pos
            qpos[self._cube_qpos_adr + 3 : self._cube_qpos_adr + 7] = cube_quat

        if full_qvel is not None:
            qvel = np.asarray(full_qvel, dtype=np.float64)
            if qvel.shape != self.data.qvel.shape:
                raise ValueError(
                    f"Expected qvel shape {self.data.qvel.shape}, got {qvel.shape}"
                )
        else:
            qvel = np.zeros_like(self.data.qvel)
            if "cube_qvel" in options:
                cube_qvel = np.asarray(options["cube_qvel"], dtype=np.float64)
                if cube_qvel.shape != (6,):
                    raise ValueError(
                        f"Expected cube_qvel shape (6,), got {cube_qvel.shape}"
                    )
                qvel[self._cube_qvel_adr : self._cube_qvel_adr + 6] = cube_qvel

        ctrl = self._compose_ctrl_from_qpos(qpos)
        self.hand.reset(qpos=qpos, qvel=qvel, ctrl=ctrl)

        settle_steps = int(options.get("settle_steps", 0))
        for _ in range(settle_steps):
            self.hand.step(nstep=1)

        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.hand.step(action)
        self._elapsed_steps += 1

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    @staticmethod
    def _normalize_quat(quat: np.ndarray) -> np.ndarray:
        if quat.shape != (4,):
            raise ValueError(f"Expected quaternion shape (4,), got {quat.shape}")
        norm = np.linalg.norm(quat)
        if norm <= 0:
            raise ValueError("Quaternion must have non-zero norm.")
        return quat / norm

    @staticmethod
    def _validate_initial_red_face(initial_red_face: str) -> str:
        if initial_red_face not in {"down", "random"}:
            raise ValueError(
                "initial_red_face must be one of {'down', 'random'}."
            )
        return initial_red_face

    @staticmethod
    def _normalize_xy_jitter(
        jitter: float | tuple[float, float] | list[float] | np.ndarray,
    ) -> np.ndarray:
        jitter_array = np.asarray(jitter, dtype=np.float64)
        if jitter_array.shape == ():
            if float(jitter_array) < 0:
                raise ValueError("cube_pos_xy_jitter must be non-negative.")
            return np.array([float(jitter_array), float(jitter_array)], dtype=np.float64)
        if jitter_array.shape != (2,):
            raise ValueError(
                f"Expected cube_pos_xy_jitter shape () or (2,), got {jitter_array.shape}"
            )
        if np.any(jitter_array < 0):
            raise ValueError("cube_pos_xy_jitter must be non-negative.")
        return jitter_array

    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        half_angle = angle_rad / 2.0
        return np.array(
            [
                np.cos(half_angle),
                axis[0] * np.sin(half_angle),
                axis[1] * np.sin(half_angle),
                axis[2] * np.sin(half_angle),
            ],
            dtype=np.float64,
        )

    @classmethod
    def _sample_random_nonsolved_quaternion(cls, rng: np.random.Generator) -> np.ndarray:
        candidates = []
        for quat in cls._axis_aligned_quaternions():
            red_face_up_alignment = cls._red_face_up_alignment_for_quat(quat)
            if red_face_up_alignment >= 0.95:
                continue
            candidates.append(quat)
        return candidates[int(rng.integers(len(candidates)))].copy()

    @classmethod
    def _axis_aligned_quaternions(cls) -> list[np.ndarray]:
        if not hasattr(cls, "_AXIS_ALIGNED_QUATERNIONS"):
            quaternions: list[np.ndarray] = []
            seen: set[tuple[float, ...]] = set()
            for rx in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi):
                for ry in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi):
                    for rz in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi):
                        quat = cls._quat_multiply(
                            cls._quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), rz),
                            cls._quat_multiply(
                                cls._quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), ry),
                                cls._quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), rx),
                            ),
                        )
                        quat = cls._normalize_quat(quat)
                        if quat[0] < 0:
                            quat = -quat
                        key = tuple(np.round(quat, decimals=8))
                        if key in seen:
                            continue
                        seen.add(key)
                        quaternions.append(quat)
            cls._AXIS_ALIGNED_QUATERNIONS = quaternions
        return [quat.copy() for quat in cls._AXIS_ALIGNED_QUATERNIONS]

    @classmethod
    def _red_face_up_alignment_for_quat(cls, quat: np.ndarray) -> float:
        quat = cls._normalize_quat(quat)
        w, x, y, z = quat
        red_face_world_normal = np.array(
            [
                2.0 * (x * z + y * w),
                2.0 * (y * z - x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
            dtype=np.float64,
        )
        return float(np.dot(red_face_world_normal, cls.WORLD_UP))
