from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from orca_sim.envs import BaseOrcaHandEnv


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
        ctrl = np.zeros(self.model.nu, dtype=np.float32)
        for actuator_id, qpos_idx in enumerate(self._actuator_qpos_indices):
            ctrl[actuator_id] = float(
                np.clip(
                    source_qpos[qpos_idx],
                    self.action_low[actuator_id],
                    self.action_high[actuator_id],
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
