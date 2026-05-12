from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from orca_sim.envs_mjx import BaseOrcaHandMjxEnv


class OrcaHandRightCubeOrientationMjx(BaseOrcaHandMjxEnv):
    """MJX-vectorized in-hand cube reorientation task with a single red target face."""

    DEFAULT_INITIAL_RED_FACE = "down"
    DEFAULT_CUBE_POS_XY_JITTER = np.array([0.0, 0.0], dtype=np.float64)
    RED_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    RED_FACE_LOCAL_NORMAL = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    SUCCESS_HOLD_STEPS = 10

    def __init__(
        self,
        num_envs: int = 1,
        version: str | None = None,
        render_mode: str | None = None,
        render_index: int = 0,
        *,
        scene_file: str = "scene_right_cube_orientation.xml",
        cube_joint_name: str = "cube_freejoint",
        cube_body_name: str = "task_cube",
        hand_pose_by_joint: Mapping[str, float] | None = None,
        initial_red_face: str = DEFAULT_INITIAL_RED_FACE,
        cube_pos_xy_jitter: float | tuple[float, float] = 0.0,
        max_episode_steps: int = 200,
        success_tolerance_rad: float = float(np.deg2rad(15.0)),
        drop_height: float = 0.05,
        timestep: float = 0.005,
        frame_skip: int = 2,
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
        self._np_random = np.random.default_rng()

        super().__init__(
            scene_file,
            num_envs=num_envs,
            version=version,
            frame_skip=frame_skip,
            render_mode=render_mode,
            render_index=render_index,
            timestep=timestep,
        )

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

        # Cache the broadcasted initial mjx_data once. Re-broadcasting the whole
        # tree on every reset() call (incl. derived buffers) is expensive; we
        # only ever overwrite qpos/qvel/ctrl, so caching is safe.
        self._cached_initial_batch = self._make_initial_batch()

        # On-device templates for per-env auto-reset inside training rollouts.
        # See `_sample_per_env_reset` below.
        qpos_template_np = np.concatenate(
            [
                self._default_hand_qpos.astype(np.float64),
                self._default_cube_pos.astype(np.float64),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),  # placeholder
            ]
        )
        ctrl_template_np = np.clip(
            self._default_hand_qpos[self._actuator_qpos_indices],
            self.action_low,
            self.action_high,
        )
        self._reset_qpos_template_jax = jnp.asarray(
            qpos_template_np, dtype=self._mjx_data0.qpos.dtype
        )
        self._reset_qvel_template_jax = jnp.zeros(
            (int(self.model.nv),), dtype=self._mjx_data0.qvel.dtype
        )
        self._reset_ctrl_template_jax = jnp.asarray(
            ctrl_template_np, dtype=self._mjx_data0.ctrl.dtype
        )
        self._reset_time_template_jax = jnp.zeros(
            (), dtype=self._mjx_data0.time.dtype
        )
        bank_np = (
            self._reset_quat_bank_np
            if self.initial_red_face == "random"
            else self._down_quat_bank_np
        )
        self._reset_quat_bank_jax = jnp.asarray(
            bank_np, dtype=self._mjx_data0.qpos.dtype
        )
        self._success_counts = jnp.zeros(self.num_envs, dtype=jnp.int32)

    # ---- jit-time setup --------------------------------------------------

    def _setup_task(self) -> None:
        self._cube_joint_id = self.model.joint(self.cube_joint_name).id
        self._cube_qpos_adr = int(self.model.jnt_qposadr[self._cube_joint_id])
        self._cube_qvel_adr = int(self.model.jnt_dofadr[self._cube_joint_id])
        self._cube_body_id = self.model.body(self.cube_body_name).id
        self._success_alignment = jnp.float32(np.cos(self.success_tolerance_rad))
        self._drop_height_jax = jnp.float32(self.drop_height)
        dt = float(self.model.opt.timestep)
        self._max_episode_time = jnp.float32(
            self.max_episode_steps * self.frame_skip * dt
        )
        self._world_up_jax = jnp.array(self.WORLD_UP, dtype=jnp.float32)

        # Bank of valid (non-solved) starting quats. Used both by the host-side
        # `reset()` (vectorized sampling) and by the on-device auto-reset path
        # consumed by training rollouts.
        valid_quats = [
            q
            for q in self._axis_aligned_quaternions()
            if self._red_face_up_alignment_for_quat(q) < 0.95
        ]
        self._reset_quat_bank_np = np.stack(valid_quats, axis=0).astype(np.float64)
        self._down_quat_bank_np = self.RED_DOWN_QUAT.reshape(1, 4).astype(np.float64)

    # ---- per-env (jnp) cube helpers, mirror task_envs.py:186-218 ---------

    def _cube_quat(self, mjx_data) -> jnp.ndarray:
        return mjx_data.qpos[self._cube_qpos_adr + 3 : self._cube_qpos_adr + 7]

    def _cube_pos(self, mjx_data) -> jnp.ndarray:
        return mjx_data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3]

    def _cube_qvel(self, mjx_data) -> jnp.ndarray:
        return mjx_data.qvel[self._cube_qvel_adr : self._cube_qvel_adr + 6]

    def _cube_red_face_world_normal(self, mjx_data) -> jnp.ndarray:
        quat = self._cube_quat(mjx_data)
        norm = jnp.linalg.norm(quat)
        w = quat[0] / norm
        x = quat[1] / norm
        y = quat[2] / norm
        z = quat[3] / norm
        return jnp.stack(
            [
                2.0 * (x * z + y * w),
                2.0 * (y * z - x * w),
                1.0 - 2.0 * (x * x + y * y),
            ]
        )

    def _red_face_up_alignment(self, mjx_data) -> jnp.ndarray:
        return jnp.dot(self._cube_red_face_world_normal(mjx_data), self._world_up_jax)

    def _red_face_up_angle_rad(self, mjx_data) -> jnp.ndarray:
        return jnp.arccos(jnp.clip(self._red_face_up_alignment(mjx_data), -1.0, 1.0))

    def _cube_dropped(self, mjx_data) -> jnp.ndarray:
        return mjx_data.qpos[self._cube_qpos_adr + 2] < self._drop_height_jax

    # ---- task hooks (jit-traced); mirror _get_obs / _get_reward / etc ----

    def _obs_fn(self, mjx_data):
        base = jnp.concatenate([mjx_data.qpos, mjx_data.qvel])
        return jnp.concatenate(
            [
                base,
                self._cube_red_face_world_normal(mjx_data),
                jnp.array([self._red_face_up_alignment(mjx_data)]),
            ]
        )

    def _reward_fn(self, mjx_data):
        alignment = self._red_face_up_alignment(mjx_data)
        cube_z = mjx_data.qpos[self._cube_qpos_adr + 2]
        alignment_reward = 0.5 * (alignment + 1.0)
        lift_bonus = jnp.clip(cube_z - 0.12, 0.0, 0.12) / 0.12
        drop_penalty = jnp.where(cube_z < self._drop_height_jax, 1.0, 0.0)
        return jnp.float32(alignment_reward + 0.10 * lift_bonus - drop_penalty)

    def _terminated_fn(self, mjx_data):
        return self._cube_dropped(mjx_data)

    def _truncated_fn(self, mjx_data):
        return mjx_data.time >= self._max_episode_time

    def step(self, actions):
        obs, rewards, terminateds, truncateds, infos = super().step(actions)

        alignment = infos["red_face_up_alignment"]
        is_aligned = alignment >= self._success_alignment
        self._success_counts = jnp.where(
            is_aligned, self._success_counts + 1, jnp.zeros_like(self._success_counts)
        )
        goal_reached = self._success_counts >= self.SUCCESS_HOLD_STEPS
        terminateds = terminateds | goal_reached
        infos["is_success"] = goal_reached

        return obs, rewards, terminateds, truncateds, infos

    # ---- on-device per-env reset (used by training rollouts) -------------

    def _sample_per_env_reset(self, rng_key):
        """Per-env reset state for on-device auto-reset inside a jitted scan.

        Returns (qpos, qvel, ctrl, time) for a single env. Designed to be
        vmapped to produce ``(num_envs, ...)`` reset states without leaving
        the GPU.
        """
        bank_size = self._reset_quat_bank_jax.shape[0]
        idx = jax.random.randint(rng_key, (), 0, bank_size)
        quat = self._reset_quat_bank_jax[idx]
        qpos = self._reset_qpos_template_jax.at[
            self._cube_qpos_adr + 3 : self._cube_qpos_adr + 7
        ].set(quat)
        return (
            qpos,
            self._reset_qvel_template_jax,
            self._reset_ctrl_template_jax,
            self._reset_time_template_jax,
        )

    # ---- batched info, mirror task_envs.py:244-255 -----------------------

    def _get_infos(self) -> dict[str, Any]:
        qpos = self.mjx_data.qpos
        qvel = self.mjx_data.qvel
        adr = self._cube_qpos_adr
        vadr = self._cube_qvel_adr

        cube_pos = qpos[:, adr : adr + 3]
        cube_quat = qpos[:, adr + 3 : adr + 7]
        cube_qvel = qvel[:, vadr : vadr + 6]

        norm = jnp.linalg.norm(cube_quat, axis=-1, keepdims=True)
        nq = cube_quat / norm
        w = nq[:, 0]
        x = nq[:, 1]
        y = nq[:, 2]
        z = nq[:, 3]
        red_normal = jnp.stack(
            [
                2.0 * (x * z + y * w),
                2.0 * (y * z - x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
            axis=-1,
        )
        alignment = jnp.sum(red_normal * self._world_up_jax, axis=-1)
        angle = jnp.arccos(jnp.clip(alignment, -1.0, 1.0))
        cube_z = qpos[:, adr + 2]
        is_success = alignment >= self._success_alignment
        dropped = cube_z < self._drop_height_jax
        dt = float(self.model.opt.timestep)
        elapsed_steps = jnp.round(
            self.mjx_data.time / (self.frame_skip * dt)
        ).astype(jnp.int32)

        return {
            "cube_pos": cube_pos,
            "cube_quat": cube_quat,
            "cube_qvel": cube_qvel,
            "red_face_world_normal": red_normal,
            "red_face_up_alignment": alignment,
            "red_face_up_angle_rad": angle,
            "is_success": is_success,
            "dropped": dropped,
            "elapsed_steps": elapsed_steps,
        }

    # ---- reset-options helpers (host-side); mirror task_envs.py:87-184 --

    def _resolve_actuator_qpos_indices(self) -> np.ndarray:
        indices = np.empty(self.model.nu, dtype=np.int32)
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(
                    f"Actuator {self.model.actuator(actuator_id).name!r} is attached to a "
                    "non-hinge joint, which this environment does not support."
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
            cube_pos[:2] += self._np_random.uniform(low=-jitter_xy, high=jitter_xy)
        return cube_pos

    def _resolve_initial_cube_quat(self, options: dict[str, Any]) -> np.ndarray:
        if "cube_quat" in options:
            return self._normalize_quat(np.asarray(options["cube_quat"], dtype=np.float64))
        initial_red_face = self._validate_initial_red_face(
            options.get("initial_red_face", self.initial_red_face)
        )
        if initial_red_face == "down":
            return self._default_cube_quat.copy()
        return self._sample_random_nonsolved_quaternion(self._np_random)

    def _compose_ctrl_from_qpos_batched(self, qpos_batch: np.ndarray) -> np.ndarray:
        ctrl = qpos_batch[:, self._actuator_qpos_indices].astype(np.float32)
        return np.clip(ctrl, self.action_low, self.action_high)

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
        cube_pos_xy_jitter: float
        | tuple[float, float]
        | list[float]
        | np.ndarray
        | None = None,
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

    def _broadcast_to_envs(
        self, value: Any, single_shape: tuple[int, ...], name: str
    ) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape == single_shape:
            return np.broadcast_to(arr, (self.num_envs, *single_shape)).copy()
        if arr.shape == (self.num_envs, *single_shape):
            return arr.copy()
        raise ValueError(
            f"Expected {name} shape {single_shape} or "
            f"{(self.num_envs, *single_shape)}, got {arr.shape}"
        )

    # ---- batched reset; mirror task_envs.py:257-335 ----------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[jax.Array, dict[str, Any]]:
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        options = {} if options is None else dict(options)

        nq = int(self.model.nq)
        nv = int(self.model.nv)

        full_qpos = options.get("qpos")
        full_qvel = options.get("qvel")

        if full_qpos is not None:
            qpos_batch = self._broadcast_to_envs(full_qpos, (nq,), "qpos")
        else:
            if "hand_qpos" in options:
                hand_slice = self._broadcast_to_envs(
                    options["hand_qpos"], (self._cube_qpos_adr,), "hand_qpos"
                )
            elif "hand_pose_by_joint" in options:
                hand_single = self._build_hand_qpos(options["hand_pose_by_joint"])
                hand_slice = np.broadcast_to(
                    hand_single, (self.num_envs, self._cube_qpos_adr)
                ).copy()
            else:
                hand_slice = np.broadcast_to(
                    self._default_hand_qpos, (self.num_envs, self._cube_qpos_adr)
                ).copy()

            if "cube_pos" in options:
                cube_pos_batch = self._broadcast_to_envs(
                    options["cube_pos"], (3,), "cube_pos"
                )
            else:
                jitter_xy = (
                    self._normalize_xy_jitter(options["cube_pos_xy_jitter"])
                    if "cube_pos_xy_jitter" in options
                    else self.cube_pos_xy_jitter
                )
                cube_pos_batch = np.broadcast_to(
                    self._default_cube_pos, (self.num_envs, 3)
                ).copy()
                if np.any(jitter_xy):
                    jitter = self._np_random.uniform(
                        low=-jitter_xy, high=jitter_xy, size=(self.num_envs, 2)
                    )
                    cube_pos_batch[:, :2] += jitter

            if "cube_quat" in options:
                cube_quat_batch = self._broadcast_to_envs(
                    options["cube_quat"], (4,), "cube_quat"
                )
                norms = np.linalg.norm(cube_quat_batch, axis=-1, keepdims=True)
                if np.any(norms <= 0):
                    raise ValueError("Quaternion must have non-zero norm.")
                cube_quat_batch = cube_quat_batch / norms
            else:
                initial_red_face = self._validate_initial_red_face(
                    options.get("initial_red_face", self.initial_red_face)
                )
                if initial_red_face == "down":
                    cube_quat_batch = np.broadcast_to(
                        self._default_cube_quat, (self.num_envs, 4)
                    ).copy()
                else:
                    indices = self._np_random.integers(
                        self._reset_quat_bank_np.shape[0], size=self.num_envs
                    )
                    cube_quat_batch = self._reset_quat_bank_np[indices].copy()

            qpos_batch = np.concatenate(
                [hand_slice, cube_pos_batch, cube_quat_batch], axis=-1
            )

        if full_qvel is not None:
            qvel_batch = self._broadcast_to_envs(full_qvel, (nv,), "qvel")
        else:
            qvel_batch = np.zeros((self.num_envs, nv), dtype=np.float64)
            if "cube_qvel" in options:
                cube_qvel_batch = self._broadcast_to_envs(
                    options["cube_qvel"], (6,), "cube_qvel"
                )
                qvel_batch[:, self._cube_qvel_adr : self._cube_qvel_adr + 6] = (
                    cube_qvel_batch
                )

        ctrl_batch = self._compose_ctrl_from_qpos_batched(qpos_batch)

        fresh = self._cached_initial_batch
        self.mjx_data = fresh.replace(
            qpos=jnp.asarray(qpos_batch, dtype=fresh.qpos.dtype),
            qvel=jnp.asarray(qvel_batch, dtype=fresh.qvel.dtype),
            ctrl=jnp.asarray(ctrl_batch, dtype=fresh.ctrl.dtype),
            time=jnp.zeros_like(fresh.time),
        )

        settle_steps = int(options.get("settle_steps", 0))
        if settle_steps > 0:
            ctrl_jax = jnp.asarray(ctrl_batch, dtype=fresh.ctrl.dtype)
            for _ in range(settle_steps):
                self.mjx_data, _, _, _, _ = self._jit_vstep(self.mjx_data, ctrl_jax)
            # Don't count settle steps toward truncation.
            self.mjx_data = self.mjx_data.replace(
                time=jnp.zeros_like(self.mjx_data.time)
            )

        obs = self._jit_vobs(self.mjx_data)
        self._success_counts = jnp.zeros(self.num_envs, dtype=jnp.int32)

        if self.render_mode is not None:
            self._sync_render_data()
            if self.render_mode == "human":
                self.render()

        return obs, self._get_infos()

    # ---- static math; mirror task_envs.py:361-472 ------------------------

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
            raise ValueError("initial_red_face must be one of {'down', 'random'}.")
        return initial_red_face

    @staticmethod
    def _normalize_xy_jitter(
        jitter: float | tuple[float, float] | list[float] | np.ndarray,
    ) -> np.ndarray:
        jitter_array = np.asarray(jitter, dtype=np.float64)
        if jitter_array.shape == ():
            if float(jitter_array) < 0:
                raise ValueError("cube_pos_xy_jitter must be non-negative.")
            return np.array(
                [float(jitter_array), float(jitter_array)], dtype=np.float64
            )
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
    def _sample_random_nonsolved_quaternion(
        cls, rng: np.random.Generator
    ) -> np.ndarray:
        candidates = []
        for quat in cls._axis_aligned_quaternions():
            if cls._red_face_up_alignment_for_quat(quat) >= 0.95:
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
