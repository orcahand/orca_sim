import sys
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from gymnasium import spaces
from mujoco import mjx

from orca_sim.versions import resolve_scene_path


def _prepare_mj_model_for_mjx(mj_model: mujoco.MjModel) -> mujoco.MjModel:
    # MJX prefers CG with low iteration counts; the orca scenes don't pin a solver
    # so the CPU default (Newton) leaks through. Set CG before mjx.put_model.
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    mj_model.opt.iterations = 6
    mj_model.opt.ls_iterations = 6
    # MJX does not implement margin/gap for plane<->mesh contacts. The orca hand
    # mjcf defaults geom margin to 0.5mm; zero it (and gap) so plane-vs-finger
    # contact compiles. Runtime-only — CPU envs in envs.py are unaffected.
    mj_model.geom_margin[:] = 0.0
    mj_model.geom_gap[:] = 0.0
    return mj_model


def _make_step_fn(mjx_model, frame_skip: int):
    def step(mjx_data, ctrl):
        mjx_data = mjx_data.replace(ctrl=ctrl)

        def body(d, _):
            return mjx.step(mjx_model, d), None

        final, _ = jax.lax.scan(body, mjx_data, xs=None, length=frame_skip)
        return final

    return step


class BaseOrcaHandMjxEnv(gym.Env[np.ndarray, np.ndarray]):
    """Single-env Gymnasium wrapper around MJX physics on GPU."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        scene_file: str,
        version: str | None = None,
        frame_skip: int = 5,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.scene_path = resolve_scene_path(scene_file, version=version)
        self.version = self.scene_path.parent.name
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        _prepare_mj_model_for_mjx(self.model)

        self._render_data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self._render_data)

        self.mjx_model = mjx.put_model(self.model)
        self.mjx_data = mjx.put_data(self.model, mujoco.MjData(self.model))

        self._step_fn = _make_step_fn(self.mjx_model, self.frame_skip)
        self._jit_step = jax.jit(self._step_fn)

        self._renderer: mujoco.Renderer | None = None
        self._viewer: Any | None = None

        ctrl_range = self.model.actuator_ctrlrange.copy()
        self.action_low = ctrl_range[:, 0].astype(np.float32)
        self.action_high = ctrl_range[:, 1].astype(np.float32)
        self.action_space = spaces.Box(
            low=self.action_low,
            high=self.action_high,
            dtype=np.float32,
        )

        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float64,
        )

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            [np.asarray(self.mjx_data.qpos), np.asarray(self.mjx_data.qvel)]
        )

    def _get_reward(self) -> float:
        return 0.0

    def _get_terminated(self) -> bool:
        return False

    def _get_truncated(self) -> bool:
        return False

    def _get_info(self) -> dict[str, Any]:
        return {}

    def _sync_render_data(self, mjx_data_single=None) -> None:
        if mjx_data_single is None:
            mjx_data_single = self.mjx_data
        cpu = mjx.get_data(self.model, mjx_data_single)
        self._render_data.qpos[:] = cpu.qpos
        self._render_data.qvel[:] = cpu.qvel
        self._render_data.ctrl[:] = cpu.ctrl
        self._render_data.time = float(cpu.time)
        mujoco.mj_forward(self.model, self._render_data)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        print("Resetting")
        super().reset(seed=seed)
        self.mjx_data = mjx.make_data(self.mjx_model)
        print("made data")

        if options and "qpos" in options:
            qpos = np.asarray(options["qpos"], dtype=np.float64)
            if qpos.shape != self.mjx_data.qpos.shape:
                raise ValueError(
                    f"Expected qpos shape {self.mjx_data.qpos.shape}, got {qpos.shape}"
                )
            self.mjx_data = self.mjx_data.replace(qpos=jnp.asarray(qpos))
        print("made q pos")

        if options and "qvel" in options:
            qvel = np.asarray(options["qvel"], dtype=np.float64)
            if qvel.shape != self.mjx_data.qvel.shape:
                raise ValueError(
                    f"Expected qvel shape {self.mjx_data.qvel.shape}, got {qvel.shape}"
                )
            self.mjx_data = self.mjx_data.replace(qvel=jnp.asarray(qvel))
        print("Forward")
        self.mjx_data = mjx.forward(self.mjx_model, self.mjx_data)
        print("Forward done")


        self._sync_render_data()
        if self.render_mode == "human":
            self.render()
        print("finished resetting")

        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {action.shape}"
            )

        clipped = np.clip(action, self.action_low, self.action_high)
        self.mjx_data = self._jit_step(self.mjx_data, jnp.asarray(clipped))

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        info = self._get_info()

        self._sync_render_data()
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model)
            self._renderer.update_scene(self._render_data)
            return self._renderer.render()

        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer

                try:
                    self._viewer = viewer.launch_passive(self.model, self._render_data)
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


class OrcaHandLeftMjx(BaseOrcaHandMjxEnv):
    def __init__(
        self,
        render_mode: str | None = None,
        version: str | None = None,
    ) -> None:
        super().__init__(
            "scene_left.xml",
            version=version,
            frame_skip=5,
            render_mode=render_mode,
        )


class OrcaHandRightMjx(BaseOrcaHandMjxEnv):
    def __init__(
        self,
        render_mode: str | None = None,
        version: str | None = None,
    ) -> None:
        super().__init__(
            "scene_right.xml",
            version=version,
            frame_skip=5,
            render_mode=render_mode,
        )


class OrcaHandCombinedMjx(BaseOrcaHandMjxEnv):
    def __init__(
        self,
        render_mode: str | None = None,
        version: str | None = None,
    ) -> None:
        super().__init__(
            "scene_combined.xml",
            version=version,
            frame_skip=5,
            render_mode=render_mode,
        )


class OrcaHandLeftExtendedMjx(BaseOrcaHandMjxEnv):
    def __init__(
        self,
        render_mode: str | None = None,
        version: str | None = None,
    ) -> None:
        super().__init__(
            "scene_left_extended.xml",
            version=version,
            frame_skip=5,
            render_mode=render_mode,
        )


class OrcaHandRightExtendedMjx(BaseOrcaHandMjxEnv):
    def __init__(
        self,
        render_mode: str | None = None,
        version: str | None = None,
    ) -> None:
        super().__init__(
            "scene_right_extended.xml",
            version=version,
            frame_skip=5,
            render_mode=render_mode,
        )


class OrcaHandCombinedExtendedMjx(BaseOrcaHandMjxEnv):
    def __init__(
        self,
        render_mode: str | None = None,
        version: str | None = None,
    ) -> None:
        super().__init__(
            "scene_combined_extended.xml",
            version=version,
            frame_skip=5,
            render_mode=render_mode,
        )


class OrcaHandMjxVectorEnv:
    """Batched MJX rollouts for many parallel hands on the GPU.

    Custom batched API (not gymnasium.VectorEnv): `reset` / `step` operate on
    leading batch axis. `render()` shows a single chosen env (default 0) so the
    standard MuJoCo viewer / Renderer can be used unchanged.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        scene_file: str,
        num_envs: int,
        version: str | None = None,
        frame_skip: int = 5,
        render_mode: str | None = None,
        render_index: int = 0,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        if not 0 <= render_index < num_envs:
            raise ValueError(
                f"render_index {render_index} out of range for num_envs={num_envs}"
            )

        self.scene_path = resolve_scene_path(scene_file, version=version)
        self.version = self.scene_path.parent.name
        self.num_envs = num_envs
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.render_index = render_index

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        _prepare_mj_model_for_mjx(self.model)

        self._render_data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self._render_data)

        self.mjx_model = mjx.put_model(self.model)
        self._mjx_data0 = mjx.put_data(self.model, mujoco.MjData(self.model))

        self._step_fn = _make_step_fn(self.mjx_model, self.frame_skip)
        self._jit_vstep = jax.jit(jax.vmap(self._step_fn, in_axes=(0, 0)))

        self.mjx_data = self._make_initial_batch()

        self._renderer: mujoco.Renderer | None = None
        self._viewer: Any | None = None

        ctrl_range = self.model.actuator_ctrlrange.copy()
        self.action_low = ctrl_range[:, 0].astype(np.float32)
        self.action_high = ctrl_range[:, 1].astype(np.float32)
        self.single_action_space = spaces.Box(
            low=self.action_low,
            high=self.action_high,
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.broadcast_to(self.action_low, (num_envs, self.action_low.size)).copy(),
            high=np.broadcast_to(self.action_high, (num_envs, self.action_high.size)).copy(),
            dtype=np.float32,
        )

        obs = self._get_obs()
        self.single_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape[1:],
            dtype=np.float64,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float64,
        )

    def _make_initial_batch(self):
        return jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (self.num_envs,) + x.shape).copy(),
            self._mjx_data0,
        )

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            [np.asarray(self.mjx_data.qpos), np.asarray(self.mjx_data.qvel)],
            axis=-1,
        )

    def _get_rewards(self) -> np.ndarray:
        return np.zeros(self.num_envs, dtype=np.float64)

    def _get_terminateds(self) -> np.ndarray:
        return np.zeros(self.num_envs, dtype=bool)

    def _get_truncateds(self) -> np.ndarray:
        return np.zeros(self.num_envs, dtype=bool)

    def _get_infos(self) -> dict[str, Any]:
        return {}

    def _slice_render_env(self):
        return jax.tree_util.tree_map(
            lambda x: x[self.render_index], self.mjx_data
        )

    def _sync_render_data(self) -> None:
        single = self._slice_render_env()
        cpu = mjx.get_data(self.model, single)
        self._render_data.qpos[:] = cpu.qpos
        self._render_data.qvel[:] = cpu.qvel
        self._render_data.ctrl[:] = cpu.ctrl
        self._render_data.time = float(cpu.time)
        mujoco.mj_forward(self.model, self._render_data)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del seed, options
        self.mjx_data = self._make_initial_batch()
        self._sync_render_data()
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._get_infos()

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.float32)
        expected = (self.num_envs, self.action_low.size)
        if actions.shape != expected:
            raise ValueError(
                f"Expected action shape {expected}, got {actions.shape}"
            )

        clipped = np.clip(actions, self.action_low, self.action_high)
        self.mjx_data = self._jit_vstep(self.mjx_data, jnp.asarray(clipped))

        obs = self._get_obs()
        rewards = self._get_rewards()
        terminateds = self._get_terminateds()
        truncateds = self._get_truncateds()
        infos = self._get_infos()

        self._sync_render_data()
        if self.render_mode == "human":
            self.render()

        return obs, rewards, terminateds, truncateds, infos

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model)
            self._renderer.update_scene(self._render_data)
            return self._renderer.render()

        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer

                try:
                    self._viewer = viewer.launch_passive(self.model, self._render_data)
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
