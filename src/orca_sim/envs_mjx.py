import os
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from gymnasium import spaces
from mujoco import mjx

from orca_sim.versions import resolve_scene_path


def _enable_jax_compilation_cache() -> None:
    """Point JAX at a persistent on-disk compile cache.

    First MJX compile of an orca scene takes ~60-120s. With a populated cache,
    subsequent runs of the same model+JAX+CUDA combination skip the compile
    almost entirely. Override / disable via env vars:
      ORCA_SIM_JAX_CACHE=0          # skip wiring the cache
      ORCA_SIM_JAX_CACHE_DIR=/path  # custom location (default: ~/.cache/orca_sim/jax)
    Already-set JAX_COMPILATION_CACHE_DIR is honored and never overwritten.
    """
    if os.environ.get("ORCA_SIM_JAX_CACHE", "1") == "0":
        return
    if os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        return  # user already configured one; don't override.
    if jax.config.jax_compilation_cache_dir:
        return  # already set elsewhere in this process.

    cache_dir = os.environ.get("ORCA_SIM_JAX_CACHE_DIR")
    if cache_dir is None:
        cache_dir = str(Path.home() / ".cache" / "orca_sim" / "jax")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)


_enable_jax_compilation_cache()


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


def _make_step_fn(mjx_model, frame_skip, obs_fn, reward_fn, terminated_fn, truncated_fn):
    """Per-env step returning (mjx_data, obs, reward, terminated, truncated).

    Pure jnp inside; meant to be vmapped over the batch axis and jitted once.
    The four output callables run on a single (non-batched) mjx_data; vmap
    handles batching.
    """

    def step(mjx_data, ctrl):
        mjx_data = mjx_data.replace(ctrl=ctrl)

        def body(d, _):
            return mjx.step(mjx_model, d), None

        final, _ = jax.lax.scan(body, mjx_data, xs=None, length=frame_skip)
        return (
            final,
            obs_fn(final),
            reward_fn(final),
            terminated_fn(final),
            truncated_fn(final),
        )

    return step


class BaseOrcaHandMjxEnv:
    """Vectorized MJX hand env on the GPU, designed for RL training.

    Physics, observation, reward and termination are computed inside a single
    jitted+vmapped step so all outputs stay on-device as ``jax.Array`` of shape
    ``(num_envs, ...)``. Subclasses define a task by overriding ``_obs_fn``,
    ``_reward_fn``, ``_terminated_fn`` and ``_truncated_fn`` — pure jnp
    functions of a single ``mjx_data``. The base wires them into the jit at
    ``__init__`` time, so anything they close over (targets, constants, …)
    must be a ``jnp`` array or a static scalar at construction.

    Renderer / viewer are instantiated lazily; when ``render_mode is None`` no
    GPU→CPU sync runs in ``step`` / ``reset``.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        scene_file: str,
        num_envs: int = 1,
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

        self._step_fn = _make_step_fn(
            self.mjx_model,
            self.frame_skip,
            obs_fn=self._obs_fn,
            reward_fn=self._reward_fn,
            terminated_fn=self._terminated_fn,
            truncated_fn=self._truncated_fn,
        )
        self._jit_vstep = jax.jit(jax.vmap(self._step_fn, in_axes=(0, 0)))
        self._jit_vobs = jax.jit(jax.vmap(self._obs_fn))

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

        sample_obs = self._jit_vobs(self.mjx_data)
        self.single_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=sample_obs.shape[1:],
            dtype=np.float64,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=sample_obs.shape,
            dtype=np.float64,
        )

    # ---- override hooks --------------------------------------------------
    # Pure jnp functions of a single (non-batched) mjx_data. Subclasses replace
    # these to define a task. They run inside jax.vmap+jax.jit at step time, so
    # any captured constants must be jnp arrays / static at __init__.

    def _obs_fn(self, mjx_data):
        return jnp.concatenate([mjx_data.qpos, mjx_data.qvel])

    def _reward_fn(self, mjx_data):
        return jnp.float32(0.0)

    def _terminated_fn(self, mjx_data):
        return jnp.bool_(False)

    def _truncated_fn(self, mjx_data):
        return jnp.bool_(False)

    def _get_infos(self) -> dict[str, Any]:
        return {}

    # ---- batched state helpers ------------------------------------------

    def _make_initial_batch(self):
        return jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (self.num_envs,) + x.shape).copy(),
            self._mjx_data0,
        )

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

    # ---- public API ------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[jax.Array, dict[str, Any]]:
        del seed, options
        self.mjx_data = self._make_initial_batch()
        obs = self._jit_vobs(self.mjx_data)

        if self.render_mode is not None:
            self._sync_render_data()
            if self.render_mode == "human":
                self.render()

        return obs, self._get_infos()

    def step(
        self, actions: np.ndarray
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.float32)
        expected = (self.num_envs, self.action_low.size)
        if actions.shape != expected:
            raise ValueError(
                f"Expected action shape {expected}, got {actions.shape}"
            )

        clipped = np.clip(actions, self.action_low, self.action_high)
        self.mjx_data, obs, rewards, terminateds, truncateds = self._jit_vstep(
            self.mjx_data, jnp.asarray(clipped)
        )
        infos = self._get_infos()

        if self.render_mode is not None:
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


class OrcaHandLeftMjx(BaseOrcaHandMjxEnv):
    def __init__(
        self,
        num_envs: int = 1,
        version: str | None = None,
        render_mode: str | None = None,
        render_index: int = 0,
    ) -> None:
        super().__init__(
            scene_file="scene_left.xml",
            num_envs=num_envs,
            version=version,
            frame_skip=5,
            render_mode=render_mode,
            render_index=render_index,
        )


class OrcaHandRightMjx(BaseOrcaHandMjxEnv):
    def __init__(
        self,
        num_envs: int = 1,
        version: str | None = None,
        render_mode: str | None = None,
        render_index: int = 0,
    ) -> None:
        super().__init__(
            scene_file="scene_right.xml",
            num_envs=num_envs,
            version=version,
            frame_skip=5,
            render_mode=render_mode,
            render_index=render_index,
        )


class OrcaHandCombinedMjx(BaseOrcaHandMjxEnv):
    def __init__(
        self,
        num_envs: int = 1,
        version: str | None = None,
        render_mode: str | None = None,
        render_index: int = 0,
    ) -> None:
        super().__init__(
            scene_file="scene_combined.xml",
            num_envs=num_envs,
            version=version,
            frame_skip=5,
            render_mode=render_mode,
            render_index=render_index,
        )


class OrcaHandLeftExtendedMjx(BaseOrcaHandMjxEnv):
    def __init__(
        self,
        num_envs: int = 1,
        version: str | None = None,
        render_mode: str | None = None,
        render_index: int = 0,
    ) -> None:
        super().__init__(
            scene_file="scene_left_extended.xml",
            num_envs=num_envs,
            version=version,
            frame_skip=5,
            render_mode=render_mode,
            render_index=render_index,
        )


class OrcaHandRightExtendedMjx(BaseOrcaHandMjxEnv):
    def __init__(
        self,
        num_envs: int = 1,
        version: str | None = None,
        render_mode: str | None = None,
        render_index: int = 0,
    ) -> None:
        super().__init__(
            scene_file="scene_right_extended.xml",
            num_envs=num_envs,
            version=version,
            frame_skip=5,
            render_mode=render_mode,
            render_index=render_index,
        )


class OrcaHandCombinedExtendedMjx(BaseOrcaHandMjxEnv):
    def __init__(
        self,
        num_envs: int = 1,
        version: str | None = None,
        render_mode: str | None = None,
        render_index: int = 0,
    ) -> None:
        super().__init__(
            scene_file="scene_combined_extended.xml",
            num_envs=num_envs,
            version=version,
            frame_skip=5,
            render_mode=render_mode,
            render_index=render_index,
        )
