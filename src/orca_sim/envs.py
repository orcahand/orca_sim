import sys
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from orca_sim.versions import (
    resolve_version,
    resolve_scene_path,
)


class BaseOrcaHandEnv(gym.Env[np.ndarray, np.ndarray]):
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

        self.version = resolve_version(version)
        self.scene_path = resolve_scene_path(scene_file, version=self.version)
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)

        self._default_camera = "closeup"
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
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    def _get_reward(self) -> float:
        return 0.0

    def _get_terminated(self) -> bool:
        return False

    def _get_truncated(self) -> bool:
        return False

    def _get_info(self) -> dict[str, Any]:
        return {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        if options and "qpos" in options:
            qpos = np.asarray(options["qpos"], dtype=np.float64)
            if qpos.shape != self.data.qpos.shape:
                raise ValueError(
                    f"Expected qpos shape {self.data.qpos.shape}, got {qpos.shape}"
                )
            self.data.qpos[:] = qpos

        if options and "qvel" in options:
            qvel = np.asarray(options["qvel"], dtype=np.float64)
            if qvel.shape != self.data.qvel.shape:
                raise ValueError(
                    f"Expected qvel shape {self.data.qvel.shape}, got {qvel.shape}"
                )
            self.data.qvel[:] = qvel

        mujoco.mj_forward(self.model, self.data)

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {action.shape}"
            )

        self.data.ctrl[:] = np.clip(action, self.action_low, self.action_high)
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

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
                            "`mjpython`, not plain `python3`. Run "
                            "`mjpython scripts/smoke_test_env.py --render-mode human` "
                            "for the interactive viewer, or use "
                            "`python3 scripts/smoke_test_env.py --render-mode rgb_array` "
                            "for an offscreen smoke test."
                        ) from exc
                    raise
                # Force viewer free-camera to scene.xml defaults on startup.
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


class OrcaHandLeft(BaseOrcaHandEnv):
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


class OrcaHandRight(BaseOrcaHandEnv):
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


class OrcaHandCombined(BaseOrcaHandEnv):
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


class OrcaHandLeftExtended(BaseOrcaHandEnv):
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


class OrcaHandRightExtended(BaseOrcaHandEnv):
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


class OrcaHandCombinedExtended(BaseOrcaHandEnv):
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

