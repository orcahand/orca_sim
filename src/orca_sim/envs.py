from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from orca_sim.hand import SimOrcaHand

RENDER_FPS = 30

class BaseOrcaHandEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": RENDER_FPS}

    def __init__(
        self,
        scene_file: str,
        version: str | None = None,
        frame_skip: int = 5,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.hand = SimOrcaHand(
            scene_file=scene_file,
            version=version,
            frame_skip=frame_skip,
            render_mode=render_mode,
        )
        self.scene_path = self.hand.scene_path
        self.version = self.hand.version
        self.frame_skip = self.hand.frame_skip
        self.render_mode = self.hand.render_mode
        self.model = self.hand.model
        self.data = self.hand.data
        self.action_low = self.hand.action_low
        self.action_high = self.hand.action_high
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
        return self.hand.observe()

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
        options = {} if options is None else dict(options)
        self.hand.reset(qpos=options.get("qpos"), qvel=options.get("qvel"))

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.hand.step(action)

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        return self.hand.render()

    def close(self) -> None:
        self.hand.close()


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

