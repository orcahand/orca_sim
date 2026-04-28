"""ManiSkill PushCube task variant using the orca_arm bimanual robot.

Table + cube + goal target; same physics as ManiSkill's stock PushCube-v1,
but with orca_arm loaded in place of the panda.
"""
from typing import Any

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from orca_sim.maniskill.agent import OrcaArm


@register_env("PushCubeOrcaArm-v1", max_episode_steps=200)
class PushCubeOrcaArmEnv(BaseEnv):
    """Push a cube into a goal region using the orca_arm bimanual robot."""

    SUPPORTED_ROBOTS = ["orca_arm"]
    agent: OrcaArm

    goal_radius = 0.1
    cube_half_size = 0.025

    def __init__(self, *args, robot_uids="orca_arm", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**18,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 0.0, 0.8], target=[0.0, 0.0, 0.2])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.2, 1.0, 1.0], [0.0, 0.0, 0.4])
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=512,
            height=512,
            fov=1.0,
            near=0.01,
            far=100,
        )

    def _load_agent(self, options: dict):
        # The orca_arm "world" link sits at z=0 in URDF coordinates and the
        # base column extends upward, so we place it just behind the table
        # edge so the hands hover over the table surface.
        super()._load_agent(options, sapien.Pose(p=[-0.7, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

        self.obj = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 1e-3]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.cube_half_size
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=[1, 0, 0, 0]))

            target_xyz = xyz + torch.tensor([0.1 + self.goal_radius, 0, 0])
            target_xyz[..., 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=target_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(
                self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        ) & (self.obj.pose.p[..., 2] < self.cube_half_size + 5e-3)
        return {"success": is_obj_placed}

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp_pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                goal_pos=self.goal_region.pose.p,
                obj_pose=self.obj.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action, info: dict):
        # Distance from right-hand TCP to a push pose just behind the cube.
        push_pose = Pose.create_from_pq(
            p=self.obj.pose.p
            + torch.tensor(
                [-self.cube_half_size - 0.01, 0, 0], device=self.device
            )
        )
        tcp_to_push = push_pose.p - self.agent.tcp_pose.p
        reach_reward = 1 - torch.tanh(5 * torch.linalg.norm(tcp_to_push, axis=1))
        reward = reach_reward

        obj_to_goal = torch.linalg.norm(
            self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal)
        reward += place_reward

        reward[info["success"]] = 4
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action, info: dict):
        return self.compute_dense_reward(obs, action, info) / 4.0
