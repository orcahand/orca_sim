"""Run the PushCubeOrcaArm-v1 environment with random actions.

Usage:
    python -m orca_sim.maniskill.run                  # interactive viewer
    python -m orca_sim.maniskill.run --no-render      # headless smoke test
    python -m orca_sim.maniskill.run --record out.mp4 # offscreen video

Requires the ``maniskill`` optional dependency group:
    pip install 'orca_sim[maniskill]'
"""
import argparse

import gymnasium as gym
import numpy as np

# Importing the subpackage registers the agent and env with ManiSkill.
import orca_sim.maniskill  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Path to save an mp4 of the rollout (rgb_array mode).",
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.record or args.no_render:
        render_mode = "rgb_array"
    else:
        render_mode = "human"

    env = gym.make(
        "PushCubeOrcaArm-v1",
        obs_mode="state",
        control_mode="pd_joint_pos",
        render_mode=render_mode,
    )
    obs, info = env.reset(seed=args.seed)
    print(f"action space: {env.action_space}")
    print(f"obs keys: {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")

    rng = np.random.default_rng(args.seed)
    action_low = env.action_space.low
    action_high = env.action_space.high

    frames = []
    for step in range(args.steps):
        # Small jitter around the rest pose so the robot doesn't fly apart.
        action = rng.uniform(action_low, action_high) * 0.05
        obs, reward, terminated, truncated, info = env.step(action)
        if args.record:
            frame = env.render()
            if hasattr(frame, "cpu"):
                frame = frame.cpu().numpy()
            if frame.ndim == 4:
                frame = frame[0]
            frames.append(frame)
        elif not args.no_render:
            env.render()
        if step % 30 == 0:
            print(
                f"step {step:4d}  reward={float(reward):+.3f}  "
                f"success={bool(info.get('success', False))}"
            )
        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    if args.record and frames:
        import imageio.v2 as imageio

        imageio.mimsave(args.record, frames, fps=30)
        print(f"saved {len(frames)} frames to {args.record}")

    print("done.")


if __name__ == "__main__":
    main()
