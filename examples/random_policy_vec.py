import argparse
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from orca_sim.envs_mjx import (
        OrcaHandCombinedExtendedMjx,
        OrcaHandCombinedMjx,
        OrcaHandLeftExtendedMjx,
        OrcaHandLeftMjx,
        OrcaHandRightExtendedMjx,
        OrcaHandRightMjx,
    )
    from orca_sim.task_envs_mjx import OrcaHandRightCubeOrientationMjx
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing runtime dependency '{exc.name}'. Activate the conda env "
        "with JAX + mujoco-mjx (e.g. `conda activate orca`)."
    ) from exc


ENV_BUILDERS = {
    "left": OrcaHandLeftMjx,
    "left_extended": OrcaHandLeftExtendedMjx,
    "right": OrcaHandRightMjx,
    "right_cube_orientation": OrcaHandRightCubeOrientationMjx,
    "right_extended": OrcaHandRightExtendedMjx,
    "combined": OrcaHandCombinedMjx,
    "combined_extended": OrcaHandCombinedExtendedMjx,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a random policy in an MJX-backed ORCA environment."
    )
    parser.add_argument(
        "--env",
        choices=sorted(ENV_BUILDERS),
        default="left",
        help="Environment variant to load.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Embodiment version, e.g. 'v1' or 'v2'.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel envs in the batched MJX rollout.",
    )
    parser.add_argument(
        "--render-mode",
        choices=["human", "rgb_array", "headless"],
        default="human",
        help="'headless' skips all GPU<->CPU sync and renderer setup.",
    )
    parser.add_argument(
        "--render-index",
        type=int,
        default=0,
        help="Which env in the batch to display when rendering.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Number of random-action steps. 0 = run until Ctrl+C.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for the action-space RNG.",
    )
    args = parser.parse_args()

    render_mode = None if args.render_mode == "headless" else args.render_mode
    env = ENV_BUILDERS[args.env](
        num_envs=args.num_envs,
        version=args.version,
        render_mode=render_mode,
        render_index=args.render_index,
    )

    if args.seed is not None:
        env.action_space.seed(args.seed)

    obs, info = env.reset()
    print(f"env={args.env} version={env.version} num_envs={args.num_envs}")
    print(f"obs_shape={obs.shape}")
    print(f"action_shape={env.action_space.shape}")
    print(f"info={info}")

    step = 0
    try:
        while True:
            if args.steps and step >= args.steps:
                break

            action = env.action_space.sample()
            t0 = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            dt = time.perf_counter() - t0

            mean_reward = float(reward.mean())
            if render_mode == "rgb_array":
                frame = env.render()
                print(
                    f"step={step} frame_shape={None if frame is None else frame.shape} "
                    f"step_dt={dt*1000:.2f}ms mean_reward={mean_reward:.4f}"
                )
            else:
                print(
                    f"step={step} step_dt={dt*1000:.2f}ms "
                    f"mean_reward={mean_reward:.4f} "
                    f"any_terminated={bool(terminated.any())} "
                    f"any_truncated={bool(truncated.any())}"
                )
                if render_mode == "human":
                    time.sleep(max(0.0, 1.0 / env.metadata["render_fps"] - dt))

            step += 1
    except KeyboardInterrupt:
        print("stopped by user")
    finally:
        env.close()


if __name__ == "__main__":
    main()
