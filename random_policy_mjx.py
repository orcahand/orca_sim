import argparse
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from orca_sim.envs_mjx import (
        OrcaHandCombinedExtendedMjx,
        OrcaHandCombinedMjx,
        OrcaHandLeftExtendedMjx,
        OrcaHandLeftMjx,
        OrcaHandMjxVectorEnv,
        OrcaHandRightExtendedMjx,
        OrcaHandRightMjx,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing runtime dependency '{exc.name}'. Activate the conda env "
        "with JAX + mujoco-mjx (e.g. `conda activate orca`)."
    ) from exc


SINGLE_ENV_BUILDERS = {
    "left": OrcaHandLeftMjx,
    "left_extended": OrcaHandLeftExtendedMjx,
    "right": OrcaHandRightMjx,
    "right_extended": OrcaHandRightExtendedMjx,
    "combined": OrcaHandCombinedMjx,
    "combined_extended": OrcaHandCombinedExtendedMjx,
}

ENV_TO_SCENE = {
    "left": "scene_left.xml",
    "left_extended": "scene_left_extended.xml",
    "right": "scene_right.xml",
    "right_extended": "scene_right_extended.xml",
    "combined": "scene_combined.xml",
    "combined_extended": "scene_combined_extended.xml",
}

def run_single(args: argparse.Namespace) -> None:
    print("Running single")
    env_cls = SINGLE_ENV_BUILDERS[args.env]
    env = env_cls(render_mode=args.render_mode, version=args.version)

    print("resetting")
    obs, info = env.reset()
    print("finished resetting")
    print(f"mode=single env={args.env} version={env.version}")
    print(f"obs_shape={obs.shape} action_shape={env.action_space.shape}")
    print(f"qpos_device={env.mjx_data.qpos.devices()}")

    step = 0
    try:
        while True:
            if args.steps and step >= args.steps:
                break
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if args.render_mode == "rgb_array":
                frame = env.render()
                print(
                    f"step={step} frame_shape={None if frame is None else frame.shape} "
                    f"reward={reward}"
                )
            else:
                print(f"step={step} reward={reward}")
                time.sleep(1.0 / env.metadata["render_fps"])
            if terminated or truncated:
                obs, info = env.reset()
            step += 1
    except KeyboardInterrupt:
        print("stopped by user")
    finally:
        env.close()


def run_vector(args: argparse.Namespace) -> None:
    scene = ENV_TO_SCENE[args.env]
    env = OrcaHandMjxVectorEnv(
        scene,
        num_envs=args.num_envs,
        version=args.version,
        render_mode=args.render_mode,
        render_index=args.render_index,
    )

    obs, info = env.reset()
    print(f"mode=vector env={args.env} version={env.version} num_envs={env.num_envs}")
    print(f"obs_shape={obs.shape} action_shape={env.action_space.shape}")
    print(f"qpos_device={env.mjx_data.qpos.devices()}")

    step = 0
    try:
        while True:
            if args.steps and step >= args.steps:
                break
            actions = env.action_space.sample()
            t0 = time.perf_counter()
            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            dt = time.perf_counter() - t0
            if args.render_mode == "rgb_array":
                frame = env.render()
                print(
                    f"step={step} frame_shape={None if frame is None else frame.shape} "
                    f"step_dt={dt*1000:.2f}ms"
                )
            else:
                print(f"step={step} step_dt={dt*1000:.2f}ms")
                time.sleep(max(0.0, 1.0 / env.metadata["render_fps"] - dt))
            step += 1
    except KeyboardInterrupt:
        print("stopped by user")
    finally:
        env.close()


def main() -> None:
    print("Main loop")
    parser = argparse.ArgumentParser(
        description="Run a random policy in an MJX-backed ORCA environment."
    )
    parser.add_argument(
        "--env",
        choices=sorted(SINGLE_ENV_BUILDERS),
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
        help="Number of parallel envs. 1 -> single-env path; >1 -> vectorized.",
    )
    parser.add_argument(
        "--render-mode",
        choices=["human", "rgb_array"],
        default="human",
    )
    parser.add_argument(
        "--render-index",
        type=int,
        default=0,
        help="Which env to render in vectorized mode.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Number of steps. 0 = run until Ctrl+C.",
    )
    args = parser.parse_args()

    print("Args parsed")

    if args.num_envs == 1:
        run_single(args)
    else:
        run_vector(args)


if __name__ == "__main__":
    main()
