import argparse
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from orca_sim import (
        OrcaHandCombined,
        OrcaHandCombinedExtended,
        OrcaHandLeft,
        OrcaHandLeftExtended,
        OrcaHandRight,
        OrcaHandRightCubeOrientation,
        OrcaHandRightExtended,
    )
except ModuleNotFoundError as exc:
    if exc.name in {"gymnasium", "mujoco", "numpy"}:
        raise SystemExit(
            "Missing runtime dependency "
            f"'{exc.name}'. Activate your environment and run `uv pip install -e .`."
        ) from exc
    raise


ENV_BUILDERS = {
    "left": OrcaHandLeft,
    "left_extended": OrcaHandLeftExtended,
    "right": OrcaHandRight,
    "right_cube_orientation": OrcaHandRightCubeOrientation,
    "right_extended": OrcaHandRightExtended,
    "combined": OrcaHandCombined,
    "combined_extended": OrcaHandCombinedExtended,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a random policy in an ORCA MuJoCo environment."
    )
    parser.add_argument(
        "--env",
        choices=sorted(ENV_BUILDERS),
        default="combined",
        help="Environment variant to load.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Embodiment version to load, for example 'v1' or 'v2'.",
    )
    parser.add_argument(
        "--render-mode",
        choices=["human", "rgb_array"],
        default="human",
        help="Use 'human' for the MuJoCo viewer or 'rgb_array' for offscreen rendering.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Number of random-action steps to run. Use 0 to run until Ctrl+C.",
    )
    args = parser.parse_args()

    env_cls = ENV_BUILDERS[args.env]
    env = env_cls(render_mode=args.render_mode, version=args.version)

    obs, info = env.reset()
    print(f"env={args.env}")
    print(f"version={env.version}")
    print(f"obs_shape={obs.shape}")
    print(f"action_shape={env.action_space.shape}")
    print(f"info={info}")

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
                    f"reward={reward} terminated={terminated} truncated={truncated}"
                )
            else:
                print(
                    f"step={step} reward={reward} "
                    f"terminated={terminated} truncated={truncated}"
                )
                time.sleep(1.0 / env.metadata["render_fps"])

            if terminated or truncated:
                obs, info = env.reset()
                print(f"reset step={step} info={info}")

            step += 1
    except KeyboardInterrupt:
        print("stopped by user")
    finally:
        env.close()


if __name__ == "__main__":
    main()
