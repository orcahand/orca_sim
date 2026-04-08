import argparse
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from orca_sim import SimOrcaHand
except ModuleNotFoundError as exc:
    if exc.name in {"gymnasium", "mujoco", "numpy", "orca_core"}:
        raise SystemExit(
            "Missing runtime dependency "
            f"'{exc.name}'. Activate your environment and run `uv pip install -e .`."
        ) from exc
    raise


SCENE_CHOICES = (
    "scene_left.xml",
    "scene_right.xml",
    "scene_combined.xml",
    "scene_left_extended.xml",
    "scene_right_extended.xml",
    "scene_combined_extended.xml",
    "scene_right_cube_orientation.xml",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate the shared orca_core hand tooling on top of the MuJoCo sim. "
            "The script samples random actions and sends the hand back to neutral "
            "every N steps while rendering."
        )
    )
    parser.add_argument(
        "--scene-file",
        choices=SCENE_CHOICES,
        default="scene_right.xml",
        help="Scene XML to load.",
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
        "--cycles",
        type=int,
        default=10,
        help="Demo cycles to run.",
    )
    
    args = parser.parse_args()

    hand = SimOrcaHand(
        scene_file=args.scene_file,
        version=args.version,
        render_mode=args.render_mode,
    )

    import logging
    logging.info(f"scene={args.scene_file}")
    logging.info(f"version={hand.version}")
    logging.info(f"num_joints={len(hand.config.joint_ids)}")
    logging.info(f"joint_ids={list(hand.config.joint_ids)}")

    try:
        hand.run_demo(demo_name="main", cycles=args.cycles, step_size=0.01)
        

    except KeyboardInterrupt:
        logging.info("WARNING! Demo stopped by user")
    finally:
        hand.close()


if __name__ == "__main__":
    main()
