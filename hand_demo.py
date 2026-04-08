import argparse
import logging

from orca_core.demo_presets import DEMO_POSE_FRACTIONS, DEMO_SEQUENCES

from orca_sim import SimOrcaHand


SCENE_CHOICES = (
    "scene_left.xml",
    "scene_right.xml",
    "scene_combined.xml",
    "scene_left_extended.xml",
    "scene_right_extended.xml",
    "scene_combined_extended.xml",
    "scene_right_cube_orientation.xml",
)


def _expand_fractions_bimanual(fractions: dict[str, float]) -> dict[str, float]:
    """Prefix bare canonical joint fractions with both ``left_`` and ``right_``."""
    return {
        f"{side}_{joint}": value
        for side in ("left", "right")
        for joint, value in fractions.items()
    }


def run_demo(
    hand: SimOrcaHand,
    demo_name: str = "main",
    cycles: int = 1,
    num_steps: int = 25,
    step_size: float = 0.05,
    return_to_neutral: bool = True,
) -> None:
    """Run a named demo sequence on any scene, including combined (bimanual) ones.

    For single-hand scenes this delegates directly to ``hand.run_demo``.  For
    combined scenes the bare canonical joint names in the demo presets are
    automatically expanded to ``left_*`` / ``right_*`` so both hands move in
    sync.
    """
    is_bimanual = hand.config.type is None  # bimanual hand has no "left" or "right"type
    if not is_bimanual:
        return hand.run_demo(
            demo_name=demo_name,
            cycles=cycles,
            num_steps=num_steps,
            step_size=step_size,
            return_to_neutral=return_to_neutral,
        )

    if demo_name not in DEMO_POSE_FRACTIONS:
        available = ", ".join(sorted(DEMO_SEQUENCES))
        raise ValueError(f"Unknown demo '{demo_name}'. Available demos: {available}.")

    for name, fractions in DEMO_POSE_FRACTIONS[demo_name].items():
        hand.register_position(
            name,
            hand.pose_from_fractions(_expand_fractions_bimanual(fractions)),
        )

    hand.play_named_positions(
        DEMO_SEQUENCES[demo_name],
        cycles=cycles,
        num_steps=num_steps,
        step_size=step_size,
        return_to_neutral=return_to_neutral,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play back orca_core demo presets on SimOrcaHand."
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
        help="Embodiment version to load, e.g. 'v1' or 'v2'.",
    )
    parser.add_argument(
        "--render-mode",
        choices=["human", "rgb_array"],
        default="human",
        help="Use 'human' for the MuJoCo viewer or 'rgb_array' for offscreen rendering.",
    )
    parser.add_argument(
        "--demo-name",
        choices=list(DEMO_SEQUENCES),
        default="main",
        help="Which demo sequence to play.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=10,
        help="Number of times to repeat the demo sequence.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=25,
        help="Interpolation steps between poses. More steps = smoother motion.",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.05,
        help="Sleep in seconds between interpolation steps. Increase to slow down playback.",
    )
    args = parser.parse_args()

    hand = SimOrcaHand(
        scene_file=args.scene_file,
        version=args.version,
        render_mode=args.render_mode,
    )
    logging.info(f"scene={args.scene_file} version={hand.version} num_joints={len(hand.config.joint_ids)}")

    try:
        hand.reset()
        run_demo(
            hand,
            demo_name=args.demo_name,
            cycles=args.cycles,
            num_steps=args.num_steps,
            step_size=args.step_size,
        )
    except KeyboardInterrupt:
        logging.info("Demo stopped by user.")
    finally:
        hand.close()


if __name__ == "__main__":
    main()
