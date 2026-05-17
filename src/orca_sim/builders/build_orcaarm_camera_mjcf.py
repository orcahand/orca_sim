from __future__ import annotations

import argparse
from pathlib import Path

from orca_sim.builders.orcaarm_camera_mjcf import build_orcaarm_camera_mjcf


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "scenes"
        / "includes"
        / "orcabot_with_cameras.xml"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug-sites",
        action="store_true",
        help="Add visible marker sites for camera mounts and aim targets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help="Path to write the built MJCF include.",
    )
    args = parser.parse_args()

    print(build_orcaarm_camera_mjcf(args.output, debug_sites=args.debug_sites))


if __name__ == "__main__":
    main()
