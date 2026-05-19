from __future__ import annotations

from pathlib import Path

from orca_sim.builders.orcapanda_mjcf import build_orcapanda_mjcf


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "scenes"
        / "includes"
        / "orcapanda_namespaced.xml"
    )


def main() -> None:
    print(build_orcapanda_mjcf(default_output_path()))


if __name__ == "__main__":
    main()
