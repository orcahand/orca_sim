from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SCENES_ROOT = PACKAGE_ROOT / "scenes"
LATEST_VERSION = "v2"


def list_versions() -> tuple[str, ...]:
    return tuple(_discover_versions())


def latest_version() -> str:
    versions = set(list_versions())
    if not versions:
        raise FileNotFoundError(f"No embodiment versions found under {SCENES_ROOT}")
    if LATEST_VERSION not in versions:
        known_versions = ", ".join(sorted(versions)) or "none"
        raise FileNotFoundError(
            f"LATEST_VERSION is set to '{LATEST_VERSION}', but that version was not found. "
            f"Available versions: {known_versions}"
        )
    return LATEST_VERSION


def resolve_version(version: str | None = None) -> str:
    if version is None:
        return latest_version()

    versions = set(list_versions())
    if version not in versions:
        known_versions = ", ".join(sorted(versions)) or "none"
        raise FileNotFoundError(
            f"Unknown embodiment version '{version}'. Available versions: {known_versions}"
        )
    return version


def resolve_scene_path(scene_file: str, *, version: str | None = None) -> Path:
    if version is not None:
        resolved_version = resolve_version(version)
        scene_path = SCENES_ROOT / resolved_version / scene_file
        if not scene_path.exists():
            raise FileNotFoundError(
                f"Embodiment version '{resolved_version}' is missing required scene file: "
                f"{scene_file}"
            )
        return scene_path

    for candidate_version in _scene_resolution_order():
        scene_path = SCENES_ROOT / candidate_version / scene_file
        if scene_path.exists():
            return scene_path

    known_versions = ", ".join(list_versions()) or "none"
    raise FileNotFoundError(
        f"No embodiment version provides scene file '{scene_file}'. "
        f"Available versions: {known_versions}"
    )


def _discover_versions() -> list[str]:
    discovered: list[str] = []
    if not SCENES_ROOT.exists():
        return discovered

    for version_dir in sorted(SCENES_ROOT.iterdir(), key=lambda path: _version_sort_key(path.name)):
        if not version_dir.is_dir():
            continue
        if not (version_dir / "scene_left.xml").exists():
            continue
        discovered.append(version_dir.name)

    return discovered


def _scene_resolution_order() -> list[str]:
    versions = list(list_versions())
    preferred: list[str] = []
    if LATEST_VERSION in versions:
        preferred.append(LATEST_VERSION)
    preferred.extend(version for version in reversed(versions) if version != LATEST_VERSION)
    return preferred


def _version_sort_key(version: str) -> tuple[int, int | str]:
    if version.startswith("v") and version[1:].isdigit():
        return (0, int(version[1:]))
    return (1, version)
