import os
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCE = ROOT / "src" / "orca_sim"
DATA_SUFFIXES = {".mjcf", ".stl", ".xml"}


def test_wheel_contains_model_data_and_initializes_versions(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(dist_dir),
            str(ROOT),
        ],
        check=True,
    )

    [wheel] = dist_dir.glob("*.whl")
    expected_data = {
        path.relative_to(PACKAGE_SOURCE.parent).as_posix()
        for path in PACKAGE_SOURCE.rglob("*")
        if path.is_file() and path.suffix in DATA_SUFFIXES
    }
    with zipfile.ZipFile(wheel) as archive:
        packaged_data = {
            name for name in archive.namelist() if Path(name).suffix in DATA_SUFFIXES
        }

    assert expected_data <= packaged_data

    venv_dir = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        check=True,
    )
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    venv_python = venv_dir / scripts_dir / ("python.exe" if os.name == "nt" else "python")
    subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            str(wheel),
        ],
        check=True,
    )

    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    smoke_test = f"""
from pathlib import Path
import orca_sim
from orca_sim import OrcaHandRight

source_root = Path({str(ROOT)!r})
assert source_root not in Path(orca_sim.__file__).resolve().parents

for version in ("v1", "v2"):
    env = OrcaHandRight(version=version)
    try:
        observation, info = env.reset()
        assert observation.shape == (34,)
        assert info == {{}}
    finally:
        env.close()
"""
    subprocess.run(
        [str(venv_python), "-c", smoke_test],
        cwd=tmp_path,
        env=clean_env,
        check=True,
    )
