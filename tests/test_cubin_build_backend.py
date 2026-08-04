import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_local_version_is_applied_before_package_import(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    isolated_repository = tmp_path / "repository"
    cubin_package = isolated_repository / "flashinfer-cubin"
    shutil.copytree(repository / "flashinfer-cubin", cubin_package)
    shutil.copy2(repository / "build_utils.py", isolated_repository)
    shutil.copy2(repository / "version.txt", isolated_repository)
    (isolated_repository / ".git").mkdir()

    stale_metadata = cubin_package / "flashinfer_cubin" / "_build_meta.py"
    stale_metadata.write_text(
        '__version__ = "0.0.0+stale"\n__git_version__ = "stale"\n',
        encoding="utf-8",
    )

    output = isolated_repository / "metadata"
    output.mkdir()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["FLASHINFER_LOCAL_VERSION"] = "metadata1"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import build_backend, sys; "
            "build_backend.prepare_metadata_for_build_wheel(sys.argv[1])",
            str(output),
        ],
        cwd=cubin_package,
        env=environment,
        check=True,
    )

    metadata_files = list(output.glob("*.dist-info/METADATA"))
    assert len(metadata_files) == 1
    expected_version = (repository / "version.txt").read_text().strip()
    assert f"Version: {expected_version}+metadata1\n" in metadata_files[0].read_text()
