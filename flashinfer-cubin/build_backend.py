"""
Custom build backend that downloads cubins before building the package.
"""

import os
import shutil
import sys
from pathlib import Path
from setuptools import build_meta as _orig

# Add parent directory to path to import artifacts module
sys.path.insert(0, str(Path(__file__).parent.parent))

from build_utils import get_git_version

# Skip version check when building flashinfer-cubin package
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"


def _copy_tree_contents(source: Path, destination: Path):
    """Populate destination from source, using hardlinks when possible."""
    if not source.exists():
        return

    for path in source.rglob("*"):
        rel_path = path.relative_to(source)
        if any(
            part.endswith(".lock") or part.endswith(".tmp")
            for part in rel_path.parts
        ):
            continue

        target = destination / rel_path
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            try:
                if os.path.samefile(path, target):
                    continue
            except OSError:
                pass
            target.unlink()

        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)


def _download_cubins():
    """Download cubins to the source directory before building."""
    # Create cubins directory in the source tree
    cubin_dir = Path(__file__).parent / "flashinfer_cubin" / "cubins"
    cubin_dir.mkdir(parents=True, exist_ok=True)
    cache_dir_value = os.environ.get("FLASHINFER_CUBIN_CACHE_DIR")
    cache_dir = Path(cache_dir_value) if cache_dir_value else None

    # Set environment variable to download to our package directory
    original_cubin_dir = os.environ.get("FLASHINFER_CUBIN_DIR")
    os.environ["FLASHINFER_CUBIN_DIR"] = str(cubin_dir)

    try:
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Restoring cached cubins from {cache_dir} to {cubin_dir}...")
            _copy_tree_contents(cache_dir, cubin_dir)

        from flashinfer import artifacts
        from flashinfer.jit import cubin_loader
        from flashinfer.jit import env as jit_env

        # These modules cache FLASHINFER_CUBIN_DIR at import time.
        jit_env.FLASHINFER_CUBIN_DIR = cubin_dir
        cubin_loader.FLASHINFER_CUBIN_DIR = cubin_dir
        artifacts.FLASHINFER_CUBIN_DIR = cubin_dir

        print(f"Downloading cubins to {cubin_dir}...")
        artifacts.download_artifacts()
        print(f"Successfully downloaded cubins to {cubin_dir}")

        # Count the downloaded files
        cubin_files = list(cubin_dir.rglob("*.cubin"))
        print(f"Downloaded {len(cubin_files)} cubin files")

        if cache_dir is not None:
            print(f"Updating cubin cache at {cache_dir}...")
            _copy_tree_contents(cubin_dir, cache_dir)

    finally:
        # Restore original environment variable
        if original_cubin_dir:
            os.environ["FLASHINFER_CUBIN_DIR"] = original_cubin_dir
        else:
            os.environ.pop("FLASHINFER_CUBIN_DIR", None)


def _create_build_metadata():
    """Create build metadata file with version information."""
    version_file = Path(__file__).parent.parent / "version.txt"
    if version_file.exists():
        with open(version_file, "r") as f:
            version = f.read().strip()
    else:
        version = "0.0.0+unknown"

    # Add dev suffix if specified
    dev_suffix = os.environ.get("FLASHINFER_DEV_RELEASE_SUFFIX", "")
    if dev_suffix:
        version = f"{version}.dev{dev_suffix}"

    # Get git version
    git_version = get_git_version(cwd=Path(__file__).parent.parent)

    # Append local version suffix if available
    local_version = os.environ.get("FLASHINFER_LOCAL_VERSION")
    if local_version:
        # Use + to create a local version identifier that will appear in wheel name
        version = f"{version}+{local_version}"

    # Create build metadata in the source tree
    package_dir = Path(__file__).parent / "flashinfer_cubin"
    build_meta_file = package_dir / "_build_meta.py"

    # Check if we're in a git repository
    git_dir = Path(__file__).parent.parent / ".git"
    in_git_repo = git_dir.exists()

    # If file exists and not in git repo (installing from sdist), keep existing file
    if build_meta_file.exists() and not in_git_repo:
        print("Build metadata file already exists (not in git repo), keeping it")
        return version

    # In git repo (editable) or file doesn't exist, create/update it
    with open(build_meta_file, "w") as f:
        f.write('"""Build metadata for flashinfer-cubin package."""\n')
        f.write(f'__version__ = "{version}"\n')
        f.write(f'__git_version__ = "{git_version}"\n')

    print(f"Created build metadata file with version {version}")
    return version


# Create build metadata as soon as this module is imported
_create_build_metadata()


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build a wheel, downloading cubins first."""
    _download_cubins()
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Build an editable install, downloading cubins first."""
    _download_cubins()
    return _orig.build_editable(wheel_directory, config_settings, metadata_directory)


# Pass through all other hooks
get_requires_for_build_wheel = _orig.get_requires_for_build_wheel
get_requires_for_build_editable = _orig.get_requires_for_build_editable
prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel
prepare_metadata_for_build_editable = _orig.prepare_metadata_for_build_editable
