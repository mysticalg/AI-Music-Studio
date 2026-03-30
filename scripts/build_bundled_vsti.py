from __future__ import annotations

import argparse
import os
import platform
import shutil
import stat
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def ensure_juce_clone(juce_dir: Path) -> None:
    if juce_dir.exists():
        return

    juce_dir.parent.mkdir(parents=True, exist_ok=True)
    run(
        ["git", "clone", "--depth", "1", "https://github.com/juce-framework/JUCE.git", str(juce_dir)],
        cwd=juce_dir.parent,
    )


def _remove_readonly(func, path, excinfo) -> None:
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, onexc=_remove_readonly)


def copy_tree_filtered(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".git", ".gitignore", ".gitmodules", ".gitattributes"),
    )


def default_build_dir(repo_root: Path) -> Path:
    system = platform.system()
    if system == "Windows":
        suffix = "windows"
    elif system == "Darwin":
        suffix = "macos"
    else:
        suffix = "linux"
    return repo_root / "plugins" / "AdvancedVSTi" / f"build-{suffix}"


def collect_plugin_bundles(build_dir: Path) -> list[Path]:
    bundles = [path for path in build_dir.rglob("*.vst3")]
    bundles.sort(key=lambda path: (len(path.parts), str(path).lower()))
    return bundles


def main() -> int:
    parser = argparse.ArgumentParser(description="Build bundled AdvancedVSTi plugins and copy VST3 bundles into an output folder.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--juce-dir", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    juce_dir = (args.juce_dir or (repo_root / ".cache" / "JUCE")).resolve()
    build_dir = (args.build_dir or default_build_dir(repo_root)).resolve()
    output_dir = (args.output_dir or (repo_root / "vsti")).resolve()
    plugin_root = repo_root / "plugins" / "AdvancedVSTi"

    ensure_juce_clone(juce_dir)

    configure_command = [
        "cmake",
        "-S",
        str(plugin_root),
        "-B",
        str(build_dir),
        f"-DJUCE_DIR={juce_dir}",
    ]
    if platform.system() != "Windows":
        configure_command.append("-DCMAKE_BUILD_TYPE=Release")

    run(configure_command, cwd=repo_root)
    run(["cmake", "--build", str(build_dir), "--config", "Release"], cwd=repo_root)

    plugin_bundles = collect_plugin_bundles(build_dir)
    if not plugin_bundles:
        raise FileNotFoundError(f"No .vst3 bundles were produced under {build_dir}")

    if output_dir.exists():
        remove_tree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for bundle in plugin_bundles:
        destination = output_dir / bundle.name
        if bundle.is_dir():
            copy_tree_filtered(bundle, destination)
        else:
            shutil.copy2(bundle, destination)

    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
