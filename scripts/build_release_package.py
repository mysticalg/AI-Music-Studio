from __future__ import annotations

import argparse
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

from build_bundled_vsti import ensure_juce_clone, run


def normalize_arch() -> str:
    machine = platform.machine().lower()
    if machine in {"amd64", "x86_64", "x64"}:
        return "x64"
    if machine in {"arm64", "aarch64"}:
        return "arm64"
    return machine or "unknown"


def default_build_dir(repo_root: Path) -> Path:
    system = platform.system()
    if system == "Windows":
        suffix = "native-app"
    elif system == "Darwin":
        suffix = "native-app-macos"
    else:
        suffix = "native-app-linux"
    return repo_root / "build" / suffix


def locate_app_artifact(build_dir: Path) -> Path:
    system = platform.system()
    if system == "Windows":
        candidates = [path for path in build_dir.rglob("Mutagen.exe") if path.is_file()]
    elif system == "Darwin":
        candidates = [path for path in build_dir.rglob("Mutagen.app") if path.is_dir()]
    else:
        candidates = [path for path in build_dir.rglob("Mutagen") if path.is_file()]

    candidates.sort(key=lambda path: (len(path.parts), str(path).lower()))
    if not candidates:
        raise FileNotFoundError(f"Could not locate built Mutagen artifact under {build_dir}")
    return candidates[0]


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


def stage_release_tree(repo_root: Path, stage_dir: Path, app_artifact: Path, release_version: str) -> None:
    if stage_dir.exists():
        remove_tree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    app_destination = stage_dir / app_artifact.name
    if app_artifact.is_dir():
        copy_tree_filtered(app_artifact, app_destination)
    else:
        shutil.copy2(app_artifact, app_destination)

    bundled_vsti_dir = repo_root / "vsti"
    if bundled_vsti_dir.exists():
        copy_tree_filtered(bundled_vsti_dir, stage_dir / "vsti")

    support_dir = stage_dir / "support"
    support_dir.mkdir(parents=True, exist_ok=True)
    ace_step_installer_script = repo_root / "scripts" / "install_ace_step_backend.ps1"
    if ace_step_installer_script.exists():
        shutil.copy2(ace_step_installer_script, support_dir / ace_step_installer_script.name)

    shutil.copy2(repo_root / "README.md", stage_dir / "README.md")
    guide_path = repo_root / "GUIDE.md"
    if guide_path.exists():
        shutil.copy2(guide_path, stage_dir / "GUIDE.md")
    (stage_dir / "VERSION.txt").write_text(f"{release_version}\n", encoding="ascii")


def create_archive(stage_dir: Path, archive_path: Path) -> None:
    if archive_path.exists():
        archive_path.unlink()

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(stage_dir.rglob("*")):
                if path.is_dir():
                    continue
                archive.write(path, arcname=path.relative_to(stage_dir))
    else:
        with tarfile.open(archive_path, "w:gz") as archive:
            for path in sorted(stage_dir.iterdir()):
                archive.add(path, arcname=path.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and package a native Mutagen release for the current platform.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--release-version", default="dev")
    parser.add_argument("--configuration", default="Release")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--juce-dir", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--skip-bundled-vst-build", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = (args.output_dir or (repo_root / "dist")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    juce_dir = (args.juce_dir or (repo_root / ".cache" / "JUCE")).resolve()
    ensure_juce_clone(juce_dir)

    if not args.skip_bundled_vst_build:
        build_bundled_vsti_script = Path(__file__).with_name("build_bundled_vsti.py")
        subprocess.run(
            [
                sys.executable,
                str(build_bundled_vsti_script),
                "--repo-root",
                str(repo_root),
                "--juce-dir",
                str(juce_dir),
                "--output-dir",
                str(repo_root / "vsti"),
            ],
            cwd=repo_root,
            check=True,
        )

    build_dir = (args.build_dir or default_build_dir(repo_root)).resolve()
    source_dir = repo_root / "native" / "app"
    configure_command = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        f"-DJUCE_DIR={juce_dir}",
    ]
    if platform.system() != "Windows":
        configure_command.append(f"-DCMAKE_BUILD_TYPE={args.configuration}")

    run(configure_command, cwd=repo_root)
    run(
        ["cmake", "--build", str(build_dir), "--config", args.configuration, "--target", "AIMusicStudioNative"],
        cwd=repo_root,
    )

    app_artifact = locate_app_artifact(build_dir)
    stage_dir = output_dir / "Mutagen"
    stage_release_tree(repo_root, stage_dir, app_artifact, args.release_version)

    system = platform.system()
    arch = normalize_arch()
    release_label = "".join(char if char.isalnum() or char in "._-" else "-" for char in args.release_version)
    if system == "Windows":
        archive_name = f"Mutagen-{release_label}-windows-{arch}.zip"
    elif system == "Darwin":
        archive_name = f"Mutagen-{release_label}-macos-{arch}.zip"
    else:
        archive_name = f"Mutagen-{release_label}-linux-{arch}.tar.gz"

    archive_path = output_dir / archive_name
    create_archive(stage_dir, archive_path)
    print(archive_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
