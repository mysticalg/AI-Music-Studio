#!/usr/bin/env python3
from __future__ import annotations

import argparse
import posixpath
import re
import shutil
import subprocess
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / ".cache" / "OpenInstrumentSamples"

SFZ_OPCODE_RE = re.compile(r"([A-Za-z0-9_]+)=([^<>\r\n]*?)(?=(?:\s+[A-Za-z0-9_]+=)|$)")

BASS_RAW_ROOT = "https://raw.githubusercontent.com/freepats/electric-bass-YR/main/"

SSO_SPARSE_DIRS = [
    "Sonatina Symphonic Orchestra/Strings - Performance",
    "Sonatina Symphonic Orchestra/Woodwinds - Performance",
    "Sonatina Symphonic Orchestra/Samples/1st Violins",
    "Sonatina Symphonic Orchestra/Samples/Violas",
    "Sonatina Symphonic Orchestra/Samples/Celli",
    "Sonatina Symphonic Orchestra/Samples/Basses",
    "Sonatina Symphonic Orchestra/Samples/Violin 2",
    "Sonatina Symphonic Orchestra/Samples/Flute 1",
    "Sonatina Symphonic Orchestra/Samples/Flute 2",
    "Sonatina Symphonic Orchestra/Samples/Alto Flute",
    "Sonatina Symphonic Orchestra/Samples-looped/Violin 1",
    "Sonatina Symphonic Orchestra/Samples-looped/Violin 2",
    "Sonatina Symphonic Orchestra/Samples-looped/Flute 1",
]

BASS_TOP_LEVEL_FILES = [
    "FingerBassYR 20190930.sfz",
    "PickedBassYR 20190930.sfz",
    "LICENSE.txt",
    "README.txt",
]

ARCHIVES = [
    (
        "https://freepats.zenvoid.org/Reed/TenorSaxophone/TenorSaxophone-small-SFZ-20200717.tar.bz2",
        "Saxophone",
        "TenorSaxophone-small-SFZ-20200717.tar.bz2",
    ),
    (
        "https://freepats.zenvoid.org/Organ/ChurchOrganEmulation/ChurchOrganEmulation-SFZ-20190924.tar.xz",
        "Organ",
        "ChurchOrganEmulation-SFZ-20190924.tar.xz",
    ),
]


def quote_repo_path(path: str) -> str:
    return urllib.parse.quote(path.replace("\\", "/"), safe="/()+")


def normalize_repo_path(path: str | Path) -> str:
    return posixpath.normpath(str(path).replace("\\", "/"))


def fetch_text(url: str) -> str:
    try:
        with urllib.request.urlopen(url) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch text: {url}") from exc


def download_to(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except Exception as exc:
        raise RuntimeError(f"Failed to download: {url}") from exc


def iter_sfz_opcodes(line: str):
    for match in SFZ_OPCODE_RE.finditer(line):
        yield match.group(1).strip().lower(), match.group(2).strip()


def sfz_dependencies(text: str):
    includes: list[str] = []
    samples: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        include_match = re.match(r'#include\s+"([^"]+)"', line)
        if include_match:
            includes.append(include_match.group(1).strip())
            continue
        for key, value in iter_sfz_opcodes(raw_line):
            if key == "sample" and value:
                samples.append(value)
    return includes, samples


def run_git(args: list[str], cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def download_sso_tree(root: Path) -> None:
    checkout_dir = root / "_downloads" / "sso-sparse"
    source_dir = checkout_dir / "Sonatina Symphonic Orchestra"
    destination_dir = root / "Sonatina Symphonic Orchestra"

    if not checkout_dir.exists():
        run_git(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                "https://github.com/peastman/sso.git",
                str(checkout_dir),
            ]
        )

    run_git(
        ["git", "sparse-checkout", "set", "--cone", *SSO_SPARSE_DIRS],
        cwd=checkout_dir,
    )
    run_git(["git", "checkout", "master"], cwd=checkout_dir)

    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    shutil.copytree(source_dir, destination_dir)


def download_bass_tree(root: Path) -> None:
    seen: set[str] = set()

    def download_dependency(rel_path: str) -> None:
        normalized = normalize_repo_path(rel_path)
        if normalized in seen:
            return
        seen.add(normalized)

        url = BASS_RAW_ROOT + quote_repo_path(normalized)
        destination = root / "BassGuitar" / normalized

        if normalized.lower().endswith(".sfz"):
            text = fetch_text(url)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
            _, samples = sfz_dependencies(text)
            base_dir = Path(normalized).parent
            for sample_path in samples:
                child = normalize_repo_path(base_dir / sample_path)
                download_to(BASS_RAW_ROOT + quote_repo_path(child), root / "BassGuitar" / child)
        else:
            download_to(url, destination)

    for rel_path in BASS_TOP_LEVEL_FILES:
        download_dependency(rel_path)


def extract_archive(url: str, subdir: str, archive_name: str, root: Path) -> None:
    destination_root = root / subdir
    archive_path = root / "_downloads" / archive_name
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        download_to(url, archive_path)

    marker = destination_root / ".extracted"
    if marker.exists():
        return

    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path) as archive:
        archive.extractall(destination_root)

    marker.write_text(url, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch open acoustic sample subsets for AI Music Studio native VSTs.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Destination cache directory.")
    parser.add_argument("--clean", action="store_true", help="Remove the destination before downloading.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if args.clean and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    print(f"Fetching open instrument samples into {root}")
    download_sso_tree(root)
    download_bass_tree(root)
    for url, subdir, archive_name in ARCHIVES:
        extract_archive(url, subdir, archive_name, root)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
