#!/usr/bin/env python3
"""Inspect Access Virus TI bank dumps stored as Standard MIDI files.

This helper is intentionally stdlib-only so it can run in CI or a clean
developer shell without extra Python packages.

It extracts SysEx messages from a MIDI file, checks for the Access
manufacturer ID, and prints patch slot/name metadata for each single dump.
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import sys
from dataclasses import dataclass


ACCESS_MANUFACTURER_ID = bytes((0x00, 0x20, 0x33))
PATCH_NAME_OFFSET = 248
PATCH_NAME_LENGTH = 10


@dataclass(frozen=True)
class VirusPatchDump:
    slot: int
    name: str
    payload_size: int
    file_name: str


def read_vlq(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    while True:
        if offset >= len(data):
            raise ValueError("Unexpected end of track while reading VLQ")
        byte = data[offset]
        offset += 1
        value = (value << 7) | (byte & 0x7F)
        if (byte & 0x80) == 0:
            return value, offset


def extract_sysex_messages(midi_path: pathlib.Path) -> list[bytes]:
    raw = midi_path.read_bytes()
    if raw[:4] != b"MThd":
        raise ValueError(f"{midi_path} is not a Standard MIDI file")

    if len(raw) < 14:
        raise ValueError(f"{midi_path} is too short to be a valid MIDI file")

    header_length = struct.unpack(">I", raw[4:8])[0]
    offset = 8 + header_length
    if offset > len(raw):
        raise ValueError(f"{midi_path} has an invalid MIDI header length")

    messages: list[bytes] = []

    while offset + 8 <= len(raw):
        if raw[offset:offset + 4] != b"MTrk":
            break

        track_length = struct.unpack(">I", raw[offset + 4:offset + 8])[0]
        track = raw[offset + 8:offset + 8 + track_length]
        offset += 8 + track_length

        index = 0
        running_status = 0

        while index < len(track):
            _, index = read_vlq(track, index)
            if index >= len(track):
                break

            status = track[index]

            if status < 0x80:
                if running_status == 0:
                    raise ValueError(f"{midi_path} uses running status before any status byte")
                status = running_status
            else:
                index += 1
                if status < 0xF0:
                    running_status = status

            if status in (0xF0, 0xF7):
                length, index = read_vlq(track, index)
                payload = track[index:index + length]
                if len(payload) != length:
                    raise ValueError(f"{midi_path} has a truncated SysEx event")
                messages.append(payload)
                index += length
                continue

            if status == 0xFF:
                if index >= len(track):
                    raise ValueError(f"{midi_path} has a truncated meta event")
                index += 1  # meta type
                length, index = read_vlq(track, index)
                index += length
                continue

            high_nibble = status & 0xF0
            if high_nibble in (0xC0, 0xD0):
                index += 1
            else:
                index += 2

    return messages


def decode_patch_name(payload: bytes) -> str:
    name_bytes = payload[PATCH_NAME_OFFSET:PATCH_NAME_OFFSET + PATCH_NAME_LENGTH]
    chars = [chr(byte) if 32 <= byte <= 126 else " " for byte in name_bytes]
    return "".join(chars).rstrip() or "(unnamed)"


def decode_patch_dump(midi_path: pathlib.Path, payload: bytes) -> VirusPatchDump | None:
    if len(payload) < PATCH_NAME_OFFSET + PATCH_NAME_LENGTH:
        return None
    if not payload.startswith(ACCESS_MANUFACTURER_ID):
        return None

    slot = payload[7] if len(payload) > 7 else -1
    return VirusPatchDump(
        slot=slot,
        name=decode_patch_name(payload),
        payload_size=len(payload),
        file_name=midi_path.name,
    )


def inspect_file(midi_path: pathlib.Path) -> list[VirusPatchDump]:
    patches: list[VirusPatchDump] = []
    for payload in extract_sysex_messages(midi_path):
        dump = decode_patch_dump(midi_path, payload)
        if dump is not None:
            patches.append(dump)
    return patches


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Inspect Access Virus TI bank MIDI dumps.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more .mid files or directories containing Virus bank dumps.",
    )
    args = parser.parse_args(argv)

    midi_files: list[pathlib.Path] = []
    for raw_path in args.paths:
        path = pathlib.Path(raw_path)
        if path.is_dir():
            midi_files.extend(sorted(p for p in path.rglob("*.mid") if "__MACOSX" not in p.parts))
        elif path.suffix.lower() == ".mid":
            midi_files.append(path)

    if not midi_files:
        print("No MIDI files found.", file=sys.stderr)
        return 1

    for midi_path in midi_files:
        try:
            patches = inspect_file(midi_path)
        except Exception as exc:  # pragma: no cover - best-effort utility
            print(f"\n{midi_path.name}: ERROR: {exc}", file=sys.stderr)
            continue

        print(f"\n{midi_path.name}: {len(patches)} Access patch dumps")
        if not patches:
            continue

        for patch in patches[:16]:
            print(f"  {patch.slot:03d}  {patch.name}")

        if len(patches) > 16:
            print(f"  ... {len(patches) - 16} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
