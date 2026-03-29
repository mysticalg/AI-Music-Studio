from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from pywinauto import Application
from pywinauto.keyboard import send_keys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = ROOT / "build" / "native-app" / "AIMusicStudioNative_artefacts" / "Release" / "Mutagen.exe"
PROFILE_DIR = ROOT / "dist" / "native-profile"

RACK_NAMES = [
    "AI TB303",
    "AI VEC1 Drum Pads",
    "AI Bass Guitar",
    "AI Strings",
    "AI Organ",
    "AI Piano",
    "AI Flute",
    "AI Saxophone",
    "AI Violin",
    "Virus Synth",
]

TRACK_COLOURS = [
    "#3B82F6",
    "#F59E0B",
    "#22C55E",
    "#EF4444",
    "#8B5CF6",
    "#06B6D4",
    "#F97316",
    "#10B981",
]

TICKS_PER_BEAT = 480
TICKS_PER_BAR = TICKS_PER_BEAT * 4


def make_note_pattern(track_index: int, bar_count: int) -> list[dict]:
    notes: list[dict] = []
    pitches = [36, 43, 48, 55, 60, 64, 67, 72]
    base_pitch = pitches[track_index % len(pitches)]
    step = TICKS_PER_BEAT // 2
    duration = step
    total_steps = bar_count * 8
    for step_index in range(total_steps):
        pitch_offset = (step_index + track_index) % 5
        notes.append(
            {
                "start_tick": step_index * step,
                "duration_tick": duration,
                "pitch": max(24, min(96, base_pitch + pitch_offset)),
                "velocity": 92 + ((step_index + track_index) % 24),
                "selected": False,
            }
        )
    return notes


def make_mix_automation(target: str, values: list[float], bar_count: int) -> dict:
    points = []
    if len(values) < 2:
        values = values + values
    max_tick = bar_count * TICKS_PER_BAR
    segment = max_tick // (len(values) - 1)
    for index, value in enumerate(values):
        points.append({"tick": min(max_tick, index * segment), "value": value})
    return {"target": target, "enabled": True, "points": points}


def make_tb303_parameter_automation(parameter_name: str, bar_count: int, phase: int) -> dict:
    points = []
    bar_ticks = TICKS_PER_BAR
    values = [18.0, 64.0, 87.0, 35.0, 92.0, 41.0, 76.0, 28.0]
    for bar in range(bar_count + 1):
        points.append(
            {
                "tick": min(bar * bar_ticks, bar_count * bar_ticks),
                "value": values[(bar + phase) % len(values)],
            }
        )
    return {"target": f"vsti_parameter:{parameter_name}", "enabled": True, "points": points}


def make_track(track_index: int, bar_count: int) -> dict:
    rack_name = RACK_NAMES[track_index % len(RACK_NAMES)]
    automation = [
        make_mix_automation("volume", [0.58, 0.92, 0.71, 0.88, 0.63], bar_count),
        make_mix_automation("pan", [-0.4, 0.35, -0.25, 0.4, 0.0], bar_count),
        make_mix_automation("vsti_output_gain_db", [-2.0, 1.5, -1.0, 2.5, 0.0], bar_count),
    ]
    if rack_name == "AI TB303":
        automation.append(make_tb303_parameter_automation("Cutoff", bar_count, track_index % 4))
        automation.append(make_tb303_parameter_automation("Resonance", bar_count, (track_index + 2) % 4))

    return {
        "name": f"{rack_name} {track_index + 1}",
        "track_type": "instrument",
        "volume": 0.82,
        "pan": 0.0,
        "instrument": rack_name,
        "instrument_mode": "VSTI Rack",
        "rack_vsti": rack_name,
        "midi_program": 0,
        "midi_channel": track_index % 16,
        "synth_profile": "synth",
        "mute": False,
        "solo": False,
        "live_armed": False,
        "color_hex": TRACK_COLOURS[track_index % len(TRACK_COLOURS)],
        "notes": make_note_pattern(track_index, bar_count),
        "automation_lanes": automation,
    }


def make_project(track_count: int, bar_count: int) -> dict:
    right_tick = bar_count * TICKS_PER_BAR
    return {
        "format": "ai_music_studio_project",
        "version": 1,
        "project": {
            "bpm": 126,
            "quantize_enabled": True,
            "quantize_div": 16,
            "quantize_triplet": False,
            "loop_enabled": True,
            "metronome_enabled": False,
            "left_locator_tick": 0,
            "right_locator_tick": right_tick,
            "playhead_tick": 0,
            "left_locator_sec": 0.0,
            "right_locator_sec": float(bar_count * 2),
            "playhead_sec": 0.0,
            "tracks": [make_track(track_index, bar_count) for track_index in range(track_count)],
            "vsti_paths": [],
            "vsti_folder_paths": [],
            "sample_paths": [],
            "vsti_rack": [],
            "sample_assets": [],
            "sample_clips": [],
            "midi_sections": [],
        },
        "ui": {},
    }


def wait_for_window(process: subprocess.Popen[str], timeout: float = 30.0):
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        for backend in ("win32", "uia"):
            try:
                app = Application(backend=backend).connect(process=process.pid)
                window = app.window(title_re="Mutagen.*")
                window.wait("visible ready", timeout=1.5)
                return window
            except Exception as exc:  # noqa: PERF203
                last_error = exc
        time.sleep(0.4)
    raise RuntimeError(f"Could not attach to native window: {last_error}")


def run_profile(exe_path: Path, track_count: int, bar_count: int, playback_seconds: float) -> tuple[Path, Path]:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    project_path = PROFILE_DIR / f"stress_{track_count}tracks_{bar_count}bars.aims"
    log_path = PROFILE_DIR / f"profile_{track_count}tracks_{bar_count}bars.log"
    project_path.write_text(json.dumps(make_project(track_count, bar_count), indent=2), encoding="utf-8")
    if log_path.exists():
        log_path.unlink()

    env = os.environ.copy()
    env["AIMS_NATIVE_PROFILE_LOG"] = str(log_path)
    process = subprocess.Popen(
        [str(exe_path), "--project", str(project_path)],
        cwd=str(ROOT),
        env=env,
    )

    window = wait_for_window(process)
    time.sleep(2.0)
    window.set_focus()
    window.click_input()
    send_keys("{SPACE}")
    time.sleep(playback_seconds)
    send_keys("{SPACE}")
    time.sleep(2.0)
    send_keys("%{F4}")

    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=10)

    return project_path, log_path


def parse_profile_log(log_path: Path) -> list[str]:
    if not log_path.exists():
        return ["No profile log was produced."]

    lines = log_path.read_text(encoding="utf-8").splitlines()
    summary_lines = []
    for line in lines:
        if not line or line.startswith("["):
            summary_lines.append(line)
            continue

        parts = {}
        name, _, rest = line.partition(":")
        for token in rest.strip().split():
            if "=" in token:
                key, value = token.split("=", 1)
                parts[key] = value
        total_ms = float(parts.get("total_ms", "0"))
        avg_us = float(parts.get("avg_us", "0"))
        max_us = int(float(parts.get("max_us", "0")))
        summary_lines.append(
            f"{name}: total_ms={total_ms:8.2f} avg_us={avg_us:8.2f} max_us={max_us}"
        )
    return summary_lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a native Mutagen stress profile.")
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--tracks", type=int, default=20)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--playback-seconds", type=float, default=12.0)
    args = parser.parse_args()

    exe_path = args.exe.resolve()
    if not exe_path.exists():
        print(f"Native exe not found: {exe_path}", file=sys.stderr)
        return 1

    project_path, log_path = run_profile(exe_path, args.tracks, args.bars, args.playback_seconds)
    print(f"Project: {project_path}")
    print(f"Profile log: {log_path}")
    for line in parse_profile_log(log_path):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
