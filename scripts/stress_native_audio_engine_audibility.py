from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.native_audio_engine import NativeAudioEngineClient
from scripts.native_vst_host_bridge import NativeVstHostBridge


TICKS_PER_BEAT = 480


def available_plugins() -> list[tuple[str, Path]]:
    preferred = [
        "AI TB303",
        "Virus Synth",
        "AI Organ",
        "AI Piano",
        "AI Flute",
        "AI Violin",
    ]
    result: list[tuple[str, Path]] = []
    for name in preferred:
        path = ROOT / "vsti" / f"{name}.vst3"
        if path.exists():
            result.append((name, path))
    return result


def build_notes(track_index: int) -> list[dict[str, int]]:
    base_pitch = 48 + (track_index * 4)
    notes: list[dict[str, int]] = []
    for step in range(16):
        start = int(step * (TICKS_PER_BEAT // 2))
        duration = int(TICKS_PER_BEAT if (step % 4) == 0 else (TICKS_PER_BEAT // 2))
        notes.append(
            {
                "start_tick": start,
                "end_tick": start + duration,
                "pitch": int(min(96, base_pitch + (step % 5) * 2)),
                "velocity": 96,
                "channel": int(min(16, track_index + 1)),
            }
        )
    return notes


def build_tracks(plugin_specs: list[tuple[str, Path]]) -> list[dict[str, object]]:
    tracks: list[dict[str, object]] = []
    for index, (name, path) in enumerate(plugin_specs):
        tracks.append(
            {
                "name": str(name),
                "track_type": "instrument",
                "plugin_path": str(path),
                "state_path": "",
                "state_mtime_ns": 0,
                "parameters": {},
                "volume": 0.72,
                "output_gain_db": 0.0,
                "pan": float(max(-0.8, min(0.8, -0.5 + (index * 0.2)))),
                "audible": True,
                "notes": build_notes(index),
            }
        )
    return tracks


def run_stress(cycles: int) -> int:
    plugins = available_plugins()
    if len(plugins) < 4:
        raise RuntimeError(f"Need at least 4 bundled plugins, found {plugins}")

    tracks = build_tracks(plugins[:6])
    toggle_track_index = 1
    toggle_track = tracks[toggle_track_index]

    bridge = NativeVstHostBridge(
        restore_last_plugin_on_startup=False,
        bridge_mode=False,
        hidden=True,
        prefer_in_process=False,
        sample_rate=48000,
        buffer_size=256,
    )
    bridge.start(startup_timeout=12.0)
    client = NativeAudioEngineClient(bridge)

    try:
        client.set_tracks(
            tracks,
            bpm=118,
            ticks_per_beat=TICKS_PER_BEAT,
            loop_enabled=True,
            loop_start_tick=0,
            loop_end_tick=TICKS_PER_BEAT * 8,
            metronome_enabled=False,
            preserve_transport=False,
        )
        start_status = client.play(0)
        if not bool(start_status.get("audio_engine_running", False)):
            raise RuntimeError("Audio engine failed to start")
        time.sleep(0.25)

        muted = False
        positions: list[int] = []
        command_times_ms: list[float] = []
        for _cycle in range(max(1, int(cycles))):
            muted = not muted
            started = time.perf_counter()
            client.set_track_mix(
                toggle_track_index,
                volume=float(toggle_track["volume"]),
                output_gain_db=float(toggle_track["output_gain_db"]),
                pan=float(toggle_track["pan"]),
                audible=not muted,
            )
            command_times_ms.append((time.perf_counter() - started) * 1000.0)
            status = client.status()
            if not bool(status.get("audio_engine_running", False)):
                raise RuntimeError("Audio engine stopped during audibility stress")
            positions.append(int(status.get("audio_engine_position_frame", 0) or 0))
            time.sleep(0.03)

        if any(current <= previous for previous, current in zip(positions, positions[1:])):
            raise RuntimeError(f"Audio engine position failed to advance monotonically: {positions}")

        print(
            json.dumps(
                {
                    "ok": True,
                    "cycles": int(cycles),
                    "avg_command_ms": sum(command_times_ms) / len(command_times_ms),
                    "max_command_ms": max(command_times_ms),
                    "positions": positions,
                },
                indent=2,
            )
        )
        return 0
    finally:
        try:
            client.stop(allow_tail=False)
        except Exception:
            pass
        bridge.stop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=24)
    args = parser.parse_args()
    return run_stress(args.cycles)


if __name__ == "__main__":
    raise SystemExit(main())
