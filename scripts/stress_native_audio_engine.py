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


def available_plugins() -> list[tuple[str, str]]:
    names = [
        "AI Bass Synth",
        "AI Lead Synth",
        "AI Pad Synth",
        "AI String Synth",
        "AI TB303",
    ]
    result: list[tuple[str, str]] = []
    for name in names:
        path = ROOT / "vsti" / f"{name}.vst3"
        if path.exists():
            result.append((name, str(path)))
    return result


def build_track_payload(plugin_name: str, plugin_path: str, track_index: int, note_count: int) -> dict[str, object]:
    base_pitch = 48 + (track_index * 7)
    notes = []
    for step in range(note_count):
        duration = TICKS_PER_BEAT // 2 if (step % 3) else TICKS_PER_BEAT
        notes.append(
            {
                "start_tick": int(step * (TICKS_PER_BEAT // 2)),
                "end_tick": int((step * (TICKS_PER_BEAT // 2)) + duration),
                "pitch": int(min(96, base_pitch + (step % 5) * 2)),
                "velocity": int(88 + ((track_index + step) % 28)),
                "channel": int((track_index % 15) + 1),
            }
        )
    return {
        "name": f"Stress Track {track_index + 1} {plugin_name}",
        "plugin_path": plugin_path,
        "state_path": "",
        "parameters": {},
        "gain": 0.65,
        "pan": -0.35 + (track_index * 0.25),
        "notes": notes,
    }


def frames_to_tick(frame: int, sample_rate: int, bpm: int) -> int:
    return int((max(0, int(frame)) * max(1, int(bpm)) * TICKS_PER_BEAT) / (max(1, int(sample_rate)) * 60))


def run_stress(runs: int) -> int:
    plugins = available_plugins()
    if len(plugins) < 3:
        raise RuntimeError(f"Need at least 3 bundled VST3 plugins, found {plugins}")

    bridge = NativeVstHostBridge(
        restore_last_plugin_on_startup=False,
        bridge_mode=False,
        hidden=True,
        prefer_in_process=False,
        sample_rate=48000,
        buffer_size=512,
    )
    bridge.start(startup_timeout=10.0)
    client = NativeAudioEngineClient(bridge)
    results: list[dict[str, object]] = []

    try:
        configurations = [1, 2, 4, 3]
        for iteration in range(max(1, int(runs))):
            track_count = configurations[iteration % len(configurations)]
            note_count = 8 + (iteration % 4) * 4
            loop_enabled = (iteration % 2) == 0
            loop_start_tick = 0
            loop_end_tick = max(TICKS_PER_BEAT * 4, note_count * (TICKS_PER_BEAT // 2))
            bpm = 92 + (iteration % 5) * 12

            tracks = [
                build_track_payload(name, path, index, note_count)
                for index, (name, path) in enumerate(plugins[:track_count])
            ]

            configure_status = client.set_tracks(
                tracks,
                bpm=bpm,
                ticks_per_beat=TICKS_PER_BEAT,
                loop_enabled=loop_enabled,
                loop_start_tick=loop_start_tick,
                loop_end_tick=loop_end_tick,
            )
            configured_track_count = int(configure_status.get("audio_engine_track_count", 0) or 0)
            if configured_track_count != track_count:
                raise RuntimeError(
                    f"Configured track count mismatch iteration={iteration + 1}: expected {track_count}, got {configured_track_count}"
                )

            start_tick = 0 if not loop_enabled else (iteration % 3) * (TICKS_PER_BEAT // 2)
            start_status = client.play(start_tick=start_tick)
            if not bool(start_status.get("audio_engine_running", False)):
                raise RuntimeError(f"Audio engine did not enter running state on iteration {iteration + 1}")

            sample_rate = int(start_status.get("sample_rate", 48000) or 48000)
            observed_frames: list[int] = [int(start_status.get("audio_engine_position_frame", 0) or 0)]
            observed_ticks: list[int] = [frames_to_tick(observed_frames[-1], sample_rate, bpm)]

            for _poll in range(6):
                time.sleep(0.14)
                status = client.status()
                if not bool(status.get("audio_engine_running", False)):
                    raise RuntimeError(f"Audio engine stopped unexpectedly on iteration {iteration + 1}")
                frame = int(status.get("audio_engine_position_frame", 0) or 0)
                observed_frames.append(frame)
                observed_ticks.append(frames_to_tick(frame, sample_rate, bpm))

            if max(observed_frames) <= min(observed_frames):
                raise RuntimeError(f"Audio engine position did not advance on iteration {iteration + 1}: {observed_frames}")

            if loop_enabled:
                lower_bound = loop_start_tick
                upper_bound = loop_end_tick + TICKS_PER_BEAT
                if not all(lower_bound <= tick <= upper_bound for tick in observed_ticks):
                    raise RuntimeError(
                        f"Looping ticks escaped expected range on iteration {iteration + 1}: {observed_ticks}"
                    )

            stop_status = client.stop()
            if bool(stop_status.get("audio_engine_running", False)):
                raise RuntimeError(f"Audio engine still running after stop on iteration {iteration + 1}")

            results.append(
                {
                    "iteration": iteration + 1,
                    "track_count": track_count,
                    "loop_enabled": loop_enabled,
                    "bpm": bpm,
                    "sample_rate": sample_rate,
                    "observed_frames": observed_frames,
                    "observed_ticks": observed_ticks,
                }
            )

        print(json.dumps({"ok": True, "runs": len(results), "results": results}, indent=2))
        return 0
    finally:
        bridge.stop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    return run_stress(args.runs)


if __name__ == "__main__":
    raise SystemExit(main())
