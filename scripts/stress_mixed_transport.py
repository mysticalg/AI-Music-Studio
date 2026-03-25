from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6 import QtCore, QtWidgets  # noqa: E402

import app  # noqa: E402


def pump_events(qt_app: QtWidgets.QApplication, seconds: float) -> None:
    deadline = time.perf_counter() + max(0.0, float(seconds))
    while time.perf_counter() < deadline:
        qt_app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
        time.sleep(0.01)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=8)
    args = parser.parse_args()

    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = app.MainWindow()
    window.hide()
    results: list[dict[str, object]] = []

    def wait_for_mixed_transport(timeout: float = 8.0) -> dict[str, object] | None:
        deadline = time.perf_counter() + max(0.0, float(timeout))
        latest_status: dict[str, object] | None = None
        while time.perf_counter() < deadline:
            qt_app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
            bridge = getattr(window, "_native_output_bridge", None)
            status = window._native_output_status(bridge) if bridge is not None else None
            if isinstance(status, dict):
                latest_status = status
            if (
                window._native_audio_engine_active()
                and isinstance(status, dict)
                and bool(status.get("audio_engine_running", False))
                and int(status.get("audio_engine_position_frame", 0) or 0) > 0
            ):
                return status
            time.sleep(0.01)
        return latest_status

    try:
        sample_path = ROOT / "sample_timeline.wav"
        if not sample_path.exists():
            raise FileNotFoundError(sample_path)

        with wave.open(str(sample_path), "rb") as source:
            sample_rate = int(source.getframerate())
            duration_sec = float(source.getnframes()) / float(max(1, sample_rate))

        window.new_project()
        window.set_prefer_prerendered_vst_playback(False)
        pump_events(qt_app, 0.15)

        while len(window.project.tracks) < 2:
            window.add_track(preferred_type="instrument", ask=False)
        window.add_track(preferred_type="sample", ask=False)
        pump_events(qt_app, 0.1)

        synth_names = [
            name for name in [
                "AI Bass Synth",
                "AI String Synth",
                "AI Lead Synth",
                "AI Pad Synth",
                "AI TB303",
            ]
            if window._rack_vsti_entry(name) is not None
        ]
        if len(synth_names) < 3:
            raise RuntimeError(f"Not enough bundled VSTs available for mixed stress test: {synth_names}")

        step_ticks = app.TICKS_PER_BEAT // 4
        patterns = [
            [36, 39, 43, 46, 48, 51, 55, 58],
            [60, 63, 67, 70, 72, 75, 79, 82],
        ]
        for track_index in range(2):
            track = window.project.tracks[track_index]
            if not window._assign_rack_vsti_to_track(track_index, synth_names[track_index]):
                raise RuntimeError(f"Failed to assign {synth_names[track_index]} to track {track_index}")
            track.notes.clear()
            for step in range(16):
                track.notes.append(
                    app.MidiNote(
                        start_tick=step * step_ticks,
                        duration_tick=step_ticks,
                        pitch=patterns[track_index][step % len(patterns[track_index])],
                        velocity=95 + ((step + track_index) % 24),
                    )
                )

        sample_track_index = 2
        window.project.sample_assets.append(
            app.SampleAsset(
                path=str(sample_path),
                duration_sec=duration_sec,
                sample_rate=sample_rate,
            )
        )
        window.place_sample_asset_on_track(len(window.project.sample_assets) - 1, sample_track_index, 0.0)

        window.project.bpm = 124
        if hasattr(window, "tempo_spin"):
            window.tempo_spin.blockSignals(True)
            window.tempo_spin.setValue(124)
            window.tempo_spin.blockSignals(False)
        window.project.loop_enabled = True
        window.project.left_locator_sec = 0.0
        window.project.right_locator_sec = 2.0
        window._sync_playback_loop_state()
        window._set_locator_spin_values(window.project.left_locator_sec, window.project.right_locator_sec)
        window.set_metronome_enabled(True)
        window.on_notes_changed()
        pump_events(qt_app, 0.25)

        for iteration in range(1, max(1, int(args.runs)) + 1):
            if window.project.sample_clips:
                clip = window.project.sample_clips[0]
                clip.start_sec = 0.0 if iteration % 2 else 0.125
            sample_track = window.project.tracks[sample_track_index]
            sample_track.volume = 0.85 if iteration % 3 else 0.65
            window.set_metronome_enabled(iteration % 2 == 1)

            swap_name = synth_names[(iteration + 1) % len(synth_names)]
            if not window._assign_rack_vsti_to_track(1, swap_name):
                raise RuntimeError(f"Failed to assign {swap_name} during mixed stress iteration {iteration}")
            window._invalidate_playback_caches(clear_track_audio=False)
            window.sample_timeline.refresh()
            pump_events(qt_app, 0.12)

            window.start_playback()
            pump_events(qt_app, 0.1)
            status = wait_for_mixed_transport(timeout=5.0)
            if not (
                isinstance(status, dict)
                and window._native_audio_engine_active()
                and bool(status.get("audio_engine_running", False))
                and int(status.get("audio_engine_position_frame", 0) or 0) > 0
            ):
                raise RuntimeError(f"Mixed transport did not become active on iteration {iteration}: {status}")

            start_frame = int(status.get("audio_engine_position_frame", 0) or 0)
            pump_events(qt_app, 0.35)
            bridge = getattr(window, "_native_output_bridge", None)
            later_status = window._native_output_status(bridge) if bridge is not None else None
            if not isinstance(later_status, dict):
                raise RuntimeError(f"Missing mixed transport status on iteration {iteration}")
            later_frame = int(later_status.get("audio_engine_position_frame", 0) or 0)
            loop_end_frame = int(window.project.right_locator_sec * window._playback_sample_rate)
            advanced_frames = later_frame - start_frame
            if window.project.loop_enabled and loop_end_frame > 0 and advanced_frames <= 0:
                advanced_frames = (loop_end_frame - start_frame) + later_frame
            if advanced_frames <= 0:
                raise RuntimeError(
                    f"Audio engine position did not advance on iteration {iteration}: {start_frame} -> {later_frame}"
                )
            if int(later_status.get("audio_engine_track_count", 0) or 0) < 3:
                raise RuntimeError(f"Unified engine did not keep the mixed arrangement loaded on iteration {iteration}: {later_status}")

            results.append(
                {
                    "iteration": iteration,
                    "metronome_enabled": bool(window.project.metronome_enabled),
                    "sample_clip_start_sec": float(window.project.sample_clips[0].start_sec),
                    "sample_track_volume": float(sample_track.volume),
                    "track_rows": list(getattr(window, "_native_audio_engine_track_rows", [])),
                    "start_frame": start_frame,
                    "later_frame": later_frame,
                    "advanced_frames": advanced_frames,
                    "engine_track_count": int(later_status.get("audio_engine_track_count", 0) or 0),
                    "swap_name": swap_name,
                }
            )

            window.stop_playback()
            pump_events(qt_app, 0.12)

        print(json.dumps({"ok": True, "runs": int(args.runs), "results": results}, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc), "results": results}, indent=2))
        return 1
    finally:
        try:
            window._shutdown_runtime_resources()
        except Exception:
            pass
        try:
            window.close()
        except Exception:
            pass
        pump_events(qt_app, 0.1)


if __name__ == "__main__":
    raise SystemExit(main())
