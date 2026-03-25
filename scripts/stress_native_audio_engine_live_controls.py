from __future__ import annotations

import argparse
import json
import os
import sys
import time
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


def wait_for_engine(window: app.MainWindow, qt_app: QtWidgets.QApplication, timeout: float = 5.0) -> dict[str, object] | None:
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
        ):
            return status
        time.sleep(0.01)
    return latest_status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=6)
    args = parser.parse_args()

    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = app.MainWindow()
    window.hide()
    results: list[dict[str, object]] = []

    try:
        synth_names = [
            name
            for name in [
                "AI Bass Synth",
                "AI String Synth",
                "AI Lead Synth",
                "AI Pad Synth",
                "AI TB303",
            ]
            if window._rack_vsti_entry(name) is not None
        ]
        if len(synth_names) < 2:
            raise RuntimeError(f"Not enough bundled VSTs available for live-control stress test: {synth_names}")

        for iteration in range(1, max(1, int(args.runs)) + 1):
            window.new_project()
            window.set_prefer_prerendered_vst_playback(False)
            pump_events(qt_app, 0.15)

            track = window.project.tracks[0]
            synth_name = synth_names[(iteration - 1) % len(synth_names)]
            if not window._assign_rack_vsti_to_track(0, synth_name):
                raise RuntimeError(f"Failed to assign {synth_name} on iteration {iteration}")
            track.notes.clear()
            step_ticks = app.TICKS_PER_BEAT // 4
            for step in range(64):
                track.notes.append(
                    app.MidiNote(
                        start_tick=step * step_ticks,
                        duration_tick=step_ticks,
                        pitch=48 + ((step + iteration) % 7) * 2,
                        velocity=92 + ((step + iteration) % 28),
                    )
                )
            window.project.loop_enabled = False
            window.on_notes_changed()
            pump_events(qt_app, 0.2)

            window.start_playback()
            status = wait_for_engine(window, qt_app, timeout=5.0)
            if not (
                isinstance(status, dict)
                and window._native_audio_engine_active()
                and bool(status.get("audio_engine_running", False))
            ):
                raise RuntimeError(f"Clean engine did not become active on iteration {iteration}: {status}")

            buffer_size = max(512, int(status.get("buffer_size", 0) or 0))
            seek_target_sec = 1.0 + (0.25 * (iteration % 3))
            seek_target_tick = int(window._transport_seconds_to_tick(seek_target_sec))
            seek_target_frame = int(
                app.tick_to_sample_frame(seek_target_tick, window._playback_sample_rate, window.project.bpm)
            )
            window.set_playhead_position(seek_target_sec)
            pump_events(qt_app, 0.05)
            seek_status = wait_for_engine(window, qt_app, timeout=1.5)
            if not isinstance(seek_status, dict):
                raise RuntimeError(f"Missing engine status after seek on iteration {iteration}")
            seek_frame = int(seek_status.get("audio_engine_position_frame", 0) or 0)
            seek_tolerance = max(buffer_size * 4, 4096)
            if abs(seek_frame - seek_target_frame) > seek_tolerance:
                raise RuntimeError(
                    f"Seek drift too large on iteration {iteration}: target_frame={seek_target_frame} actual={seek_frame} tolerance={seek_tolerance}"
                )

            loop_sec = 0.5 if iteration % 2 else 0.75
            window.set_loop_enabled(True)
            window.set_left_locator_position(0.0)
            window.set_right_locator_position(loop_sec)
            pump_events(qt_app, loop_sec + 0.3)
            loop_status = wait_for_engine(window, qt_app, timeout=1.5)
            if not isinstance(loop_status, dict):
                raise RuntimeError(f"Missing engine status after loop update on iteration {iteration}")
            loop_frame = int(loop_status.get("audio_engine_position_frame", 0) or 0)
            loop_end_frame = int(
                app.tick_to_sample_frame(int(window.project.right_locator_tick), window._playback_sample_rate, window.project.bpm)
            )
            loop_tolerance = max(buffer_size * 3, 3072)
            if loop_frame >= (loop_end_frame + loop_tolerance):
                raise RuntimeError(
                    f"Live loop update did not wrap on iteration {iteration}: frame={loop_frame} loop_end={loop_end_frame}"
                )

            window.set_right_locator_position(1.0)
            pump_events(qt_app, 0.08)
            new_bpm = 150 if iteration % 2 else 180
            window.update_tempo(new_bpm)
            loop_end_frame_after_tempo = int(
                app.tick_to_sample_frame(int(window.project.right_locator_tick), window._playback_sample_rate, window.project.bpm)
            )
            loop_duration_sec_after_tempo = loop_end_frame_after_tempo / float(max(1, window._playback_sample_rate))
            pump_events(qt_app, min(0.9, loop_duration_sec_after_tempo + 0.12))
            tempo_status = wait_for_engine(window, qt_app, timeout=1.5)
            if not isinstance(tempo_status, dict):
                raise RuntimeError(f"Missing engine status after tempo change on iteration {iteration}")
            tempo_frame = int(tempo_status.get("audio_engine_position_frame", 0) or 0)
            tempo_tolerance = max(buffer_size * 3, 3072)
            if tempo_frame >= (loop_end_frame_after_tempo + tempo_tolerance):
                raise RuntimeError(
                    f"Tempo change did not update engine loop duration on iteration {iteration}: frame={tempo_frame} loop_end={loop_end_frame_after_tempo}"
                )

            results.append(
                {
                    "iteration": iteration,
                    "synth_name": synth_name,
                    "seek_target_sec": seek_target_sec,
                    "seek_target_frame": seek_target_frame,
                    "seek_frame": seek_frame,
                    "loop_sec": loop_sec,
                    "loop_frame": loop_frame,
                    "loop_end_frame": loop_end_frame,
                    "new_bpm": new_bpm,
                    "tempo_frame": tempo_frame,
                    "tempo_loop_end_frame": loop_end_frame_after_tempo,
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
