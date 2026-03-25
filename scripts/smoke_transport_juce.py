from __future__ import annotations

import json
import os
import sys
import time
import traceback
import wave
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PySide6 import QtCore, QtWidgets  # noqa: E402

import app as aims  # noqa: E402


def pump_events(app: QtWidgets.QApplication, duration_ms: int) -> None:
    deadline = time.perf_counter() + (max(0, int(duration_ms)) / 1000.0)
    while time.perf_counter() < deadline:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
        time.sleep(0.01)


def pump_until(
    app: QtWidgets.QApplication,
    predicate,
    *,
    timeout_ms: int = 5000,
    poll_ms: int = 20,
) -> bool:
    deadline = time.perf_counter() + (max(0, int(timeout_ms)) / 1000.0)
    while time.perf_counter() < deadline:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if predicate():
            return True
        time.sleep(max(0.001, poll_ms / 1000.0))
    app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
    return bool(predicate())


def transport_snapshot(window: aims.MainWindow) -> dict[str, object]:
    bridge = getattr(window, "_native_output_bridge", None)
    status = window._native_output_status(bridge) if bridge is not None else None
    return {
        "pending_tick": getattr(window, "_pending_playback_start_tick", None),
        "playback_active": bool(getattr(window, "_playback_active", False)),
        "graph_transport": bool(window._graph_native_transport_active()),
        "direct_transport": bool(window._direct_native_transport_active()),
        "audio_engine_transport": bool(window._native_audio_engine_active()),
        "audio_engine_track_rows": list(getattr(window, "_native_audio_engine_track_rows", [])),
        "audio_engine_running": None if status is None else bool(status.get("audio_engine_running", False)),
        "audio_engine_track_count": None if status is None else int(status.get("audio_engine_track_count", 0) or 0),
        "audio_stream_enabled": None if status is None else bool(status.get("audio_stream_enabled", False)),
        "queue_frames": int(getattr(window, "_native_output_queued_frames", 0)),
        "status_message": window.statusBar().currentMessage(),
    }


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = aims.MainWindow()
    window.hide()
    results: list[dict[str, object]] = []

    def record_step(
        name: str,
        action,
        *,
        settle_ms: int = 1000,
        wait_for_transport: bool = False,
    ) -> None:
        started_at = time.perf_counter()
        action()
        elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
        if wait_for_transport:
            pump_until(
                app,
                lambda: (
                    getattr(window, "_pending_playback_start_tick", None) is None
                    and (
                        bool(getattr(window, "_playback_active", False))
                        or "cancelled" in window.statusBar().currentMessage().lower()
                        or "stopped" in window.statusBar().currentMessage().lower()
                    )
                ),
                timeout_ms=max(2500, settle_ms),
            )
        pump_events(app, settle_ms)
        snapshot = transport_snapshot(window)
        snapshot["step"] = name
        snapshot["call_ms"] = elapsed_ms
        results.append(snapshot)

    try:
        window.set_prefer_prerendered_vst_playback(False)
        pump_events(app, 150)
        track = window.project.tracks[0]
        track.notes = [
            aims.MidiNote(
                start_tick=0,
                duration_tick=aims.TICKS_PER_BEAT,
                pitch=60,
                velocity=100,
            )
        ]
        window.on_notes_changed()
        pump_events(app, 150)

        record_step("start_default_track", window.start_playback, settle_ms=1200, wait_for_transport=True)
        record_step("stop_default_track", window.stop_playback, settle_ms=1000)
        def prepare_single_vsti_project() -> None:
            window.new_project()
            window.set_prefer_prerendered_vst_playback(False)
            track = window.project.tracks[0]
            track.notes = [
                aims.MidiNote(
                    start_tick=0,
                    duration_tick=aims.TICKS_PER_BEAT,
                    pitch=60,
                    velocity=100,
                )
            ]
            if not window._assign_rack_vsti_to_track(0, "AI Bass Synth"):
                raise RuntimeError("Failed to assign AI Bass Synth on a fresh project")
            window.on_notes_changed()

        record_step(
            "assign_single_vsti",
            prepare_single_vsti_project,
            settle_ms=1000,
        )
        record_step("start_single_vsti", window.start_playback, settle_ms=1500, wait_for_transport=True)
        record_step("stop_single_vsti", window.stop_playback, settle_ms=1200)

        def add_second_track() -> None:
            window.add_track(preferred_type="instrument", ask=False)
            track2 = window.project.tracks[1]
            track2.notes = [
                aims.MidiNote(
                    start_tick=0,
                    duration_tick=aims.TICKS_PER_BEAT,
                    pitch=64,
                    velocity=100,
                )
            ]
            window._assign_rack_vsti_to_track(1, "AI Bass Synth")
            window.on_notes_changed()

        record_step("append_second_vsti_track", add_second_track, settle_ms=1200)
        record_step("start_two_vsti", window.start_playback, settle_ms=1800, wait_for_transport=True)
        record_step("stop_two_vsti", window.stop_playback, settle_ms=1200)

        def add_sample_clip_and_metronome() -> None:
            sample_path = REPO_ROOT / "sample_timeline.wav"
            if not sample_path.exists():
                raise FileNotFoundError(sample_path)
            window.add_track(preferred_type="sample", ask=False)
            sample_row = len(window.project.tracks) - 1
            with wave.open(str(sample_path), "rb") as source:
                sample_rate = int(source.getframerate())
                duration_sec = float(source.getnframes()) / float(max(1, sample_rate))
            window.project.sample_assets.append(
                aims.SampleAsset(
                    path=str(sample_path),
                    duration_sec=duration_sec,
                    sample_rate=sample_rate,
                )
            )
            window.place_sample_asset_on_track(len(window.project.sample_assets) - 1, sample_row, 0.0)
            window.set_metronome_enabled(True)

        record_step("append_sample_track_and_metronome", add_sample_clip_and_metronome, settle_ms=1200)

        def start_mixed_transport() -> None:
            window.start_playback()
            if not pump_until(
                app,
                lambda: (
                    bool(getattr(window, "_playback_active", False))
                    and window._native_audio_engine_active()
                    and int(getattr(window, "_native_audio_engine_position_frame", 0) or 0) > 0
                ),
                timeout_ms=5000,
            ):
                raise RuntimeError("Mixed engine transport did not become active on the unified native engine")

        record_step("start_vsti_sample_metronome", start_mixed_transport, settle_ms=1400)
        record_step("stop_vsti_sample_metronome", window.stop_playback, settle_ms=1200)

        mixed_snapshot = next(
            (item for item in results if item.get("step") == "start_vsti_sample_metronome"),
            None,
        )
        if not mixed_snapshot or not bool(mixed_snapshot.get("audio_engine_transport")):
            raise RuntimeError("Mixed transport did not use the clean native audio engine")
        if int(mixed_snapshot.get("audio_engine_track_count", 0) or 0) < 3:
            raise RuntimeError(f"Unified engine did not load the full mixed arrangement: {mixed_snapshot}")

        print(json.dumps({"ok": True, "results": results}, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "results": results,
                },
                indent=2,
            )
        )
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
        pump_events(app, 100)


if __name__ == "__main__":
    raise SystemExit(main())
