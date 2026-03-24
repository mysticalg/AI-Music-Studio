from __future__ import annotations

import json
import os
import sys
import time
import traceback
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
    return {
        "pending_tick": getattr(window, "_pending_playback_start_tick", None),
        "playback_active": bool(getattr(window, "_playback_active", False)),
        "graph_transport": bool(window._graph_native_transport_active()),
        "direct_transport": bool(window._direct_native_transport_active()),
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
        record_step(
            "assign_single_vsti",
            lambda: window._assign_rack_vsti_to_track(0, "AI Bass Synth"),
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
