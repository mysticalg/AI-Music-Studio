from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_worker(iteration: int) -> int:
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--iteration", str(iteration)]
    completed = subprocess.run(command, cwd=str(ROOT), env=env)
    return int(completed.returncode)


def _exercise_window(iteration: int) -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from PySide6 import QtCore, QtWidgets
    import app

    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = app.MainWindow()
    window.hide()

    def pump(seconds: float) -> None:
        deadline = time.perf_counter() + max(0.0, float(seconds))
        while time.perf_counter() < deadline:
            qt_app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
            time.sleep(0.01)

    def wait_for_clean_engine(timeout: float = 4.0) -> bool:
        deadline = time.perf_counter() + max(0.0, float(timeout))
        while time.perf_counter() < deadline:
            qt_app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
            if bool(getattr(window, "_playback_active", False)) and window._native_audio_engine_active():
                return True
            time.sleep(0.01)
        qt_app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
        return bool(getattr(window, "_playback_active", False)) and window._native_audio_engine_active()

    try:
        window.new_project()
        window.set_prefer_prerendered_vst_playback(False)
        pump(0.1)
        available = [
            name for name in [
                "AI Bass Synth",
                "AI String Synth",
                "AI Lead Synth",
                "AI Pad Synth",
                "AI TB303",
            ]
            if window._rack_vsti_entry(name) is not None
        ]
        if len(available) < 3:
            raise RuntimeError(f"Not enough bundled VSTs available for stress test: {available}")

        cold_track = window.project.tracks[0]
        cold_track.notes = [
            app.MidiNote(
                start_tick=0,
                duration_tick=app.TICKS_PER_BEAT,
                pitch=48,
                velocity=100,
            )
        ]
        if not window._assign_rack_vsti_to_track(0, available[0]):
            raise RuntimeError(f"Failed cold-start assignment for {available[0]}")
        window.on_notes_changed()
        pump(0.1)
        window.start_playback()
        if not wait_for_clean_engine(4.0):
            raise RuntimeError("Clean audio engine did not become active for fresh-project single-VST cold start")
        pump(0.18)
        window.stop_playback()
        pump(0.08)

        window.new_project()
        window.set_prefer_prerendered_vst_playback(False)
        pump(0.1)
        while len(window.project.tracks) < 4:
            window.add_track(preferred_type="instrument", ask=False)

        bpm = 120
        window.project.bpm = bpm
        if hasattr(window, "tempo_spin"):
            window.tempo_spin.blockSignals(True)
            window.tempo_spin.setValue(bpm)
            window.tempo_spin.blockSignals(False)
        window.project.loop_enabled = True
        window.project.left_locator_sec = 0.0
        window.project.right_locator_sec = 2.0
        window._sync_playback_loop_state()
        window._set_locator_spin_values(window.project.left_locator_sec, window.project.right_locator_sec)

        patterns = [
            [36, 39, 43, 46, 48, 51, 55, 58],
            [48, 51, 55, 58, 60, 63, 67, 70],
            [60, 63, 67, 70, 72, 75, 79, 82],
            [72, 75, 79, 82, 84, 87, 91, 94],
        ]
        steps_per_bar = 16
        total_bars = 2
        total_steps = steps_per_bar * total_bars
        step_ticks = app.TICKS_PER_BEAT // 4

        for track_index in range(4):
            track = window.project.tracks[track_index]
            vst_name = available[track_index % len(available)]
            if not window._assign_rack_vsti_to_track(track_index, vst_name):
                raise RuntimeError(f"Failed to assign {vst_name} to track {track_index}")
            track.notes.clear()
            pitches = patterns[track_index % len(patterns)]
            for step in range(total_steps):
                track.notes.append(
                    app.MidiNote(
                        start_tick=step * step_ticks,
                        duration_tick=step_ticks,
                        pitch=pitches[step % len(pitches)],
                        velocity=92 + ((step + track_index) % 28),
                    )
                )

        window.on_notes_changed()
        pump(0.2)

        for cycle in range(12):
            window.start_playback()
            if not wait_for_clean_engine(4.0):
                raise RuntimeError(f"Clean audio engine did not become active on cycle {cycle + 1}")
            pump(0.35 if cycle < 6 else 0.25)
            window.stop_playback()
            pump(0.08)

        swap_pairs = [
            (0, "AI String Synth"),
            (1, "AI Lead Synth"),
            (2, "AI Pad Synth"),
            (3, "AI TB303"),
            (0, "AI Bass Synth"),
            (1, "AI String Synth"),
            (2, "AI Lead Synth"),
            (3, "AI Pad Synth"),
        ]
        for row, vst_name in swap_pairs:
            if window._rack_vsti_entry(vst_name) is None:
                continue
            ok = window._assign_rack_vsti_to_track(row, vst_name)
            if not ok:
                raise RuntimeError(f"Failed swap assignment row={row} vst={vst_name}")
            pump(0.06)
            window.start_playback()
            if not wait_for_clean_engine(4.0):
                raise RuntimeError(f"Clean audio engine inactive after swap row={row} vst={vst_name}")
            pump(0.22)
            window.stop_playback()
            pump(0.06)

        window.start_playback()
        if not wait_for_clean_engine(4.0):
            raise RuntimeError("Clean audio engine inactive before live swaps")
        pump(0.18)
        live_swaps = ["AI String Synth", "AI Lead Synth", "AI Pad Synth", "AI Bass Synth"]
        for offset, vst_name in enumerate(live_swaps):
            row = offset % 3
            if window._rack_vsti_entry(vst_name) is None:
                continue
            ok = window._assign_rack_vsti_to_track(row, vst_name)
            if not ok:
                raise RuntimeError(f"Failed live assignment row={row} vst={vst_name}")
            if not wait_for_clean_engine(4.0):
                raise RuntimeError(f"Clean audio engine inactive during live swap row={row} vst={vst_name}")
            pump(0.14)
        window.stop_playback()
        pump(0.08)

        print(
            {
                "iteration": iteration,
                "tracks": len(window.project.tracks),
                "vsts": [track.rack_vsti for track in window.project.tracks[:4]],
                "audio_engine_rows": list(getattr(window, "_native_audio_engine_track_rows", [])),
                "status": "ok",
            }
        )
        return 0
    finally:
        try:
            window._shutdown_runtime_resources()
        except Exception:
            pass
        try:
            window.close()
        except Exception:
            pass
        QtWidgets.QApplication.processEvents()
        qt_app.quit()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--runs", type=int, default=8)
    args = parser.parse_args()

    if args.worker:
        return _exercise_window(args.iteration)

    failures: list[tuple[int, int]] = []
    for iteration in range(1, max(1, int(args.runs)) + 1):
        code = _run_worker(iteration)
        print(f"run {iteration}: exit={code}")
        if code != 0:
            failures.append((iteration, code))
    if failures:
        print({"status": "failed", "failures": failures})
        return 1
    print({"status": "ok", "runs": int(args.runs)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
