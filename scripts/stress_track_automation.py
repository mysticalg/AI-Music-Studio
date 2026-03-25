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


def wait_for_predicate(
    qt_app: QtWidgets.QApplication,
    predicate,
    *,
    timeout: float = 5.0,
) -> bool:
    deadline = time.perf_counter() + max(0.0, float(timeout))
    while time.perf_counter() < deadline:
        qt_app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if predicate():
            return True
        time.sleep(0.01)
    qt_app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
    return bool(predicate())


def rms(values) -> float:
    if app.np is not None:
        array = app.np.asarray(values, dtype=app.np.float32)
        if array.size <= 0:
            return 0.0
        return float(app.np.sqrt(app.np.mean(app.np.square(array))))
    sequence = [float(value) for value in values]
    if not sequence:
        return 0.0
    return (sum(value * value for value in sequence) / float(len(sequence))) ** 0.5


def stereo_channel_energy(stereo, start: int, count: int) -> tuple[float, float]:
    end = max(start, int(start) + max(1, int(count)))
    if app.np is not None:
        audio = app.np.asarray(stereo, dtype=app.np.float32)
        left = audio[0, start:end]
        right = audio[1, start:end]
    else:
        left = stereo[0][start:end]
        right = stereo[1][start:end]
    return rms(left), rms(right)


def wait_for_engine(window: app.MainWindow, qt_app: QtWidgets.QApplication, timeout: float = 5.0) -> dict[str, object] | None:
    latest_status: dict[str, object] | None = None

    def predicate() -> bool:
        nonlocal latest_status
        bridge = getattr(window, "_native_output_bridge", None)
        status = window._native_output_status(bridge) if bridge is not None else None
        if isinstance(status, dict):
            latest_status = status
        return bool(
            window._native_audio_engine_active()
            and isinstance(status, dict)
            and bool(status.get("audio_engine_running", False))
        )

    if wait_for_predicate(qt_app, predicate, timeout=timeout):
        return latest_status
    return latest_status


def ensure_sample_clip(window: app.MainWindow, track_index: int) -> None:
    sample_path = ROOT / "sample_timeline.wav"
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)
    data, sample_rate = app.load_wav_samples(sample_path)
    duration_sec = (
        float(data.shape[0] if app.np is not None and hasattr(data, "shape") else len(data))
        / float(max(1, sample_rate))
    )
    asset = app.SampleAsset(
        path=str(sample_path),
        duration_sec=duration_sec,
        sample_rate=sample_rate,
    )
    window.project.sample_assets.append(asset)
    window.place_sample_asset_on_track(len(window.project.sample_assets) - 1, track_index, 0.0)


def build_param_lane(window: app.MainWindow, track: app.TrackState) -> app.AutomationLane | None:
    if not track.rack_vsti:
        return None
    parameter_names = window.vsti_parameter_names_for_rack(track.rack_vsti)
    if not parameter_names:
        return None
    target = app.automation_target_for_vst_parameter(parameter_names[0])
    return app.AutomationLane(
        target=target,
        enabled=True,
        points=[
            app.AutomationPoint(tick=0, value=10.0),
            app.AutomationPoint(tick=app.TICKS_PER_BAR, value=90.0),
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
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
            raise RuntimeError(f"Need at least two rack VST instruments for automation stress: {synth_names}")

        for iteration in range(1, max(1, int(args.runs)) + 1):
            window.new_project()
            window.set_prefer_prerendered_vst_playback(False)
            window.project.loop_enabled = False
            window.project.right_locator_tick = app.TICKS_PER_BAR * 2
            window.project.right_locator_sec = app.tick_to_seconds(window.project.right_locator_tick, window.project.bpm)
            pump_events(qt_app, 0.15)

            track0 = window.project.tracks[0]
            synth_a = synth_names[(iteration - 1) % len(synth_names)]
            if not window._assign_rack_vsti_to_track(0, synth_a):
                raise RuntimeError(f"Failed to assign {synth_a} on iteration {iteration}")
            track0.notes = [
                app.MidiNote(
                    start_tick=step * (app.TICKS_PER_BEAT // 2),
                    duration_tick=app.TICKS_PER_BEAT // 2,
                    pitch=48 + ((step + iteration) % 5) * 2,
                    velocity=96,
                )
                for step in range(16)
            ]
            track0.automation_lanes = [
                app.AutomationLane(
                    target=app.AUTOMATION_TARGET_VOLUME,
                    enabled=True,
                    points=[
                        app.AutomationPoint(tick=0, value=0.2),
                        app.AutomationPoint(tick=app.TICKS_PER_BAR, value=1.0),
                    ],
                ),
                app.AutomationLane(
                    target=app.AUTOMATION_TARGET_PAN,
                    enabled=True,
                    points=[
                        app.AutomationPoint(tick=0, value=-1.0),
                        app.AutomationPoint(tick=app.TICKS_PER_BAR, value=1.0),
                    ],
                ),
            ]
            param_lane = build_param_lane(window, track0)
            if param_lane is not None:
                track0.automation_lanes.append(param_lane)
            window.on_track_automation_changed(0, {lane.target for lane in track0.automation_lanes})
            window.on_notes_changed()
            pump_events(qt_app, 0.2)

            analysis_frames = int(app.tick_to_sample_frame(app.TICKS_PER_BAR * 2, window._playback_sample_rate, window.project.bpm))
            if not window._build_playback_mix(start_frame=0, frame_count=analysis_frames):
                raise RuntimeError(f"Playback mix build failed on iteration {iteration}")
            analysis_mix = window._playback_mix_samples
            if analysis_mix is None:
                raise RuntimeError(f"Playback mix was empty on iteration {iteration}")
            early_left, early_right = stereo_channel_energy(analysis_mix, 0, max(1, analysis_frames // 4))
            late_left, late_right = stereo_channel_energy(analysis_mix, analysis_frames // 2, max(1, analysis_frames // 4))
            if not (early_left > early_right and late_right > late_left):
                raise RuntimeError(
                    f"Pan automation did not swing across the stereo field on iteration {iteration}: "
                    f"early=({early_left:.4f},{early_right:.4f}) late=({late_left:.4f},{late_right:.4f})"
                )

            payload = window._project_payload()
            project_copy, _blobs, _ui_state = window._project_from_payload(payload)
            copied_track0 = project_copy.tracks[0]
            if len(copied_track0.automation_lanes) < 2:
                raise RuntimeError(f"Automation lanes were not preserved by save/load on iteration {iteration}")

            window.add_track(preferred_type="instrument", ask=False)
            track1 = window.project.tracks[1]
            synth_b = synth_names[iteration % len(synth_names)]
            if not window._assign_rack_vsti_to_track(1, synth_b):
                raise RuntimeError(f"Failed to assign {synth_b} on iteration {iteration}")
            track1.notes = [
                app.MidiNote(
                    start_tick=step * (app.TICKS_PER_BEAT // 2),
                    duration_tick=app.TICKS_PER_BEAT // 3,
                    pitch=60 + ((step + iteration) % 4) * 3,
                    velocity=88,
                )
                for step in range(16)
            ]
            track1.automation_lanes = [
                app.AutomationLane(
                    target=app.AUTOMATION_TARGET_VOLUME,
                    enabled=True,
                    points=[
                        app.AutomationPoint(tick=0, value=0.9),
                        app.AutomationPoint(tick=app.TICKS_PER_BAR, value=0.35),
                    ],
                ),
            ]
            param_lane_track1 = build_param_lane(window, track1)
            if param_lane_track1 is not None:
                track1.automation_lanes.append(param_lane_track1)
            window.on_track_automation_changed(1, {lane.target for lane in track1.automation_lanes})
            window.on_notes_changed()

            window.add_track(preferred_type="sample", ask=False)
            sample_row = len(window.project.tracks) - 1
            sample_track = window.project.tracks[sample_row]
            sample_track.automation_lanes = [
                app.AutomationLane(
                    target=app.AUTOMATION_TARGET_PAN,
                    enabled=True,
                    points=[
                        app.AutomationPoint(tick=0, value=1.0),
                        app.AutomationPoint(tick=app.TICKS_PER_BAR, value=-0.8),
                    ],
                ),
            ]
            ensure_sample_clip(window, sample_row)
            window.on_track_automation_changed(sample_row, {lane.target for lane in sample_track.automation_lanes})
            pump_events(qt_app, 0.3)

            window.start_playback()
            status = wait_for_engine(window, qt_app, timeout=5.0)
            if not (
                isinstance(status, dict)
                and window._native_audio_engine_active()
                and bool(status.get("audio_engine_running", False))
            ):
                raise RuntimeError(f"Clean engine did not become active on iteration {iteration}: {status}")

            track0.automation_lanes[0].points[-1].value = 0.55
            window.on_track_automation_changed(0, {track0.automation_lanes[0].target})
            sample_track.automation_lanes[0].points[-1].value = 0.75
            window.on_track_automation_changed(sample_row, {sample_track.automation_lanes[0].target})
            pump_events(qt_app, 0.15)
            replay_status = wait_for_engine(window, qt_app, timeout=5.0)
            if not (
                isinstance(replay_status, dict)
                and window._native_audio_engine_active()
                and bool(replay_status.get("audio_engine_running", False))
            ):
                raise RuntimeError(f"Playback did not recover after live automation edit on iteration {iteration}: {replay_status}")

            results.append(
                {
                    "iteration": iteration,
                    "synth_a": synth_a,
                    "synth_b": synth_b,
                    "track0_lane_count": len(track0.automation_lanes),
                    "track1_lane_count": len(track1.automation_lanes),
                    "sample_lane_count": len(sample_track.automation_lanes),
                    "early_left": round(early_left, 6),
                    "early_right": round(early_right, 6),
                    "late_left": round(late_left, 6),
                    "late_right": round(late_right, 6),
                    "engine_track_rows": list(getattr(window, "_native_audio_engine_track_rows", [])),
                    "queued_frames": int(getattr(window, "_native_output_queued_frames", 0) or 0),
                }
            )

            window.stop_playback()
            pump_events(qt_app, 0.2)

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
