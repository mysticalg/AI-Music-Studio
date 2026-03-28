from __future__ import annotations

from typing import Any


class NativeAudioEngineClient:
    def __init__(self, bridge: Any) -> None:
        self.bridge = bridge

    def set_tracks(
        self,
        tracks: list[dict[str, object]],
        *,
        bpm: float,
        ticks_per_beat: int,
        loop_enabled: bool,
        loop_start_tick: int,
        loop_end_tick: int,
        metronome_enabled: bool = False,
        preserve_transport: bool = False,
    ) -> dict[str, Any]:
        return self.bridge.command(
            "set_audio_engine_state",
            tracks=list(tracks),
            bpm=float(bpm),
            ticks_per_beat=int(ticks_per_beat),
            loop_enabled=bool(loop_enabled),
            loop_start_tick=int(loop_start_tick),
            loop_end_tick=int(loop_end_tick),
            metronome_enabled=bool(metronome_enabled),
            preserve_transport=bool(preserve_transport),
        )

    def set_track_notes(
        self,
        track_index: int,
        notes: list[dict[str, object]],
    ) -> dict[str, Any]:
        return self.bridge.command(
            "set_audio_engine_track_notes",
            track_index=int(track_index),
            notes=list(notes),
        )

    def play(self, start_tick: int = 0) -> dict[str, Any]:
        return self.bridge.command("start_audio_engine", start_tick=int(start_tick))

    def seek(self, start_tick: int = 0) -> dict[str, Any]:
        return self.bridge.command("seek_audio_engine", start_tick=int(start_tick))

    def update_transport(
        self,
        *,
        bpm: float,
        ticks_per_beat: int,
        loop_enabled: bool,
        loop_start_tick: int,
        loop_end_tick: int,
        metronome_enabled: bool = False,
    ) -> dict[str, Any]:
        return self.bridge.command(
            "set_audio_engine_transport",
            bpm=float(bpm),
            ticks_per_beat=int(ticks_per_beat),
            loop_enabled=bool(loop_enabled),
            loop_start_tick=int(loop_start_tick),
            loop_end_tick=int(loop_end_tick),
            metronome_enabled=bool(metronome_enabled),
        )

    def stop(self, *, allow_tail: bool = False) -> dict[str, Any]:
        return self.bridge.command("stop_audio_engine", allow_tail=bool(allow_tail))

    def status(self) -> dict[str, Any]:
        return self.bridge.command("status")

    def get_position_frame(self) -> int:
        status = self.status()
        return int(status.get("audio_engine_position_frame", 0) or 0)
