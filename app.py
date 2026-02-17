from __future__ import annotations

import dataclasses
import json
import math
import os
import struct
import subprocess
import sys
import urllib.error
import urllib.request
import wave
from pathlib import Path

import mido
from PySide6 import QtCore, QtGui, QtWidgets

TICKS_PER_BEAT = 480
DEFAULT_BPM = 120
PITCH_MIN = 36
PITCH_MAX = 84
OPENAI_API_URL = "https://api.openai.com/v1/responses"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-codex")
RENDER_DIR = Path("renders")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def midi_to_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def load_wav_preview(path: Path, max_points: int = 800) -> tuple[list[float], int, float]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(frames)

    if sampwidth != 2:
        raise RuntimeError("Only 16-bit PCM WAV is supported for waveform preview.")

    sample_count = max(1, len(raw) // 2)
    unpacked = struct.unpack(f"<{sample_count}h", raw)
    mono: list[float] = []
    for i in range(0, len(unpacked), channels):
        mono.append(unpacked[i] / 32768.0)

    if not mono:
        return [0.0], sample_rate, 0.0

    bucket = max(1, len(mono) // max_points)
    preview: list[float] = []
    for i in range(0, len(mono), bucket):
        window = mono[i : i + bucket]
        preview.append(max(abs(v) for v in window))

    duration = len(mono) / sample_rate
    return preview, sample_rate, duration


def convert_audio(input_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for mp3 conversion but was not found in PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.decode("utf-8", errors="ignore")) from exc


def load_wav_samples(path: Path) -> tuple[list[float], int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(frames)

    if sampwidth != 2:
        raise RuntimeError("Only 16-bit PCM WAV is supported.")

    sample_count = max(1, len(raw) // 2)
    unpacked = struct.unpack(f"<{sample_count}h", raw)
    mono: list[float] = []
    for i in range(0, len(unpacked), channels):
        mono.append(unpacked[i] / 32768.0)
    return mono, sample_rate


def write_wav_samples(path: Path, samples: list[float], sample_rate: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for value in samples:
            clipped = int(clamp(value, -1.0, 1.0) * 32767)
            frames.extend(struct.pack("<h", clipped))
        wf.writeframes(frames)


@dataclasses.dataclass
class MidiNote:
    start_tick: int
    duration_tick: int
    pitch: int
    velocity: int = 100
    selected: bool = False


@dataclasses.dataclass
class TrackState:
    name: str
    track_type: str = "instrument"
    notes: list[MidiNote] = dataclasses.field(default_factory=list)
    volume: float = 0.8
    pan: float = 0.0
    instrument: str = "Default Synth"
    instrument_mode: str = "AI Synth"
    rack_vsti: str = ""
    plugins: list[str] = dataclasses.field(default_factory=list)
    midi_program: int = 0
    midi_channel: int = 0
    synth_profile: str = "synth"
    rendered_audio_path: str = ""
    mute: bool = False
    solo: bool = False


@dataclasses.dataclass
class VSTInstrument:
    name: str
    path: str


@dataclasses.dataclass
class SampleAsset:
    path: str
    duration_sec: float
    sample_rate: int = 44100
    waveform_preview: list[float] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class SampleClip:
    path: str
    track_index: int
    start_sec: float
    duration_sec: float
    sample_rate: int = 44100
    waveform_preview: list[float] = dataclasses.field(default_factory=list)


class ProjectState:
    def __init__(self) -> None:
        self.tracks: list[TrackState] = [TrackState(name="Track 1")]
        self.bpm = DEFAULT_BPM
        self.quantize_div = 16
        self.vsti_paths: list[str] = []
        self.vsti_rack: list[VSTInstrument] = []
        self.sample_assets: list[SampleAsset] = []
        self.sample_clips: list[SampleClip] = []
        self.left_locator_sec = 0.0
        self.right_locator_sec = 8.0


class OpenAIClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "")

    def is_enabled(self) -> bool:
        return bool(self.api_key)

    def run_json_prompt(self, system_instruction: str, user_instruction: str) -> dict:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Set it in your environment first.")

        payload = {
            "model": OPENAI_MODEL,
            "input": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_instruction},
            ],
        }

        req = urllib.request.Request(
            OPENAI_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenAI API error: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI network error: {exc}") from exc

        result = json.loads(raw)
        text = result.get("output_text", "").strip()
        if not text:
            for item in result.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text += content.get("text", "")

        if not text:
            raise RuntimeError("No text returned by OpenAI response.")

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Model response was not valid JSON: {text[:300]}") from exc


class OpenAIComposer:
    def __init__(self, client: OpenAIClient) -> None:
        self.client = client

    def compose(self, prompt: str, bars: int, bpm: int) -> dict:
        system_instruction = (
            "You are a MIDI composer for a DAW. Return strict JSON only with the schema: "
            "{\"tracks\": [{\"name\": str, \"instrument\": str, \"notes\": "
            "[{\"start_beat\": number, \"duration_beat\": number, \"pitch\": int, \"velocity\": int}]}]}. "
            "Keep pitches in MIDI range 36..84 and fit inside requested bars."
        )
        user_instruction = (
            f"Create a multi-track arrangement. Prompt: {prompt}. Bars: {bars}. BPM: {bpm}. "
            "Use 2-5 tracks and musically coherent note timing."
        )
        return self.client.run_json_prompt(system_instruction, user_instruction)


class InstrumentIntelligence:
    FAMILY_PROFILES = {
        "strings": "saw_pad",
        "horn": "brass_stack",
        "brass": "brass_stack",
        "woodwind": "reed_breath",
        "piano": "e_piano",
        "bass": "sub_bass",
        "guitar": "pluck",
        "organ": "organ",
        "synth": "synth",
        "drums": "noise_kit",
    }

    def __init__(self, client: OpenAIClient) -> None:
        self.client = client

    def gm_instrument_name(self, program: int) -> str:
        p = int(clamp(program, 0, 127))
        if p < 8:
            return "Piano"
        if p < 16:
            return "Chromatic"
        if p < 24:
            return "Organ"
        if p < 32:
            return "Guitar"
        if p < 40:
            return "Bass"
        if p < 48:
            return "Strings"
        if p < 56:
            return "Ensemble"
        if p < 64:
            return "Brass"
        if p < 72:
            return "Reed"
        if p < 80:
            return "Pipe"
        if p < 88:
            return "Lead"
        if p < 96:
            return "Pad"
        if p < 104:
            return "FX"
        if p < 112:
            return "Ethnic"
        if p < 120:
            return "Percussive"
        return "SFX"

    def _fallback_family(self, program: int, channel: int, track_name: str) -> str:
        if channel == 9:
            return "drums"
        name = track_name.lower()
        for token in ["string", "violin", "cello"]:
            if token in name:
                return "strings"
        for token in ["horn", "trumpet", "trombone", "brass"]:
            if token in name:
                return "horn"
        for token in ["piano", "keys"]:
            if token in name:
                return "piano"
        if 32 <= program <= 39:
            return "bass"
        if 40 <= program <= 51:
            return "strings"
        if 56 <= program <= 63:
            return "brass"
        if 24 <= program <= 31:
            return "guitar"
        if 16 <= program <= 23:
            return "organ"
        return "synth"

    def classify_family(self, program: int, channel: int, track_name: str) -> str:
        fallback = self._fallback_family(program, channel, track_name)
        if not self.client.is_enabled():
            return fallback

        system_instruction = (
            "You classify MIDI tracks into one family. Return JSON only: "
            "{\"family\": one_of:[\"strings\",\"horn\",\"brass\",\"woodwind\",\"piano\",\"bass\",\"guitar\",\"organ\",\"synth\",\"drums\"]}."
        )
        user_instruction = (
            f"Track name: {track_name}. MIDI program: {program}. Channel: {channel}. "
            f"GM guess: {self.gm_instrument_name(program)}."
        )
        try:
            result = self.client.run_json_prompt(system_instruction, user_instruction)
            family = str(result.get("family", "")).lower().strip()
            if family in self.FAMILY_PROFILES:
                return family
        except Exception:
            pass
        return fallback


class AISynthRenderer:
    def __init__(self, sample_rate: int = 44100) -> None:
        self.sample_rate = sample_rate

    def _adsr(self, t: float, duration: float, a: float, d: float, s: float, r: float) -> float:
        if t < 0 or duration <= 0:
            return 0.0
        if t < a:
            return t / max(a, 1e-6)
        if t < a + d:
            return 1.0 - (1.0 - s) * ((t - a) / max(d, 1e-6))
        if t < max(0.0, duration - r):
            return s
        if t < duration:
            return s * (1.0 - (t - (duration - r)) / max(r, 1e-6))
        return 0.0

    def _wave(self, phase: float, profile: str) -> float:
        if profile == "sub_bass":
            return math.sin(phase)
        if profile == "pluck":
            return 0.7 * math.sin(phase) + 0.3 * math.sin(2.0 * phase)
        if profile == "organ":
            return 0.6 * math.sin(phase) + 0.25 * math.sin(2.0 * phase) + 0.15 * math.sin(3.0 * phase)
        if profile == "saw_pad":
            frac = (phase / (2 * math.pi)) % 1.0
            return 2.0 * frac - 1.0
        if profile == "brass_stack":
            return 0.5 * math.sin(phase) + 0.35 * math.sin(2.0 * phase) + 0.15 * math.sin(3.0 * phase)
        if profile == "reed_breath":
            return 0.8 * math.sin(phase) + 0.2 * math.sin(4.0 * phase)
        if profile == "noise_kit":
            return math.sin(phase * 13.0) * math.sin(phase * 7.0)
        if profile == "e_piano":
            return 0.8 * math.sin(phase) + 0.2 * math.sin(6.0 * phase)
        return math.sin(phase) + 0.2 * math.sin(2.0 * phase)

    def _profile_envelope(self, profile: str) -> tuple[float, float, float, float]:
        if profile in {"pluck", "e_piano"}:
            return (0.005, 0.08, 0.45, 0.12)
        if profile in {"strings", "saw_pad", "organ"}:
            return (0.03, 0.25, 0.75, 0.18)
        if profile in {"brass_stack", "reed_breath"}:
            return (0.015, 0.1, 0.65, 0.1)
        if profile == "noise_kit":
            return (0.001, 0.02, 0.2, 0.03)
        if profile == "sub_bass":
            return (0.01, 0.06, 0.75, 0.08)
        return (0.01, 0.1, 0.65, 0.12)

    def render_track(self, track: TrackState, bpm: int, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        max_tick = max((n.start_tick + n.duration_tick for n in track.notes), default=0)
        max_sec = (max_tick / TICKS_PER_BEAT) * (60.0 / max(1, bpm)) + 1.0
        total_samples = int(max_sec * self.sample_rate)
        data = [0.0] * max(1, total_samples)

        a, d, s, r = self._profile_envelope(track.synth_profile)
        for note in track.notes:
            start_sec = (note.start_tick / TICKS_PER_BEAT) * (60.0 / max(1, bpm))
            dur_sec = (note.duration_tick / TICKS_PER_BEAT) * (60.0 / max(1, bpm))
            start_idx = int(start_sec * self.sample_rate)
            note_samples = max(1, int(dur_sec * self.sample_rate))
            freq = midi_to_hz(note.pitch)
            amp = (note.velocity / 127.0) * track.volume * 0.4

            for i in range(note_samples):
                idx = start_idx + i
                if idx >= len(data):
                    break
                t = i / self.sample_rate
                phase = 2.0 * math.pi * freq * t
                env = self._adsr(t, dur_sec, a, d, s, r)
                sample = self._wave(phase, track.synth_profile) * env * amp
                data[idx] += sample

        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            frames = bytearray()
            for value in data:
                clipped = int(clamp(value, -1.0, 1.0) * 32767)
                frames.extend(struct.pack("<h", clipped))
            wf.writeframes(frames)


class PianoRollWidget(QtWidgets.QGraphicsView):
    noteChanged = QtCore.Signal()

    def __init__(self, project: ProjectState, get_track_index_callable) -> None:
        super().__init__()
        self.project = project
        self.get_track_index = get_track_index_callable
        self.scene_obj = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene_obj)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.RubberBandDrag)
        self.cell_w = 24
        self.cell_h = 14
        self.total_beats = 64
        self._drawing = False
        self._draw_start: QtCore.QPointF | None = None
        self._current_rect_item: QtWidgets.QGraphicsRectItem | None = None
        self.refresh()

    def current_track(self) -> TrackState:
        return self.project.tracks[self.get_track_index()]

    def refresh(self) -> None:
        self.scene_obj.clear()
        width = self.total_beats * self.cell_w
        pitch_count = PITCH_MAX - PITCH_MIN + 1
        height = pitch_count * self.cell_h

        for i in range(pitch_count):
            y = i * self.cell_h
            pitch = PITCH_MAX - i
            color = QtGui.QColor(40, 40, 40) if pitch % 12 in (1, 3, 6, 8, 10) else QtGui.QColor(55, 55, 55)
            self.scene_obj.addRect(0, y, width, self.cell_h, QtGui.QPen(QtCore.Qt.PenStyle.NoPen), QtGui.QBrush(color))

        for beat in range(self.total_beats + 1):
            x = beat * self.cell_w
            pen = QtGui.QPen(QtGui.QColor(120, 120, 120) if beat % 4 == 0 else QtGui.QColor(80, 80, 80))
            self.scene_obj.addLine(x, 0, x, height, pen)

        for i in range(pitch_count + 1):
            y = i * self.cell_h
            self.scene_obj.addLine(0, y, width, y, QtGui.QPen(QtGui.QColor(80, 80, 80)))

        for note in self.current_track().notes:
            self._draw_note(note)

        self.setSceneRect(0, 0, width, height)

    def _draw_note(self, note: MidiNote) -> None:
        x = note.start_tick / TICKS_PER_BEAT * self.cell_w
        w = max(1, note.duration_tick / TICKS_PER_BEAT * self.cell_w)
        y_idx = PITCH_MAX - note.pitch
        y = y_idx * self.cell_h
        rect = QtCore.QRectF(x, y, w, self.cell_h)
        color = QtGui.QColor(255, 165, 0) if note.selected else QtGui.QColor(60, 180, 240)
        item = self.scene_obj.addRect(rect, QtGui.QPen(QtGui.QColor(0, 0, 0)), QtGui.QBrush(color))
        item.setData(0, note)
        item.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and (event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier):
            self._drawing = True
            self._draw_start = self.mapToScene(event.position().toPoint())
            self._current_rect_item = self.scene_obj.addRect(QtCore.QRectF(self._draw_start, self._draw_start), QtGui.QPen(QtGui.QColor("yellow")))
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drawing and self._current_rect_item is not None and self._draw_start is not None:
            pos = self.mapToScene(event.position().toPoint())
            rect = QtCore.QRectF(self._draw_start, pos).normalized()
            self._current_rect_item.setRect(rect)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drawing and self._draw_start is not None:
            self._drawing = False
            end = self.mapToScene(event.position().toPoint())
            rect = QtCore.QRectF(self._draw_start, end).normalized()
            if self._current_rect_item is not None:
                self.scene_obj.removeItem(self._current_rect_item)
                self._current_rect_item = None

            start_beat = max(0.0, rect.left() / self.cell_w)
            end_beat = max(start_beat + 0.125, rect.right() / self.cell_w)
            pitch_idx = int(rect.top() // self.cell_h)
            pitch = max(PITCH_MIN, min(PITCH_MAX, PITCH_MAX - pitch_idx))

            note = MidiNote(
                start_tick=int(start_beat * TICKS_PER_BEAT),
                duration_tick=max(60, int((end_beat - start_beat) * TICKS_PER_BEAT)),
                pitch=pitch,
            )
            self.current_track().notes.append(note)
            self.refresh()
            self.noteChanged.emit()
            return
        super().mouseReleaseEvent(event)

    def sync_selection(self) -> None:
        for item in self.scene_obj.items():
            note = item.data(0)
            if isinstance(note, MidiNote):
                note.selected = item.isSelected()

    def delete_selected(self) -> None:
        self.sync_selection()
        track = self.current_track()
        track.notes = [n for n in track.notes if not n.selected]
        self.refresh()
        self.noteChanged.emit()

    def quantize_selected(self) -> None:
        self.sync_selection()
        grid = TICKS_PER_BEAT * 4 // self.project.quantize_div
        for note in self.current_track().notes:
            if note.selected:
                note.start_tick = round(note.start_tick / grid) * grid
                note.duration_tick = max(grid, round(note.duration_tick / grid) * grid)
        self.refresh()
        self.noteChanged.emit()


class TimelineWidget(QtWidgets.QTableWidget):
    def __init__(self, project: ProjectState) -> None:
        super().__init__(0, 8)
        self.project = project
        self.setHorizontalHeaderLabels(["Track", "Type", "Instrument", "Mode", "Profile", "Mute", "Solo", "Notes"])
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.refresh()

    def refresh(self) -> None:
        self.setRowCount(len(self.project.tracks))
        for i, track in enumerate(self.project.tracks):
            self.setItem(i, 0, QtWidgets.QTableWidgetItem(track.name))
            self.setItem(i, 1, QtWidgets.QTableWidgetItem(track.track_type.title()))
            self.setItem(i, 2, QtWidgets.QTableWidgetItem(track.instrument))
            self.setItem(i, 3, QtWidgets.QTableWidgetItem(track.instrument_mode))
            self.setItem(i, 4, QtWidgets.QTableWidgetItem(track.synth_profile))
            self.setItem(i, 5, QtWidgets.QTableWidgetItem('Yes' if track.mute else 'No'))
            self.setItem(i, 6, QtWidgets.QTableWidgetItem('Yes' if track.solo else 'No'))
            self.setItem(i, 7, QtWidgets.QTableWidgetItem(str(len(track.notes))))


class SampleTimelineWidget(QtWidgets.QGraphicsView):
    def __init__(self, project: ProjectState, get_sample_track_indices, on_drop_sample) -> None:
        super().__init__()
        self.project = project
        self.get_sample_track_indices = get_sample_track_indices
        self.on_drop_sample = on_drop_sample
        self.scene_obj = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene_obj)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        self.pixels_per_second = 80
        self.lane_height = 110
        self.setAcceptDrops(True)
        self.refresh()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasText() and event.mimeData().text().startswith('sample_asset:'):
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        self.dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        payload = event.mimeData().text()
        if not payload.startswith('sample_asset:'):
            event.ignore()
            return
        try:
            sample_idx = int(payload.split(':', 1)[1])
        except ValueError:
            event.ignore()
            return
        pos = self.mapToScene(event.position().toPoint())
        sample_tracks = self.get_sample_track_indices()
        if not sample_tracks:
            QtWidgets.QMessageBox.information(self, 'No sample track', 'Create a sample track first, then drag a sample here.')
            event.ignore()
            return
        lane = int(pos.y() // self.lane_height)
        lane = max(0, min(lane, len(sample_tracks) - 1))
        start_sec = max(0.0, pos.x() / self.pixels_per_second)
        self.on_drop_sample(sample_idx, sample_tracks[lane], start_sec)
        event.acceptProposedAction()

    def refresh(self) -> None:
        self.scene_obj.clear()
        sample_tracks = self.get_sample_track_indices()
        lane_count = max(1, len(sample_tracks))
        duration = max(8.0, self.project.right_locator_sec + 1.0)
        for clip in self.project.sample_clips:
            duration = max(duration, clip.start_sec + clip.duration_sec + 1.0)

        width = duration * self.pixels_per_second
        height = self.lane_height * lane_count
        self.scene_obj.addRect(0, 0, width, height, QtGui.QPen(QtGui.QColor(70, 70, 70)), QtGui.QBrush(QtGui.QColor(35, 35, 35)))

        for lane in range(lane_count):
            y0 = lane * self.lane_height
            self.scene_obj.addLine(0, y0, width, y0, QtGui.QPen(QtGui.QColor(65, 65, 65)))

        sec = 0
        while sec <= int(duration) + 1:
            x = sec * self.pixels_per_second
            pen = QtGui.QPen(QtGui.QColor(120, 120, 120) if sec % 4 == 0 else QtGui.QColor(80, 80, 80))
            self.scene_obj.addLine(x, 0, x, height, pen)
            sec += 1

        for locator_sec, color in ((self.project.left_locator_sec, QtGui.QColor(0, 200, 160)), (self.project.right_locator_sec, QtGui.QColor(240, 200, 0))):
            x = locator_sec * self.pixels_per_second
            self.scene_obj.addLine(x, 0, x, height, QtGui.QPen(color, 2))

        for clip in self.project.sample_clips:
            if clip.track_index not in sample_tracks:
                continue
            lane = sample_tracks.index(clip.track_index)
            x = clip.start_sec * self.pixels_per_second
            w = max(1, clip.duration_sec * self.pixels_per_second)
            y = lane * self.lane_height + 26
            h = 70
            self.scene_obj.addRect(x, y, w, h, QtGui.QPen(QtGui.QColor(0, 0, 0)), QtGui.QBrush(QtGui.QColor(72, 130, 200)))
            if clip.waveform_preview:
                path = QtGui.QPainterPath()
                step = w / max(1, len(clip.waveform_preview) - 1)
                mid = y + h / 2
                amp = h / 2 - 6
                path.moveTo(x, mid)
                for i, v in enumerate(clip.waveform_preview):
                    path.lineTo(x + i * step, mid - (v * amp))
                self.scene_obj.addPath(path, QtGui.QPen(QtGui.QColor(230, 240, 255)))
            label = self.scene_obj.addText(Path(clip.path).name)
            label.setDefaultTextColor(QtGui.QColor(240, 240, 240))
            label.setPos(x + 4, y + 4)

        self.setSceneRect(0, 0, width, height)


class MixerWidget(QtWidgets.QWidget):
    def __init__(self, project: ProjectState, current_track_callable) -> None:
        super().__init__()
        self.project = project
        self.current_track_callable = current_track_callable

        layout = QtWidgets.QFormLayout(self)
        self.volume = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.pan = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.pan.setRange(-100, 100)
        self.pan.setValue(0)
        self.mute = QtWidgets.QCheckBox('Mute')
        self.solo = QtWidgets.QCheckBox('Solo')
        self.rendered_path = QtWidgets.QLineEdit()
        self.rendered_path.setReadOnly(True)
        layout.addRow("Volume", self.volume)
        layout.addRow("Pan", self.pan)
        layout.addRow("Track state", self.mute)
        layout.addRow("", self.solo)
        layout.addRow("Rendered audio", self.rendered_path)

        self.volume.valueChanged.connect(self.apply_changes)
        self.pan.valueChanged.connect(self.apply_changes)
        self.mute.toggled.connect(self.apply_changes)
        self.solo.toggled.connect(self.apply_changes)

    def load_track(self) -> None:
        track = self.current_track_callable()
        self.volume.setValue(int(track.volume * 100))
        self.pan.setValue(int(track.pan * 100))
        self.mute.setChecked(track.mute)
        self.solo.setChecked(track.solo)
        self.rendered_path.setText(track.rendered_audio_path)

    def apply_changes(self) -> None:
        track = self.current_track_callable()
        track.volume = self.volume.value() / 100
        track.pan = self.pan.value() / 100
        track.mute = self.mute.isChecked()
        track.solo = self.solo.isChecked()


class InstrumentFxWidget(QtWidgets.QWidget):
    def __init__(self, project: ProjectState, current_track_callable, refresh_vsti_choices_callable) -> None:
        super().__init__()
        self.project = project
        self.current_track_callable = current_track_callable
        self.refresh_vsti_choices_callable = refresh_vsti_choices_callable

        root = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.instrument_mode = QtWidgets.QComboBox()
        self.instrument_mode.addItems(["AI Synth", "General MIDI", "VSTI Rack"])
        self.instrument = QtWidgets.QComboBox()
        self.instrument.addItems(["Default Synth", "Piano", "Bass", "Lead", "Sampler"])
        self.vsti_selector = QtWidgets.QComboBox()
        self.profile = QtWidgets.QLineEdit()
        self.profile.setReadOnly(True)
        form.addRow("Instrument type", self.instrument_mode)
        form.addRow("Instrument", self.instrument)
        form.addRow("VSTI rack", self.vsti_selector)
        form.addRow("AI synth profile", self.profile)

        self.fx_controls: dict[str, QtWidgets.QSlider] = {}
        for fx in ["EQ", "Compression", "Distortion", "Phaser", "Flanger", "Delay", "Reverb"]:
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(30)
            form.addRow(fx, slider)
            self.fx_controls[fx] = slider

        root.addLayout(form)
        self.instrument.currentTextChanged.connect(self.apply_changes)
        self.instrument_mode.currentTextChanged.connect(self.apply_changes)
        self.vsti_selector.currentTextChanged.connect(self.apply_changes)

    def reload_vsti_choices(self) -> None:
        current = self.vsti_selector.currentText()
        self.vsti_selector.clear()
        self.vsti_selector.addItem('None')
        for vst in self.project.vsti_rack:
            self.vsti_selector.addItem(vst.name)
        idx = self.vsti_selector.findText(current)
        if idx >= 0:
            self.vsti_selector.setCurrentIndex(idx)

    def load_track(self) -> None:
        self.reload_vsti_choices()
        track = self.current_track_callable()
        idx_mode = self.instrument_mode.findText(track.instrument_mode)
        if idx_mode >= 0:
            self.instrument_mode.setCurrentIndex(idx_mode)
        idx = self.instrument.findText(track.instrument)
        if idx >= 0:
            self.instrument.setCurrentIndex(idx)
        rack_idx = self.vsti_selector.findText(track.rack_vsti or 'None')
        if rack_idx >= 0:
            self.vsti_selector.setCurrentIndex(rack_idx)
        self.profile.setText(track.synth_profile)

    def apply_changes(self) -> None:
        track = self.current_track_callable()
        track.instrument_mode = self.instrument_mode.currentText()
        track.instrument = self.instrument.currentText()
        track.rack_vsti = '' if self.vsti_selector.currentText() == 'None' else self.vsti_selector.currentText()
        track.plugins = [f"{name}:{slider.value()}" for name, slider in self.fx_controls.items()]


class SampleLibraryWidget(QtWidgets.QListWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setDragEnabled(True)

    def mimeData(self, items):
        mime = super().mimeData(items)
        if items:
            payload = items[0].data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(payload, str):
                mime.setText(payload)
        return mime


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.project = ProjectState()
        self.ai_client = OpenAIClient()
        self.composer = OpenAIComposer(self.ai_client)
        self.instrument_ai = InstrumentIntelligence(self.ai_client)
        self.renderer = AISynthRenderer()
        self.setWindowTitle("AI Music Studio")
        self.resize(1500, 900)

        self.track_list = QtWidgets.QListWidget()
        self.track_list.currentRowChanged.connect(self._track_changed)

        self.timeline = TimelineWidget(self.project)
        self.piano_roll = PianoRollWidget(self.project, self.current_track_index)
        self.piano_roll.noteChanged.connect(self.on_notes_changed)

        self.mixer = MixerWidget(self.project, self.current_track)
        self.instruments = InstrumentFxWidget(self.project, self.current_track, self.refresh_vsti_rack_ui)
        self.sample_timeline = SampleTimelineWidget(self.project, self.sample_track_indices, self.place_sample_asset_on_track)
        self.sample_library = SampleLibraryWidget()

        quantize_box = QtWidgets.QComboBox()
        quantize_box.addItems(["1/4", "1/8", "1/16", "1/32"])
        quantize_box.setCurrentText("1/16")
        quantize_box.currentTextChanged.connect(lambda text: setattr(self.project, "quantize_div", int(text.split("/")[1])))

        add_track_btn = QtWidgets.QPushButton("+ Track (Sample/Instrument)")
        add_track_btn.clicked.connect(self.add_track)

        import_btn = QtWidgets.QPushButton("Import MIDI + AI Instrument Render")
        import_btn.clicked.connect(self.import_midi)
        export_btn = QtWidgets.QPushButton("Export MIDI")
        export_btn.clicked.connect(self.export_midi)
        render_btn = QtWidgets.QPushButton("Render AI Audio Stems")
        render_btn.clicked.connect(self.render_all_tracks)
        import_sample_btn = QtWidgets.QPushButton("Import Sample (WAV/MP3)")
        import_sample_btn.clicked.connect(self.import_sample)
        place_sample_btn = QtWidgets.QPushButton("Place Selected Sample On Timeline")
        place_sample_btn.clicked.connect(self.place_selected_sample)
        export_audio_btn = QtWidgets.QPushButton("Export Sample Timeline Audio (WAV/MP3)")
        export_audio_btn.clicked.connect(self.export_sample_timeline_audio)
        ai_btn = QtWidgets.QPushButton("AI Compose (OpenAI Codex)")
        ai_btn.clicked.connect(self.compose_with_ai)

        
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.addWidget(QtWidgets.QLabel("Tracks"))
        left_layout.addWidget(self.track_list)
        left_layout.addWidget(add_track_btn)
        left_layout.addWidget(QtWidgets.QLabel("Quantize"))
        left_layout.addWidget(quantize_box)
        left_layout.addWidget(import_btn)
        left_layout.addWidget(export_btn)
        left_layout.addWidget(render_btn)
        left_layout.addWidget(QtWidgets.QLabel("Samples Toolbox"))
        left_layout.addWidget(self.sample_library)
        left_layout.addWidget(import_sample_btn)
        left_layout.addWidget(place_sample_btn)
        left_layout.addWidget(export_audio_btn)
        left_layout.addWidget(ai_btn)
        left_layout.addStretch()

        right_tabs = QtWidgets.QTabWidget()
        right_tabs.addTab(self.timeline, "Timeline")
        right_tabs.addTab(self.sample_timeline, "Sample Timeline")
        right_tabs.addTab(self.mixer, "Mixer")
        right_tabs.addTab(self.instruments, "Instruments / FX")

        splitter_vertical = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter_vertical.addWidget(self.piano_roll)
        splitter_vertical.addWidget(right_tabs)
        splitter_vertical.setSizes([550, 320])

        splitter_main = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter_main.addWidget(left_panel)
        splitter_main.addWidget(splitter_vertical)
        splitter_main.setSizes([320, 1180])

        self.setCentralWidget(splitter_main)
        self._setup_menus()
        self._setup_floating_transport()
        self._setup_shortcuts()
        self._populate_track_list()
        self.track_list.setCurrentRow(0)
        self._setup_virtual_piano_dock()
        self.refresh_sample_library()

    def _setup_menus(self) -> None:
        settings = self.menuBar().addMenu('Settings')
        instruments_menu = settings.addMenu('Instruments')
        add_vsti = QtGui.QAction('Add VSTI Path', self)
        add_vsti.triggered.connect(self.add_vsti_path)
        instruments_menu.addAction(add_vsti)
        self.vsti_menu = instruments_menu
        self.refresh_vsti_rack_ui()

    def _setup_floating_transport(self) -> None:
        transport = QtWidgets.QToolBar('Transport', self)
        transport.setFloatable(True)
        transport.setMovable(True)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, transport)
        play_action = transport.addAction('Play')
        stop_action = transport.addAction('Stop')
        play_action.triggered.connect(lambda: self.statusBar().showMessage('Playback started (simulation)'))
        stop_action.triggered.connect(lambda: self.statusBar().showMessage('Playback stopped'))
        transport.addSeparator()
        self.left_locator = QtWidgets.QDoubleSpinBox()
        self.left_locator.setRange(0.0, 3600.0)
        self.left_locator.setValue(self.project.left_locator_sec)
        self.right_locator = QtWidgets.QDoubleSpinBox()
        self.right_locator.setRange(0.0, 3600.0)
        self.right_locator.setValue(self.project.right_locator_sec)
        self.left_locator.valueChanged.connect(self.update_locators)
        self.right_locator.valueChanged.connect(self.update_locators)
        transport.addWidget(QtWidgets.QLabel('L'))
        transport.addWidget(self.left_locator)
        transport.addWidget(QtWidgets.QLabel('R'))
        transport.addWidget(self.right_locator)

    def update_locators(self) -> None:
        self.project.left_locator_sec = min(self.left_locator.value(), self.right_locator.value())
        self.project.right_locator_sec = max(self.left_locator.value(), self.right_locator.value())
        self.sample_timeline.refresh()

    def add_vsti_path(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose VST instrument', str(Path.cwd()), 'VST Plugins (*.dll *.vst3 *.so);;All files (*)')
        if not path:
            return
        self.project.vsti_paths.append(path)
        self.project.vsti_rack.append(VSTInstrument(name=Path(path).stem, path=path))
        self.refresh_vsti_rack_ui()
        self.statusBar().showMessage(f'Added VSTI to rack: {Path(path).name}')

    def refresh_vsti_rack_ui(self) -> None:
        if hasattr(self, 'vsti_menu'):
            existing = [a for a in self.vsti_menu.actions() if a.property('rack_item')]
            for action in existing:
                self.vsti_menu.removeAction(action)
            if self.project.vsti_rack:
                self.vsti_menu.addSeparator()
                for vst in self.project.vsti_rack:
                    action = QtGui.QAction(f'Rack: {vst.name}', self)
                    action.setProperty('rack_item', True)
                    action.setEnabled(False)
                    self.vsti_menu.addAction(action)
        self.instruments.reload_vsti_choices()
        self._populate_track_list()

    def sample_track_indices(self) -> list[int]:
        return [i for i, track in enumerate(self.project.tracks) if track.track_type == 'sample']

    def place_sample_asset_on_track(self, asset_index: int, track_index: int, start_sec: float) -> None:
        if asset_index < 0 or asset_index >= len(self.project.sample_assets):
            return
        asset = self.project.sample_assets[asset_index]
        clip = SampleClip(path=asset.path, track_index=track_index, start_sec=start_sec, duration_sec=asset.duration_sec, sample_rate=asset.sample_rate, waveform_preview=asset.waveform_preview)
        self.project.sample_clips.append(clip)
        self.refresh_sample_library()

    def _setup_shortcuts(self) -> None:
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+N"), self, self.new_project)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+O"), self, self.import_midi)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, self.export_midi)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+O"), self, self.import_sample)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+E"), self, self.export_sample_timeline_audio)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self, self.render_all_tracks)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Q"), self, self.piano_roll.quantize_selected)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+G"), self, self.compose_with_ai)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Delete), self, self.piano_roll.delete_selected)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Space), self, lambda: self.statusBar().showMessage("Play/Stop toggled"))

    def _setup_virtual_piano_dock(self) -> None:
        key_map = {"Z": 60, "X": 62, "C": 64, "V": 65, "B": 67, "N": 69, "M": 71, ",": 72}
        dock = QtWidgets.QDockWidget("Virtual Piano Input")
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(panel)
        layout.addWidget(QtWidgets.QLabel("Use keys Z X C V B N M , to input notes"))
        dock.setWidget(panel)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)

        for key, pitch in key_map.items():
            QtGui.QShortcut(QtGui.QKeySequence(key), self, lambda p=pitch: self.insert_live_note(p))

    def compose_with_ai(self) -> None:
        prompt, ok = QtWidgets.QInputDialog.getText(self, "AI Composition Prompt", "Describe the song/arrangement:")
        if not ok or not prompt.strip():
            return
        bars, ok = QtWidgets.QInputDialog.getInt(self, "Bars", "Song length (bars):", 8, 1, 256)
        if not ok:
            return

        self.statusBar().showMessage("Requesting OpenAI Codex composition...")
        try:
            result = self.composer.compose(prompt=prompt.strip(), bars=bars, bpm=self.project.bpm)
            self._apply_ai_result(result)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "AI composition failed", str(exc))
            self.statusBar().showMessage("AI composition failed")

    def _apply_ai_result(self, result: dict) -> None:
        tracks = result.get("tracks", [])
        if not isinstance(tracks, list) or not tracks:
            raise RuntimeError("AI returned no tracks.")

        built_tracks: list[TrackState] = []
        for idx, track in enumerate(tracks, start=1):
            if not isinstance(track, dict):
                continue
            state = TrackState(name=str(track.get("name") or f"AI Track {idx}"))
            state.instrument = str(track.get("instrument") or "Default Synth")
            state.synth_profile = "synth"
            for note in track.get("notes", []):
                if not isinstance(note, dict):
                    continue
                start_beat = float(note.get("start_beat", 0.0))
                duration_beat = max(0.125, float(note.get("duration_beat", 0.5)))
                pitch = int(note.get("pitch", 60))
                velocity = int(note.get("velocity", 100))
                state.notes.append(
                    MidiNote(
                        start_tick=max(0, int(start_beat * TICKS_PER_BEAT)),
                        duration_tick=max(1, int(duration_beat * TICKS_PER_BEAT)),
                        pitch=max(PITCH_MIN, min(PITCH_MAX, pitch)),
                        velocity=max(1, min(127, velocity)),
                    )
                )
            built_tracks.append(state)

        if not built_tracks:
            raise RuntimeError("AI returned invalid track data.")

        self.project.tracks = built_tracks
        self._populate_track_list()
        self.track_list.setCurrentRow(0)
        self.on_notes_changed()
        self.refresh_sample_library()
        self.statusBar().showMessage(f"AI generated {len(built_tracks)} track(s)")

    def _classify_and_assign_track_sound(self, track: TrackState) -> None:
        family = self.instrument_ai.classify_family(track.midi_program, track.midi_channel, track.name)
        profile = InstrumentIntelligence.FAMILY_PROFILES.get(family, "synth")
        gm_name = self.instrument_ai.gm_instrument_name(track.midi_program)
        track.instrument = f"{gm_name} ({family})"
        track.synth_profile = profile

    def render_all_tracks(self) -> None:
        if not self.project.tracks:
            return
        stem_paths: list[str] = []
        for index, track in enumerate(self.project.tracks, start=1):
            stem_name = f"track_{index:02d}_{track.name.replace(' ', '_')}.wav"
            stem_path = RENDER_DIR / stem_name
            self.renderer.render_track(track, self.project.bpm, stem_path)
            track.rendered_audio_path = str(stem_path)
            stem_paths.append(str(stem_path))

        self.timeline.refresh()
        self.mixer.load_track()
        self.statusBar().showMessage(f"Rendered {len(stem_paths)} AI synthesis stems to {RENDER_DIR}/")
        QtWidgets.QMessageBox.information(self, "AI synthesis rendered", "\n".join(stem_paths))

    def refresh_sample_library(self) -> None:
        self.sample_library.clear()
        for idx, asset in enumerate(self.project.sample_assets):
            item = QtWidgets.QListWidgetItem(f"{Path(asset.path).name} ({asset.duration_sec:.2f}s)")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, f"sample_asset:{idx}")
            self.sample_library.addItem(item)
        self.sample_timeline.refresh()

    def import_sample(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import sample",
            str(Path.cwd()),
            "Audio files (*.wav *.mp3)",
        )
        if not path:
            return

        src = Path(path)
        sample_wav = src
        if src.suffix.lower() == ".mp3":
            converted = Path.cwd() / "renders" / f"{src.stem}_import.wav"
            convert_audio(src, converted)
            sample_wav = converted

        preview, sample_rate, duration = load_wav_preview(sample_wav)
        asset = SampleAsset(
            path=str(sample_wav),
            duration_sec=duration,
            sample_rate=sample_rate,
            waveform_preview=preview,
        )
        self.project.sample_assets.append(asset)
        self.refresh_sample_library()
        self.statusBar().showMessage(f"Imported sample asset: {src.name}")

    def place_selected_sample(self) -> None:
        row = self.sample_library.currentRow()
        if row < 0 or row >= len(self.project.sample_assets):
            QtWidgets.QMessageBox.information(self, "No sample selected", "Select a sample from the samples toolbox first.")
            return
        sample_tracks = self.sample_track_indices()
        if not sample_tracks:
            QtWidgets.QMessageBox.information(self, "No sample track", "Create a sample track before placing samples.")
            return
        start_sec, ok = QtWidgets.QInputDialog.getDouble(self, "Place sample", "Start time (seconds):", 0.0, 0.0, 3600.0, 2)
        if not ok:
            return
        self.place_sample_asset_on_track(row, sample_tracks[0], float(start_sec))

    def export_sample_timeline_audio(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export sample timeline audio",
            str(Path.cwd() / "sample_timeline.wav"),
            "Audio files (*.wav *.mp3)",
        )
        if not path:
            return

        if not self.project.sample_clips:
            QtWidgets.QMessageBox.information(self, "No samples", "No samples are placed on the timeline.")
            return

        sample_rate = 44100
        max_end = 1.0
        loaded: list[tuple[SampleClip, list[float], int]] = []
        for clip in self.project.sample_clips:
            wav_path = Path(clip.path)
            if wav_path.suffix.lower() == ".mp3":
                converted = Path.cwd() / "renders" / f"{wav_path.stem}_mix.wav"
                convert_audio(wav_path, converted)
                wav_path = converted
            data, sr = load_wav_samples(wav_path)
            loaded.append((clip, data, sr))
            clip_len = len(data) / sr
            max_end = max(max_end, clip.start_sec + clip_len)

        mix = [0.0] * int(max_end * sample_rate)
        for clip, data, sr in loaded:
            if sr != sample_rate:
                ratio = sr / sample_rate
                resampled = []
                for i in range(int(len(data) / ratio)):
                    resampled.append(data[min(len(data) - 1, int(i * ratio))])
                data = resampled
            offset = int(clip.start_sec * sample_rate)
            for i, v in enumerate(data):
                idx = offset + i
                if idx >= len(mix):
                    break
                mix[idx] += v * 0.7

        mix = [clamp(v, -1.0, 1.0) for v in mix]
        out = Path(path)
        if out.suffix.lower() == ".mp3":
            temp_wav = Path.cwd() / "renders" / "sample_timeline_export.wav"
            write_wav_samples(temp_wav, mix, sample_rate)
            convert_audio(temp_wav, out)
        else:
            write_wav_samples(out, mix, sample_rate)

        self.statusBar().showMessage(f"Exported sample timeline audio: {out.name}")


    def _build_tracks_from_midi(self, midi: mido.MidiFile) -> list[TrackState]:
        built: list[TrackState] = []
        for track_idx, mtrack in enumerate(midi.tracks):
            abs_tick = 0
            channel_program: dict[int, int] = {ch: 0 for ch in range(16)}
            active_notes: dict[tuple[int, int], tuple[int, int]] = {}
            channel_data: dict[int, TrackState] = {}

            for msg in mtrack:
                abs_tick += msg.time
                if hasattr(msg, "channel"):
                    channel = int(msg.channel)
                else:
                    channel = 0

                if msg.type == "program_change":
                    channel_program[channel] = int(msg.program)
                    if channel not in channel_data:
                        channel_data[channel] = TrackState(
                            name=f"{mtrack.name or f'Track {track_idx + 1}'} [Ch {channel + 1}]",
                            midi_program=channel_program[channel],
                            midi_channel=channel,
                        )
                    else:
                        channel_data[channel].midi_program = channel_program[channel]
                    continue

                if msg.type == "note_on" and msg.velocity > 0:
                    program = channel_program.get(channel, 0)
                    active_notes[(channel, msg.note)] = (abs_tick, program)
                    if channel not in channel_data:
                        channel_data[channel] = TrackState(
                            name=f"{mtrack.name or f'Track {track_idx + 1}'} [Ch {channel + 1}]",
                            midi_program=program,
                            midi_channel=channel,
                        )
                    continue

                if (msg.type in {"note_off", "note_on"} and msg.velocity == 0) or msg.type == "note_off":
                    key = (channel, msg.note)
                    if key in active_notes:
                        start_tick, program = active_notes.pop(key)
                        if channel not in channel_data:
                            channel_data[channel] = TrackState(
                                name=f"{mtrack.name or f'Track {track_idx + 1}'} [Ch {channel + 1}]",
                                midi_program=program,
                                midi_channel=channel,
                            )
                        state = channel_data[channel]
                        state.midi_program = program
                        state.notes.append(
                            MidiNote(
                                start_tick=start_tick,
                                duration_tick=max(1, abs_tick - start_tick),
                                pitch=int(msg.note),
                                velocity=int(getattr(msg, "velocity", 100) or 100),
                            )
                        )

            for state in channel_data.values():
                if state.notes:
                    self._classify_and_assign_track_sound(state)
                    built.append(state)

        return built

    def insert_live_note(self, pitch: int) -> None:
        track = self.current_track()
        cursor_tick = max((n.start_tick + n.duration_tick for n in track.notes), default=0)
        track.notes.append(MidiNote(start_tick=cursor_tick, duration_tick=TICKS_PER_BEAT // 2, pitch=pitch))
        self.on_notes_changed()

    def current_track_index(self) -> int:
        row = self.track_list.currentRow()
        return 0 if row < 0 else row

    def current_track(self) -> TrackState:
        return self.project.tracks[self.current_track_index()]

    def _populate_track_list(self) -> None:
        self.track_list.clear()
        for track in self.project.tracks:
            extra = f"VST:{track.rack_vsti}" if track.rack_vsti else track.instrument
            self.track_list.addItem(f"{track.name} • {track.track_type} • {extra}")

    def _track_changed(self, row: int) -> None:
        if row < 0:
            return
        track = self.current_track()
        self.piano_roll.setEnabled(track.track_type == 'instrument')
        self.piano_roll.refresh()
        self.mixer.load_track()
        self.instruments.load_track()

    def add_track(self) -> None:
        track_type, ok = QtWidgets.QInputDialog.getItem(self, 'Add track', 'Track type:', ['instrument', 'sample'], 0, False)
        if not ok:
            return
        idx = len(self.project.tracks) + 1
        state = TrackState(name=f"Track {idx}", track_type=track_type)
        if track_type == 'sample':
            state.instrument = 'Sample Track'
            state.instrument_mode = 'Sample'
        self.project.tracks.append(state)
        self._populate_track_list()
        self.track_list.setCurrentRow(idx - 1)
        self.timeline.refresh()
        self.sample_timeline.refresh()

    def new_project(self) -> None:
        self.project = ProjectState()
        self.timeline.project = self.project
        self.piano_roll.project = self.project
        self.sample_timeline.project = self.project
        self._populate_track_list()
        self.track_list.setCurrentRow(0)
        self.on_notes_changed()

    def on_notes_changed(self) -> None:
        self.piano_roll.refresh()
        self.timeline.refresh()
        self.sample_timeline.refresh()

    def import_midi(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import MIDI", str(Path.cwd()), "MIDI files (*.mid *.midi)")
        if not path:
            return

        midi = mido.MidiFile(path)
        self.project.tracks = self._build_tracks_from_midi(midi)

        if not self.project.tracks:
            self.project.tracks = [TrackState(name="Track 1")]

        self._populate_track_list()
        self.track_list.setCurrentRow(0)
        self.on_notes_changed()

        do_render = QtWidgets.QMessageBox.question(
            self,
            "AI synthesis render",
            "Imported MIDI and assigned AI instrument profiles per channel. Render synthesized audio stems now?",
        )
        if do_render == QtWidgets.QMessageBox.StandardButton.Yes:
            self.render_all_tracks()

        self.statusBar().showMessage(f"Imported MIDI with {len(self.project.tracks)} track(s): {Path(path).name}")

    def export_midi(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export MIDI", str(Path.cwd() / "project.mid"), "MIDI files (*.mid)")
        if not path:
            return

        midi = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
        for track_state in self.project.tracks:
            mtrack = mido.MidiTrack()
            mtrack.name = track_state.name
            midi.tracks.append(mtrack)
            mtrack.append(mido.Message("program_change", channel=track_state.midi_channel, program=int(clamp(track_state.midi_program, 0, 127)), time=0))

            events: list[tuple[int, mido.Message]] = []
            for note in track_state.notes:
                events.append(
                    (
                        note.start_tick,
                        mido.Message("note_on", channel=track_state.midi_channel, note=note.pitch, velocity=note.velocity, time=0),
                    )
                )
                events.append(
                    (
                        note.start_tick + note.duration_tick,
                        mido.Message("note_off", channel=track_state.midi_channel, note=note.pitch, velocity=0, time=0),
                    )
                )

            events.sort(key=lambda x: x[0])
            current = 0
            for abs_tick, msg in events:
                msg.time = max(0, abs_tick - current)
                mtrack.append(msg)
                current = abs_tick

        midi.save(path)
        self.statusBar().showMessage(f"Exported MIDI: {Path(path).name}")


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(30, 30, 30))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(25, 25, 25))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(220, 220, 220))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
