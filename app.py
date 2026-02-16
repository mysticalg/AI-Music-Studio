from __future__ import annotations

import dataclasses
import json
import math
import os
import struct
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
    notes: list[MidiNote] = dataclasses.field(default_factory=list)
    volume: float = 0.8
    pan: float = 0.0
    instrument: str = "Default Synth"
    plugins: list[str] = dataclasses.field(default_factory=list)
    midi_program: int = 0
    midi_channel: int = 0
    synth_profile: str = "synth"
    rendered_audio_path: str = ""


class ProjectState:
    def __init__(self) -> None:
        self.tracks: list[TrackState] = [TrackState(name="Track 1")]
        self.bpm = DEFAULT_BPM
        self.quantize_div = 16


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
        super().__init__(0, 5)
        self.project = project
        self.setHorizontalHeaderLabels(["Track", "Instrument", "Profile", "Length (beats)", "Notes"])
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.refresh()

    def refresh(self) -> None:
        self.setRowCount(len(self.project.tracks))
        for i, track in enumerate(self.project.tracks):
            max_tick = max((n.start_tick + n.duration_tick for n in track.notes), default=0)
            length_beats = max_tick / TICKS_PER_BEAT
            self.setItem(i, 0, QtWidgets.QTableWidgetItem(track.name))
            self.setItem(i, 1, QtWidgets.QTableWidgetItem(track.instrument))
            self.setItem(i, 2, QtWidgets.QTableWidgetItem(track.synth_profile))
            self.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{length_beats:.2f}"))
            self.setItem(i, 4, QtWidgets.QTableWidgetItem(str(len(track.notes))))


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
        self.rendered_path = QtWidgets.QLineEdit()
        self.rendered_path.setReadOnly(True)
        layout.addRow("Volume", self.volume)
        layout.addRow("Pan", self.pan)
        layout.addRow("Rendered audio", self.rendered_path)

        self.volume.valueChanged.connect(self.apply_changes)
        self.pan.valueChanged.connect(self.apply_changes)

    def load_track(self) -> None:
        track = self.current_track_callable()
        self.volume.setValue(int(track.volume * 100))
        self.pan.setValue(int(track.pan * 100))
        self.rendered_path.setText(track.rendered_audio_path)

    def apply_changes(self) -> None:
        track = self.current_track_callable()
        track.volume = self.volume.value() / 100
        track.pan = self.pan.value() / 100


class InstrumentFxWidget(QtWidgets.QWidget):
    def __init__(self, project: ProjectState, current_track_callable) -> None:
        super().__init__()
        self.project = project
        self.current_track_callable = current_track_callable

        root = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.instrument = QtWidgets.QComboBox()
        self.instrument.addItems(["Default Synth", "Piano", "Bass", "Lead", "Sampler", "External VST (placeholder)"])
        self.profile = QtWidgets.QLineEdit()
        self.profile.setReadOnly(True)
        form.addRow("Instrument", self.instrument)
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

    def load_track(self) -> None:
        track = self.current_track_callable()
        idx = self.instrument.findText(track.instrument)
        if idx >= 0:
            self.instrument.setCurrentIndex(idx)
        self.profile.setText(track.synth_profile)

    def apply_changes(self) -> None:
        track = self.current_track_callable()
        track.instrument = self.instrument.currentText()
        track.plugins = [f"{name}:{slider.value()}" for name, slider in self.fx_controls.items()]


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
        self.instruments = InstrumentFxWidget(self.project, self.current_track)

        quantize_box = QtWidgets.QComboBox()
        quantize_box.addItems(["1/4", "1/8", "1/16", "1/32"])
        quantize_box.setCurrentText("1/16")
        quantize_box.currentTextChanged.connect(lambda text: setattr(self.project, "quantize_div", int(text.split("/")[1])))

        add_track_btn = QtWidgets.QPushButton("+ Track")
        add_track_btn.clicked.connect(self.add_track)

        import_btn = QtWidgets.QPushButton("Import MIDI + AI Instrument Render")
        import_btn.clicked.connect(self.import_midi)
        export_btn = QtWidgets.QPushButton("Export MIDI")
        export_btn.clicked.connect(self.export_midi)
        render_btn = QtWidgets.QPushButton("Render AI Audio Stems")
        render_btn.clicked.connect(self.render_all_tracks)
        ai_btn = QtWidgets.QPushButton("AI Compose (OpenAI Codex)")
        ai_btn.clicked.connect(self.compose_with_ai)

        play_btn = QtWidgets.QPushButton("Play")
        stop_btn = QtWidgets.QPushButton("Stop")
        play_btn.clicked.connect(lambda: self.statusBar().showMessage("Playback started (simulation)"))
        stop_btn.clicked.connect(lambda: self.statusBar().showMessage("Playback stopped"))

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
        left_layout.addWidget(ai_btn)
        left_layout.addWidget(play_btn)
        left_layout.addWidget(stop_btn)
        left_layout.addStretch()

        right_tabs = QtWidgets.QTabWidget()
        right_tabs.addTab(self.timeline, "Timeline")
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
        self._setup_shortcuts()
        self._populate_track_list()
        self.track_list.setCurrentRow(0)
        self._setup_virtual_piano_dock()

    def _setup_shortcuts(self) -> None:
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+N"), self, self.new_project)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+O"), self, self.import_midi)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, self.export_midi)
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
            self.track_list.addItem(f"{track.name} • {track.instrument}")

    def _track_changed(self, row: int) -> None:
        if row < 0:
            return
        self.piano_roll.refresh()
        self.mixer.load_track()
        self.instruments.load_track()

    def add_track(self) -> None:
        idx = len(self.project.tracks) + 1
        self.project.tracks.append(TrackState(name=f"Track {idx}"))
        self._populate_track_list()
        self.track_list.setCurrentRow(idx - 1)
        self.timeline.refresh()

    def new_project(self) -> None:
        self.project = ProjectState()
        self.timeline.project = self.project
        self.piano_roll.project = self.project
        self._populate_track_list()
        self.track_list.setCurrentRow(0)
        self.on_notes_changed()

    def on_notes_changed(self) -> None:
        self.piano_roll.refresh()
        self.timeline.refresh()

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
