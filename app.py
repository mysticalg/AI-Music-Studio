from __future__ import annotations

import dataclasses
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import mido
from PySide6 import QtCore, QtGui, QtWidgets

TICKS_PER_BEAT = 480
DEFAULT_BPM = 120
PITCH_MIN = 36
PITCH_MAX = 84
OPENAI_API_URL = "https://api.openai.com/v1/responses"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-codex")


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


class ProjectState:
    def __init__(self) -> None:
        self.tracks: list[TrackState] = [TrackState(name="Track 1")]
        self.bpm = DEFAULT_BPM
        self.quantize_div = 16


class OpenAIComposer:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "")

    def compose(self, prompt: str, bars: int, bpm: int) -> dict:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Set it in your environment first.")

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
        super().__init__(0, 3)
        self.project = project
        self.setHorizontalHeaderLabels(["Track", "Length (beats)", "Notes"])
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
            self.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{length_beats:.2f}"))
            self.setItem(i, 2, QtWidgets.QTableWidgetItem(str(len(track.notes))))


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
        layout.addRow("Volume", self.volume)
        layout.addRow("Pan", self.pan)

        self.volume.valueChanged.connect(self.apply_changes)
        self.pan.valueChanged.connect(self.apply_changes)

    def load_track(self) -> None:
        track = self.current_track_callable()
        self.volume.setValue(int(track.volume * 100))
        self.pan.setValue(int(track.pan * 100))

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
        form.addRow("Instrument", self.instrument)

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

    def apply_changes(self) -> None:
        track = self.current_track_callable()
        track.instrument = self.instrument.currentText()
        track.plugins = [f"{name}:{slider.value()}" for name, slider in self.fx_controls.items()]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.project = ProjectState()
        self.composer = OpenAIComposer()
        self.setWindowTitle("AI Music Studio")
        self.resize(1400, 850)

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

        import_btn = QtWidgets.QPushButton("Import MIDI")
        import_btn.clicked.connect(self.import_midi)
        export_btn = QtWidgets.QPushButton("Export MIDI")
        export_btn.clicked.connect(self.export_midi)
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
        splitter_vertical.setSizes([500, 300])

        splitter_main = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter_main.addWidget(left_panel)
        splitter_main.addWidget(splitter_vertical)
        splitter_main.setSizes([250, 1100])

        self.setCentralWidget(splitter_main)
        self._setup_shortcuts()
        self._populate_track_list()
        self.track_list.setCurrentRow(0)
        self._setup_virtual_piano_dock()

    def _setup_shortcuts(self) -> None:
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+N"), self, self.new_project)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+O"), self, self.import_midi)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, self.export_midi)
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
            self.track_list.addItem(track.name)

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
        self.project.tracks = []
        for i, mtrack in enumerate(midi.tracks):
            state = TrackState(name=mtrack.name or f"Track {i + 1}")
            abs_tick = 0
            active: dict[int, int] = {}
            for msg in mtrack:
                abs_tick += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    active[msg.note] = abs_tick
                elif msg.type in {"note_off", "note_on"} and msg.note in active:
                    start = active.pop(msg.note)
                    state.notes.append(
                        MidiNote(start_tick=start, duration_tick=max(1, abs_tick - start), pitch=msg.note, velocity=getattr(msg, "velocity", 100))
                    )
            self.project.tracks.append(state)

        if not self.project.tracks:
            self.project.tracks = [TrackState(name="Track 1")]

        self._populate_track_list()
        self.track_list.setCurrentRow(0)
        self.on_notes_changed()
        self.statusBar().showMessage(f"Imported MIDI: {Path(path).name}")

    def export_midi(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export MIDI", str(Path.cwd() / "project.mid"), "MIDI files (*.mid)")
        if not path:
            return

        midi = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
        for track_state in self.project.tracks:
            mtrack = mido.MidiTrack()
            mtrack.name = track_state.name
            midi.tracks.append(mtrack)

            events: list[tuple[int, mido.Message]] = []
            for note in track_state.notes:
                events.append((note.start_tick, mido.Message("note_on", note=note.pitch, velocity=note.velocity, time=0)))
                events.append((note.start_tick + note.duration_tick, mido.Message("note_off", note=note.pitch, velocity=0, time=0)))

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
