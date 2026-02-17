from __future__ import annotations

import base64
import ctypes
import dataclasses
import hashlib
import json
import math
import os
import secrets
import shutil
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
import webbrowser
from pathlib import Path

import mido
from PySide6 import QtCore, QtGui, QtMultimedia, QtWidgets

try:
    import numpy as np
    from pedalboard import load_plugin
    PEDALBOARD_AVAILABLE = True
except Exception:
    np = None
    load_plugin = None
    PEDALBOARD_AVAILABLE = False

TICKS_PER_BEAT = 480
DEFAULT_BPM = 120
PITCH_MIN = 36
PITCH_MAX = 84
OPENAI_API_URL = "https://api.openai.com/v1/responses"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-codex")
RENDER_DIR = Path("renders")
APP_PREFS_PATH = Path(".ai_music_studio_prefs.json")


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
    vsti_parameters: dict[str, float] = dataclasses.field(default_factory=dict)
    vsti_state_path: str = ""
    vsti_output_gain_db: float = 0.0
    vsti_wet_mix: float = 100.0
    carla_automation_enabled: bool = True
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


class VSTBinaryLoader:
    def __init__(self) -> None:
        self._handles: dict[str, object] = {}
        self._errors: dict[str, str] = {}

    def is_loaded(self, path: str) -> bool:
        return path in self._handles

    def load(self, path: str) -> tuple[bool, str]:
        normalized = str(Path(path))
        if normalized in self._handles:
            return True, 'Already loaded'
        try:
            suffix = Path(normalized).suffix.lower()
            if os.name == 'nt' and suffix == '.dll':
                handle = ctypes.WinDLL(normalized)
            else:
                handle = ctypes.CDLL(normalized)
            self._handles[normalized] = handle
            self._errors.pop(normalized, None)
            return True, 'Loaded successfully'
        except Exception as exc:
            self._errors[normalized] = str(exc)
            return False, str(exc)

    def last_error(self, path: str) -> str:
        return self._errors.get(str(Path(path)), '')


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


@dataclasses.dataclass
class MidiSection:
    track_index: int
    start_sec: float
    duration_sec: float
    name: str = "MIDI Part"


class ProjectState:
    def __init__(self) -> None:
        self.tracks: list[TrackState] = [TrackState(name="Track 1")]
        self.bpm = DEFAULT_BPM
        self.quantize_div = 16
        self.quantize_triplet = False
        self.vsti_paths: list[str] = []
        self.sample_paths: list[str] = []
        self.vsti_rack: list[VSTInstrument] = []
        self.carla_host_path: str = ''
        self.sample_assets: list[SampleAsset] = []
        self.sample_clips: list[SampleClip] = []
        self.midi_sections: list[MidiSection] = []
        self.left_locator_sec = 0.0
        self.right_locator_sec = 8.0
        self.playhead_sec = 0.0


class OpenAIClient:
    AUTH_PATH = Path('.openai_auth.json')

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.oauth_access_token = ""
        self.oauth_refresh_token = ""
        self.oauth_expires_at = 0.0
        self._load_saved_auth()

    def _load_saved_auth(self) -> None:
        if not self.AUTH_PATH.exists():
            return
        try:
            payload = json.loads(self.AUTH_PATH.read_text())
        except Exception:
            return
        self.api_key = payload.get('api_key', self.api_key)
        self.oauth_access_token = payload.get('oauth_access_token', '')
        self.oauth_refresh_token = payload.get('oauth_refresh_token', '')
        self.oauth_expires_at = float(payload.get('oauth_expires_at', 0.0) or 0.0)

    def _save_auth(self) -> None:
        payload = {
            'api_key': self.api_key,
            'oauth_access_token': self.oauth_access_token,
            'oauth_refresh_token': self.oauth_refresh_token,
            'oauth_expires_at': self.oauth_expires_at,
        }
        self.AUTH_PATH.write_text(json.dumps(payload, indent=2))

    def is_enabled(self) -> bool:
        return bool(self.api_key or self.oauth_access_token)

    def auth_status(self) -> str:
        if self.oauth_access_token:
            if self.oauth_expires_at > time.time():
                mins = int((self.oauth_expires_at - time.time()) / 60)
                return f"OpenAI connected via OAuth (expires in ~{max(0, mins)} min)"
            return "OpenAI connected via OAuth"
        if self.api_key:
            return "OpenAI connected via API key"
        return "OpenAI not connected"

    def set_api_key(self, api_key: str) -> None:
        self.api_key = api_key.strip()
        self.oauth_access_token = ''
        self.oauth_refresh_token = ''
        self.oauth_expires_at = 0.0
        self._save_auth()

    def set_oauth_tokens(self, access_token: str, refresh_token: str = '', expires_in: int = 3600) -> None:
        self.api_key = ''
        self.oauth_access_token = access_token.strip()
        self.oauth_refresh_token = refresh_token.strip()
        self.oauth_expires_at = time.time() + max(0, int(expires_in or 0))
        self._save_auth()

    def clear_auth(self) -> None:
        self.api_key = ''
        self.oauth_access_token = ''
        self.oauth_refresh_token = ''
        self.oauth_expires_at = 0.0
        self._save_auth()

    def _authorization_header(self) -> str:
        token = self.oauth_access_token or self.api_key
        return f"Bearer {token}"

    def run_json_prompt(self, system_instruction: str, user_instruction: str) -> dict:
        if not self.is_enabled():
            raise RuntimeError("OpenAI is not connected. Use Settings > OpenAI to connect via API key or OAuth.")

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
                "Authorization": self._authorization_header(),
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

    def __init__(self, project: ProjectState, get_track_index_callable, set_playhead_callable=None, set_left_locator_callable=None, set_right_locator_callable=None) -> None:
        super().__init__()
        self.project = project
        self.get_track_index = get_track_index_callable
        self.set_playhead = set_playhead_callable or (lambda sec: setattr(self.project, 'playhead_sec', max(0.0, float(sec))))
        self.set_left_locator = set_left_locator_callable or (lambda sec: setattr(self.project, 'left_locator_sec', max(0.0, float(sec))))
        self.set_right_locator = set_right_locator_callable or (lambda sec: setattr(self.project, 'right_locator_sec', max(0.0, float(sec))))
        self.scene_obj = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene_obj)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.RubberBandDrag)
        self.cell_w = 24
        self.cell_h = 14
        self.total_beats = 64
        self.tool = 'pencil'
        self.note_length_div = 8
        self._line_start: QtCore.QPointF | None = None
        self._pencil_note: MidiNote | None = None
        self._pencil_anchor_tick = 0
        self._drag_anchor_tick = 0
        self._drag_anchor_pitch = 0
        self._drag_selected_snapshot: list[tuple[MidiNote, int, int]] = []
        self._drag_playhead = False
        self._locator_ruler_height = 16
        self.refresh()

    def current_track(self) -> TrackState:
        return self.project.tracks[self.get_track_index()]

    def _quantize_ticks(self) -> int:
        beats = 4.0 / max(1, self.project.quantize_div)
        if getattr(self.project, 'quantize_triplet', False):
            beats *= 2.0 / 3.0
        return max(1, int(round(beats * TICKS_PER_BEAT)))

    def _grid_tick(self) -> int:
        return self._quantize_ticks()

    def _pos_to_beat_pitch(self, pos: QtCore.QPointF) -> tuple[float, int]:
        beat = max(0.0, pos.x() / self.cell_w)
        pitch_idx = int(pos.y() // self.cell_h)
        pitch = max(PITCH_MIN, min(PITCH_MAX, PITCH_MAX - pitch_idx))
        return beat, pitch

    def _length_ticks(self) -> int:
        return max(1, self._quantize_ticks())

    def set_tool(self, tool: str) -> None:
        self.tool = tool
        if self.tool == 'select':
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.RubberBandDrag)
        else:
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)

    def set_note_length_div(self, div: int) -> None:
        self.note_length_div = max(1, div)

    def _set_headers(self) -> None:
        locator_info = f"L {self.project.left_locator_sec:.2f}s  R {self.project.right_locator_sec:.2f}s"
        self.setHorizontalHeaderLabels([
            f"Track ({locator_info})", "Type", "Instrument", "Mode", "Profile", "Mute", "Solo", "Notes"
        ])

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

        self.scene_obj.addRect(0, 0, width, self._locator_ruler_height, QtGui.QPen(QtGui.QColor(88, 88, 88)), QtGui.QBrush(QtGui.QColor(28, 28, 28, 220)))
        sec = 0
        max_sec = int((self.total_beats * 60.0) / max(1, self.project.bpm)) + 1
        while sec <= max_sec:
            x = sec * self.project.bpm / 60.0 * self.cell_w
            self.scene_obj.addLine(x, 0, x, self._locator_ruler_height, QtGui.QPen(QtGui.QColor(130, 130, 130)))
            if sec % 2 == 0:
                label = self.scene_obj.addSimpleText(f"{sec}s")
                label.setBrush(QtGui.QBrush(QtGui.QColor(210, 210, 210)))
                label.setPos(x + 2, 0)
            sec += 1

        for i in range(pitch_count + 1):
            y = i * self.cell_h
            self.scene_obj.addLine(0, y, width, y, QtGui.QPen(QtGui.QColor(80, 80, 80)))

        for note in self.current_track().notes:
            self._draw_note(note)

        bpm = max(1, self.project.bpm)
        for label, locator_sec, color in (('L', self.project.left_locator_sec, QtGui.QColor(0, 200, 160)), ('R', self.project.right_locator_sec, QtGui.QColor(240, 200, 0))):
            locator_x = locator_sec * (bpm / 60.0) * self.cell_w
            self.scene_obj.addLine(locator_x, 0, locator_x, height, QtGui.QPen(color, 2))
            tag = self.scene_obj.addSimpleText(label)
            tag.setBrush(QtGui.QBrush(color))
            tag.setPos(locator_x + 2, 0)

        playhead_beat = self.project.playhead_sec * (bpm / 60.0)
        playhead_x = playhead_beat * self.cell_w
        self.scene_obj.addLine(playhead_x, 0, playhead_x, height, QtGui.QPen(QtGui.QColor(255, 90, 90), 2))

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

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            step = 2 if delta > 0 else -2
            self.cell_w = max(8, min(96, self.cell_w + step))
            self.refresh()
            event.accept()
            return
        super().wheelEvent(event)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        menu = QtWidgets.QMenu(self)
        tools_menu = menu.addMenu('Editor Tools')
        actions: dict[str, QtGui.QAction] = {}
        group = QtGui.QActionGroup(menu)
        group.setExclusive(True)
        for key, label in [
            ('select', 'Selector'),
            ('pencil', 'Pencil'),
            ('scissors', 'Scissors'),
            ('eraser', 'Eraser'),
            ('line', 'Line Tool'),
        ]:
            action = tools_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(self.tool == key)
            group.addAction(action)
            actions[key] = action

        length_menu = menu.addMenu('Note Length')
        length_group = QtGui.QActionGroup(menu)
        length_group.setExclusive(True)
        for div in [1, 2, 4, 8, 16, 32, 64]:
            action = length_menu.addAction(f'1/{div}')
            action.setCheckable(True)
            action.setChecked(self.note_length_div == div)
            length_group.addAction(action)
            action.triggered.connect(lambda checked=False, d=div: self.set_note_length_div(d))

        chosen = menu.exec(event.globalPos())
        if chosen is None:
            return
        for key, action in actions.items():
            if chosen == action:
                self.set_tool(key)
                break

    def _find_note_at(self, scene_pos: QtCore.QPointF) -> MidiNote | None:
        for item in self.items(self.mapFromScene(scene_pos)):
            note = item.data(0)
            if isinstance(note, MidiNote):
                return note
        return None

    def _insert_note_at(self, scene_pos: QtCore.QPointF) -> MidiNote:
        beat, pitch = self._pos_to_beat_pitch(scene_pos)
        grid = self._grid_tick()
        start_tick = int(round((beat * TICKS_PER_BEAT) / grid) * grid)
        note = MidiNote(start_tick=start_tick, duration_tick=self._length_ticks(), pitch=pitch)
        self.current_track().notes.append(note)
        self.refresh()
        self.noteChanged.emit()
        return note

    def _erase_note_at(self, scene_pos: QtCore.QPointF) -> None:
        note = self._find_note_at(scene_pos)
        if note is None:
            return
        track = self.current_track()
        track.notes = [n for n in track.notes if n is not note]
        self.refresh()
        self.noteChanged.emit()

    def _slice_note_at(self, scene_pos: QtCore.QPointF) -> None:
        note = self._find_note_at(scene_pos)
        if note is None:
            return
        beat, _ = self._pos_to_beat_pitch(scene_pos)
        cut_tick = int(beat * TICKS_PER_BEAT)
        start = note.start_tick
        end = note.start_tick + note.duration_tick
        if cut_tick <= start or cut_tick >= end:
            return
        left = cut_tick - start
        right = end - cut_tick
        if left < 1 or right < 1:
            return
        note.duration_tick = left
        self.current_track().notes.append(MidiNote(start_tick=cut_tick, duration_tick=right, pitch=note.pitch, velocity=note.velocity))
        self.refresh()
        self.noteChanged.emit()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        scene_pos = self.mapToScene(event.position().toPoint())
        sec = max(0.0, scene_pos.x() / self.cell_w * (60.0 / max(1, self.project.bpm)))

        if scene_pos.y() <= self._locator_ruler_height and event.button() in (QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.RightButton):
            if event.button() == QtCore.Qt.MouseButton.RightButton or bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier):
                self.set_right_locator(sec)
            else:
                self.set_left_locator(sec)
            return

        if abs(scene_pos.x() - (self.project.playhead_sec * (self.project.bpm / 60.0) * self.cell_w)) <= 6 and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_playhead = True
            self.set_playhead(sec)
            return

        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.tool == 'pencil':
                clicked = self._find_note_at(scene_pos)
                if clicked is not None:
                    for note in self.current_track().notes:
                        note.selected = (note is clicked)
                    self.refresh()
                    self.noteChanged.emit()
                    return
                note = self._insert_note_at(scene_pos)
                self._pencil_note = note
                self._pencil_anchor_tick = note.start_tick
                return
            if self.tool == 'eraser':
                self._erase_note_at(scene_pos)
                return
            if self.tool == 'scissors':
                self._slice_note_at(scene_pos)
                return
            if self.tool == 'line':
                self._line_start = scene_pos
                return
            if self.tool == 'select':
                clicked = self._find_note_at(scene_pos)
                if clicked is not None:
                    self.sync_selection()
                    if not clicked.selected:
                        for note in self.current_track().notes:
                            note.selected = False
                        clicked.selected = True
                    self._drag_anchor_tick = clicked.start_tick
                    self._drag_anchor_pitch = clicked.pitch
                    self._drag_selected_snapshot = [
                        (n, n.start_tick, n.pitch) for n in self.current_track().notes if n.selected
                    ]
                    self.refresh()
                    self.noteChanged.emit()
                    return
                for note in self.current_track().notes:
                    note.selected = False
                self.refresh()
                self.noteChanged.emit()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        scene_pos = self.mapToScene(event.position().toPoint())
        if self._drag_playhead:
            sec = max(0.0, scene_pos.x() / self.cell_w * (60.0 / max(1, self.project.bpm)))
            self.set_playhead(sec)
            return
        if self.tool == 'pencil' and self._pencil_note is not None:
            beat, _ = self._pos_to_beat_pitch(scene_pos)
            grid = self._grid_tick()
            end_tick = int(round((beat * TICKS_PER_BEAT) / grid) * grid)
            new_duration = max(grid, end_tick - self._pencil_anchor_tick + grid)
            self._pencil_note.duration_tick = new_duration
            self.refresh()
            self.noteChanged.emit()
            return

        if self.tool == 'select' and self._drag_selected_snapshot:
            beat, pitch = self._pos_to_beat_pitch(scene_pos)
            grid = self._grid_tick()
            current_tick = int(round((beat * TICKS_PER_BEAT) / grid) * grid)
            delta_tick = current_tick - self._drag_anchor_tick
            delta_pitch = pitch - self._drag_anchor_pitch
            for note, start_tick, start_pitch in self._drag_selected_snapshot:
                note.start_tick = max(0, start_tick + delta_tick)
                note.pitch = max(PITCH_MIN, min(PITCH_MAX, start_pitch + delta_pitch))
            self.refresh()
            self.noteChanged.emit()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.tool == 'line' and self._line_start is not None and event.button() == QtCore.Qt.MouseButton.LeftButton:
            end_pos = self.mapToScene(event.position().toPoint())
            start_beat, start_pitch = self._pos_to_beat_pitch(self._line_start)
            end_beat, end_pitch = self._pos_to_beat_pitch(end_pos)
            if end_beat < start_beat:
                start_beat, end_beat = end_beat, start_beat
                start_pitch, end_pitch = end_pitch, start_pitch
            note_len_beats = 4 / max(1, self.note_length_div)
            count = max(1, int((end_beat - start_beat) / max(0.001, note_len_beats)) + 1)
            track = self.current_track()
            for i in range(count):
                t = 0.0 if count == 1 else i / (count - 1)
                beat = start_beat + (end_beat - start_beat) * t
                pitch = int(round(start_pitch + (end_pitch - start_pitch) * t))
                grid = self._grid_tick()
                start_tick = round((beat * TICKS_PER_BEAT) / grid) * grid
                track.notes.append(MidiNote(start_tick=int(start_tick), duration_tick=self._length_ticks(), pitch=max(PITCH_MIN, min(PITCH_MAX, pitch))))
            self._line_start = None
            self.refresh()
            self.noteChanged.emit()
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_playhead = False
            self._pencil_note = None
            self._drag_selected_snapshot = []
        self._line_start = None
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
        grid = self._grid_tick()
        for note in self.current_track().notes:
            if note.selected:
                note.start_tick = round(note.start_tick / grid) * grid
                note.duration_tick = max(grid, round(note.duration_tick / grid) * grid)
        self.refresh()
        self.noteChanged.emit()

    def duplicate_selected_by_grid(self) -> None:
        self.sync_selection()
        grid = self._grid_tick()
        selected = [n for n in self.current_track().notes if n.selected]
        if not selected:
            return
        for note in self.current_track().notes:
            note.selected = False
        for note in selected:
            duplicated = MidiNote(
                start_tick=max(0, note.start_tick + grid),
                duration_tick=note.duration_tick,
                pitch=note.pitch,
                velocity=note.velocity,
                selected=True,
            )
            self.current_track().notes.append(duplicated)
        self.refresh()
        self.noteChanged.emit()


class VelocityEditorWidget(QtWidgets.QGraphicsView):
    velocityChanged = QtCore.Signal()

    def __init__(self, project: ProjectState, get_track_index_callable) -> None:
        super().__init__()
        self.project = project
        self.get_track_index = get_track_index_callable
        self.scene_obj = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene_obj)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        self.setFixedHeight(140)
        self.cell_w = 24
        self.total_beats = 64
        self._drag_note: MidiNote | None = None
        self.refresh()

    def current_track(self) -> TrackState:
        return self.project.tracks[self.get_track_index()]

    def refresh(self) -> None:
        self.scene_obj.clear()
        width = self.total_beats * self.cell_w
        height = 120
        self.scene_obj.addRect(0, 0, width, height, QtGui.QPen(QtGui.QColor(70, 70, 70)), QtGui.QBrush(QtGui.QColor(28, 28, 28)))

        for note in self.current_track().notes:
            x = note.start_tick / TICKS_PER_BEAT * self.cell_w
            w = max(2, note.duration_tick / TICKS_PER_BEAT * self.cell_w)
            h = max(2, int((note.velocity / 127.0) * (height - 12)))
            y = height - h
            color = QtGui.QColor(255, 175, 80) if note.selected else QtGui.QColor(110, 210, 110)
            rect = self.scene_obj.addRect(x, y, w, h, QtGui.QPen(QtCore.Qt.PenStyle.NoPen), QtGui.QBrush(color))
            rect.setData(0, note)

        bpm = max(1, self.project.bpm)
        for label, locator_sec, color in (('L', self.project.left_locator_sec, QtGui.QColor(0, 200, 160)), ('R', self.project.right_locator_sec, QtGui.QColor(240, 200, 0))):
            locator_x = locator_sec * (bpm / 60.0) * self.cell_w
            self.scene_obj.addLine(locator_x, 0, locator_x, height, QtGui.QPen(color, 2))
            tag = self.scene_obj.addSimpleText(label)
            tag.setBrush(QtGui.QBrush(color))
            tag.setPos(locator_x + 2, 0)

        playhead_beat = self.project.playhead_sec * (bpm / 60.0)
        playhead_x = playhead_beat * self.cell_w
        self.scene_obj.addLine(playhead_x, 0, playhead_x, height, QtGui.QPen(QtGui.QColor(255, 90, 90), 2))

        self.setSceneRect(0, 0, width, height)

    def _apply_velocity_from_pos(self, scene_pos: QtCore.QPointF) -> None:
        if self._drag_note is None:
            return
        height = 120
        y = max(0.0, min(float(height), scene_pos.y()))
        vel = int(round((1.0 - (y / height)) * 127.0))
        self._drag_note.velocity = max(1, min(127, vel))
        self.refresh()
        self.velocityChanged.emit()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            item = self.itemAt(event.position().toPoint())
            if item is not None:
                note = item.data(0)
                if isinstance(note, MidiNote):
                    self._drag_note = note
                    self._apply_velocity_from_pos(self.mapToScene(event.position().toPoint()))
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_note is not None:
            self._apply_velocity_from_pos(self.mapToScene(event.position().toPoint()))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_note = None
        super().mouseReleaseEvent(event)


class TimelineWidget(QtWidgets.QTableWidget):
    def __init__(self, project: ProjectState) -> None:
        super().__init__(0, 8)
        self.project = project
        self._set_headers()
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.refresh()

    def _set_headers(self) -> None:
        locator_info = f"L {self.project.left_locator_sec:.2f}s  R {self.project.right_locator_sec:.2f}s"
        self.setHorizontalHeaderLabels([
            f"Track ({locator_info})", "Type", "Instrument", "Mode", "Profile", "Mute", "Solo", "Notes"
        ])

    def refresh(self) -> None:
        self._set_headers()
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
    def __init__(self, project: ProjectState, get_sample_track_indices, on_drop_sample, set_locator_callable) -> None:
        super().__init__()
        self.project = project
        self.get_sample_track_indices = get_sample_track_indices
        self.on_drop_sample = on_drop_sample
        self.set_locator_callable = set_locator_callable
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

        playhead_x = self.project.playhead_sec * self.pixels_per_second
        self.scene_obj.addLine(playhead_x, 0, playhead_x, height, QtGui.QPen(QtGui.QColor(255, 90, 90), 2))

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

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            sec = max(0.0, self.mapToScene(event.position().toPoint()).x() / self.pixels_per_second)
            self.set_locator_callable(sec)
        super().mousePressEvent(event)


class ArrangementOverviewWidget(QtWidgets.QGraphicsView):
    locatorChanged = QtCore.Signal(float)

    def __init__(self, project: ProjectState, set_locator_callable, set_left_locator_callable, set_right_locator_callable, on_section_moved_callable, get_bpm_callable) -> None:
        super().__init__()
        self.project = project
        self.set_locator_callable = set_locator_callable
        self.set_left_locator_callable = set_left_locator_callable
        self.set_right_locator_callable = set_right_locator_callable
        self.on_section_moved = on_section_moved_callable
        self.get_bpm = get_bpm_callable
        self.scene_obj = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene_obj)
        self.pixels_per_second = 80
        self.lane_height = 56
        self._drag_index: int | None = None
        self._drag_offset_sec = 0.0
        self._drag_origin_start_sec = 0.0
        self._drag_origin_track_index = 0
        self.arrangement_quantize_mode = "beat"
        self._drag_playhead = False
        self._locator_ruler_height = 16
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        self.setMouseTracking(True)
        self.refresh()

    def _duration_seconds(self) -> float:
        duration = max(8.0, self.project.right_locator_sec + 1.0)
        for section in self.project.midi_sections:
            duration = max(duration, section.start_sec + section.duration_sec + 1.0)
        return duration

    def _lane_count(self) -> int:
        return max(1, len(self.project.tracks))

    def refresh(self) -> None:
        self.scene_obj.clear()
        duration = self._duration_seconds()
        lane_count = self._lane_count()
        width = duration * self.pixels_per_second
        height = lane_count * self.lane_height

        self.scene_obj.addRect(0, 0, width, height, QtGui.QPen(QtGui.QColor(70, 70, 70)), QtGui.QBrush(QtGui.QColor(33, 33, 33)))

        for lane in range(lane_count):
            y = lane * self.lane_height
            self.scene_obj.addLine(0, y, width, y, QtGui.QPen(QtGui.QColor(62, 62, 62)))
            if lane < len(self.project.tracks):
                label = self.scene_obj.addText(self.project.tracks[lane].name)
                label.setDefaultTextColor(QtGui.QColor(188, 188, 188))
                label.setPos(4, y + 2)

        self.scene_obj.addRect(0, 0, width, self._locator_ruler_height, QtGui.QPen(QtGui.QColor(88, 88, 88)), QtGui.QBrush(QtGui.QColor(28, 28, 28, 220)))
        sec = 0
        while sec <= int(duration) + 1:
            x = sec * self.pixels_per_second
            self.scene_obj.addLine(x, 0, x, height, QtGui.QPen(QtGui.QColor(96, 96, 96) if sec % 4 == 0 else QtGui.QColor(74, 74, 74)))
            if sec % 2 == 0:
                label = self.scene_obj.addSimpleText(f"{sec}s")
                label.setBrush(QtGui.QBrush(QtGui.QColor(210, 210, 210)))
                label.setPos(x + 2, 0)
            sec += 1

        for idx, section in enumerate(self.project.midi_sections):
            if section.track_index >= lane_count:
                continue
            x = section.start_sec * self.pixels_per_second
            w = max(10, section.duration_sec * self.pixels_per_second)
            y = section.track_index * self.lane_height + 20
            h = self.lane_height - 24
            rect = self.scene_obj.addRect(x, y, w, h, QtGui.QPen(QtGui.QColor(0, 0, 0)), QtGui.QBrush(QtGui.QColor(125, 88, 220)))
            rect.setData(0, idx)
            label = self.scene_obj.addText(section.name)
            label.setDefaultTextColor(QtGui.QColor(242, 242, 242))
            label.setPos(x + 4, y + 2)

        locator_x = self.project.playhead_sec * self.pixels_per_second
        self.scene_obj.addLine(locator_x, 0, locator_x, height, QtGui.QPen(QtGui.QColor(255, 80, 80), 2))

        for locator_sec, color in ((self.project.left_locator_sec, QtGui.QColor(0, 200, 160)), (self.project.right_locator_sec, QtGui.QColor(240, 200, 0))):
            x = locator_sec * self.pixels_per_second
            self.scene_obj.addLine(x, 0, x, height, QtGui.QPen(color, 1))

        self.setSceneRect(0, 0, width, height)

    def _snap_seconds(self, sec: float) -> float:
        bpm = max(1, int(self.get_bpm()))
        beat_sec = 60.0 / bpm
        grid = beat_sec * 4.0 if self.arrangement_quantize_mode == 'bar' else beat_sec
        if grid <= 0:
            return max(0.0, sec)
        return max(0.0, round(sec / grid) * grid)

    def _set_playhead_from_event(self, event: QtGui.QMouseEvent) -> None:
        sec = max(0.0, self.mapToScene(event.position().toPoint()).x() / self.pixels_per_second)
        self.set_locator_callable(sec)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        scene_pos = self.mapToScene(event.position().toPoint())
        sec = max(0.0, scene_pos.x() / self.pixels_per_second)

        if scene_pos.y() <= self._locator_ruler_height and event.button() in (QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.RightButton):
            if event.button() == QtCore.Qt.MouseButton.RightButton or bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier):
                self.set_right_locator_callable(sec)
            else:
                self.set_left_locator_callable(sec)
            return

        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if abs(scene_pos.x() - (self.project.playhead_sec * self.pixels_per_second)) <= 6:
                self._drag_playhead = True
                self.set_locator_callable(sec)
                return
            item = self.itemAt(event.position().toPoint())
            if item is not None and item.data(0) is not None:
                self._drag_index = int(item.data(0))
                section = self.project.midi_sections[self._drag_index]
                x_sec = self.mapToScene(event.position().toPoint()).x() / self.pixels_per_second
                self._drag_offset_sec = max(0.0, x_sec - section.start_sec)
                self._drag_origin_start_sec = section.start_sec
                self._drag_origin_track_index = section.track_index
            else:
                self._drag_index = None
                self._set_playhead_from_event(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_playhead:
            sec = max(0.0, self.mapToScene(event.position().toPoint()).x() / self.pixels_per_second)
            self.set_locator_callable(sec)
            return
        if self._drag_index is not None and 0 <= self._drag_index < len(self.project.midi_sections):
            pos = self.mapToScene(event.position().toPoint())
            x_sec = pos.x() / self.pixels_per_second
            lane = int(max(0, pos.y()) // self.lane_height)
            lane = max(0, min(self._lane_count() - 1, lane))
            section = self.project.midi_sections[self._drag_index]
            section.start_sec = self._snap_seconds(max(0.0, x_sec - self._drag_offset_sec))
            section.track_index = lane
            self.refresh()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self._drag_playhead:
                self._drag_playhead = False
            elif self._drag_index is None:
                self._set_playhead_from_event(event)
            elif 0 <= self._drag_index < len(self.project.midi_sections):
                section = self.project.midi_sections[self._drag_index]
                self.on_section_moved(
                    self._drag_index,
                    self._drag_origin_start_sec,
                    section.start_sec,
                    self._drag_origin_track_index,
                    section.track_index,
                )
            self._drag_index = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            step = 8 if delta > 0 else -8
            self.pixels_per_second = max(24, min(320, self.pixels_per_second + step))
            self.refresh()
            event.accept()
            return
        super().wheelEvent(event)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        menu = QtWidgets.QMenu(self)
        quantize_menu = menu.addMenu('Arrangement Quantize')
        beat_action = quantize_menu.addAction('Beat')
        bar_action = quantize_menu.addAction('Bar')
        beat_action.setCheckable(True)
        bar_action.setCheckable(True)
        beat_action.setChecked(self.arrangement_quantize_mode == 'beat')
        bar_action.setChecked(self.arrangement_quantize_mode == 'bar')
        chosen = menu.exec(event.globalPos())
        if chosen == beat_action:
            self.arrangement_quantize_mode = 'beat'
        elif chosen == bar_action:
            self.arrangement_quantize_mode = 'bar'


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
    def __init__(self, project: ProjectState, current_track_callable, refresh_vsti_choices_callable, on_track_updated_callable=None, load_selected_vsti_callable=None, open_vsti_gui_callable=None, vsti_param_names_callable=None) -> None:
        super().__init__()
        self.project = project
        self.current_track_callable = current_track_callable
        self.refresh_vsti_choices_callable = refresh_vsti_choices_callable
        self.on_track_updated = on_track_updated_callable
        self.load_selected_vsti = load_selected_vsti_callable
        self.open_vsti_gui = open_vsti_gui_callable
        self.vsti_param_names_callable = vsti_param_names_callable

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
        self.midi_channel = QtWidgets.QSpinBox()
        self.midi_channel.setRange(1, 16)
        self.midi_program = QtWidgets.QSpinBox()
        self.midi_program.setRange(0, 127)
        form.addRow("MIDI channel", self.midi_channel)
        form.addRow("MIDI program", self.midi_program)

        self.fx_controls: dict[str, QtWidgets.QSlider] = {}
        for fx in ["EQ", "Compression", "Distortion", "Phaser", "Flanger", "Delay", "Reverb"]:
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(30)
            form.addRow(fx, slider)
            self.fx_controls[fx] = slider

        root.addLayout(form)

        btn_row = QtWidgets.QHBoxLayout()
        self.assign_rack_btn = QtWidgets.QPushButton('Use Selected Rack VSTI')
        self.load_vsti_btn = QtWidgets.QPushButton('Load Selected VSTI Binary')
        self.open_vsti_gui_btn = QtWidgets.QPushButton('Open VSTI GUI')
        self.edit_vsti_params_btn = QtWidgets.QPushButton('Edit VSTI Parameters')
        btn_row.addWidget(self.assign_rack_btn)
        btn_row.addWidget(self.load_vsti_btn)
        btn_row.addWidget(self.open_vsti_gui_btn)
        btn_row.addWidget(self.edit_vsti_params_btn)
        root.addLayout(btn_row)

        self.instrument.currentTextChanged.connect(self.apply_changes)
        self.instrument_mode.currentTextChanged.connect(self.apply_changes)
        self.vsti_selector.currentTextChanged.connect(self.apply_changes)
        self.midi_channel.valueChanged.connect(self.apply_changes)
        self.midi_program.valueChanged.connect(self.apply_changes)
        self.assign_rack_btn.clicked.connect(self.assign_selected_rack_vsti)
        self.load_vsti_btn.clicked.connect(self.load_selected_vsti_binary)
        self.open_vsti_gui_btn.clicked.connect(self.open_selected_vsti_gui)
        self.edit_vsti_params_btn.clicked.connect(self.edit_vsti_parameters)
        for slider in self.fx_controls.values():
            slider.valueChanged.connect(self.apply_changes)

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

        self.instrument_mode.blockSignals(True)
        self.instrument.blockSignals(True)
        self.vsti_selector.blockSignals(True)
        self.midi_channel.blockSignals(True)
        self.midi_program.blockSignals(True)

        idx_mode = self.instrument_mode.findText(track.instrument_mode)
        if idx_mode >= 0:
            self.instrument_mode.setCurrentIndex(idx_mode)

        idx = self.instrument.findText(track.instrument)
        if idx < 0 and track.instrument:
            self.instrument.addItem(track.instrument)
            idx = self.instrument.findText(track.instrument)
        if idx >= 0:
            self.instrument.setCurrentIndex(idx)

        rack_idx = self.vsti_selector.findText(track.rack_vsti or 'None')
        if rack_idx >= 0:
            self.vsti_selector.setCurrentIndex(rack_idx)

        self.midi_channel.setValue(int(track.midi_channel) + 1)
        self.midi_program.setValue(int(track.midi_program))

        self.instrument_mode.blockSignals(False)
        self.instrument.blockSignals(False)
        self.vsti_selector.blockSignals(False)
        self.midi_channel.blockSignals(False)
        self.midi_program.blockSignals(False)
        self.profile.setText(track.synth_profile)

    def apply_changes(self) -> None:
        track = self.current_track_callable()
        track.instrument_mode = self.instrument_mode.currentText()
        selected_rack = '' if self.vsti_selector.currentText() == 'None' else self.vsti_selector.currentText()
        track.rack_vsti = selected_rack
        if track.instrument_mode == 'VSTI Rack' and selected_rack:
            track.instrument = selected_rack
        else:
            track.instrument = self.instrument.currentText()
        track.midi_channel = int(self.midi_channel.value()) - 1
        track.midi_program = int(self.midi_program.value())
        track.plugins = [f"{name}:{slider.value()}" for name, slider in self.fx_controls.items()]
        if callable(self.on_track_updated):
            self.on_track_updated()


    def assign_selected_rack_vsti(self) -> None:
        selected = '' if self.vsti_selector.currentText() == 'None' else self.vsti_selector.currentText()
        if not selected:
            QtWidgets.QMessageBox.information(self, 'No rack VSTI', 'Select a rack VSTI first.')
            return
        self.instrument_mode.setCurrentText('VSTI Rack')
        self.vsti_selector.setCurrentText(selected)
        self.apply_changes()

    def load_selected_vsti_binary(self) -> None:
        if not callable(self.load_selected_vsti):
            return
        selected = '' if self.vsti_selector.currentText() == 'None' else self.vsti_selector.currentText()
        if not selected:
            QtWidgets.QMessageBox.information(self, 'No rack VSTI', 'Select a rack VSTI first.')
            return
        self.load_selected_vsti(selected)

    def open_selected_vsti_gui(self) -> None:
        if not callable(self.open_vsti_gui):
            return
        selected = '' if self.vsti_selector.currentText() == 'None' else self.vsti_selector.currentText()
        if not selected:
            QtWidgets.QMessageBox.information(self, 'No rack VSTI', 'Select a rack VSTI first.')
            return
        self.open_vsti_gui(selected)

    def edit_vsti_parameters(self) -> None:
        track = self.current_track_callable()
        if track.instrument_mode != 'VSTI Rack' or not track.rack_vsti:
            QtWidgets.QMessageBox.information(self, 'No VSTI assigned', 'Assign a rack VSTI to this track first.')
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f'VSTI Parameters - {track.rack_vsti}')
        layout = QtWidgets.QFormLayout(dialog)
        param_names: list[str] = []
        if callable(self.vsti_param_names_callable):
            param_names = self.vsti_param_names_callable(track.rack_vsti)
        if not param_names:
            param_names = [f'Param {i}' for i in range(1, 9)]

        sliders: dict[str, QtWidgets.QSlider] = {}
        for key in param_names[:12]:
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(int(track.vsti_parameters.get(key, 50)))
            layout.addRow(key, slider)
            sliders[key] = slider

        gain_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        gain_slider.setRange(-240, 240)
        gain_slider.setValue(int(track.vsti_output_gain_db * 10.0))
        gain_label = QtWidgets.QLabel(f'{track.vsti_output_gain_db:.1f} dB')
        gain_slider.valueChanged.connect(lambda value: gain_label.setText(f'{value / 10.0:.1f} dB'))
        gain_row = QtWidgets.QHBoxLayout()
        gain_row.addWidget(gain_slider)
        gain_row.addWidget(gain_label)
        gain_widget = QtWidgets.QWidget()
        gain_widget.setLayout(gain_row)
        layout.addRow('Output gain', gain_widget)

        wet_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        wet_slider.setRange(0, 100)
        wet_slider.setValue(int(track.vsti_wet_mix))
        wet_label = QtWidgets.QLabel(f'{track.vsti_wet_mix:.0f}%')
        wet_slider.valueChanged.connect(lambda value: wet_label.setText(f'{value:.0f}%'))
        wet_row = QtWidgets.QHBoxLayout()
        wet_row.addWidget(wet_slider)
        wet_row.addWidget(wet_label)
        wet_widget = QtWidgets.QWidget()
        wet_widget.setLayout(wet_row)
        layout.addRow('Wet mix', wet_widget)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        track.vsti_parameters = {name: float(slider.value()) for name, slider in sliders.items()}
        track.vsti_output_gain_db = gain_slider.value() / 10.0
        track.vsti_wet_mix = float(wet_slider.value())
        if callable(self.on_track_updated):
            self.on_track_updated()


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


class OpenAIConnectDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('Connect OpenAI')
        self.resize(620, 420)
        self.code_verifier = ''

        layout = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)

        api_key_tab = QtWidgets.QWidget()
        api_key_form = QtWidgets.QFormLayout(api_key_tab)
        self.api_key_input = QtWidgets.QLineEdit()
        self.api_key_input.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        api_key_form.addRow('API key', self.api_key_input)
        tabs.addTab(api_key_tab, 'API Key')

        oauth_tab = QtWidgets.QWidget()
        oauth_form = QtWidgets.QFormLayout(oauth_tab)
        self.client_id_input = QtWidgets.QLineEdit(os.getenv('OPENAI_OAUTH_CLIENT_ID', ''))
        self.auth_url_input = QtWidgets.QLineEdit(os.getenv('OPENAI_OAUTH_AUTHORIZE_URL', 'https://auth.openai.com/oauth/authorize'))
        self.token_url_input = QtWidgets.QLineEdit(os.getenv('OPENAI_OAUTH_TOKEN_URL', 'https://auth.openai.com/oauth/token'))
        self.redirect_uri_input = QtWidgets.QLineEdit(os.getenv('OPENAI_OAUTH_REDIRECT_URI', 'http://127.0.0.1:8765/callback'))
        self.scope_input = QtWidgets.QLineEdit(os.getenv('OPENAI_OAUTH_SCOPE', 'openid profile email'))
        self.auth_code_input = QtWidgets.QLineEdit()
        self.auth_code_input.setPlaceholderText('Paste the authorization code from redirect URL here')
        oauth_form.addRow('Client ID', self.client_id_input)
        oauth_form.addRow('Authorize URL', self.auth_url_input)
        oauth_form.addRow('Token URL', self.token_url_input)
        oauth_form.addRow('Redirect URI', self.redirect_uri_input)
        oauth_form.addRow('Scope', self.scope_input)
        oauth_form.addRow('Authorization code', self.auth_code_input)

        oauth_buttons = QtWidgets.QHBoxLayout()
        self.open_browser_btn = QtWidgets.QPushButton('Open OAuth Login')
        self.open_browser_btn.clicked.connect(self.open_oauth_login)
        oauth_buttons.addWidget(self.open_browser_btn)
        tabs.addTab(oauth_tab, 'OAuth')
        oauth_form.addRow('', oauth_buttons)

        self.status_label = QtWidgets.QLabel('')
        layout.addWidget(self.status_label)

        buttons = QtWidgets.QDialogButtonBox()
        self.connect_btn = buttons.addButton('Connect', QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        cancel_btn = buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.connect_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(buttons)

        self.tabs = tabs

    def open_oauth_login(self) -> None:
        self.code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(48)).decode().rstrip('=')
        challenge = base64.urlsafe_b64encode(hashlib.sha256(self.code_verifier.encode()).digest()).decode().rstrip('=')
        params = {
            'response_type': 'code',
            'client_id': self.client_id_input.text().strip(),
            'redirect_uri': self.redirect_uri_input.text().strip(),
            'scope': self.scope_input.text().strip(),
            'code_challenge': challenge,
            'code_challenge_method': 'S256',
            'state': secrets.token_urlsafe(16),
        }
        url = f"{self.auth_url_input.text().strip()}?{urllib.parse.urlencode(params)}"
        webbrowser.open(url)
        self.status_label.setText('Browser opened. After login, paste the returned authorization code and click Connect.')

    def auth_payload(self) -> dict:
        return {
            'mode': 'api_key' if self.tabs.currentIndex() == 0 else 'oauth',
            'api_key': self.api_key_input.text().strip(),
            'client_id': self.client_id_input.text().strip(),
            'token_url': self.token_url_input.text().strip(),
            'redirect_uri': self.redirect_uri_input.text().strip(),
            'auth_code': self.auth_code_input.text().strip(),
            'code_verifier': self.code_verifier,
        }


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.project = ProjectState()
        self.ai_client = OpenAIClient()
        self.composer = OpenAIComposer(self.ai_client)
        self.instrument_ai = InstrumentIntelligence(self.ai_client)
        self.renderer = AISynthRenderer()
        self.vsti_binary_loader = VSTBinaryLoader()
        self.vsti_plugin_metadata: dict[str, list[str]] = {}
        self._load_preferences()
        self.audio_output = QtMultimedia.QAudioOutput(self)
        self.media_player = QtMultimedia.QMediaPlayer(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.mediaStatusChanged.connect(self._on_media_status_changed)
        self.selected_audio_output_id = ""
        self.playback_mix_path = Path.cwd() / "renders" / "_playback_mix.wav"
        self.carla_bridge_state_path = Path.cwd() / "renders" / "carla_bridge_state.json"
        self._playback_loop_ms = 0
        self.setWindowTitle("AI Music Studio")
        self.resize(1500, 900)

        self.track_list = QtWidgets.QListWidget()
        self._selected_track_index = 0
        self.track_list.currentRowChanged.connect(self._track_changed)
        self.track_list.viewport().installEventFilter(self)
        self.last_added_track_type = 'instrument'

        self.timeline = TimelineWidget(self.project)
        self.piano_roll = PianoRollWidget(self.project, self.current_track_index, self.set_playhead_position, self.set_left_locator_position, self.set_right_locator_position)
        self.piano_roll.noteChanged.connect(self.on_notes_changed)
        self.velocity_editor = VelocityEditorWidget(self.project, self.current_track_index)
        self.velocity_editor.velocityChanged.connect(self.on_notes_changed)

        self.mixer = MixerWidget(self.project, self.current_track)
        self.instruments = InstrumentFxWidget(self.project, self.current_track, self.refresh_vsti_rack_ui, self.on_track_instrument_changed, self.load_vsti_binary_by_name, self.open_vsti_gui_by_name, self.vsti_parameter_names_for_rack)
        self.sample_timeline = SampleTimelineWidget(self.project, self.sample_track_indices, self.place_sample_asset_on_track, self.set_playhead_position)
        self.arrangement_overview = ArrangementOverviewWidget(self.project, self.set_playhead_position, self.set_left_locator_position, self.set_right_locator_position, self.apply_arrangement_section_move, lambda: self.project.bpm)
        self.sample_library = SampleLibraryWidget()

        self.quantize_box = QtWidgets.QComboBox()
        quantize_values = [
            "1/1", "1/2", "1/2T", "1/4", "1/4T", "1/8", "1/8T", "1/16", "1/16T", "1/32", "1/32T", "1/64", "1/64T"
        ]
        self.quantize_box.addItems(quantize_values)
        self.quantize_box.setCurrentText("1/16")
        self.quantize_box.currentTextChanged.connect(self.on_quantize_changed)
        self.quantize_snap_btn = QtWidgets.QPushButton("Snap")
        self.quantize_snap_btn.clicked.connect(self.piano_roll.quantize_selected)

        add_track_btn = QtWidgets.QPushButton("+ Track (Sample/Instrument)")
        add_track_btn.clicked.connect(self.add_track)

        import_btn = QtWidgets.QPushButton("Import MIDI + AI Instrument Render")
        import_btn.clicked.connect(self.import_midi)
        render_btn = QtWidgets.QPushButton("Render AI Audio Stems")
        render_btn.clicked.connect(self.render_all_tracks)
        import_sample_btn = QtWidgets.QPushButton("Import Sample (WAV/MP3)")
        import_sample_btn.clicked.connect(self.import_sample)
        place_sample_btn = QtWidgets.QPushButton("Place Selected Sample On Timeline")
        place_sample_btn.clicked.connect(self.place_selected_sample)
        ai_btn = QtWidgets.QPushButton("AI Compose (OpenAI Codex)")
        ai_btn.clicked.connect(self.compose_with_ai)

        
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.addWidget(QtWidgets.QLabel("Tracks"))
        left_layout.addWidget(self.track_list)
        left_layout.addWidget(add_track_btn)
        left_layout.addWidget(QtWidgets.QLabel("Quantize"))
        quantize_row = QtWidgets.QHBoxLayout()
        quantize_row.addWidget(self.quantize_box)
        quantize_row.addWidget(self.quantize_snap_btn)
        left_layout.addLayout(quantize_row)
        left_layout.addWidget(import_btn)
        left_layout.addWidget(render_btn)
        left_layout.addWidget(QtWidgets.QLabel("Samples Toolbox"))
        left_layout.addWidget(self.sample_library)
        left_layout.addWidget(import_sample_btn)
        left_layout.addWidget(place_sample_btn)
        left_layout.addWidget(ai_btn)
        left_layout.addStretch()

        right_tabs = QtWidgets.QTabWidget()
        right_tabs.addTab(self.timeline, "Timeline")
        right_tabs.addTab(self.arrangement_overview, "Arrangement Overview")
        right_tabs.addTab(self.sample_timeline, "Sample Timeline")
        right_tabs.addTab(self.mixer, "Mixer")
        right_tabs.addTab(self.instruments, "Instruments / FX")

        note_editor = QtWidgets.QWidget()
        note_editor_layout = QtWidgets.QVBoxLayout(note_editor)
        note_editor_layout.setContentsMargins(0, 0, 0, 0)
        note_editor_layout.setSpacing(4)
        note_editor_layout.addWidget(self.piano_roll)
        note_editor_layout.addWidget(QtWidgets.QLabel('Velocity Editor'))
        note_editor_layout.addWidget(self.velocity_editor)

        splitter_vertical = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter_vertical.addWidget(note_editor)
        splitter_vertical.addWidget(right_tabs)
        splitter_vertical.setSizes([600, 320])

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
        self.scan_sample_paths()

    def _setup_menus(self) -> None:
        file_menu = self.menuBar().addMenu('File')
        import_midi_action = QtGui.QAction('Import MIDI + AI Instrument Render', self)
        import_midi_action.triggered.connect(self.import_midi)
        export_midi_action = QtGui.QAction('Export MIDI', self)
        export_midi_action.triggered.connect(self.export_midi)
        export_audio_action = QtGui.QAction('Export Sample Timeline Audio (WAV/MP3)', self)
        export_audio_action.triggered.connect(self.export_sample_timeline_audio)
        file_menu.addAction(import_midi_action)
        file_menu.addSeparator()
        file_menu.addAction(export_midi_action)
        file_menu.addAction(export_audio_action)

        settings = self.menuBar().addMenu('Settings')

        instruments_menu = settings.addMenu('Instruments')
        add_vsti = QtGui.QAction('Add VSTI Path', self)
        add_vsti.triggered.connect(self.add_vsti_path)
        instruments_menu.addAction(add_vsti)
        add_vsti_folder = QtGui.QAction('Add VSTI Folder', self)
        add_vsti_folder.triggered.connect(self.add_vsti_folder)
        instruments_menu.addAction(add_vsti_folder)
        add_vsti_to_rack = QtGui.QAction('Add Discovered VSTI To Rack', self)
        add_vsti_to_rack.triggered.connect(self.add_discovered_vsti_to_rack)
        instruments_menu.addAction(add_vsti_to_rack)
        instruments_menu.addSeparator()
        set_carla_host = QtGui.QAction('Set Carla Host Binary…', self)
        set_carla_host.triggered.connect(self.set_carla_host_binary)
        verify_carla_host = QtGui.QAction('Verify Carla Host', self)
        verify_carla_host.triggered.connect(self.verify_carla_host)
        clear_carla_host = QtGui.QAction('Use PATH Carla Detection', self)
        clear_carla_host.triggered.connect(self.clear_carla_host_binary)
        instruments_menu.addAction(set_carla_host)
        instruments_menu.addAction(verify_carla_host)
        instruments_menu.addAction(clear_carla_host)
        instruments_menu.addSeparator()
        export_carla_session = QtGui.QAction('Export Carla Session Snapshot…', self)
        export_carla_session.triggered.connect(self.export_carla_session_snapshot)
        import_carla_session = QtGui.QAction('Import Carla Session Snapshot…', self)
        import_carla_session.triggered.connect(self.import_carla_session_snapshot)
        instruments_menu.addAction(export_carla_session)
        instruments_menu.addAction(import_carla_session)
        instruments_menu.addSeparator()
        self.carla_transport_bridge_action = QtGui.QAction('Enable Carla Transport Bridge', self)
        self.carla_transport_bridge_action.setCheckable(True)
        self.carla_transport_bridge_action.setChecked(True)
        self.carla_transport_bridge_action.toggled.connect(self.toggle_carla_transport_bridge)
        instruments_menu.addAction(self.carla_transport_bridge_action)
        self.vsti_menu = instruments_menu

        tracks_menu = settings.addMenu('Tracks')
        assign_track_instrument = QtGui.QAction('Assign Instrument To Selected Track', self)
        assign_track_instrument.triggered.connect(self.assign_instrument_to_selected_track)
        tracks_menu.addAction(assign_track_instrument)

        openai_menu = settings.addMenu('OpenAI')
        connect_openai = QtGui.QAction('Connect', self)
        connect_openai.triggered.connect(self.connect_openai)
        disconnect_openai = QtGui.QAction('Disconnect', self)
        disconnect_openai.triggered.connect(self.disconnect_openai)
        codex_tracks = QtGui.QAction('Prompt Codex About Tracks', self)
        codex_tracks.triggered.connect(self.codex_track_assistant)
        openai_menu.addAction(connect_openai)
        openai_menu.addAction(disconnect_openai)
        openai_menu.addSeparator()
        openai_menu.addAction(codex_tracks)
        self.openai_status_action = QtGui.QAction(self.ai_client.auth_status(), self)
        self.openai_status_action.setEnabled(False)
        openai_menu.addAction(self.openai_status_action)

        samples_menu = settings.addMenu('Sample Paths')
        add_sample_path = QtGui.QAction('Add Sample Folder', self)
        add_sample_path.triggered.connect(self.add_sample_path)
        scan_sample_paths = QtGui.QAction('Scan Sample Folders', self)
        scan_sample_paths.triggered.connect(self.scan_sample_paths)
        samples_menu.addAction(add_sample_path)
        samples_menu.addAction(scan_sample_paths)

        self.audio_output_menu = settings.addMenu('Audio Output')
        self.refresh_audio_output_menu()

        self.refresh_vsti_rack_ui()
        self.refresh_openai_status()

    def _setup_floating_transport(self) -> None:
        self.playback_timer = QtCore.QTimer(self)
        self.playback_timer.setInterval(33)
        self.playback_timer.timeout.connect(self._tick_playback)
        self._playback_started_at = 0.0
        self._playback_origin_sec = 0.0
        self._playback_rate = 1.0

        transport = QtWidgets.QToolBar('Transport', self)
        transport.setFloatable(True)
        transport.setMovable(True)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, transport)
        play_action = transport.addAction('Play')
        stop_action = transport.addAction('Stop')
        play_action.triggered.connect(self.start_playback)
        stop_action.triggered.connect(self.stop_playback)
        transport.addSeparator()
        self.playhead_spin = QtWidgets.QDoubleSpinBox()
        self.playhead_spin.setRange(0.0, 3600.0)
        self.playhead_spin.setDecimals(2)
        self.playhead_spin.setSingleStep(0.1)
        self.playhead_spin.setValue(self.project.playhead_sec)
        self.playhead_spin.valueChanged.connect(self.set_playhead_position)
        transport.addWidget(QtWidgets.QLabel('Playhead'))
        transport.addWidget(self.playhead_spin)
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
        transport.addSeparator()
        self.tempo_spin = QtWidgets.QSpinBox()
        self.tempo_spin.setRange(20, 300)
        self.tempo_spin.setValue(self.project.bpm)
        self.tempo_spin.valueChanged.connect(self.update_tempo)
        transport.addWidget(QtWidgets.QLabel('Tempo'))
        transport.addWidget(self.tempo_spin)

    def on_quantize_changed(self, text: str) -> None:
        value = text.strip().upper()
        triplet = value.endswith('T')
        if triplet:
            value = value[:-1]
        try:
            div = int(value.split('/')[1])
        except Exception:
            div = 16
            triplet = False
        self.project.quantize_div = max(1, div)
        self.project.quantize_triplet = triplet
        self.piano_roll.quantize_selected()

    def set_left_locator_position(self, sec: float) -> None:
        self.project.left_locator_sec = min(max(0.0, float(sec)), self.project.right_locator_sec)
        self.left_locator.blockSignals(True)
        self.left_locator.setValue(self.project.left_locator_sec)
        self.left_locator.blockSignals(False)
        self.update_locators()

    def set_right_locator_position(self, sec: float) -> None:
        self.project.right_locator_sec = max(max(0.0, float(sec)), self.project.left_locator_sec)
        self.right_locator.blockSignals(True)
        self.right_locator.setValue(self.project.right_locator_sec)
        self.right_locator.blockSignals(False)
        self.update_locators()

    def update_locators(self) -> None:
        self.project.left_locator_sec = min(self.left_locator.value(), self.right_locator.value())
        self.project.right_locator_sec = max(self.left_locator.value(), self.right_locator.value())
        self.sample_timeline.refresh()
        self.arrangement_overview.refresh()
        self.piano_roll.refresh()
        self.velocity_editor.refresh()
        self.timeline.refresh()

    def set_playhead_position(self, sec: float) -> None:
        self.project.playhead_sec = max(0.0, float(sec))
        if hasattr(self, 'playhead_spin'):
            self.playhead_spin.blockSignals(True)
            self.playhead_spin.setValue(self.project.playhead_sec)
            self.playhead_spin.blockSignals(False)
        self.sample_timeline.refresh()
        self.arrangement_overview.refresh()
        self.piano_roll.refresh()
        self.velocity_editor.refresh()

    def start_playback(self) -> None:
        if not self._build_playback_mix(self.playback_mix_path):
            QtWidgets.QMessageBox.information(self, 'Nothing to play', 'No playable audio was found. Add notes or sample clips first.')
            return
        self._playback_rate = max(0.2, self.project.bpm / DEFAULT_BPM)
        self._playback_loop_ms = max(1, int((self.project.right_locator_sec - self.project.left_locator_sec) * 1000))
        self.media_player.setSource(QtCore.QUrl.fromLocalFile(str(self.playback_mix_path.resolve())))
        self.media_player.setPlaybackRate(self._playback_rate)
        seek_sec = max(0.0, self.project.playhead_sec - self.project.left_locator_sec)
        seek_ms = int(seek_sec * 1000) % self._playback_loop_ms
        self.media_player.setPosition(seek_ms)
        self.media_player.play()
        self._playback_started_at = time.time()
        self._playback_origin_sec = self.project.playhead_sec
        self.playback_timer.start()
        self._apply_carla_parameter_bridge()
        self._write_carla_bridge_state()
        self.statusBar().showMessage(f'Playback started at {self.project.bpm} BPM ({self._playback_rate:.2f}x)')

    def stop_playback(self) -> None:
        should_reset = hasattr(self, 'playback_timer') and (not self.playback_timer.isActive())
        if hasattr(self, 'playback_timer'):
            self.playback_timer.stop()
        self.media_player.stop()
        self._write_carla_bridge_state()
        if should_reset:
            self.set_playhead_position(0.0)
            self.statusBar().showMessage('Playback reset to 0.00s')
            return
        self.statusBar().showMessage('Playback stopped')

    def _tick_playback(self) -> None:
        if self.media_player.playbackState() == QtMultimedia.QMediaPlayer.PlaybackState.PlayingState and self._playback_loop_ms > 0:
            left = self.project.left_locator_sec
            loop_pos_ms = self.media_player.position() % self._playback_loop_ms
            new_pos = left + (loop_pos_ms / 1000.0)
        else:
            elapsed = time.time() - self._playback_started_at
            new_pos = self._playback_origin_sec + (elapsed * self._playback_rate)
        if new_pos > self.project.right_locator_sec:
            new_pos = self.project.left_locator_sec
            self._playback_started_at = time.time()
            self._playback_origin_sec = new_pos
            if self.media_player.playbackState() == QtMultimedia.QMediaPlayer.PlaybackState.PlayingState:
                self.media_player.setPosition(0)
                self.media_player.play()
        self.set_playhead_position(new_pos)
        self._apply_carla_parameter_bridge()
        self._write_carla_bridge_state()

    def toggle_carla_transport_bridge(self, enabled: bool) -> None:
        state = 'enabled' if enabled else 'disabled'
        self.statusBar().showMessage(f'Carla transport bridge {state}')
        self._write_carla_bridge_state()

    def _apply_carla_parameter_bridge(self) -> None:
        if not hasattr(self, 'carla_transport_bridge_action') or not self.carla_transport_bridge_action.isChecked():
            return
        for track in self.project.tracks:
            if track.instrument_mode != 'VSTI Rack' or not track.rack_vsti or not track.carla_automation_enabled:
                continue
            track.vsti_parameters['Param 1'] = float(max(0.0, min(100.0, track.volume * 100.0)))
            track.vsti_parameters['Param 2'] = float(max(0.0, min(100.0, (track.pan + 1.0) * 50.0)))

    def _write_carla_bridge_state(self) -> None:
        enabled = hasattr(self, 'carla_transport_bridge_action') and self.carla_transport_bridge_action.isChecked()
        payload = {
            'enabled': bool(enabled),
            'playing': bool(self.playback_timer.isActive()) if hasattr(self, 'playback_timer') else False,
            'bpm': int(self.project.bpm),
            'playhead_sec': float(self.project.playhead_sec),
            'left_locator_sec': float(self.project.left_locator_sec),
            'right_locator_sec': float(self.project.right_locator_sec),
            'timestamp': time.time(),
            'tracks': [],
        }
        for idx, track in enumerate(self.project.tracks):
            if track.instrument_mode != 'VSTI Rack' or not track.rack_vsti:
                continue
            payload['tracks'].append({
                'index': idx,
                'name': track.name,
                'rack_vsti': track.rack_vsti,
                'vsti_state_path': track.vsti_state_path,
                'automation_enabled': track.carla_automation_enabled,
                'mapped_params': {
                    'Param 1': float(track.vsti_parameters.get('Param 1', 50.0)),
                    'Param 2': float(track.vsti_parameters.get('Param 2', 50.0)),
                },
            })
        try:
            self.carla_bridge_state_path.parent.mkdir(parents=True, exist_ok=True)
            self.carla_bridge_state_path.write_text(json.dumps(payload, indent=2))
        except Exception:
            pass

    def _on_media_status_changed(self, status: QtMultimedia.QMediaPlayer.MediaStatus) -> None:
        if status == QtMultimedia.QMediaPlayer.MediaStatus.EndOfMedia and self.playback_timer.isActive():
            self.media_player.setPosition(0)
            self.media_player.play()

    def update_tempo(self, bpm: int) -> None:
        self.project.bpm = int(bpm)
        self.statusBar().showMessage(f'Tempo set to {self.project.bpm} BPM')

    def refresh_audio_output_menu(self) -> None:
        if not hasattr(self, 'audio_output_menu'):
            return
        self.audio_output_menu.clear()
        group = QtGui.QActionGroup(self.audio_output_menu)
        group.setExclusive(True)

        default_action = self.audio_output_menu.addAction('System Default Soundcard')
        default_action.setCheckable(True)
        default_action.setChecked(not self.selected_audio_output_id)
        default_action.triggered.connect(lambda: self.set_audio_output_device(''))
        group.addAction(default_action)

        self.audio_output_menu.addSeparator()
        for device in QtMultimedia.QMediaDevices.audioOutputs():
            action = self.audio_output_menu.addAction(device.description())
            action.setCheckable(True)
            device_id = bytes(device.id()).hex()
            action.setChecked(self.selected_audio_output_id == device_id)
            action.triggered.connect(lambda _checked=False, d=device: self.set_audio_output_device(bytes(d.id()).hex()))
            group.addAction(action)

    def set_audio_output_device(self, device_id: str) -> None:
        self.selected_audio_output_id = device_id
        if not device_id:
            self.audio_output.setDevice(QtMultimedia.QMediaDevices.defaultAudioOutput())
            self.statusBar().showMessage('Audio output set to system default soundcard')
            self.refresh_audio_output_menu()
            return

        for device in QtMultimedia.QMediaDevices.audioOutputs():
            if bytes(device.id()).hex() == device_id:
                self.audio_output.setDevice(device)
                self.statusBar().showMessage(f'Audio output set to {device.description()}')
                self.refresh_audio_output_menu()
                return

        self.selected_audio_output_id = ''
        self.audio_output.setDevice(QtMultimedia.QMediaDevices.defaultAudioOutput())
        self.refresh_audio_output_menu()

    def _rack_vsti_path(self, rack_name: str) -> str:
        for vst in self.project.vsti_rack:
            if vst.name == rack_name:
                return vst.path
        return ''

    def vsti_parameter_names_for_rack(self, rack_name: str) -> list[str]:
        plugin_path = self._rack_vsti_path(rack_name)
        if not plugin_path:
            return []
        return list(self.vsti_plugin_metadata.get(plugin_path, []))

    def _capture_vsti_metadata(self, plugin_path: str) -> None:
        if not PEDALBOARD_AVAILABLE or plugin_path in self.vsti_plugin_metadata:
            return
        try:
            plugin = load_plugin(plugin_path)
            names = [str(name) for name in plugin.parameters.keys()]
            self.vsti_plugin_metadata[plugin_path] = names
        except Exception:
            self.vsti_plugin_metadata[plugin_path] = []

    def _process_track_with_vsti(self, track: TrackState, data: list[float], sample_rate: int) -> list[float]:
        if not PEDALBOARD_AVAILABLE:
            return data
        if track.instrument_mode != 'VSTI Rack' or not track.rack_vsti:
            return data

        plugin_path = self._rack_vsti_path(track.rack_vsti)
        if not plugin_path or not Path(plugin_path).exists():
            return data

        try:
            plugin = load_plugin(plugin_path)
            param_names = [str(name) for name in plugin.parameters.keys()]
            self.vsti_plugin_metadata[plugin_path] = param_names
            for idx, param in enumerate(plugin.parameters.values()):
                key = param_names[idx] if idx < len(param_names) else f'Param {idx + 1}'
                param_value = track.vsti_parameters.get(key, track.vsti_parameters.get(f'Param {idx + 1}', 50.0))
                try:
                    param.raw_value = max(0.0, min(1.0, float(param_value) / 100.0))
                except Exception:
                    pass

            dry_audio = np.asarray(data, dtype=np.float32)[None, :]
            block = 1024
            chunks = []
            for start in range(0, dry_audio.shape[1], block):
                piece = dry_audio[:, start : start + block]
                processed = plugin.process(piece, sample_rate, reset=(start == 0))
                chunks.append(processed)
            if not chunks:
                return data
            wet_audio = np.concatenate(chunks, axis=1)
            wet_mix = max(0.0, min(1.0, float(track.vsti_wet_mix) / 100.0))
            merged = (wet_audio * wet_mix) + (dry_audio * (1.0 - wet_mix))
            gain_linear = 10.0 ** (float(track.vsti_output_gain_db) / 20.0)
            merged = merged * gain_linear
            return np.clip(merged[0], -1.0, 1.0).astype(np.float32).tolist()
        except Exception as exc:
            self.statusBar().showMessage(f'VST process fallback to synth for {track.name}: {exc}')
            return data

    def _build_playback_mix(self, out_path: Path) -> bool:
        left = self.project.left_locator_sec
        right = self.project.right_locator_sec
        if right <= left:
            return False

        sample_rate = 44100
        mix = [0.0] * max(1, int((right - left) * sample_rate))
        has_audio = False

        solo_tracks = {idx for idx, t in enumerate(self.project.tracks) if t.solo}
        for idx, track in enumerate(self.project.tracks):
            if track.track_type != 'instrument' or not track.notes:
                continue
            if solo_tracks and idx not in solo_tracks:
                continue
            if track.mute:
                continue

            if track.instrument_mode == 'VSTI Rack' and track.rendered_audio_path and Path(track.rendered_audio_path).exists():
                data, sr = load_wav_samples(Path(track.rendered_audio_path))
            else:
                stem_path = Path.cwd() / 'renders' / f'_play_track_{idx}.wav'
                self.renderer.render_track(track, self.project.bpm, stem_path)
                data, sr = load_wav_samples(stem_path)
                data = self._process_track_with_vsti(track, data, sr)
            if sr != sample_rate:
                ratio = sr / sample_rate
                data = [data[min(len(data) - 1, int(i * ratio))] for i in range(max(1, int(len(data) / ratio)))]
            for i, value in enumerate(data):
                if i >= len(mix):
                    break
                mix[i] += value
            has_audio = True

        for clip in self.project.sample_clips:
            wav_path = Path(clip.path)
            if wav_path.suffix.lower() == '.mp3':
                converted = Path.cwd() / 'renders' / f'{wav_path.stem}_play.wav'
                convert_audio(wav_path, converted)
                wav_path = converted
            data, sr = load_wav_samples(wav_path)
            if sr != sample_rate:
                ratio = sr / sample_rate
                data = [data[min(len(data) - 1, int(i * ratio))] for i in range(max(1, int(len(data) / ratio)))]

            clip_start = clip.start_sec
            clip_end = clip.start_sec + (len(data) / sample_rate)
            if clip_end <= left or clip_start >= right:
                continue
            overlap_start = max(left, clip_start)
            src_start = int((overlap_start - clip_start) * sample_rate)
            dst_start = int((overlap_start - left) * sample_rate)
            count = min(len(data) - src_start, len(mix) - dst_start)
            for i in range(max(0, count)):
                mix[dst_start + i] += data[src_start + i] * 0.7
            has_audio = True

        if not has_audio:
            return False

        write_wav_samples(out_path, [clamp(v, -1.0, 1.0) for v in mix], sample_rate)
        return True


    def apply_arrangement_section_move(self, section_index: int, old_start_sec: float, new_start_sec: float, old_track_index: int, new_track_index: int) -> None:
        if not (0 <= old_track_index < len(self.project.tracks)):
            return
        if not (0 <= new_track_index < len(self.project.tracks)):
            new_track_index = old_track_index

        old_track = self.project.tracks[old_track_index]
        if old_track.track_type != 'instrument':
            return

        new_track = self.project.tracks[new_track_index]
        if new_track.track_type != 'instrument':
            new_track_index = old_track_index
            new_track = old_track

        sec_per_tick = 60.0 / max(1, self.project.bpm) / TICKS_PER_BEAT
        delta_tick = int(round((new_start_sec - old_start_sec) / max(1e-9, sec_per_tick)))

        moved_notes = list(old_track.notes)
        if delta_tick != 0:
            for note in moved_notes:
                note.start_tick = max(0, note.start_tick + delta_tick)

        if new_track_index != old_track_index:
            new_track.notes.extend(moved_notes)
            old_track.notes = []

        self.on_notes_changed()

    def _load_preferences(self) -> None:
        if not APP_PREFS_PATH.exists():
            return
        try:
            payload = json.loads(APP_PREFS_PATH.read_text())
        except Exception:
            return

        self.project.vsti_paths = [p for p in payload.get('vsti_paths', []) if isinstance(p, str)]
        carla_path = payload.get('carla_host_path', '')
        self.project.carla_host_path = carla_path if isinstance(carla_path, str) else ''
        self.project.sample_paths = [p for p in payload.get('sample_paths', []) if isinstance(p, str)]
        rack_paths = [p for p in payload.get('vsti_rack_paths', []) if isinstance(p, str)]
        self.project.vsti_paths = [p for p in self.project.vsti_paths if Path(p).exists()]
        rack = []
        for path in rack_paths:
            if Path(path).exists():
                rack.append(VSTInstrument(name=Path(path).stem, path=path))
        self.project.vsti_rack = rack
        for vst in self.project.vsti_rack:
            self.vsti_binary_loader.load(vst.path)
            self._capture_vsti_metadata(vst.path)

    def _save_preferences(self) -> None:
        payload = {
            'vsti_paths': self.project.vsti_paths,
            'vsti_rack_paths': [v.path for v in self.project.vsti_rack],
            'sample_paths': self.project.sample_paths,
            'carla_host_path': self.project.carla_host_path,
        }
        APP_PREFS_PATH.write_text(json.dumps(payload, indent=2))

    def _discover_vstis_in_folder(self, folder: Path) -> list[Path]:
        discovered: list[Path] = []
        for path in folder.rglob('*'):
            name = path.name.lower()
            if path.is_file() and path.suffix.lower() in {'.dll', '.so', '.vst3'}:
                discovered.append(path)
            elif path.is_dir() and name.endswith('.vst3'):
                discovered.append(path)
        unique: list[Path] = []
        seen: set[str] = set()
        for path in discovered:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        return unique

    def add_vsti_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose VST folder', str(Path.cwd()))
        if not folder:
            return
        found = self._discover_vstis_in_folder(Path(folder))
        added = 0
        for plugin in found:
            pstr = str(plugin)
            if pstr in self.project.vsti_paths:
                continue
            self.project.vsti_paths.append(pstr)
            added += 1
        self._save_preferences()
        self.statusBar().showMessage(f'Discovered {added} VSTI plugin(s) from folder. Add desired plugins to rack from Settings > Instruments > Add Discovered VSTI To Rack.')

    def add_sample_path(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose sample folder', str(Path.cwd()))
        if not folder:
            return
        if folder not in self.project.sample_paths:
            self.project.sample_paths.append(folder)
            self._save_preferences()
        self.statusBar().showMessage(f'Added sample folder: {Path(folder).name}')

    def scan_sample_paths(self) -> None:
        seen = {Path(a.path).resolve() for a in self.project.sample_assets if Path(a.path).exists()}
        added = 0
        for root in self.project.sample_paths:
            root_path = Path(root)
            if not root_path.exists():
                continue
            for ext in ('*.wav', '*.mp3'):
                for file in root_path.rglob(ext):
                    resolved = file.resolve()
                    if resolved in seen:
                        continue
                    try:
                        src = file
                        sample_wav = src
                        if src.suffix.lower() == '.mp3':
                            converted = Path.cwd() / 'renders' / f'{src.stem}_import.wav'
                            convert_audio(src, converted)
                            sample_wav = converted
                        preview, sample_rate, duration = load_wav_preview(sample_wav)
                        self.project.sample_assets.append(SampleAsset(path=str(sample_wav), duration_sec=duration, sample_rate=sample_rate, waveform_preview=preview))
                        seen.add(resolved)
                        added += 1
                    except Exception:
                        continue
        self.refresh_sample_library()
        self.statusBar().showMessage(f'Scanned sample folders. Added {added} sample(s).')

    def _load_vsti_binary_path(self, path: str, show_message: bool = True) -> bool:
        ok, detail = self.vsti_binary_loader.load(path)
        if ok:
            self._capture_vsti_metadata(path)
        if show_message:
            name = Path(path).name
            if ok:
                mode = 'native wrapper + pedalboard' if PEDALBOARD_AVAILABLE else 'binary-only (install pedalboard for audio processing)'
                self.statusBar().showMessage(f'Loaded VSTI binary: {name} ({mode})')
            else:
                QtWidgets.QMessageBox.warning(self, 'VSTI load failed', f'Could not load {name}\n\n{detail}')
        return ok

    def load_vsti_binary_by_name(self, vsti_name: str) -> None:
        for vst in self.project.vsti_rack:
            if vst.name == vsti_name:
                self._load_vsti_binary_path(vst.path, show_message=True)
                return
        QtWidgets.QMessageBox.information(self, 'VSTI not found', f'No rack VSTI named {vsti_name}.')

    def export_carla_session_snapshot(self) -> None:
        default_path = Path.cwd() / 'renders' / 'carla_session_snapshot.json'
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export Carla session snapshot', str(default_path), 'JSON (*.json)')
        if not path:
            return
        data = {
            'carla_host_path': self.project.carla_host_path,
            'tracks': [
                {
                    'name': t.name,
                    'instrument_mode': t.instrument_mode,
                    'rack_vsti': t.rack_vsti,
                    'vsti_state_path': t.vsti_state_path,
                    'vsti_parameters': t.vsti_parameters,
                    'carla_automation_enabled': t.carla_automation_enabled,
                }
                for t in self.project.tracks
            ],
        }
        try:
            Path(path).write_text(json.dumps(data, indent=2))
            self.statusBar().showMessage(f'Exported Carla snapshot: {Path(path).name}')
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, 'Export failed', str(exc))

    def import_carla_session_snapshot(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Import Carla session snapshot', str(Path.cwd()), 'JSON (*.json)')
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, 'Import failed', str(exc))
            return

        host = data.get('carla_host_path', '')
        if isinstance(host, str):
            self.project.carla_host_path = host

        tracks_data = data.get('tracks', [])
        if isinstance(tracks_data, list):
            for idx, payload in enumerate(tracks_data):
                if idx >= len(self.project.tracks) or not isinstance(payload, dict):
                    continue
                track = self.project.tracks[idx]
                track.instrument_mode = str(payload.get('instrument_mode', track.instrument_mode))
                track.rack_vsti = str(payload.get('rack_vsti', track.rack_vsti))
                track.vsti_state_path = str(payload.get('vsti_state_path', track.vsti_state_path))
                params = payload.get('vsti_parameters', {})
                if isinstance(params, dict):
                    normalized: dict[str, float] = {}
                    for k, v in params.items():
                        try:
                            normalized[str(k)] = float(v)
                        except Exception:
                            continue
                    if normalized:
                        track.vsti_parameters = normalized
                auto_enabled = payload.get('carla_automation_enabled', track.carla_automation_enabled)
                track.carla_automation_enabled = bool(auto_enabled)

        self._save_preferences()
        self.refresh_vsti_rack_ui()
        self.on_track_instrument_changed()
        self._write_carla_bridge_state()
        self.statusBar().showMessage(f'Imported Carla snapshot: {Path(path).name}')

    def _carla_single_binary(self) -> str:
        configured = self.project.carla_host_path.strip() if hasattr(self.project, 'carla_host_path') else ''
        if configured and Path(configured).exists():
            return configured
        return shutil.which('carla-single') or shutil.which('carla') or ''

    def set_carla_host_binary(self) -> None:
        current = self._carla_single_binary() or str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Carla host binary', current, 'Executable (*)')
        if not path:
            return
        chosen = Path(path)
        if not chosen.exists() or not chosen.is_file():
            QtWidgets.QMessageBox.warning(self, 'Invalid Carla host', 'Selected path is not a valid executable file.')
            return
        self.project.carla_host_path = str(chosen)
        self._save_preferences()
        self.statusBar().showMessage(f'Configured Carla host: {chosen.name}')
        self.refresh_vsti_rack_ui()

    def clear_carla_host_binary(self) -> None:
        self.project.carla_host_path = ''
        self._save_preferences()
        self.refresh_vsti_rack_ui()
        self.statusBar().showMessage('Carla host configuration cleared. Using PATH detection.')

    def verify_carla_host(self) -> None:
        host = self._carla_single_binary()
        if not host:
            QtWidgets.QMessageBox.information(self, 'Carla not found', 'Carla host was not found. Install carla-single/carla or set a custom binary path in Settings > Instruments > Set Carla Host Binary…')
            return
        source = 'custom path' if self.project.carla_host_path and Path(self.project.carla_host_path).exists() and Path(self.project.carla_host_path) == Path(host) else 'PATH discovery'
        version_line = ''
        try:
            result = subprocess.run([host, '--version'], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
            raw = (result.stdout or result.stderr).decode('utf-8', errors='ignore').strip()
            version_line = raw.splitlines()[0] if raw else ''
        except Exception:
            version_line = ''
        self.statusBar().showMessage(f'Carla host ready ({source}): {host}')
        details = f'Using Carla host: {host}\nSource: {source}'
        if version_line:
            details += f'\nVersion: {version_line}'
        QtWidgets.QMessageBox.information(self, 'Carla host ready', details)

    def open_vsti_gui_by_name(self, vsti_name: str) -> None:
        for vst in self.project.vsti_rack:
            if vst.name != vsti_name:
                continue

            track = self.current_track()
            self._capture_vsti_metadata(vst.path)
            param_names = self.vsti_parameter_names_for_rack(vsti_name)
            if not param_names:
                param_names = [f'Param {i}' for i in range(1, 9)]

            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle(f'Native VSTI Wrapper - {vst.name}')
            dialog.resize(560, 480)
            layout = QtWidgets.QVBoxLayout(dialog)
            info = QtWidgets.QLabel(
                'This is a built-in VSTI wrapper UI. Parameter changes are applied during playback/render via pedalboard.\n'
                "Use Carla only if you need the plugin's original vendor GUI window."
            )
            info.setWordWrap(True)
            layout.addWidget(info)

            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            body = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(body)
            sliders: dict[str, QtWidgets.QSlider] = {}

            for key in param_names[:24]:
                slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
                slider.setRange(0, 100)
                slider.setValue(int(track.vsti_parameters.get(key, 50)))
                value_label = QtWidgets.QLabel(f"{slider.value()}%")
                slider.valueChanged.connect(lambda v, lbl=value_label: lbl.setText(f"{v}%"))
                row = QtWidgets.QHBoxLayout()
                row.addWidget(slider)
                row.addWidget(value_label)
                row_widget = QtWidgets.QWidget()
                row_widget.setLayout(row)
                form.addRow(key, row_widget)
                sliders[key] = slider

            scroll.setWidget(body)
            layout.addWidget(scroll)

            buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)

            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return

            for key, slider in sliders.items():
                track.vsti_parameters[key] = float(slider.value())
            self.statusBar().showMessage(f'Updated VSTI wrapper controls: {vst.name}')
            self.on_track_instrument_changed()
            return
        QtWidgets.QMessageBox.information(self, 'VSTI not found', f'No rack VSTI named {vsti_name}.')

    def add_discovered_vsti_to_rack(self) -> None:
        available = [path for path in self.project.vsti_paths if Path(path).exists() and path not in {v.path for v in self.project.vsti_rack}]
        if not available:
            QtWidgets.QMessageBox.information(self, 'No discovered VSTI', 'No discovered VST instruments are available to add to the rack.')
            return

        labels = [Path(path).stem for path in available]
        selected, ok = QtWidgets.QInputDialog.getItem(self, 'Add VSTI To Rack', 'Choose instrument:', labels, 0, False)
        if not ok:
            return
        idx = labels.index(selected)
        chosen_path = available[idx]
        self.project.vsti_rack.append(VSTInstrument(name=Path(chosen_path).stem, path=chosen_path))
        self._load_vsti_binary_path(chosen_path, show_message=False)
        self._save_preferences()
        self.refresh_vsti_rack_ui()
        self.statusBar().showMessage(f'Added to rack: {Path(chosen_path).stem}')

    def add_vsti_path(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose VST instrument', str(Path.cwd()), 'VST Plugins (*.dll *.vst3 *.so);;All files (*)')
        if not path:
            return
        if path not in self.project.vsti_paths:
            self.project.vsti_paths.append(path)
        if path in {v.path for v in self.project.vsti_rack}:
            self.statusBar().showMessage(f'VSTI already loaded in rack: {Path(path).name}')
            return
        self.project.vsti_rack.append(VSTInstrument(name=Path(path).stem, path=path))
        self._load_vsti_binary_path(path, show_message=False)
        self._save_preferences()
        self.refresh_vsti_rack_ui()
        self.statusBar().showMessage(f'Added VSTI to rack: {Path(path).name}')

    def refresh_vsti_rack_ui(self) -> None:
        if hasattr(self, 'vsti_menu'):
            existing = [a for a in self.vsti_menu.actions() if a.property('rack_item')]
            for action in existing:
                self.vsti_menu.removeAction(action)
            host = self._carla_single_binary()
            host_flag = '✓' if host else '⚠'
            host_label = Path(host).name if host else 'Not found (configure or install carla-single)'
            host_action = QtGui.QAction(f'Carla host: {host_flag} {host_label}', self)
            host_action.setProperty('rack_item', True)
            host_action.setEnabled(False)
            separator = self.vsti_menu.addSeparator()
            separator.setProperty('rack_item', True)
            self.vsti_menu.addAction(host_action)
            if self.project.vsti_rack:
                for vst in self.project.vsti_rack:
                    loaded_flag = '✓' if self.vsti_binary_loader.is_loaded(vst.path) else '⚠'
                    action = QtGui.QAction(f'Rack: {loaded_flag} {vst.name}', self)
                    action.setProperty('rack_item', True)
                    action.setEnabled(False)
                    self.vsti_menu.addAction(action)
        self.instruments.reload_vsti_choices()
        self._populate_track_list()

    def refresh_openai_status(self) -> None:
        if hasattr(self, 'openai_status_action'):
            self.openai_status_action.setText(self.ai_client.auth_status())

    def on_track_instrument_changed(self) -> None:
        self._populate_track_list()
        self.timeline.refresh()

    def assign_instrument_to_selected_track(self) -> None:
        if not self.project.tracks:
            return
        row = self.track_list.currentRow()
        if row < 0:
            row = 0
        track = self.project.tracks[row]

        mode, ok = QtWidgets.QInputDialog.getItem(self, 'Track Instrument Mode', 'Mode:', ['AI Synth', 'General MIDI', 'VSTI Rack'], 0, False)
        if not ok:
            return
        track.instrument_mode = mode

        if mode == 'VSTI Rack':
            if not self.project.vsti_rack:
                QtWidgets.QMessageBox.information(self, 'No rack instruments', 'Load a VST into the rack first from Settings > Instruments.')
                return
            options = [v.name for v in self.project.vsti_rack]
            chosen, ok = QtWidgets.QInputDialog.getItem(self, 'Assign Rack Instrument', 'Rack instrument:', options, 0, False)
            if not ok:
                return
            track.rack_vsti = chosen
            track.instrument = chosen
        else:
            options = [self.instruments.instrument.itemText(i) for i in range(self.instruments.instrument.count())]
            chosen, ok = QtWidgets.QInputDialog.getItem(self, 'Assign Instrument', 'Instrument:', options, 0, False)
            if not ok:
                return
            track.rack_vsti = ''
            track.instrument = chosen

        self._populate_track_list()
        self.timeline.refresh()
        self.instruments.load_track()

    def connect_openai(self) -> None:
        dialog = OpenAIConnectDialog(self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        payload = dialog.auth_payload()
        try:
            if payload['mode'] == 'api_key':
                if not payload['api_key']:
                    raise RuntimeError('Please provide an API key.')
                self.ai_client.set_api_key(payload['api_key'])
            else:
                self._exchange_oauth_code(payload)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, 'OpenAI connection failed', str(exc))
            return
        self.refresh_openai_status()
        self.statusBar().showMessage('OpenAI connected successfully')

    def _exchange_oauth_code(self, payload: dict) -> None:
        if not payload['client_id'] or not payload['token_url'] or not payload['auth_code']:
            raise RuntimeError('OAuth requires client id, token URL, and authorization code.')
        if not payload['code_verifier']:
            raise RuntimeError('Click "Open OAuth Login" first so a PKCE code verifier is generated.')

        req_body = urllib.parse.urlencode(
            {
                'grant_type': 'authorization_code',
                'client_id': payload['client_id'],
                'code': payload['auth_code'],
                'redirect_uri': payload['redirect_uri'],
                'code_verifier': payload['code_verifier'],
            }
        ).encode('utf-8')
        request = urllib.request.Request(
            payload['token_url'],
            data=req_body,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                token_payload = json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='ignore')
            raise RuntimeError(f'OAuth token exchange failed: {exc.code} {detail}') from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f'OAuth token exchange network failure: {exc}') from exc

        access_token = token_payload.get('access_token', '')
        if not access_token:
            raise RuntimeError(f'No access_token in OAuth response: {token_payload}')
        self.ai_client.set_oauth_tokens(
            access_token=access_token,
            refresh_token=token_payload.get('refresh_token', ''),
            expires_in=int(token_payload.get('expires_in', 3600) or 3600),
        )

    def disconnect_openai(self) -> None:
        self.ai_client.clear_auth()
        self.refresh_openai_status()
        self.statusBar().showMessage('OpenAI disconnected')

    def codex_track_assistant(self) -> None:
        if not self.ai_client.is_enabled():
            QtWidgets.QMessageBox.information(self, 'OpenAI not connected', 'Connect OpenAI first via Settings > OpenAI > Connect.')
            return

        prompt, ok = QtWidgets.QInputDialog.getMultiLineText(
            self,
            'Codex Track Assistant',
            'Describe how Codex should modify existing tracks:',
            'Rename tracks, set mute/solo, and adjust instrument modes for arrangement cleanup.',
        )
        if not ok or not prompt.strip():
            return

        track_context = []
        for idx, track in enumerate(self.project.tracks, start=1):
            track_context.append(
                {
                    'index': idx,
                    'name': track.name,
                    'track_type': track.track_type,
                    'instrument': track.instrument,
                    'instrument_mode': track.instrument_mode,
                    'mute': track.mute,
                    'solo': track.solo,
                    'note_count': len(track.notes),
                }
            )

        system_instruction = (
            'You are a DAW assistant. Return strict JSON with schema '
            '{"actions":[{"track_index":int,"rename":str|null,"mute":bool|null,"solo":bool|null,'
            '"instrument_mode":str|null,"instrument":str|null}]}. Do not include markdown.'
        )
        user_instruction = f"User request: {prompt}\n\nTracks:\n{json.dumps(track_context)}"
        try:
            result = self.ai_client.run_json_prompt(system_instruction, user_instruction)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, 'Codex assistant failed', str(exc))
            return

        changed = 0
        for action in result.get('actions', []):
            if not isinstance(action, dict):
                continue
            idx = int(action.get('track_index', 0)) - 1
            if idx < 0 or idx >= len(self.project.tracks):
                continue
            track = self.project.tracks[idx]
            rename = action.get('rename')
            if isinstance(rename, str) and rename.strip():
                track.name = rename.strip()
                changed += 1
            if isinstance(action.get('mute'), bool):
                track.mute = action['mute']
                changed += 1
            if isinstance(action.get('solo'), bool):
                track.solo = action['solo']
                changed += 1
            mode = action.get('instrument_mode')
            if isinstance(mode, str) and mode in {'AI Synth', 'General MIDI', 'VSTI Rack', 'Sample'}:
                track.instrument_mode = mode
                changed += 1
            instrument = action.get('instrument')
            if isinstance(instrument, str) and instrument.strip():
                track.instrument = instrument.strip()
                changed += 1

        if changed:
            self._populate_track_list()
            self.timeline.refresh()
            self.mixer.load_track()
            self.instruments.load_track()
        self.statusBar().showMessage(f'Codex applied {changed} track updates')

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
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+D"), self, self.piano_roll.duplicate_selected_by_grid)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+G"), self, self.compose_with_ai)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Delete), self, self.piano_roll.delete_selected)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Space), self, lambda: self.stop_playback() if self.playback_timer.isActive() else self.start_playback())

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
        self.refresh_vsti_rack_ui()
        self.scan_sample_paths()
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

        left = self.project.left_locator_sec
        right = self.project.right_locator_sec
        if right <= left:
            QtWidgets.QMessageBox.warning(self, "Invalid locators", "Right locator must be greater than left locator for export.")
            return

        sample_rate = 44100
        loaded: list[tuple[SampleClip, list[float], int]] = []
        for clip in self.project.sample_clips:
            wav_path = Path(clip.path)
            if wav_path.suffix.lower() == ".mp3":
                converted = Path.cwd() / "renders" / f"{wav_path.stem}_mix.wav"
                convert_audio(wav_path, converted)
                wav_path = converted
            data, sr = load_wav_samples(wav_path)
            loaded.append((clip, data, sr))

        mix_length = int((right - left) * sample_rate)
        mix = [0.0] * max(1, mix_length)
        for clip, data, sr in loaded:
            if sr != sample_rate:
                ratio = sr / sample_rate
                resampled = []
                for i in range(int(len(data) / ratio)):
                    resampled.append(data[min(len(data) - 1, int(i * ratio))])
                data = resampled

            clip_start = clip.start_sec
            clip_end = clip.start_sec + (len(data) / sample_rate)
            if clip_end <= left or clip_start >= right:
                continue

            overlap_start = max(left, clip_start)
            overlap_end = min(right, clip_end)
            src_start = int((overlap_start - clip_start) * sample_rate)
            src_end = int((overlap_end - clip_start) * sample_rate)
            dst_offset = int((overlap_start - left) * sample_rate)

            for i, v in enumerate(data[src_start:src_end]):
                idx = dst_offset + i
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

    def rebuild_midi_sections(self) -> None:
        sections: list[MidiSection] = []
        for i, track in enumerate(self.project.tracks):
            if track.track_type != 'instrument' or not track.notes:
                continue
            start_tick = min(note.start_tick for note in track.notes)
            end_tick = max(note.start_tick + note.duration_tick for note in track.notes)
            sec_per_tick = 60.0 / max(1, self.project.bpm) / TICKS_PER_BEAT
            sections.append(
                MidiSection(
                    track_index=i,
                    start_sec=start_tick * sec_per_tick,
                    duration_sec=max(0.1, (end_tick - start_tick) * sec_per_tick),
                    name=f"{track.name} Part",
                )
            )
        self.project.midi_sections = sections

    def insert_live_note(self, pitch: int) -> None:
        track = self.current_track()
        cursor_tick = max((n.start_tick + n.duration_tick for n in track.notes), default=0)
        track.notes.append(MidiNote(start_tick=cursor_tick, duration_tick=TICKS_PER_BEAT // 2, pitch=pitch))
        self.on_notes_changed()

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self.track_list.viewport() and event.type() == QtCore.QEvent.Type.MouseButtonDblClick:
            if isinstance(event, QtGui.QMouseEvent):
                item = self.track_list.itemAt(event.position().toPoint())
                if item is None:
                    self.add_track(preferred_type=self.last_added_track_type, ask=False)
                    return True
        return super().eventFilter(watched, event)

    def current_track_index(self) -> int:
        if not self.project.tracks:
            return 0
        return max(0, min(self._selected_track_index, len(self.project.tracks) - 1))

    def current_track(self) -> TrackState:
        return self.project.tracks[self.current_track_index()]

    def _populate_track_list(self) -> None:
        selected = self.current_track_index() if self.project.tracks else 0
        self.track_list.blockSignals(True)
        self.track_list.clear()
        for track in self.project.tracks:
            extra = f"VST:{track.rack_vsti}" if track.rack_vsti else track.instrument
            ch = f"Ch {track.midi_channel + 1}" if track.track_type == 'instrument' else 'Sample'
            self.track_list.addItem(f"{track.name} • {track.track_type} • {ch} • {track.instrument_mode} • {extra}")
        if self.project.tracks:
            safe_index = max(0, min(selected, len(self.project.tracks) - 1))
            self._selected_track_index = safe_index
            self.track_list.setCurrentRow(safe_index)
        self.track_list.blockSignals(False)
        if self.project.tracks:
            self._track_changed(self._selected_track_index)

    def _track_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.project.tracks):
            return
        self._selected_track_index = row
        track = self.project.tracks[row]
        self.piano_roll.setEnabled(track.track_type == 'instrument')
        self.velocity_editor.setEnabled(track.track_type == 'instrument')
        self.piano_roll.refresh()
        self.velocity_editor.refresh()
        self.mixer.load_track()
        self.instruments.load_track()

    def add_track(self, preferred_type: str | None = None, ask: bool = True) -> None:
        track_type = preferred_type or self.last_added_track_type or 'instrument'
        if ask:
            default_idx = 1 if track_type == 'sample' else 0
            chosen, ok = QtWidgets.QInputDialog.getItem(self, 'Add track', 'Track type:', ['instrument', 'sample'], default_idx, False)
            if not ok:
                return
            track_type = str(chosen)

        if track_type not in {'instrument', 'sample'}:
            track_type = 'instrument'

        idx = len(self.project.tracks) + 1
        used_channels = {t.midi_channel for t in self.project.tracks if t.track_type == 'instrument'}
        next_channel = next((ch for ch in range(16) if ch not in used_channels), idx % 16)
        state = TrackState(name=f"Track {idx}", track_type=track_type, midi_channel=next_channel)
        if track_type == 'sample':
            state.instrument = 'Sample Track'
            state.instrument_mode = 'Sample'

        self.last_added_track_type = track_type
        self.project.tracks.append(state)
        self._populate_track_list()
        self.track_list.setCurrentRow(idx - 1)
        self.timeline.refresh()
        self.sample_timeline.refresh()
        self.rebuild_midi_sections()
        self.arrangement_overview.refresh()

    def new_project(self) -> None:
        self.project = ProjectState()
        self._load_preferences()
        self.timeline.project = self.project
        self.piano_roll.project = self.project
        self.velocity_editor.project = self.project
        self.sample_timeline.project = self.project
        self.arrangement_overview.project = self.project
        if hasattr(self, 'tempo_spin'):
            self.tempo_spin.setValue(self.project.bpm)
        if hasattr(self, 'left_locator'):
            self.left_locator.setValue(self.project.left_locator_sec)
            self.right_locator.setValue(self.project.right_locator_sec)
        self.set_playhead_position(self.project.playhead_sec)
        self._populate_track_list()
        self.track_list.setCurrentRow(0)
        self.refresh_vsti_rack_ui()
        self.scan_sample_paths()
        self.on_notes_changed()

    def on_notes_changed(self) -> None:
        self.piano_roll.refresh()
        self.velocity_editor.refresh()
        self.timeline.refresh()
        self.sample_timeline.refresh()
        self.rebuild_midi_sections()
        self.arrangement_overview.refresh()

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
        self.refresh_vsti_rack_ui()
        self.scan_sample_paths()
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

        left_sec = self.project.left_locator_sec
        right_sec = self.project.right_locator_sec
        if right_sec <= left_sec:
            QtWidgets.QMessageBox.warning(self, "Invalid locators", "Right locator must be greater than left locator for export.")
            return

        sec_per_tick = 60.0 / max(1, self.project.bpm) / TICKS_PER_BEAT
        left_tick = int(left_sec / sec_per_tick)
        right_tick = int(right_sec / sec_per_tick)

        midi = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
        for track_state in self.project.tracks:
            mtrack = mido.MidiTrack()
            mtrack.name = track_state.name
            midi.tracks.append(mtrack)
            mtrack.append(mido.Message("program_change", channel=track_state.midi_channel, program=int(clamp(track_state.midi_program, 0, 127)), time=0))

            events: list[tuple[int, mido.Message]] = []
            for note in track_state.notes:
                note_start = note.start_tick
                note_end = note.start_tick + note.duration_tick
                if note_end <= left_tick or note_start >= right_tick:
                    continue

                clipped_start = max(left_tick, note_start)
                clipped_end = min(right_tick, note_end)
                start_rel = clipped_start - left_tick
                end_rel = clipped_end - left_tick

                events.append((start_rel, mido.Message("note_on", channel=track_state.midi_channel, note=note.pitch, velocity=note.velocity, time=0)))
                events.append((end_rel, mido.Message("note_off", channel=track_state.midi_channel, note=note.pitch, velocity=0, time=0)))

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
