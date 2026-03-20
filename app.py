from __future__ import annotations

import base64
import ctypes
import dataclasses
import hashlib
import io
import json
import math
import os
import secrets
import struct
import subprocess
import sys
import threading
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
except Exception:
    np = None

try:
    from pedalboard import Pedalboard, load_plugin
    PEDALBOARD_AVAILABLE = True
    PEDALBOARD_IMPORT_ERROR = ""
except Exception:
    Pedalboard = None
    load_plugin = None
    PEDALBOARD_AVAILABLE = False
    PEDALBOARD_IMPORT_ERROR = str(sys.exc_info()[1])

TICKS_PER_BEAT = 480
DEFAULT_BPM = 120
PITCH_MIN = 36
PITCH_MAX = 84
BLACK_KEY_PITCH_CLASSES = {1, 3, 6, 8, 10}
OPENAI_API_URL = "https://api.openai.com/v1/responses"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-codex")
APP_NAME = "AI Music Studio"
PROJECT_FILE_EXTENSION = ".aims"
PROJECT_FILE_FILTER = "AI Music Studio Project (*.aims);;JSON files (*.json);;All files (*)"
PROJECT_FILE_VERSION = 1
TRACK_COLOR_PALETTE = [
    "#4AB4FF",
    "#FF8A65",
    "#7ED957",
    "#F4D35E",
    "#C084FC",
    "#FF6F91",
    "#4FD1C5",
    "#F97316",
]


def application_root_directory() -> Path:
    if getattr(sys, "frozen", False):
        executable = Path(sys.executable).resolve()
        if sys.platform == "darwin":
            macos_dir = executable.parent
            contents_dir = macos_dir.parent
            if macos_dir.name == "MacOS" and contents_dir.name == "Contents":
                return contents_dir / "Resources"
        return executable.parent
    return Path(__file__).resolve().parent


def app_data_directory() -> Path:
    if sys.platform == "darwin":
        target = Path.home() / "Library" / "Application Support" / APP_NAME
    elif os.name == "nt":
        base = os.environ.get("APPDATA")
        target = Path(base) / APP_NAME if base else Path.home() / "AppData" / "Roaming" / APP_NAME
    else:
        base = os.environ.get("XDG_DATA_HOME")
        target = Path(base) / "ai-music-studio" if base else Path.home() / ".local" / "share" / "ai-music-studio"
    target.mkdir(parents=True, exist_ok=True)
    return target


def default_user_files_directory() -> Path:
    home = Path.home()
    documents = home / "Documents"
    desktop = home / "Desktop"
    for candidate in (documents, desktop, home):
        if candidate.exists():
            return candidate
    return home


APP_ROOT_DIR = application_root_directory()
APP_DATA_DIR = app_data_directory()
RENDER_DIR = APP_DATA_DIR / "renders"
APP_PREFS_PATH = APP_DATA_DIR / "preferences.json"
BUNDLED_VSTI_DIR = APP_ROOT_DIR / "vsti"
USER_VSTI_DIR = APP_DATA_DIR / "vsti"
DEFAULT_USER_FILES_DIR = default_user_files_directory()


def vst_host_candidate_paths(path: str | Path) -> list[str]:
    resolved = Path(path).expanduser().resolve()
    candidates: list[Path] = [resolved]

    # VST3 bundles are platform-specific directories. Add the likely loadable
    # module paths for each desktop OS before falling back to a recursive scan.
    if resolved.suffix.lower() == ".vst3" and resolved.is_dir():
        if os.name == "nt":
            windows_bundle_dir = resolved / "Contents" / "x86_64-win"
            expected_binary = windows_bundle_dir / f"{resolved.stem}.vst3"
            if expected_binary.exists():
                candidates.append(expected_binary)
            if windows_bundle_dir.exists():
                candidates.extend(sorted(windows_bundle_dir.glob("*.vst3")))
        elif sys.platform == "darwin":
            macos_bundle_dir = resolved / "Contents" / "MacOS"
            expected_binary = macos_bundle_dir / resolved.stem
            if expected_binary.exists():
                candidates.append(expected_binary)
            if macos_bundle_dir.exists():
                candidates.extend(sorted(child for child in macos_bundle_dir.iterdir() if child.is_file()))
        else:
            linux_bundle_dir = resolved / "Contents" / "x86_64-linux"
            expected_binary = linux_bundle_dir / f"{resolved.stem}.so"
            if expected_binary.exists():
                candidates.append(expected_binary)
            if linux_bundle_dir.exists():
                candidates.extend(sorted(child for child in linux_bundle_dir.iterdir() if child.is_file()))

        candidates.extend(sorted(child for child in resolved.rglob("*.vst3") if child != resolved))

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve())
        except Exception:
            key = str(candidate)
        if key in seen or not candidate.exists():
            continue
        seen.add(key)
        unique.append(key)
    return unique


def load_vst_plugin_with_fallback(path: str):
    if not PEDALBOARD_AVAILABLE or not load_plugin:
        raise RuntimeError(pedalboard_runtime_hint() or "Pedalboard is not available.")

    errors: list[str] = []
    for candidate in vst_host_candidate_paths(path):
        try:
            return load_plugin(candidate), candidate
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    if errors:
        raise RuntimeError("\n".join(errors))
    raise RuntimeError(f"Plugin path does not exist: {path}")


def load_startup_preferences() -> dict:
    if not APP_PREFS_PATH.exists():
        return {}
    try:
        payload = json.loads(APP_PREFS_PATH.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def pedalboard_runtime_hint() -> str:
    if PEDALBOARD_AVAILABLE:
        return ""
    detail = PEDALBOARD_IMPORT_ERROR or "missing pedalboard dependency"
    return f"VST3 hosting is unavailable in {sys.executable}: {detail}"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def tick_to_sample_frame(tick: int, sample_rate: int, bpm: int) -> int:
    tick_value = max(0, int(tick))
    rate = max(1, int(sample_rate))
    tempo = max(1, int(bpm))
    numerator = tick_value * rate * 60
    denominator = tempo * TICKS_PER_BEAT
    return (numerator + (denominator // 2)) // denominator


def sample_frame_to_seconds(frame: int, sample_rate: int) -> float:
    return max(0, int(frame)) / float(max(1, int(sample_rate)))


def seconds_to_sample_frame(sec: float, sample_rate: int) -> int:
    rate = max(1, int(sample_rate))
    value = max(0.0, float(sec)) * rate
    return max(0, int(math.floor(value + 0.5)))


def midi_to_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def _pcm16_to_mono(raw: bytes, channels: int) -> object:
    if np is not None:
        samples = np.frombuffer(raw, dtype="<i2")
        if samples.size == 0:
            return np.zeros(0, dtype=np.float32)
        if channels > 1:
            samples = samples.reshape(-1, channels)[:, 0]
        return samples.astype(np.float32) / 32768.0

    sample_count = max(1, len(raw) // 2)
    unpacked = struct.unpack(f"<{sample_count}h", raw)
    mono: list[float] = []
    for i in range(0, len(unpacked), channels):
        mono.append(unpacked[i] / 32768.0)
    return mono


def resample_samples(samples: object, source_rate: int, target_rate: int) -> object:
    if source_rate == target_rate:
        return samples

    if np is not None and isinstance(samples, np.ndarray):
        if samples.size == 0:
            return samples.copy()
        out_len = max(1, int(round(samples.shape[0] * target_rate / max(1, source_rate))))
        x_old = np.arange(samples.shape[0], dtype=np.float32)
        x_new = np.linspace(0, max(0, samples.shape[0] - 1), num=out_len, dtype=np.float32)
        return np.interp(x_new, x_old, samples).astype(np.float32)

    data = list(samples)
    if not data:
        return [0.0]
    ratio = source_rate / target_rate
    return [data[min(len(data) - 1, int(i * ratio))] for i in range(max(1, int(len(data) / ratio)))]


def load_wav_preview(path: Path, max_points: int = 800) -> tuple[list[float], int, float]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(frames)

    if sampwidth != 2:
        raise RuntimeError("Only 16-bit PCM WAV is supported for waveform preview.")

    mono = _pcm16_to_mono(raw, channels)

    if np is not None and isinstance(mono, np.ndarray):
        if mono.size == 0:
            return [0.0], sample_rate, 0.0
        bucket = max(1, mono.shape[0] // max_points)
        preview: list[float] = []
        for i in range(0, mono.shape[0], bucket):
            window = np.abs(mono[i : i + bucket])
            preview.append(float(window.max()) if window.size else 0.0)
        duration = mono.shape[0] / sample_rate
        return preview, sample_rate, duration

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


def load_wav_samples(path: Path) -> tuple[object, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(frames)

    if sampwidth != 2:
        raise RuntimeError("Only 16-bit PCM WAV is supported.")

    return _pcm16_to_mono(raw, channels), sample_rate


def write_wav_samples(path: Path, samples: object, sample_rate: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_wav_samples(samples, sample_rate))


def encode_wav_samples(samples: object, sample_rate: int = 44100) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        if np is not None and isinstance(samples, np.ndarray):
            clipped = np.clip(samples, -1.0, 1.0)
            frames = (clipped * 32767.0).astype("<i2", copy=False).tobytes()
            wf.writeframes(frames)
            return buffer.getvalue()
        frames = bytearray()
        for value in samples:
            clipped = int(clamp(value, -1.0, 1.0) * 32767)
            frames.extend(struct.pack("<h", clipped))
        wf.writeframes(frames)
    return buffer.getvalue()


def encode_pcm16_stereo_samples(samples: object) -> bytes:
    if np is not None:
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim == 1:
            left = audio
            right = audio
        elif audio.ndim == 2:
            if audio.shape[0] == 2:
                left = audio[0]
                right = audio[1] if audio.shape[0] > 1 else audio[0]
            elif audio.shape[1] == 2:
                left = audio[:, 0]
                right = audio[:, 1]
            else:
                mono = audio.mean(axis=0 if audio.shape[0] <= audio.shape[1] else 1).astype(np.float32, copy=False)
                left = mono
                right = mono
        else:
            mono = audio.reshape(-1).astype(np.float32, copy=False)
            left = mono
            right = mono
        interleaved = np.column_stack((np.clip(left, -1.0, 1.0), np.clip(right, -1.0, 1.0)))
        return (interleaved * 32767.0).astype("<i2", copy=False).tobytes()

    if samples and isinstance(samples, list) and isinstance(samples[0], (list, tuple)) and len(samples[0]) >= 2:
        frames = bytearray()
        for left, right in samples:
            frames.extend(struct.pack("<h", int(clamp(float(left), -1.0, 1.0) * 32767)))
            frames.extend(struct.pack("<h", int(clamp(float(right), -1.0, 1.0) * 32767)))
        return bytes(frames)

    frames = bytearray()
    for value in samples:
        clipped = int(clamp(float(value), -1.0, 1.0) * 32767)
        frames.extend(struct.pack("<h", clipped))
        frames.extend(struct.pack("<h", clipped))
    return bytes(frames)


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
    instrument: str = "Piano"
    instrument_mode: str = "General MIDI"
    rack_vsti: str = ""
    plugins: list[str] = dataclasses.field(default_factory=list)
    vsti_parameters: dict[str, float] = dataclasses.field(default_factory=dict)
    vsti_state_path: str = ""
    vsti_output_gain_db: float = 0.0
    vsti_wet_mix: float = 100.0
    vst_fx_chain: list[str] = dataclasses.field(default_factory=list)
    midi_program: int = 0
    midi_channel: int = 0
    synth_profile: str = "synth"
    rendered_audio_path: str = ""
    mute: bool = False
    solo: bool = False
    color_hex: str = ""


@dataclasses.dataclass
class VSTInstrument:
    name: str
    path: str
    plugin_name: str = ""
    is_instrument: bool = False
    is_effect: bool = False
    category: str = ""
    host_supported: bool = True
    host_error: str = ""


@dataclasses.dataclass
class RealtimeTrackPlaybackState:
    key: str = ""
    instrument_plugin: object | None = None
    fx_plugins: list[object] = dataclasses.field(default_factory=list)
    reset_pending: bool = True
    last_error: str = ""


def default_track_color(index: int) -> QtGui.QColor:
    return QtGui.QColor(TRACK_COLOR_PALETTE[index % len(TRACK_COLOR_PALETTE)])


def track_display_color(track: TrackState, index: int = 0) -> QtGui.QColor:
    color = QtGui.QColor(track.color_hex)
    if color.isValid():
        return color
    return default_track_color(index)


def track_text_color(color: QtGui.QColor) -> QtGui.QColor:
    luminance = (0.299 * color.red()) + (0.587 * color.green()) + (0.114 * color.blue())
    return QtGui.QColor(18, 18, 18) if luminance >= 170 else QtGui.QColor(242, 242, 242)


def qcolor_to_hex(color: QtGui.QColor) -> str:
    return color.name(QtGui.QColor.NameFormat.HexRgb) if color.isValid() else ""


class VSTBinaryLoader:
    def __init__(self) -> None:
        self._handles: dict[str, object] = {}
        self._errors: dict[str, str] = {}
        self._resolved_paths: dict[str, str] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _normalize_key(path: str) -> str:
        try:
            return str(Path(path).expanduser().resolve())
        except Exception:
            return str(Path(path))

    def is_loaded(self, path: str) -> bool:
        normalized = self._normalize_key(path)
        with self._lock:
            return normalized in self._handles

    def load(self, path: str) -> tuple[bool, str]:
        normalized = self._normalize_key(path)
        with self._lock:
            if normalized in self._handles:
                return True, 'Already loaded'
        suffix = Path(normalized).suffix.lower()
        if suffix != '.vst3':
            detail = 'Unsupported plugin format for this host. Use a VST3 plugin to get real VST playback.'
            with self._lock:
                self._errors[normalized] = detail
            return False, detail
        if not PEDALBOARD_AVAILABLE or not load_plugin:
            detail = pedalboard_runtime_hint() or 'Pedalboard is not available. Install pedalboard and restart the app to host VST3 plugins.'
            with self._lock:
                self._errors[normalized] = detail
            return False, detail
        app_instance = QtCore.QCoreApplication.instance()
        if app_instance is not None and QtCore.QThread.currentThread() != app_instance.thread():
            detail = 'Pedalboard VST3 plugins must be loaded on the main thread for reliable playback and editor support.'
            with self._lock:
                self._errors[normalized] = detail
            return False, detail
        try:
            handle, resolved_path = load_vst_plugin_with_fallback(normalized)
            with self._lock:
                self._handles[normalized] = handle
                self._resolved_paths[normalized] = resolved_path
                self._errors.pop(normalized, None)
            return True, 'Loaded via pedalboard'
        except Exception as exc:
            with self._lock:
                self._errors[normalized] = str(exc)
            return False, str(exc)

    def last_error(self, path: str) -> str:
        normalized = self._normalize_key(path)
        with self._lock:
            return self._errors.get(normalized, '')

    def handle(self, path: str):
        normalized = self._normalize_key(path)
        with self._lock:
            return self._handles.get(normalized)

    def resolved_path(self, path: str) -> str:
        normalized = self._normalize_key(path)
        with self._lock:
            return self._resolved_paths.get(normalized, '')


class VSTLoadWorkerSignals(QtCore.QObject):
    finished = QtCore.Signal(str, bool, str, list)


class VSTLoadWorker(QtCore.QRunnable):
    def __init__(self, loader: VSTBinaryLoader, plugin_path: str) -> None:
        super().__init__()
        self.loader = loader
        self.plugin_path = plugin_path
        self.signals = VSTLoadWorkerSignals()

    @QtCore.Slot()
    def run(self) -> None:
        ok, detail = self.loader.load(self.plugin_path)
        param_names: list[str] = []
        if ok:
            plugin = self.loader.handle(self.plugin_path)
            if plugin is not None:
                try:
                    param_names = [str(name) for name in plugin.parameters.keys()]
                except Exception:
                    param_names = []
        self.signals.finished.emit(self.plugin_path, ok, detail, param_names)


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
        default_bar_sec = 4.0 * (60.0 / max(1, DEFAULT_BPM))
        self.tracks: list[TrackState] = [TrackState(name="Track 1")]
        self.bpm = DEFAULT_BPM
        self.quantize_enabled = True
        self.quantize_div = 16
        self.quantize_triplet = False
        self.loop_enabled = True
        self.metronome_enabled = False
        self.vsti_paths: list[str] = []
        self.vsti_folder_paths: list[str] = []
        self.sample_paths: list[str] = []
        self.vsti_rack: list[VSTInstrument] = []
        self.sample_assets: list[SampleAsset] = []
        self.sample_clips: list[SampleClip] = []
        self.midi_sections: list[MidiSection] = []
        self.left_locator_sec = default_bar_sec
        self.right_locator_sec = default_bar_sec * 2.0
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

    def _adsr_curve(self, t: object, duration: float, a: float, d: float, s: float, r: float) -> object:
        if np is None or not isinstance(t, np.ndarray):
            return [self._adsr(float(value), duration, a, d, s, r) for value in t]

        env = np.zeros_like(t, dtype=np.float32)
        if duration <= 0:
            return env

        attack_end = max(a, 1e-6)
        decay_end = a + max(d, 1e-6)
        release_start = max(0.0, duration - r)

        attack_mask = t < a
        if attack_mask.any():
            env[attack_mask] = t[attack_mask] / attack_end

        decay_mask = (t >= a) & (t < decay_end)
        if decay_mask.any():
            env[decay_mask] = 1.0 - (1.0 - s) * ((t[decay_mask] - a) / max(d, 1e-6))

        sustain_mask = (t >= decay_end) & (t < release_start)
        if sustain_mask.any():
            env[sustain_mask] = s

        release_mask = (t >= release_start) & (t < duration)
        if release_mask.any():
            env[release_mask] = s * (1.0 - (t[release_mask] - release_start) / max(r, 1e-6))

        return env

    def _wave_curve(self, phase: object, profile: str) -> object:
        if np is None or not isinstance(phase, np.ndarray):
            return [self._wave(float(value), profile) for value in phase]

        if profile == "sub_bass":
            return np.sin(phase)
        if profile == "pluck":
            return 0.7 * np.sin(phase) + 0.3 * np.sin(2.0 * phase)
        if profile == "organ":
            return 0.6 * np.sin(phase) + 0.25 * np.sin(2.0 * phase) + 0.15 * np.sin(3.0 * phase)
        if profile == "saw_pad":
            frac = np.mod(phase / (2.0 * math.pi), 1.0)
            return 2.0 * frac - 1.0
        if profile == "brass_stack":
            return 0.5 * np.sin(phase) + 0.35 * np.sin(2.0 * phase) + 0.15 * np.sin(3.0 * phase)
        if profile == "reed_breath":
            return 0.8 * np.sin(phase) + 0.2 * np.sin(4.0 * phase)
        if profile == "noise_kit":
            return np.sin(phase * 13.0) * np.sin(phase * 7.0)
        if profile == "e_piano":
            return 0.8 * np.sin(phase) + 0.2 * np.sin(6.0 * phase)
        return np.sin(phase) + 0.2 * np.sin(2.0 * phase)

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

    def render_track_audio(self, track: TrackState, bpm: int) -> tuple[object, int]:
        max_frame = max((tick_to_sample_frame(n.start_tick + n.duration_tick, self.sample_rate, bpm) for n in track.notes), default=0)
        total_samples = max(1, max_frame + self.sample_rate)
        if np is not None:
            data: object = np.zeros(max(1, total_samples), dtype=np.float32)
        else:
            data = [0.0] * max(1, total_samples)

        a, d, s, r = self._profile_envelope(track.synth_profile)
        for note in track.notes:
            start_idx = tick_to_sample_frame(note.start_tick, self.sample_rate, bpm)
            end_idx = max(start_idx + 1, tick_to_sample_frame(note.start_tick + note.duration_tick, self.sample_rate, bpm))
            note_samples = max(1, end_idx - start_idx)
            dur_sec = note_samples / float(self.sample_rate)
            freq = midi_to_hz(note.pitch)
            amp = (note.velocity / 127.0) * track.volume * 0.4

            render_end = min(len(data), start_idx + note_samples)
            if render_end <= start_idx:
                continue

            count = render_end - start_idx
            if np is not None and isinstance(data, np.ndarray):
                t = np.arange(count, dtype=np.float32) / float(self.sample_rate)
                phase = 2.0 * math.pi * freq * t
                env = self._adsr_curve(t, dur_sec, a, d, s, r)
                waveform = self._wave_curve(phase, track.synth_profile)
                data[start_idx:render_end] += (waveform * env * amp).astype(np.float32, copy=False)
                continue

            for i in range(count):
                idx = start_idx + i
                t = i / self.sample_rate
                phase = 2.0 * math.pi * freq * t
                env = self._adsr(t, dur_sec, a, d, s, r)
                sample = self._wave(phase, track.synth_profile) * env * amp
                data[idx] += sample

        return data, self.sample_rate

    def render_track_chunk(self, track: TrackState, bpm: int, start_sec: float, duration_sec: float) -> tuple[object, int]:
        total_samples = max(1, int(round(max(0.0, duration_sec) * self.sample_rate)))
        if np is not None:
            data: object = np.zeros(total_samples, dtype=np.float32)
        else:
            data = [0.0] * total_samples
        if track.track_type != 'instrument' or not track.notes:
            return data, self.sample_rate

        chunk_start_frame = max(0, int(round(max(0.0, float(start_sec)) * self.sample_rate)))
        chunk_end_frame = chunk_start_frame + total_samples
        a, d, s, r = self._profile_envelope(track.synth_profile)

        for note in track.notes:
            note_start_frame = tick_to_sample_frame(note.start_tick, self.sample_rate, bpm)
            note_end_frame = max(
                note_start_frame + 1,
                tick_to_sample_frame(note.start_tick + note.duration_tick, self.sample_rate, bpm),
            )
            if note_end_frame <= chunk_start_frame or note_start_frame >= chunk_end_frame:
                continue

            overlap_start_frame = max(note_start_frame, chunk_start_frame)
            overlap_end_frame = min(note_end_frame, chunk_end_frame)
            dst_start = max(0, overlap_start_frame - chunk_start_frame)
            dst_end = min(total_samples, overlap_end_frame - chunk_start_frame)
            if dst_end <= dst_start:
                continue

            amp = (note.velocity / 127.0) * track.volume * 0.4
            freq = midi_to_hz(note.pitch)
            source_t = (overlap_start_frame - note_start_frame) / float(self.sample_rate)
            note_duration = max(1, note_end_frame - note_start_frame) / float(self.sample_rate)
            count = dst_end - dst_start

            if np is not None and isinstance(data, np.ndarray):
                t = source_t + (np.arange(count, dtype=np.float32) / float(self.sample_rate))
                phase = 2.0 * math.pi * freq * t
                env = self._adsr_curve(t, note_duration, a, d, s, r)
                waveform = self._wave_curve(phase, track.synth_profile)
                data[dst_start:dst_end] += (waveform * env * amp).astype(np.float32, copy=False)
                continue

            for i in range(count):
                t = source_t + (i / self.sample_rate)
                phase = 2.0 * math.pi * freq * t
                env = self._adsr(t, note_duration, a, d, s, r)
                data[dst_start + i] += self._wave(phase, track.synth_profile) * env * amp

        return data, self.sample_rate

    def render_track(self, track: TrackState, bpm: int, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data, sample_rate = self.render_track_audio(track, bpm)
        write_wav_samples(output_path, data, sample_rate)


class PianoRollWidget(QtWidgets.QGraphicsView):
    noteChanged = QtCore.Signal()
    selectionChanged = QtCore.Signal()
    notePreviewRequested = QtCore.Signal(int, int, int)
    horizontalZoomChanged = QtCore.Signal(int)
    rulerDisplayModeChanged = QtCore.Signal(str)

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
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self.setOptimizationFlag(QtWidgets.QGraphicsView.OptimizationFlag.DontSavePainterState, True)
        self.setOptimizationFlag(QtWidgets.QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.RubberBandDrag)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.cell_w = 48
        self.cell_h = 14
        self.total_beats = 64
        self.tool = 'pencil'
        self.note_length_div = 8
        self.ruler_display_mode = 'bars'
        self._line_start: QtCore.QPointF | None = None
        self._pencil_note: MidiNote | None = None
        self._pencil_anchor_tick = 0
        self._drag_anchor_tick = 0
        self._drag_anchor_pitch = 0
        self._drag_selected_snapshot: list[tuple[MidiNote, int, int]] = []
        self._resize_note: MidiNote | None = None
        self._resize_edge = 'right'
        self._resize_anchor_start_tick = 0
        self._resize_anchor_end_tick = 0
        self._resize_anchor_duration = 0
        self._drag_playhead = False
        self._drag_left_locator = False
        self._drag_right_locator = False
        self._suppress_ruler_context_menu = False
        self._interaction_dirty = False
        self._locator_ruler_height = 16
        self._left_locator_item: QtWidgets.QGraphicsLineItem | None = None
        self._left_locator_tag: QtWidgets.QGraphicsSimpleTextItem | None = None
        self._right_locator_item: QtWidgets.QGraphicsLineItem | None = None
        self._right_locator_tag: QtWidgets.QGraphicsSimpleTextItem | None = None
        self._playhead_item: QtWidgets.QGraphicsLineItem | None = None
        self._note_items: dict[int, QtWidgets.QGraphicsRectItem] = {}
        self.refresh()

    def current_track(self) -> TrackState:
        return self.project.tracks[self.get_track_index()]

    def _current_track_color(self) -> QtGui.QColor:
        return track_display_color(self.current_track(), self.get_track_index())

    def _content_height(self) -> float:
        pitch_count = PITCH_MAX - PITCH_MIN + 1
        return float(pitch_count * self.cell_h)

    def _seconds_per_tick(self) -> float:
        return 60.0 / max(1, self.project.bpm) / TICKS_PER_BEAT

    def _is_bar_ruler_mode(self) -> bool:
        return str(self.ruler_display_mode).strip().lower() == 'bars'

    def _beats_to_x(self, beats: float) -> float:
        beats = max(0.0, float(beats))
        if self._is_bar_ruler_mode():
            return beats * self.cell_w
        return beats * (60.0 / max(1, self.project.bpm)) * self.cell_w

    def _tick_to_x(self, tick: int) -> float:
        return self._beats_to_x(float(tick) / TICKS_PER_BEAT)

    def _duration_ticks_to_width(self, duration_tick: int) -> float:
        return max(1.0, self._beats_to_x(float(duration_tick) / TICKS_PER_BEAT))

    def _x_to_sec(self, x: float) -> float:
        if self._is_bar_ruler_mode():
            beats = max(0.0, float(x) / max(1.0, float(self.cell_w)))
            return beats * (60.0 / max(1, self.project.bpm))
        return max(0.0, float(x) / max(1.0, float(self.cell_w)))

    def _scene_duration_sec(self) -> float:
        base_sec = self.total_beats * (60.0 / max(1, self.project.bpm))
        note_end_tick = max((note.start_tick + note.duration_tick for note in self.current_track().notes), default=0)
        note_end_sec = note_end_tick * self._seconds_per_tick()
        padding_sec = max(1.0, 60.0 / max(1, self.project.bpm))
        return max(
            base_sec,
            self.project.right_locator_sec,
            self.project.playhead_sec,
            note_end_sec + padding_sec,
        )

    def _beat_duration_sec(self) -> float:
        return 60.0 / max(1, self.project.bpm)

    def _bar_duration_sec(self) -> float:
        return self._beat_duration_sec() * 4.0

    def _draw_snap_division_lines(self, width: float, height: float) -> None:
        if not getattr(self.project, 'quantize_enabled', True):
            return
        grid_tick = self._grid_tick()
        if grid_tick <= 0:
            return
        snap_spacing_px = self._duration_ticks_to_width(grid_tick)
        if snap_spacing_px < 4.0:
            return
        duration_ticks = max(grid_tick, int(math.ceil(self._scene_duration_sec() / max(1e-9, self._seconds_per_tick()))))
        max_tick = ((duration_ticks + grid_tick - 1) // grid_tick) * grid_tick
        pen = QtGui.QPen(QtGui.QColor(66, 66, 66))
        pen.setWidth(1)
        tick = grid_tick
        while tick <= max_tick:
            if tick % TICKS_PER_BEAT != 0:
                x = self._tick_to_x(tick)
                if x > width + 1.0:
                    break
                self.scene_obj.addLine(x, self._locator_ruler_height, x, height, pen)
            tick += grid_tick

    def _draw_seconds_ruler(self, duration_sec: float) -> None:
        sec = 0
        max_sec = int(math.ceil(duration_sec))
        while sec <= max_sec:
            x = self._locator_x(sec)
            self.scene_obj.addLine(x, 0, x, self._locator_ruler_height, QtGui.QPen(QtGui.QColor(130, 130, 130)))
            if sec % 2 == 0:
                label = self.scene_obj.addSimpleText(f"{sec}s")
                label.setBrush(QtGui.QBrush(QtGui.QColor(210, 210, 210)))
                label.setPos(x + 2, 0)
            sec += 1

    def _draw_bar_ruler(self, duration_sec: float) -> None:
        bar_duration = self._bar_duration_sec()
        if bar_duration <= 0:
            self._draw_seconds_ruler(duration_sec)
            return
        bar_count = max(1, int(math.ceil(duration_sec / bar_duration)))
        bar_spacing_px = self._locator_x(bar_duration) - self._locator_x(0.0)
        label_every = 1
        if bar_spacing_px < 48.0:
            label_every = 2
        if bar_spacing_px < 28.0:
            label_every = 4
        if bar_spacing_px < 16.0:
            label_every = 8
        for bar_idx in range(bar_count + 1):
            sec = bar_idx * bar_duration
            x = self._locator_x(sec)
            self.scene_obj.addLine(x, 0, x, self._locator_ruler_height, QtGui.QPen(QtGui.QColor(130, 130, 130)))
            if bar_idx % label_every == 0:
                label = self.scene_obj.addSimpleText(str(bar_idx + 1))
                label.setBrush(QtGui.QBrush(QtGui.QColor(210, 210, 210)))
                label.setPos(x + 2, 0)

    def _locator_x(self, sec: float) -> float:
        sec = max(0.0, float(sec))
        if self._is_bar_ruler_mode():
            return sec * (max(1, self.project.bpm) / 60.0) * self.cell_w
        return sec * self.cell_w

    def update_overlay_items(self) -> None:
        height = self._content_height()
        if self._left_locator_item is not None:
            x = self._locator_x(self.project.left_locator_sec)
            self._left_locator_item.setLine(x, 0, x, height)
            if self._left_locator_tag is not None:
                self._left_locator_tag.setPos(x + 2, 0)

        if self._right_locator_item is not None:
            x = self._locator_x(self.project.right_locator_sec)
            self._right_locator_item.setLine(x, 0, x, height)
            if self._right_locator_tag is not None:
                self._right_locator_tag.setPos(x + 2, 0)

        if self._playhead_item is not None:
            x = self._locator_x(self.project.playhead_sec)
            self._playhead_item.setLine(x, 0, x, height)

    def _quantize_ticks(self) -> int:
        beats = 4.0 / max(1, self.project.quantize_div)
        if getattr(self.project, 'quantize_triplet', False):
            beats *= 2.0 / 3.0
        return max(1, int(round(beats * TICKS_PER_BEAT)))

    def _grid_tick(self) -> int:
        return self._quantize_ticks()

    def _pos_to_beat_pitch(self, pos: QtCore.QPointF) -> tuple[float, int]:
        beat = self._x_to_sec(pos.x()) * (max(1, self.project.bpm) / 60.0)
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
        bar_duration = 4.0 * (60.0 / max(1, self.project.bpm))
        locator_info = f"L {self.project.left_locator_sec / bar_duration:.2f}b  R {self.project.right_locator_sec / bar_duration:.2f}b"
        self.setHorizontalHeaderLabels([
            f"Track ({locator_info})", "Type", "Instrument", "Mode", "Profile", "Mute", "Solo", "Notes"
        ])

    def _note_rect(self, note: MidiNote) -> QtCore.QRectF:
        x = self._tick_to_x(note.start_tick)
        w = self._duration_ticks_to_width(note.duration_tick)
        y_idx = PITCH_MAX - note.pitch
        y = y_idx * self.cell_h
        return QtCore.QRectF(x, y, w, self.cell_h)

    def _note_brush(self, note: MidiNote) -> QtGui.QBrush:
        color = self._current_track_color()
        color = color.lighter(150) if note.selected else color
        return QtGui.QBrush(color)

    def _note_pen(self, note: MidiNote) -> QtGui.QPen:
        color = self._current_track_color()
        if note.selected:
            pen_color = color.lighter(185)
            width = 2
        else:
            pen_color = color.darker(190)
            width = 1
        return QtGui.QPen(pen_color, width)

    @staticmethod
    def _note_label_text(note: MidiNote) -> str:
        names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return names[int(note.pitch) % 12]

    def _apply_note_label_style(self, item: QtWidgets.QGraphicsRectItem, note: MidiNote) -> None:
        label = item.data(1)
        if not isinstance(label, QtWidgets.QGraphicsSimpleTextItem):
            return
        label.setText(self._note_label_text(note))
        font = label.font()
        font.setPixelSize(max(7, min(10, self.cell_h - 2)))
        label.setFont(font)
        label.setBrush(QtGui.QBrush(track_text_color(self._note_brush(note).color())))
        rect = item.rect()
        label_rect = label.boundingRect()
        padding_x = 3.0
        available_width = max(0.0, rect.width() - (padding_x * 2.0))
        show_label = available_width >= (label_rect.width() + 1.0) and rect.height() >= (label_rect.height() + 2.0)
        label.setVisible(show_label)
        if show_label:
            label.setPos(rect.x() + padding_x, rect.y() + max(0.0, (rect.height() - label_rect.height()) * 0.5 - 0.5))

    def _apply_note_item_style(self, item: QtWidgets.QGraphicsRectItem, note: MidiNote) -> None:
        item.setRect(self._note_rect(note))
        item.setBrush(self._note_brush(note))
        item.setPen(self._note_pen(note))
        item.setSelected(note.selected)
        item.setZValue(20 if note.selected else 10)
        self._apply_note_label_style(item, note)

    def _resize_margin_px(self) -> float:
        return max(3.0, min(5.0, self.cell_w * 0.12))

    def _find_note_hit(self, scene_pos: QtCore.QPointF) -> tuple[MidiNote | None, str | None]:
        for item in self.items(self.mapFromScene(scene_pos)):
            note = item.data(0)
            if isinstance(note, MidiNote):
                rect = item.sceneBoundingRect()
                margin = min(self._resize_margin_px(), rect.width() * 0.5)
                near_start = (
                    scene_pos.x() >= rect.left() - 1.0
                    and scene_pos.x() <= min(rect.right(), rect.left() + margin)
                )
                near_end = (
                    scene_pos.x() >= max(rect.left(), rect.right() - margin)
                    and scene_pos.x() <= rect.right() + 1.0
                )
                if near_start and near_end:
                    hit = 'left' if scene_pos.x() < rect.center().x() else 'right'
                elif near_start:
                    hit = 'left'
                elif near_end:
                    hit = 'right'
                else:
                    hit = 'body'
                return note, hit
        return None, None

    def _begin_note_drag(self, clicked: MidiNote) -> None:
        if not clicked.selected:
            self._set_single_selected_note(clicked)
        self._drag_anchor_tick = clicked.start_tick
        self._drag_anchor_pitch = clicked.pitch
        self._drag_selected_snapshot = [
            (note, note.start_tick, note.pitch) for note in self.current_track().notes if note.selected
        ]

    def _begin_note_resize(self, clicked: MidiNote, edge: str = 'right') -> None:
        if not clicked.selected:
            self._set_single_selected_note(clicked)
        self._resize_note = clicked
        self._resize_edge = edge if edge in {'left', 'right'} else 'right'
        self._resize_anchor_start_tick = clicked.start_tick
        self._resize_anchor_end_tick = clicked.start_tick + clicked.duration_tick
        self._resize_anchor_duration = clicked.duration_tick

    def _finish_left_mouse_interaction(self) -> bool:
        commit_change = bool(self._interaction_dirty)
        self._interaction_dirty = False
        self._drag_playhead = False
        self._drag_left_locator = False
        self._pencil_note = None
        self._drag_selected_snapshot = []
        self._resize_note = None
        self._resize_edge = 'right'
        return commit_change

    def _update_hover_cursor(self, scene_pos: QtCore.QPointF | None = None) -> None:
        if scene_pos is None or self.tool not in {'pencil', 'select'}:
            self.unsetCursor()
            return
        note, hit = self._find_note_hit(scene_pos)
        if hit in {'left', 'right'} and note is not None:
            self.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)
        elif note is not None:
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        else:
            self.unsetCursor()

    def _locator_hit_margin_px(self) -> float:
        return 6.0

    def _ruler_menu_zone_height(self) -> float:
        return 6.0

    def _is_in_ruler_menu_zone(self, viewport_y: float) -> bool:
        return 0.0 <= float(viewport_y) <= self._ruler_menu_zone_height()

    def _is_in_locator_ruler(self, viewport_y: float) -> bool:
        y = float(viewport_y)
        return self._ruler_menu_zone_height() < y <= self._locator_ruler_height

    def _is_near_any_locator(self, scene_pos: QtCore.QPointF, viewport_y: float) -> bool:
        if not self._is_in_locator_ruler(viewport_y):
            return False
        margin = self._locator_hit_margin_px()
        left_x = self._locator_x(self.project.left_locator_sec)
        right_x = self._locator_x(self.project.right_locator_sec)
        return abs(scene_pos.x() - left_x) <= margin or abs(scene_pos.x() - right_x) <= margin

    def _ruler_hit_target(self, scene_pos: QtCore.QPointF, button: QtCore.Qt.MouseButton, viewport_y: float) -> str | None:
        if not self._is_in_locator_ruler(viewport_y):
            return None
        margin = self._locator_hit_margin_px()
        left_x = self._locator_x(self.project.left_locator_sec)
        right_x = self._locator_x(self.project.right_locator_sec)
        playhead_x = self._locator_x(self.project.playhead_sec)
        if button == QtCore.Qt.MouseButton.LeftButton and abs(scene_pos.x() - left_x) <= margin:
            return 'left_locator'
        if button == QtCore.Qt.MouseButton.RightButton and abs(scene_pos.x() - right_x) <= margin:
            return 'right_locator'
        if button == QtCore.Qt.MouseButton.LeftButton and abs(scene_pos.x() - playhead_x) <= margin:
            return 'playhead'
        return None

    def _update_note_item(self, note: MidiNote) -> None:
        item = self._note_items.get(id(note))
        if item is None:
            self._draw_note(note)
            return
        self._apply_note_item_style(item, note)

    def _remove_note_item(self, note: MidiNote) -> None:
        item = self._note_items.pop(id(note), None)
        if item is not None:
            self.scene_obj.removeItem(item)

    def _set_single_selected_note(self, clicked: MidiNote) -> None:
        changed = False
        for note in self.current_track().notes:
            selected = note is clicked
            if note.selected != selected:
                note.selected = selected
                self._update_note_item(note)
                changed = True
        if changed:
            self.selectionChanged.emit()

    def _clear_note_selection(self) -> None:
        changed = False
        for note in self.current_track().notes:
            if note.selected:
                note.selected = False
                self._update_note_item(note)
                changed = True
        if changed:
            self.selectionChanged.emit()

    def refresh(self) -> None:
        self.scene_obj.clear()
        self._left_locator_item = None
        self._left_locator_tag = None
        self._right_locator_item = None
        self._right_locator_tag = None
        self._playhead_item = None
        self._note_items = {}
        duration_sec = self._scene_duration_sec()
        beat_span = max(self.total_beats, int(math.ceil(duration_sec * max(1, self.project.bpm) / 60.0)))
        width = max(1, int(math.ceil(self._locator_x(duration_sec))))
        pitch_count = PITCH_MAX - PITCH_MIN + 1
        height = pitch_count * self.cell_h

        for i in range(pitch_count):
            y = i * self.cell_h
            pitch = PITCH_MAX - i
            color = QtGui.QColor(26, 26, 26) if pitch % 12 in BLACK_KEY_PITCH_CLASSES else QtGui.QColor(56, 56, 56)
            self.scene_obj.addRect(QtCore.QRectF(0.0, float(y), float(width), float(self.cell_h)), QtGui.QPen(QtCore.Qt.PenStyle.NoPen), QtGui.QBrush(color))

        self._draw_snap_division_lines(width, height)

        for beat in range(beat_span + 1):
            x = self._locator_x(beat * (60.0 / max(1, self.project.bpm)))
            pen = QtGui.QPen(QtGui.QColor(120, 120, 120) if beat % 4 == 0 else QtGui.QColor(80, 80, 80))
            self.scene_obj.addLine(x, 0, x, height, pen)

        self.scene_obj.addRect(0, 0, width, self._locator_ruler_height, QtGui.QPen(QtGui.QColor(88, 88, 88)), QtGui.QBrush(QtGui.QColor(28, 28, 28, 220)))
        if self.ruler_display_mode == 'bars':
            self._draw_bar_ruler(duration_sec)
        else:
            self._draw_seconds_ruler(duration_sec)

        for i in range(pitch_count + 1):
            y = i * self.cell_h
            self.scene_obj.addLine(0, y, width, y, QtGui.QPen(QtGui.QColor(80, 80, 80)))

        for note in self.current_track().notes:
            self._draw_note(note)

        self._left_locator_item = self.scene_obj.addLine(0, 0, 0, height, QtGui.QPen(QtGui.QColor(0, 200, 160), 2))
        self._left_locator_item.setZValue(1000)
        self._left_locator_tag = self.scene_obj.addSimpleText('L')
        self._left_locator_tag.setBrush(QtGui.QBrush(QtGui.QColor(0, 200, 160)))
        self._left_locator_tag.setZValue(1001)

        self._right_locator_item = self.scene_obj.addLine(0, 0, 0, height, QtGui.QPen(QtGui.QColor(240, 200, 0), 2))
        self._right_locator_item.setZValue(1000)
        self._right_locator_tag = self.scene_obj.addSimpleText('R')
        self._right_locator_tag.setBrush(QtGui.QBrush(QtGui.QColor(240, 200, 0)))
        self._right_locator_tag.setZValue(1001)

        self._playhead_item = self.scene_obj.addLine(0, 0, 0, height, QtGui.QPen(QtGui.QColor(255, 90, 90), 2))
        self._playhead_item.setZValue(1002)
        self.update_overlay_items()

        self.setSceneRect(0, 0, width, height)

    def _draw_note(self, note: MidiNote) -> None:
        item = self.scene_obj.addRect(self._note_rect(note), QtGui.QPen(QtGui.QColor(0, 0, 0)), self._note_brush(note))
        item.setData(0, note)
        item.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        label = QtWidgets.QGraphicsSimpleTextItem(self._note_label_text(note), item)
        label.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        label.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, False)
        item.setData(1, label)
        self._apply_note_item_style(item, note)
        self._note_items[id(note)] = item

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta:
            if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                step = 1 if delta > 0 else -1
                self.cell_h = max(10, min(28, self.cell_h + step))
            else:
                step = 2 if delta > 0 else -2
                self.cell_w = max(8, min(160, self.cell_w + step))
                self.horizontalZoomChanged.emit(self.cell_w)
            self.refresh()
            event.accept()
            return
        super().wheelEvent(event)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        if self._suppress_ruler_context_menu:
            self._suppress_ruler_context_menu = False
            event.accept()
            return
        scene_pos = self.mapToScene(event.pos())
        viewport_y = float(event.pos().y())
        menu = QtWidgets.QMenu(self)
        if self._is_in_ruler_menu_zone(viewport_y):
            ruler_menu = menu.addMenu('Ruler Display')
            ruler_group = QtGui.QActionGroup(menu)
            ruler_group.setExclusive(True)
            seconds_action = ruler_menu.addAction('Seconds')
            seconds_action.setCheckable(True)
            seconds_action.setChecked(self.ruler_display_mode == 'seconds')
            ruler_group.addAction(seconds_action)
            bars_action = ruler_menu.addAction('Bars')
            bars_action.setCheckable(True)
            bars_action.setChecked(self.ruler_display_mode == 'bars')
            ruler_group.addAction(bars_action)
            menu.addSeparator()
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
        if self._is_in_ruler_menu_zone(viewport_y):
            if chosen == seconds_action:
                self.ruler_display_mode = 'seconds'
                self.rulerDisplayModeChanged.emit(self.ruler_display_mode)
                self.refresh()
                return
            if chosen == bars_action:
                self.ruler_display_mode = 'bars'
                self.rulerDisplayModeChanged.emit(self.ruler_display_mode)
                self.refresh()
                return
        for key, action in actions.items():
            if chosen == action:
                self.set_tool(key)
                break

    def _find_note_at(self, scene_pos: QtCore.QPointF) -> MidiNote | None:
        note, _hit = self._find_note_hit(scene_pos)
        return note

    def _insert_note_at(self, scene_pos: QtCore.QPointF, commit: bool = True) -> MidiNote:
        beat, pitch = self._pos_to_beat_pitch(scene_pos)
        grid = self._grid_tick()
        start_tick = int(round((beat * TICKS_PER_BEAT) / grid) * grid)
        note = MidiNote(start_tick=start_tick, duration_tick=self._length_ticks(), pitch=pitch)
        self.current_track().notes.append(note)
        self._draw_note(note)
        self.notePreviewRequested.emit(note.pitch, note.velocity, note.duration_tick)
        if commit:
            self.noteChanged.emit()
        else:
            self._interaction_dirty = True
        return note

    def _erase_note_at(self, scene_pos: QtCore.QPointF) -> None:
        note = self._find_note_at(scene_pos)
        if note is None:
            return
        track = self.current_track()
        track.notes = [n for n in track.notes if n is not note]
        self._remove_note_item(note)
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
        self._update_note_item(note)
        sliced = MidiNote(start_tick=cut_tick, duration_tick=right, pitch=note.pitch, velocity=note.velocity)
        self.current_track().notes.append(sliced)
        self._draw_note(sliced)
        self.noteChanged.emit()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
        scene_pos = self.mapToScene(event.position().toPoint())
        viewport_y = float(event.position().y())
        sec = self._x_to_sec(scene_pos.x())
        ruler_hit = self._ruler_hit_target(scene_pos, event.button(), viewport_y)

        if ruler_hit == 'left_locator':
            self._drag_left_locator = True
            self.set_left_locator(sec)
            return
        if ruler_hit == 'right_locator':
            self._drag_right_locator = True
            self._suppress_ruler_context_menu = True
            self.set_right_locator(sec)
            return
        if ruler_hit == 'playhead':
            self._drag_playhead = True
            self.set_playhead(sec)
            return
        if self._is_near_any_locator(scene_pos, viewport_y):
            if event.button() == QtCore.Qt.MouseButton.RightButton:
                self._suppress_ruler_context_menu = True
            return

        if self._is_in_locator_ruler(viewport_y) and event.button() == QtCore.Qt.MouseButton.LeftButton:
            if bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier):
                self.set_right_locator(sec)
            else:
                self.set_left_locator(sec)
            return
        if self._is_in_locator_ruler(viewport_y) and event.button() == QtCore.Qt.MouseButton.RightButton:
            self._suppress_ruler_context_menu = True
            self.set_right_locator(sec)
            return

        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            clicked, hit = self._find_note_hit(scene_pos)
            if self.tool == 'pencil':
                if clicked is not None:
                    if hit in {'left', 'right'}:
                        self._begin_note_resize(clicked, hit)
                    else:
                        self._begin_note_drag(clicked)
                    return
                note = self._insert_note_at(scene_pos, commit=False)
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
                if clicked is not None:
                    self.sync_selection()
                    if hit in {'left', 'right'}:
                        self._begin_note_resize(clicked, hit)
                    else:
                        self._begin_note_drag(clicked)
                    return
                self._clear_note_selection()
                super().mousePressEvent(event)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        scene_pos = self.mapToScene(event.position().toPoint())
        if (
            self._drag_playhead
            or self._drag_left_locator
            or self._pencil_note is not None
            or self._resize_note is not None
            or self._drag_selected_snapshot
        ) and not bool(event.buttons() & QtCore.Qt.MouseButton.LeftButton):
            commit_change = self._finish_left_mouse_interaction()
            if commit_change:
                self.noteChanged.emit()
            self._update_hover_cursor(scene_pos)
            return
        if self._drag_right_locator and not bool(event.buttons() & QtCore.Qt.MouseButton.RightButton):
            self._drag_right_locator = False
            self._update_hover_cursor(scene_pos)
            return
        if self._drag_left_locator:
            sec = self._x_to_sec(scene_pos.x())
            self.set_left_locator(sec)
            return
        if self._drag_right_locator:
            sec = self._x_to_sec(scene_pos.x())
            self.set_right_locator(sec)
            return
        if self._drag_playhead:
            sec = self._x_to_sec(scene_pos.x())
            self.set_playhead(sec)
            return
        if self.tool == 'pencil' and self._pencil_note is not None:
            beat, _ = self._pos_to_beat_pitch(scene_pos)
            grid = self._grid_tick()
            end_tick = int(round((beat * TICKS_PER_BEAT) / grid) * grid)
            new_duration = max(grid, end_tick - self._pencil_anchor_tick + grid)
            if new_duration == self._pencil_note.duration_tick:
                return
            self._pencil_note.duration_tick = new_duration
            self._update_note_item(self._pencil_note)
            self._interaction_dirty = True
            return

        if self._resize_note is not None:
            beat, _pitch = self._pos_to_beat_pitch(scene_pos)
            grid = self._grid_tick()
            edge_tick = int(round((beat * TICKS_PER_BEAT) / grid) * grid)
            if self._resize_edge == 'left':
                max_start_tick = max(0, self._resize_anchor_end_tick - grid)
                new_start_tick = max(0, min(edge_tick, max_start_tick))
                new_duration = max(grid, self._resize_anchor_end_tick - new_start_tick)
                if new_start_tick != self._resize_note.start_tick or new_duration != self._resize_note.duration_tick:
                    self._resize_note.start_tick = new_start_tick
                    self._resize_note.duration_tick = new_duration
                    self._update_note_item(self._resize_note)
                    self._interaction_dirty = True
            else:
                new_duration = max(grid, edge_tick - self._resize_note.start_tick)
                if new_duration != self._resize_note.duration_tick:
                    self._resize_note.duration_tick = new_duration
                    self._update_note_item(self._resize_note)
                    self._interaction_dirty = True
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
                self._update_note_item(note)
            self._interaction_dirty = True
            return

        if self.tool == 'pencil' and self._drag_selected_snapshot:
            beat, pitch = self._pos_to_beat_pitch(scene_pos)
            grid = self._grid_tick()
            current_tick = int(round((beat * TICKS_PER_BEAT) / grid) * grid)
            delta_tick = current_tick - self._drag_anchor_tick
            delta_pitch = pitch - self._drag_anchor_pitch
            for note, start_tick, start_pitch in self._drag_selected_snapshot:
                note.start_tick = max(0, start_tick + delta_tick)
                note.pitch = max(PITCH_MIN, min(PITCH_MAX, start_pitch + delta_pitch))
                self._update_note_item(note)
            self._interaction_dirty = True
            return

        self._update_hover_cursor(scene_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        commit_change = False
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
                note = MidiNote(start_tick=int(start_tick), duration_tick=self._length_ticks(), pitch=max(PITCH_MIN, min(PITCH_MAX, pitch)))
                track.notes.append(note)
                self._draw_note(note)
            self._line_start = None
            commit_change = True
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            commit_change = commit_change or self._finish_left_mouse_interaction()
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self._drag_right_locator = False
        self._line_start = None
        super().mouseReleaseEvent(event)
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.tool == 'select' and not commit_change:
            self.sync_selection()
            self.selectionChanged.emit()
        if commit_change:
            self.noteChanged.emit()
        self._update_hover_cursor(self.mapToScene(event.position().toPoint()))

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self.unsetCursor()
        super().leaveEvent(event)

    def sync_selection(self) -> None:
        for item in self.scene_obj.items():
            note = item.data(0)
            if isinstance(note, MidiNote):
                note.selected = item.isSelected()
                self._update_note_item(note)

    def _selected_notes(self) -> list[MidiNote]:
        self.sync_selection()
        return [note for note in self.current_track().notes if note.selected]

    def _nudge_selected(self, *, delta_tick: int = 0, delta_pitch: int = 0) -> bool:
        selected = self._selected_notes()
        if not selected:
            return False

        applied_tick = int(delta_tick)
        if applied_tick < 0:
            applied_tick = max(applied_tick, -min(note.start_tick for note in selected))

        applied_pitch = int(delta_pitch)
        if applied_pitch > 0:
            max_up = min(PITCH_MAX - note.pitch for note in selected)
            applied_pitch = min(applied_pitch, max_up)
        elif applied_pitch < 0:
            max_down = min(note.pitch - PITCH_MIN for note in selected)
            applied_pitch = max(applied_pitch, -max_down)

        if applied_tick == 0 and applied_pitch == 0:
            return False

        for note in selected:
            note.start_tick = max(0, note.start_tick + applied_tick)
            note.pitch = max(PITCH_MIN, min(PITCH_MAX, note.pitch + applied_pitch))
            self._update_note_item(note)

        self.noteChanged.emit()
        return True

    def delete_selected(self) -> None:
        self.sync_selection()
        track = self.current_track()
        removed = [n for n in track.notes if n.selected]
        track.notes = [n for n in track.notes if not n.selected]
        for note in removed:
            self._remove_note_item(note)
        self.noteChanged.emit()

    def quantize_selected(self) -> None:
        if not getattr(self.project, 'quantize_enabled', True):
            return
        self.sync_selection()
        grid = self._grid_tick()
        for note in self.current_track().notes:
            if note.selected:
                note.start_tick = round(note.start_tick / grid) * grid
                note.duration_tick = max(grid, round(note.duration_tick / grid) * grid)
                self._update_note_item(note)
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
            self._draw_note(duplicated)
            self._update_note_item(note)
        self.noteChanged.emit()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        step_tick = self._grid_tick()
        handled = False
        if event.key() == QtCore.Qt.Key.Key_Left:
            handled = self._nudge_selected(delta_tick=-step_tick)
        elif event.key() == QtCore.Qt.Key.Key_Right:
            handled = self._nudge_selected(delta_tick=step_tick)
        elif event.key() == QtCore.Qt.Key.Key_Up:
            handled = self._nudge_selected(delta_pitch=1)
        elif event.key() == QtCore.Qt.Key.Key_Down:
            handled = self._nudge_selected(delta_pitch=-1)

        if handled:
            event.accept()
            return
        super().keyPressEvent(event)


class VelocityEditorWidget(QtWidgets.QGraphicsView):
    velocityChanged = QtCore.Signal()
    horizontalZoomChanged = QtCore.Signal(int)

    def __init__(self, project: ProjectState, get_track_index_callable, get_ruler_display_mode_callable=None) -> None:
        super().__init__()
        self.project = project
        self.get_track_index = get_track_index_callable
        self.get_ruler_display_mode = get_ruler_display_mode_callable or (lambda: 'bars')
        self.scene_obj = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene_obj)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self.setOptimizationFlag(QtWidgets.QGraphicsView.OptimizationFlag.DontSavePainterState, True)
        self.setOptimizationFlag(QtWidgets.QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)
        self.setFixedHeight(140)
        self.cell_w = 48
        self.total_beats = 64
        self._drag_note: MidiNote | None = None
        self._drag_dirty = False
        self._left_locator_item: QtWidgets.QGraphicsLineItem | None = None
        self._left_locator_tag: QtWidgets.QGraphicsSimpleTextItem | None = None
        self._right_locator_item: QtWidgets.QGraphicsLineItem | None = None
        self._right_locator_tag: QtWidgets.QGraphicsSimpleTextItem | None = None
        self._playhead_item: QtWidgets.QGraphicsLineItem | None = None
        self.refresh()

    def current_track(self) -> TrackState:
        return self.project.tracks[self.get_track_index()]

    def _current_track_color(self) -> QtGui.QColor:
        return track_display_color(self.current_track(), self.get_track_index())

    def _is_bar_ruler_mode(self) -> bool:
        return str(self.get_ruler_display_mode() or 'bars').strip().lower() == 'bars'

    def _beats_to_x(self, beats: float) -> float:
        beats = max(0.0, float(beats))
        if self._is_bar_ruler_mode():
            return beats * self.cell_w
        return beats * (60.0 / max(1, self.project.bpm)) * self.cell_w

    def _locator_x(self, sec: float) -> float:
        sec = max(0.0, float(sec))
        if self._is_bar_ruler_mode():
            return sec * (max(1, self.project.bpm) / 60.0) * self.cell_w
        return sec * self.cell_w

    def _x_to_sec(self, x: float) -> float:
        x = max(0.0, float(x))
        if self._is_bar_ruler_mode():
            return x * (60.0 / max(1, self.project.bpm)) / max(1e-9, float(self.cell_w))
        return x / max(1e-9, float(self.cell_w))

    def _seconds_per_tick(self) -> float:
        return 60.0 / max(1, self.project.bpm) / TICKS_PER_BEAT

    def _tick_to_x(self, tick: int) -> float:
        return self._beats_to_x(float(tick) / TICKS_PER_BEAT)

    def _duration_ticks_to_width(self, duration_tick: int) -> float:
        return max(2.0, self._beats_to_x(float(duration_tick) / TICKS_PER_BEAT))

    def _scene_duration_sec(self) -> float:
        base_sec = self.total_beats * (60.0 / max(1, self.project.bpm))
        note_end_tick = max((note.start_tick + note.duration_tick for note in self.current_track().notes), default=0)
        note_end_sec = note_end_tick * self._seconds_per_tick()
        padding_sec = max(1.0, 60.0 / max(1, self.project.bpm))
        return max(
            base_sec,
            self.project.right_locator_sec,
            self.project.playhead_sec,
            note_end_sec + padding_sec,
        )

    @staticmethod
    def _velocity_line_height(note: MidiNote, height: float) -> float:
        return max(6.0, float(note.velocity) / 127.0 * max(1.0, height - 12.0))

    def _velocity_line_color(self, note: MidiNote) -> QtGui.QColor:
        ratio = max(0.0, min(1.0, float(note.velocity) / 127.0))
        hue = int(round(210.0 - (210.0 * ratio)))
        color = QtGui.QColor.fromHsv(hue % 360, 210, int(round(150 + (90.0 * ratio))))
        if note.selected:
            color = color.lighter(135)
        return color

    def _velocity_line_pen(self, note: MidiNote) -> QtGui.QPen:
        pen = QtGui.QPen(self._velocity_line_color(note))
        pen.setWidth(4 if note.selected else 3)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setCosmetic(True)
        return pen

    def update_overlay_items(self) -> None:
        height = 120
        if self._left_locator_item is not None:
            x = self._locator_x(self.project.left_locator_sec)
            self._left_locator_item.setLine(x, 0, x, height)
            if self._left_locator_tag is not None:
                self._left_locator_tag.setPos(x + 2, 0)

        if self._right_locator_item is not None:
            x = self._locator_x(self.project.right_locator_sec)
            self._right_locator_item.setLine(x, 0, x, height)
            if self._right_locator_tag is not None:
                self._right_locator_tag.setPos(x + 2, 0)

        if self._playhead_item is not None:
            x = self._locator_x(self.project.playhead_sec)
            self._playhead_item.setLine(x, 0, x, height)

    def refresh(self) -> None:
        self.scene_obj.clear()
        self._left_locator_item = None
        self._left_locator_tag = None
        self._right_locator_item = None
        self._right_locator_tag = None
        self._playhead_item = None
        duration_sec = self._scene_duration_sec()
        width = max(1, int(math.ceil(self._locator_x(duration_sec))))
        height = 120
        self.scene_obj.addRect(0, 0, width, height, QtGui.QPen(QtGui.QColor(70, 70, 70)), QtGui.QBrush(QtGui.QColor(28, 28, 28)))

        for note in self.current_track().notes:
            x = self._tick_to_x(note.start_tick)
            h = self._velocity_line_height(note, height)
            y = height - h
            line = self.scene_obj.addLine(x, height - 1.0, x, y, self._velocity_line_pen(note))
            line.setData(0, note)
            line.setZValue(15 if note.selected else 10)

        self._left_locator_item = self.scene_obj.addLine(0, 0, 0, height, QtGui.QPen(QtGui.QColor(0, 200, 160), 2))
        self._left_locator_item.setZValue(1000)
        self._left_locator_tag = self.scene_obj.addSimpleText('L')
        self._left_locator_tag.setBrush(QtGui.QBrush(QtGui.QColor(0, 200, 160)))
        self._left_locator_tag.setZValue(1001)

        self._right_locator_item = self.scene_obj.addLine(0, 0, 0, height, QtGui.QPen(QtGui.QColor(240, 200, 0), 2))
        self._right_locator_item.setZValue(1000)
        self._right_locator_tag = self.scene_obj.addSimpleText('R')
        self._right_locator_tag.setBrush(QtGui.QBrush(QtGui.QColor(240, 200, 0)))
        self._right_locator_tag.setZValue(1001)

        self._playhead_item = self.scene_obj.addLine(0, 0, 0, height, QtGui.QPen(QtGui.QColor(255, 90, 90), 2))
        self._playhead_item.setZValue(1002)
        self.update_overlay_items()

        self.setSceneRect(0, 0, width, height)

    def _apply_velocity_from_pos(self, scene_pos: QtCore.QPointF, commit: bool = True) -> None:
        if self._drag_note is None:
            return
        height = 120
        y = max(0.0, min(float(height), scene_pos.y()))
        vel = int(round((1.0 - (y / height)) * 127.0))
        new_velocity = max(1, min(127, vel))
        if new_velocity == self._drag_note.velocity:
            return
        self._drag_note.velocity = new_velocity
        self.refresh()
        if commit:
            self.velocityChanged.emit()
        else:
            self._drag_dirty = True

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            item = self.itemAt(event.position().toPoint())
            if item is not None:
                note = item.data(0)
                if isinstance(note, MidiNote):
                    self._drag_note = note
                    self._drag_dirty = False
                    self._apply_velocity_from_pos(self.mapToScene(event.position().toPoint()), commit=False)
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_note is not None:
            self._apply_velocity_from_pos(self.mapToScene(event.position().toPoint()), commit=False)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self._drag_dirty:
                self._drag_dirty = False
                self.velocityChanged.emit()
            self._drag_note = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta:
            step = 2 if delta > 0 else -2
            self.cell_w = max(8, min(160, self.cell_w + step))
            self.horizontalZoomChanged.emit(self.cell_w)
            self.refresh()
            event.accept()
            return
        super().wheelEvent(event)


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
        bar_duration = 4.0 * (60.0 / max(1, self.project.bpm))
        locator_info = f"L {self.project.left_locator_sec / bar_duration:.2f}b  R {self.project.right_locator_sec / bar_duration:.2f}b"
        self.setHorizontalHeaderLabels([
            f"Track ({locator_info})", "Type", "Instrument", "Mode", "Profile", "Mute", "Solo", "Notes"
        ])

    def update_locator_header(self) -> None:
        self._set_headers()

    def refresh(self) -> None:
        self._set_headers()
        self.setRowCount(len(self.project.tracks))
        for i, track in enumerate(self.project.tracks):
            row_color = track_display_color(track, i)
            row_background = row_color.darker(240)
            text_color = track_text_color(row_background)
            values = [
                track.name,
                track.track_type.title(),
                track.instrument,
                track.instrument_mode,
                track.synth_profile,
                'Yes' if track.mute else 'No',
                'Yes' if track.solo else 'No',
                str(len(track.notes)),
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setBackground(QtGui.QBrush(row_background))
                item.setForeground(QtGui.QBrush(text_color))
                self.setItem(i, col, item)


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
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self.setOptimizationFlag(QtWidgets.QGraphicsView.OptimizationFlag.DontSavePainterState, True)
        self.setOptimizationFlag(QtWidgets.QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)
        self.pixels_per_second = 80
        self.lane_height = 110
        self.setAcceptDrops(True)
        self._left_locator_item: QtWidgets.QGraphicsLineItem | None = None
        self._right_locator_item: QtWidgets.QGraphicsLineItem | None = None
        self._playhead_item: QtWidgets.QGraphicsLineItem | None = None
        self.refresh()

    def update_overlay_items(self) -> None:
        height = self.sceneRect().height()
        if self._left_locator_item is not None:
            x = self.project.left_locator_sec * self.pixels_per_second
            self._left_locator_item.setLine(x, 0, x, height)
        if self._right_locator_item is not None:
            x = self.project.right_locator_sec * self.pixels_per_second
            self._right_locator_item.setLine(x, 0, x, height)
        if self._playhead_item is not None:
            x = self.project.playhead_sec * self.pixels_per_second
            self._playhead_item.setLine(x, 0, x, height)

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
        self._left_locator_item = None
        self._right_locator_item = None
        self._playhead_item = None
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

        self._left_locator_item = self.scene_obj.addLine(0, 0, 0, height, QtGui.QPen(QtGui.QColor(0, 200, 160), 2))
        self._left_locator_item.setZValue(1000)
        self._right_locator_item = self.scene_obj.addLine(0, 0, 0, height, QtGui.QPen(QtGui.QColor(240, 200, 0), 2))
        self._right_locator_item.setZValue(1000)
        self._playhead_item = self.scene_obj.addLine(0, 0, 0, height, QtGui.QPen(QtGui.QColor(255, 90, 90), 2))
        self._playhead_item.setZValue(1001)

        for clip in self.project.sample_clips:
            if clip.track_index not in sample_tracks:
                continue
            lane = sample_tracks.index(clip.track_index)
            x = clip.start_sec * self.pixels_per_second
            w = max(1, clip.duration_sec * self.pixels_per_second)
            y = lane * self.lane_height + 26
            h = 70
            clip_track = self.project.tracks[clip.track_index]
            clip_color = track_display_color(clip_track, clip.track_index)
            self.scene_obj.addRect(x, y, w, h, QtGui.QPen(clip_color.darker(210)), QtGui.QBrush(clip_color))
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
            label.setDefaultTextColor(track_text_color(clip_color))
            label.setPos(x + 4, y + 4)

        self.setSceneRect(0, 0, width, height)
        self.update_overlay_items()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            sec = max(0.0, self.mapToScene(event.position().toPoint()).x() / self.pixels_per_second)
            self.set_locator_callable(sec)
        super().mousePressEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta:
            step = 8 if delta > 0 else -8
            self.pixels_per_second = max(24, min(320, self.pixels_per_second + step))
            self.refresh()
            event.accept()
            return
        super().wheelEvent(event)


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
        self._drag_left_locator = False
        self._drag_right_locator = False
        self._locator_ruler_height = 16
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self.setOptimizationFlag(QtWidgets.QGraphicsView.OptimizationFlag.DontSavePainterState, True)
        self.setOptimizationFlag(QtWidgets.QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)
        self.setMouseTracking(True)
        self._left_locator_item: QtWidgets.QGraphicsLineItem | None = None
        self._right_locator_item: QtWidgets.QGraphicsLineItem | None = None
        self._playhead_item: QtWidgets.QGraphicsLineItem | None = None
        self.refresh()

    def _duration_seconds(self) -> float:
        duration = max(8.0, self.project.right_locator_sec + 1.0)
        for section in self.project.midi_sections:
            duration = max(duration, section.start_sec + section.duration_sec + 1.0)
        return duration

    def _lane_count(self) -> int:
        return max(1, len(self.project.tracks))

    def update_overlay_items(self) -> None:
        height = self.sceneRect().height()
        if self._playhead_item is not None:
            x = self.project.playhead_sec * self.pixels_per_second
            self._playhead_item.setLine(x, 0, x, height)
        if self._left_locator_item is not None:
            x = self.project.left_locator_sec * self.pixels_per_second
            self._left_locator_item.setLine(x, 0, x, height)
        if self._right_locator_item is not None:
            x = self.project.right_locator_sec * self.pixels_per_second
            self._right_locator_item.setLine(x, 0, x, height)

    def refresh(self) -> None:
        self.scene_obj.clear()
        self._left_locator_item = None
        self._right_locator_item = None
        self._playhead_item = None
        duration = self._duration_seconds()
        lane_count = self._lane_count()
        width = duration * self.pixels_per_second
        height = lane_count * self.lane_height

        self.scene_obj.addRect(0, 0, width, height, QtGui.QPen(QtGui.QColor(70, 70, 70)), QtGui.QBrush(QtGui.QColor(33, 33, 33)))

        for lane in range(lane_count):
            y = lane * self.lane_height
            self.scene_obj.addLine(0, y, width, y, QtGui.QPen(QtGui.QColor(62, 62, 62)))
            if lane < len(self.project.tracks):
                lane_track = self.project.tracks[lane]
                lane_color = track_display_color(lane_track, lane)
                label = self.scene_obj.addText(lane_track.name)
                label.setDefaultTextColor(lane_color.lighter(130))
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
            track = self.project.tracks[section.track_index]
            section_color = track_display_color(track, section.track_index)
            rect = self.scene_obj.addRect(x, y, w, h, QtGui.QPen(section_color.darker(210)), QtGui.QBrush(section_color))
            rect.setData(0, idx)
            label = self.scene_obj.addText(section.name)
            label.setDefaultTextColor(track_text_color(section_color))
            label.setPos(x + 4, y + 2)

        self._playhead_item = self.scene_obj.addLine(0, 0, 0, height, QtGui.QPen(QtGui.QColor(255, 80, 80), 2))
        self._playhead_item.setZValue(1001)
        self._left_locator_item = self.scene_obj.addLine(0, 0, 0, height, QtGui.QPen(QtGui.QColor(0, 200, 160), 1))
        self._left_locator_item.setZValue(1000)
        self._right_locator_item = self.scene_obj.addLine(0, 0, 0, height, QtGui.QPen(QtGui.QColor(240, 200, 0), 1))
        self._right_locator_item.setZValue(1000)

        self.setSceneRect(0, 0, width, height)
        self.update_overlay_items()

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

    def _locator_hit_margin_px(self) -> float:
        return 6.0

    def _is_near_any_locator(self, scene_pos: QtCore.QPointF, viewport_y: float) -> bool:
        if float(viewport_y) > self._locator_ruler_height:
            return False
        margin = self._locator_hit_margin_px()
        left_x = self.project.left_locator_sec * self.pixels_per_second
        right_x = self.project.right_locator_sec * self.pixels_per_second
        return abs(scene_pos.x() - left_x) <= margin or abs(scene_pos.x() - right_x) <= margin

    def _ruler_hit_target(self, scene_pos: QtCore.QPointF, button: QtCore.Qt.MouseButton, viewport_y: float) -> str | None:
        if float(viewport_y) > self._locator_ruler_height:
            return None
        margin = self._locator_hit_margin_px()
        left_x = self.project.left_locator_sec * self.pixels_per_second
        right_x = self.project.right_locator_sec * self.pixels_per_second
        playhead_x = self.project.playhead_sec * self.pixels_per_second
        if button == QtCore.Qt.MouseButton.LeftButton and abs(scene_pos.x() - left_x) <= margin:
            return 'left_locator'
        if button == QtCore.Qt.MouseButton.RightButton and abs(scene_pos.x() - right_x) <= margin:
            return 'right_locator'
        if button == QtCore.Qt.MouseButton.LeftButton and abs(scene_pos.x() - playhead_x) <= margin:
            return 'playhead'
        return None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        scene_pos = self.mapToScene(event.position().toPoint())
        viewport_y = float(event.position().y())
        sec = max(0.0, scene_pos.x() / self.pixels_per_second)
        ruler_hit = self._ruler_hit_target(scene_pos, event.button(), viewport_y)

        if ruler_hit == 'left_locator':
            self._drag_left_locator = True
            self.set_left_locator_callable(sec)
            return
        if ruler_hit == 'right_locator':
            self._drag_right_locator = True
            self.set_right_locator_callable(sec)
            return
        if ruler_hit == 'playhead':
            self._drag_playhead = True
            self.set_locator_callable(sec)
            return
        if self._is_near_any_locator(scene_pos, viewport_y):
            return

        if viewport_y <= self._locator_ruler_height and event.button() == QtCore.Qt.MouseButton.LeftButton:
            if bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier):
                self.set_right_locator_callable(sec)
            else:
                self.set_left_locator_callable(sec)
            return

        if event.button() == QtCore.Qt.MouseButton.LeftButton:
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
        if self._drag_left_locator:
            sec = max(0.0, self.mapToScene(event.position().toPoint()).x() / self.pixels_per_second)
            self.set_left_locator_callable(sec)
            return
        if self._drag_right_locator:
            sec = max(0.0, self.mapToScene(event.position().toPoint()).x() / self.pixels_per_second)
            self.set_right_locator_callable(sec)
            return
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
            if self._drag_left_locator:
                self._drag_left_locator = False
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
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self._drag_right_locator = False
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta:
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


class KnobInput(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(int)

    def __init__(self, minimum: int = 0, maximum: int = 100, value: int = 0, suffix: str = '', dial_size: int = 68) -> None:
        super().__init__()
        self._suffix = suffix
        self._dial_size = max(24, int(dial_size))
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2 if self._dial_size <= 40 else 4)
        self.dial = QtWidgets.QDial()
        self.dial.setNotchesVisible(True)
        self.dial.setWrapping(False)
        self.dial.setFixedSize(self._dial_size, self._dial_size)
        self.dial.setRange(int(minimum), int(maximum))
        self.value_label = QtWidgets.QLabel()
        self.value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.value_label.setMinimumWidth(max(32, self._dial_size))
        layout.addWidget(self.dial, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.value_label, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        self.dial.valueChanged.connect(self._on_value_changed)
        self.setValue(value)

    def _on_value_changed(self, value: int) -> None:
        self.value_label.setText(f'{int(value)}{self._suffix}')
        self.valueChanged.emit(int(value))

    def setRange(self, minimum: int, maximum: int) -> None:
        self.dial.setRange(int(minimum), int(maximum))

    def setValue(self, value: int) -> None:
        self.dial.setValue(int(value))

    def value(self) -> int:
        return int(self.dial.value())

    def setSuffix(self, suffix: str) -> None:
        self._suffix = suffix
        self._on_value_changed(self.value())


class MixerLevelMeterWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._level = 0.0
        self.setFixedSize(16, 210)

    def setLevel(self, level: float) -> None:
        clamped = max(0.0, min(1.0, float(level)))
        if abs(clamped - self._level) < 0.002:
            return
        self._level = clamped
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.fillRect(rect, QtGui.QColor(8, 14, 18))
        painter.setPen(QtGui.QPen(QtGui.QColor(56, 72, 88), 1))
        painter.drawRect(rect)

        if self._level <= 0.0:
            return

        fill_height = max(2, int(round(rect.height() * self._level)))
        fill_rect = QtCore.QRect(rect.left() + 1, rect.bottom() - fill_height + 1, max(1, rect.width() - 1), fill_height)
        gradient = QtGui.QLinearGradient(fill_rect.bottomLeft(), fill_rect.topLeft())
        gradient.setColorAt(0.0, QtGui.QColor(48, 220, 122))
        gradient.setColorAt(0.65, QtGui.QColor(232, 206, 64))
        gradient.setColorAt(1.0, QtGui.QColor(255, 96, 96))
        painter.fillRect(fill_rect, gradient)


class MixerChannelStrip(QtWidgets.QFrame):
    def __init__(self, row: int, on_change_callable=None, on_select_callable=None) -> None:
        super().__init__()
        self._row = int(row)
        self._on_change = on_change_callable
        self._on_select = on_select_callable
        self._loading = False
        self._selected = False
        self._accent = QtGui.QColor('#4AB4FF')
        self.setObjectName('mixerChannelStrip')
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setMinimumWidth(134)
        self.setMaximumWidth(158)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.accent_bar = QtWidgets.QFrame()
        self.accent_bar.setFixedHeight(4)
        layout.addWidget(self.accent_bar)

        self.index_label = QtWidgets.QLabel()
        self.index_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.index_label.setStyleSheet('font-size: 10px; color: #8DA2B8; letter-spacing: 0.6px;')
        layout.addWidget(self.index_label)

        self.name_label = QtWidgets.QLabel()
        self.name_label.setWordWrap(True)
        self.name_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.name_label.setStyleSheet('font-size: 12px; font-weight: 600; color: #E9EEF5;')
        layout.addWidget(self.name_label)

        self.subtitle_label = QtWidgets.QLabel()
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setStyleSheet('font-size: 10px; color: #92A0B1;')
        layout.addWidget(self.subtitle_label)

        button_row = QtWidgets.QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(6)
        self.mute_btn = QtWidgets.QToolButton()
        self.mute_btn.setText('M')
        self.mute_btn.setCheckable(True)
        self.mute_btn.setFixedSize(32, 24)
        self.solo_btn = QtWidgets.QToolButton()
        self.solo_btn.setText('S')
        self.solo_btn.setCheckable(True)
        self.solo_btn.setFixedSize(32, 24)
        button_row.addStretch(1)
        button_row.addWidget(self.mute_btn)
        button_row.addWidget(self.solo_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        center_row = QtWidgets.QHBoxLayout()
        center_row.setContentsMargins(0, 0, 0, 0)
        center_row.setSpacing(8)
        self.level_meter = MixerLevelMeterWidget()
        center_row.addWidget(self.level_meter, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)

        self.volume = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self.volume.setRange(0, 100)
        self.volume.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBothSides)
        self.volume.setTickInterval(10)
        self.volume.setSingleStep(1)
        self.volume.setPageStep(5)
        self.volume.setMinimumHeight(220)
        self.volume.setStyleSheet(
            'QSlider::groove:vertical { background: #1A2631; border: 1px solid #314456; width: 18px; border-radius: 4px; }'
            'QSlider::sub-page:vertical { background: #C9D3DF; border-radius: 3px; }'
            'QSlider::add-page:vertical { background: #10202B; border-radius: 3px; }'
            'QSlider::handle:vertical { background: #E7F0FA; border: 1px solid #4C667F; height: 20px; margin: 0 -6px; border-radius: 6px; }'
        )
        center_row.addWidget(self.volume, 1)
        layout.addLayout(center_row, 1)

        self.volume_label = QtWidgets.QLabel()
        self.volume_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.volume_label.setStyleSheet('font-size: 11px; color: #D7E1EC;')
        layout.addWidget(self.volume_label)

        pan_label = QtWidgets.QLabel('Pan')
        pan_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        pan_label.setStyleSheet('font-size: 10px; color: #92A0B1;')
        layout.addWidget(pan_label)

        self.pan = KnobInput(-100, 100, 0, '', dial_size=34)
        layout.addWidget(self.pan, 0, QtCore.Qt.AlignmentFlag.AlignCenter)

        self.mode_label = QtWidgets.QLabel()
        self.mode_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.mode_label.setStyleSheet('font-size: 10px; color: #AAB6C4;')
        self.mode_label.setWordWrap(True)
        layout.addWidget(self.mode_label)

        layout.addStretch(1)

        self.volume.valueChanged.connect(self._emit_change)
        self.pan.valueChanged.connect(self._emit_change)
        self.mute_btn.toggled.connect(self._emit_change)
        self.solo_btn.toggled.connect(self._emit_change)
        self._apply_style()

    def _apply_style(self) -> None:
        border = self._accent.name() if self._selected else '#2A3645'
        background = '#2A3240' if self._selected else '#202732'
        self.setStyleSheet(
            f'QFrame#mixerChannelStrip {{ background: {background}; border: 1px solid {border}; border-radius: 8px; }}'
            'QToolButton { background: #1A232E; border: 1px solid #3A4C60; border-radius: 5px; color: #E8EEF6; font-weight: 700; }'
            'QToolButton:checked { background: #E8EEF6; color: #11161D; border-color: #A8BACF; }'
        )
        self.accent_bar.setStyleSheet(f'background: {self._accent.name()}; border-radius: 2px;')

    def _emit_change(self, *_args) -> None:
        if self._loading:
            return
        if callable(self._on_change):
            self._on_change(self._row)
        self.volume_label.setText(f'{int(self.volume.value())}%')

    def set_track(self, track: TrackState, *, selected: bool, level: float, color: QtGui.QColor) -> None:
        self._loading = True
        try:
            self._selected = bool(selected)
            self._accent = QtGui.QColor(color)
            self.index_label.setText(f'TRACK {self._row + 1:02d}')
            self.name_label.setText(track.name)
            subtitle = track.instrument if track.instrument_mode != 'Sample' else 'Sample Track'
            self.subtitle_label.setText(subtitle)
            self.volume.setValue(int(round(float(track.volume) * 100.0)))
            self.pan.setValue(int(round(float(track.pan) * 100.0)))
            self.mute_btn.setChecked(bool(track.mute))
            self.solo_btn.setChecked(bool(track.solo))
            self.level_meter.setLevel(level)
            self.volume_label.setText(f'{int(self.volume.value())}%')
            mode_text = track.instrument_mode
            if track.track_type == 'sample':
                mode_text = 'Sample'
            self.mode_label.setText(mode_text)
            self._apply_style()
        finally:
            self._loading = False

    def set_level(self, level: float) -> None:
        self.level_meter.setLevel(level)

    def apply_to_track(self, track: TrackState) -> None:
        track.volume = self.volume.value() / 100.0
        track.pan = self.pan.value() / 100.0
        track.mute = self.mute_btn.isChecked()
        track.solo = self.solo_btn.isChecked()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and callable(self._on_select):
            self._on_select(self._row)
        super().mousePressEvent(event)


class MixerWidget(QtWidgets.QWidget):
    def __init__(
        self,
        project: ProjectState,
        current_track_callable,
        available_fx_callable=None,
        on_track_updated_callable=None,
        select_track_callable=None,
        meter_levels_callable=None,
    ) -> None:
        super().__init__()
        self.project = project
        self.current_track_callable = current_track_callable
        self.available_fx = available_fx_callable
        self.on_track_updated = on_track_updated_callable
        self.select_track_callable = select_track_callable
        self.meter_levels_callable = meter_levels_callable or (lambda: {})
        self._strips: list[MixerChannelStrip] = []

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        header = QtWidgets.QLabel('Mixer')
        header.setStyleSheet('font-size: 16px; font-weight: 700; color: #E9EEF5;')
        root.addWidget(header)

        subheader = QtWidgets.QLabel('All tracks are shown as channel strips with meter, fader, and pan.')
        subheader.setStyleSheet('font-size: 11px; color: #94A4B7;')
        root.addWidget(subheader)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.strip_container = QtWidgets.QWidget()
        self.strip_layout = QtWidgets.QHBoxLayout(self.strip_container)
        self.strip_layout.setContentsMargins(0, 0, 0, 0)
        self.strip_layout.setSpacing(10)
        self.strip_layout.addStretch(1)
        self.scroll.setWidget(self.strip_container)
        root.addWidget(self.scroll, 1)

        self._meter_timer = QtCore.QTimer(self)
        self._meter_timer.setInterval(33)
        self._meter_timer.timeout.connect(self.refresh_meters)
        self._meter_timer.start()

    def _selected_index(self) -> int:
        if not self.project.tracks:
            return -1
        try:
            selected_track = self.current_track_callable()
        except Exception:
            return -1
        for idx, track in enumerate(self.project.tracks):
            if track is selected_track:
                return idx
        return -1

    def _meter_levels(self) -> dict[int, float]:
        try:
            raw = self.meter_levels_callable()
        except Exception:
            return {}
        if isinstance(raw, dict):
            return {int(key): max(0.0, min(1.0, float(value))) for key, value in raw.items()}
        return {}

    def _on_strip_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.project.tracks):
            return
        self._strips[row].apply_to_track(self.project.tracks[row])
        if callable(self.on_track_updated):
            self.on_track_updated(row)

    def _select_track(self, row: int) -> None:
        if callable(self.select_track_callable):
            self.select_track_callable(int(row))
        self.load_track()

    def _ensure_strip_count(self, count: int) -> None:
        while len(self._strips) < count:
            row = len(self._strips)
            strip = MixerChannelStrip(row, self._on_strip_changed, self._select_track)
            self._strips.append(strip)
            self.strip_layout.insertWidget(max(0, self.strip_layout.count() - 1), strip)
        while len(self._strips) > count:
            strip = self._strips.pop()
            self.strip_layout.removeWidget(strip)
            strip.deleteLater()

    def load_track(self) -> None:
        self._ensure_strip_count(len(self.project.tracks))
        levels = self._meter_levels()
        selected_index = self._selected_index()
        for row, track in enumerate(self.project.tracks):
            color = track_display_color(track, row)
            level = levels.get(row, 0.0)
            self._strips[row].set_track(track, selected=(row == selected_index), level=level, color=color)

    def refresh_meters(self) -> None:
        if not self._strips:
            return
        levels = self._meter_levels()
        for row, strip in enumerate(self._strips):
            strip.set_level(levels.get(row, 0.0))

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self.load_track()


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
        self._updating_ui = False

        root = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.instrument_mode = QtWidgets.QComboBox()
        self.instrument_mode.addItems(["General MIDI", "VSTI Rack"])
        self.instrument = QtWidgets.QComboBox()
        self.instrument.addItems(["Piano", "Chromatic", "Organ", "Guitar", "Bass", "Strings", "Brass", "Reed", "Pipe", "Lead", "Pad", "Percussive"])
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

        self.fx_controls: dict[str, KnobInput] = {}
        fx_row_widget = QtWidgets.QWidget()
        fx_row_layout = QtWidgets.QHBoxLayout(fx_row_widget)
        fx_row_layout.setContentsMargins(0, 0, 0, 0)
        fx_row_layout.setSpacing(10)
        for fx in ["EQ", "Compression", "Distortion", "Phaser", "Flanger", "Delay", "Reverb"]:
            knob = KnobInput(0, 100, 30, '%', dial_size=34)
            self.fx_controls[fx] = knob
            fx_cell = QtWidgets.QWidget()
            fx_cell_layout = QtWidgets.QVBoxLayout(fx_cell)
            fx_cell_layout.setContentsMargins(0, 0, 0, 0)
            fx_cell_layout.setSpacing(4)
            fx_label = QtWidgets.QLabel(fx)
            fx_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            fx_label.setStyleSheet('font-size: 11px;')
            fx_cell_layout.addWidget(fx_label)
            fx_cell_layout.addWidget(knob, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
            fx_row_layout.addWidget(fx_cell)
        fx_row_layout.addStretch(1)
        form.addRow("Built-in FX", fx_row_widget)

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
        for knob in self.fx_controls.values():
            knob.valueChanged.connect(self.apply_changes)
        self._update_vsti_controls()

    def _instrument_entries(self) -> list[VSTInstrument]:
        return [entry for entry in self.project.vsti_rack if entry.host_supported and entry.is_instrument]

    def _update_vsti_controls(self, track: TrackState | None = None) -> None:
        active_track = track or self.current_track_callable()
        active_entry = None
        if active_track.rack_vsti and hasattr(self, 'project'):
            active_entry = next((entry for entry in self.project.vsti_rack if entry.name == active_track.rack_vsti), None)
        vst_enabled = active_track.track_type == 'instrument'
        use_vsti = vst_enabled and active_track.instrument_mode == 'VSTI Rack'
        self.instrument_mode.setEnabled(active_track.track_type == 'instrument')
        self.instrument.setEnabled(active_track.track_type == 'instrument' and not use_vsti)
        self.vsti_selector.setEnabled(use_vsti)
        self.assign_rack_btn.setEnabled(use_vsti)
        self.load_vsti_btn.setEnabled(use_vsti)
        self.open_vsti_gui_btn.setEnabled(use_vsti and bool(active_track.rack_vsti) and (active_entry is None or active_entry.host_supported))
        self.edit_vsti_params_btn.setEnabled(use_vsti and bool(active_track.rack_vsti) and (active_entry is None or active_entry.host_supported))
        if not self._instrument_entries():
            self.vsti_selector.setToolTip('Add at least one supported VST3 instrument plugin to the rack.')
        elif not PEDALBOARD_AVAILABLE:
            self.vsti_selector.setToolTip(pedalboard_runtime_hint())
        elif active_entry is not None and not active_entry.host_supported:
            self.vsti_selector.setToolTip(active_entry.host_error or 'This rack plugin is not supported by the current VST backend.')
        else:
            self.vsti_selector.setToolTip('Choose a VST instrument from the rack.')

    def reload_vsti_choices(self) -> None:
        current = self.vsti_selector.currentText()
        self.vsti_selector.blockSignals(True)
        self.vsti_selector.clear()
        self.vsti_selector.addItem('None')
        for vst in self._instrument_entries():
            self.vsti_selector.addItem(vst.name)
        idx = self.vsti_selector.findText(current)
        if idx >= 0:
            self.vsti_selector.setCurrentIndex(idx)
        self.vsti_selector.blockSignals(False)

    def load_track(self) -> None:
        self._updating_ui = True
        try:
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
            elif track.rack_vsti:
                self.vsti_selector.setCurrentText('None')

            self.midi_channel.setValue(int(track.midi_channel) + 1)
            self.midi_program.setValue(int(track.midi_program))

            self.instrument_mode.blockSignals(False)
            self.instrument.blockSignals(False)
            self.vsti_selector.blockSignals(False)
            self.midi_channel.blockSignals(False)
            self.midi_program.blockSignals(False)
            self.profile.setText(track.synth_profile)
            self._update_vsti_controls(track)
        finally:
            self._updating_ui = False

    def apply_changes(self) -> None:
        if self._updating_ui:
            return
        track = self.current_track_callable()
        previous_rack_vsti = track.rack_vsti
        if track.track_type == 'instrument':
            track.instrument_mode = self.instrument_mode.currentText()
        else:
            track.instrument_mode = 'Sample'

        use_vsti = track.track_type == 'instrument' and track.instrument_mode == 'VSTI Rack'
        track.rack_vsti = self.vsti_selector.currentText() if use_vsti and self.vsti_selector.currentText() != 'None' else ''
        if track.rack_vsti != previous_rack_vsti:
            track.vsti_parameters = {}
            track.vsti_state_path = ''
        track.instrument = track.rack_vsti if track.rack_vsti else self.instrument.currentText()

        sender = self.sender()
        if sender is self.instrument and not track.rack_vsti:
            default_program = self._default_gm_program(track.instrument)
            self.midi_program.blockSignals(True)
            self.midi_program.setValue(default_program)
            self.midi_program.blockSignals(False)

        track.midi_channel = int(self.midi_channel.value()) - 1
        track.midi_program = int(self.midi_program.value())
        if track.rack_vsti:
            track.synth_profile = 'vst_instrument'
        else:
            track.synth_profile = self._infer_synth_profile(track.instrument, track.midi_program)
        self.profile.setText(track.synth_profile)
        track.plugins = [f"{name}:{slider.value()}" for name, slider in self.fx_controls.items()]
        self._update_vsti_controls(track)
        if callable(self.on_track_updated):
            self.on_track_updated()

    @staticmethod
    def _default_gm_program(instrument_name: str) -> int:
        normalized = instrument_name.strip().lower()
        defaults = {
            'piano': 0,
            'chromatic': 8,
            'organ': 16,
            'guitar': 24,
            'bass': 32,
            'strings': 40,
            'brass': 56,
            'reed': 64,
            'pipe': 72,
            'lead': 80,
            'pad': 88,
            'percussive': 112,
            'ensemble': 48,
            'fx': 96,
            'ethnic': 104,
            'sfx': 120,
        }
        for token, program in defaults.items():
            if token in normalized:
                return program
        return 0

    @staticmethod
    def _infer_synth_profile(instrument_name: str, midi_program: int) -> str:
        normalized = instrument_name.strip().lower()
        category_profile_map = {
            'piano': 'e_piano',
            'chromatic': 'e_piano',
            'organ': 'organ',
            'guitar': 'pluck',
            'bass': 'sub_bass',
            'strings': 'saw_pad',
            'brass': 'brass_stack',
            'reed': 'reed_breath',
            'pipe': 'reed_breath',
            'lead': 'synth',
            'pad': 'saw_pad',
            'percussive': 'noise_kit',
            'ensemble': 'saw_pad',
            'fx': 'synth',
            'ethnic': 'pluck',
            'sfx': 'synth',
        }
        for token, profile in category_profile_map.items():
            if token in normalized:
                return profile

        program = int(clamp(midi_program, 0, 127))
        if program < 16:
            return 'e_piano'
        if program < 24:
            return 'organ'
        if program < 32:
            return 'pluck'
        if program < 40:
            return 'sub_bass'
        if program < 56:
            return 'saw_pad'
        if program < 64:
            return 'brass_stack'
        if program < 80:
            return 'reed_breath'
        if program < 96:
            return 'synth'
        if program < 104:
            return 'synth'
        if program < 120:
            return 'noise_kit'
        return 'synth'


    def assign_selected_rack_vsti(self) -> None:
        if self.vsti_selector.currentText() == 'None':
            QtWidgets.QMessageBox.information(self, 'No VSTI selected', 'Choose a rack instrument first.')
            return
        self.instrument_mode.setCurrentText('VSTI Rack')
        self.apply_changes()

    def load_selected_vsti_binary(self) -> None:
        if self.vsti_selector.currentText() == 'None':
            QtWidgets.QMessageBox.information(self, 'No VSTI selected', 'Choose a rack instrument first.')
            return
        if callable(self.load_selected_vsti):
            self.load_selected_vsti(self.vsti_selector.currentText())

    def open_selected_vsti_gui(self) -> None:
        if self.vsti_selector.currentText() == 'None':
            QtWidgets.QMessageBox.information(self, 'No VSTI selected', 'Choose a rack instrument first.')
            return
        if callable(self.open_vsti_gui):
            self.open_vsti_gui(self.vsti_selector.currentText())

    def edit_vsti_parameters(self) -> None:
        self.open_selected_vsti_gui()


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
        self.access_token_input = QtWidgets.QLineEdit()
        self.access_token_input.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.access_token_input.setPlaceholderText('Paste OpenAI access token here (no client_id required)')
        self.client_id_input = QtWidgets.QLineEdit(os.getenv('OPENAI_OAUTH_CLIENT_ID', ''))
        self.auth_url_input = QtWidgets.QLineEdit(os.getenv('OPENAI_OAUTH_AUTHORIZE_URL', 'https://auth.openai.com/oauth/authorize'))
        self.token_url_input = QtWidgets.QLineEdit(os.getenv('OPENAI_OAUTH_TOKEN_URL', 'https://auth.openai.com/oauth/token'))
        self.redirect_uri_input = QtWidgets.QLineEdit(os.getenv('OPENAI_OAUTH_REDIRECT_URI', 'http://127.0.0.1:8765/callback'))
        self.scope_input = QtWidgets.QLineEdit(os.getenv('OPENAI_OAUTH_SCOPE', 'openid profile offline_access'))
        self.auth_code_input = QtWidgets.QLineEdit()
        self.auth_code_input.setPlaceholderText('Optional: paste authorization code from redirect URL')
        oauth_form.addRow('Access token', self.access_token_input)
        oauth_form.addRow('Client ID (optional)', self.client_id_input)
        oauth_form.addRow('Authorize URL', self.auth_url_input)
        oauth_form.addRow('Token URL', self.token_url_input)
        oauth_form.addRow('Redirect URI', self.redirect_uri_input)
        oauth_form.addRow('Scope', self.scope_input)
        oauth_form.addRow('Authorization code (optional)', self.auth_code_input)

        oauth_buttons = QtWidgets.QHBoxLayout()
        self.open_browser_btn = QtWidgets.QPushButton('Open OAuth Login (Advanced)')
        self.open_browser_btn.clicked.connect(self.open_oauth_login)
        oauth_buttons.addWidget(self.open_browser_btn)
        tabs.addTab(oauth_tab, 'OAuth / Access Token')
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
        client_id = self.client_id_input.text().strip()
        auth_url = self.auth_url_input.text().strip()
        redirect_uri = self.redirect_uri_input.text().strip()
        scope = self.scope_input.text().strip()

        if not auth_url or not redirect_uri:
            msg = 'OAuth authorize URL and redirect URI are required.'
            self.status_label.setText(msg)
            QtWidgets.QMessageBox.warning(self, 'Missing OAuth configuration', msg)
            return
        if not client_id:
            msg = 'Advanced OAuth login requires Client ID. For simple setup, paste an access token and click Connect.'
            self.status_label.setText(msg)
            QtWidgets.QMessageBox.warning(self, 'Missing OAuth client_id', msg)
            return

        self.code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(48)).decode().rstrip('=')
        challenge = base64.urlsafe_b64encode(hashlib.sha256(self.code_verifier.encode()).digest()).decode().rstrip('=')
        params = {
            'response_type': 'code',
            'redirect_uri': redirect_uri,
            'scope': scope,
            'code_challenge': challenge,
            'code_challenge_method': 'S256',
            'state': secrets.token_urlsafe(16),
        }
        params['client_id'] = client_id
        url = f"{auth_url}?{urllib.parse.urlencode(params)}"
        webbrowser.open(url)
        self.status_label.setText('Browser opened. Prefer pasting an access token directly. Use auth code exchange only if your OAuth app requires it.')

    def auth_payload(self) -> dict:
        return {
            'mode': 'api_key' if self.tabs.currentIndex() == 0 else 'oauth',
            'api_key': self.api_key_input.text().strip(),
            'access_token': self.access_token_input.text().strip(),
            'client_id': self.client_id_input.text().strip(),
            'token_url': self.token_url_input.text().strip(),
            'redirect_uri': self.redirect_uri_input.text().strip(),
            'auth_code': self.auth_code_input.text().strip(),
            'code_verifier': self.code_verifier,
        }


class FloatingPanelWindow(QtWidgets.QMainWindow):
    visibilityChanged = QtCore.Signal(bool)

    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None, *, always_on_top: bool = False) -> None:
        flags = QtCore.Qt.WindowType.Tool
        if always_on_top:
            flags |= QtCore.Qt.WindowType.WindowStaysOnTopHint
        super().__init__(parent, flags)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setWindowTitle(title)
        self.resize(1180, 520)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        event.ignore()
        self.hide()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self.visibilityChanged.emit(True)

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        super().hideEvent(event)
        self.visibilityChanged.emit(False)


class VirtualPianoKeyboardWidget(QtWidgets.QWidget):
    noteTriggered = QtCore.Signal(int)

    _NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def __init__(self, key_specs: list[tuple[int, str, list[str]]], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setMouseTracking(True)
        self._key_scale = 0.5
        self._pressed_pitch: int | None = None
        self._pressed_timer = QtCore.QTimer(self)
        self._pressed_timer.setSingleShot(True)
        self._pressed_timer.timeout.connect(self._clear_pressed_pitch)
        self._key_specs: list[dict[str, object]] = [
            {
                'pitch': int(pitch),
                'primary': str(primary),
                'aliases': [str(alias) for alias in aliases],
                'is_black': int(pitch) % 12 in BLACK_KEY_PITCH_CLASSES,
                'rect': QtCore.QRectF(),
            }
            for pitch, primary, aliases in key_specs
        ]
        self._white_keys = [spec for spec in self._key_specs if not bool(spec['is_black'])]
        self._black_keys = [spec for spec in self._key_specs if bool(spec['is_black'])]
        self._update_widget_size()
        self._rebuild_key_geometry()

    def minimumSizeHint(self) -> QtCore.QSize:
        return self._natural_size()

    def sizeHint(self) -> QtCore.QSize:
        return self._natural_size()

    def key_scale(self) -> float:
        return float(self._key_scale)

    def set_key_scale(self, scale: float) -> None:
        new_scale = max(0.35, min(1.75, float(scale)))
        if abs(new_scale - self._key_scale) < 0.001:
            return
        self._key_scale = new_scale
        self._update_widget_size()
        self._rebuild_key_geometry()

    def _natural_size(self) -> QtCore.QSize:
        white_width = 54.0 * self._key_scale
        white_height = 184.0 * self._key_scale
        width = int(math.ceil((white_width * max(1, len(self._white_keys))) + 24.0))
        height = int(math.ceil(white_height + 24.0))
        return QtCore.QSize(max(280, width), max(96, height))

    def _update_widget_size(self) -> None:
        size = self._natural_size()
        self.setMinimumSize(size)
        self.resize(size)
        self.updateGeometry()

    def _note_name(self, pitch: int) -> str:
        return self._NOTE_NAMES[int(pitch) % 12]

    def _shortcut_text(self, spec: dict[str, object]) -> str:
        labels = [str(spec['primary']), *[str(alias) for alias in spec['aliases']]]
        return ' / '.join(label for label in labels if label)

    def _rebuild_key_geometry(self) -> None:
        rect = QtCore.QRectF(self.contentsRect()).adjusted(12, 12, -12, -12)
        if rect.width() <= 0.0 or rect.height() <= 0.0 or not self._white_keys:
            return
        natural = self._natural_size()
        natural_width = max(1.0, float(natural.width() - 24))
        natural_height = max(1.0, float(natural.height() - 24))
        left = rect.x() + max(0.0, (rect.width() - natural_width) / 2.0)
        top = rect.y() + max(0.0, (rect.height() - natural_height) / 2.0)
        white_width = natural_width / max(1, len(self._white_keys))
        white_height = natural_height
        black_width = white_width * 0.62
        black_height = white_height * 0.62
        white_cursor = 0
        for spec in self._key_specs:
            pitch = int(spec['pitch'])
            if pitch % 12 in BLACK_KEY_PITCH_CLASSES:
                x_center = left + (white_cursor * white_width)
                spec['rect'] = QtCore.QRectF(x_center - (black_width / 2.0), top, black_width, black_height)
            else:
                spec['rect'] = QtCore.QRectF(left + (white_cursor * white_width), top, white_width + 0.5, white_height)
                white_cursor += 1
        self.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._rebuild_key_geometry()

    def _color_for_key(self, spec: dict[str, object]) -> QtGui.QLinearGradient:
        rect = QtCore.QRectF(spec['rect'])
        gradient = QtGui.QLinearGradient(rect.topLeft(), rect.bottomLeft())
        is_black = bool(spec['is_black'])
        pitch = int(spec['pitch'])
        active = self._pressed_pitch == pitch
        if is_black:
            if active:
                gradient.setColorAt(0.0, QtGui.QColor(78, 170, 255))
                gradient.setColorAt(1.0, QtGui.QColor(20, 88, 180))
            else:
                gradient.setColorAt(0.0, QtGui.QColor(72, 82, 96))
                gradient.setColorAt(1.0, QtGui.QColor(18, 22, 28))
        else:
            if active:
                gradient.setColorAt(0.0, QtGui.QColor(202, 235, 255))
                gradient.setColorAt(1.0, QtGui.QColor(112, 186, 255))
            else:
                gradient.setColorAt(0.0, QtGui.QColor(252, 252, 252))
                gradient.setColorAt(1.0, QtGui.QColor(208, 214, 224))
        return gradient

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor(20, 22, 28))

        for spec in self._white_keys:
            rect = QtCore.QRectF(spec['rect'])
            painter.setPen(QtGui.QPen(QtGui.QColor(72, 78, 88), 1.0))
            painter.setBrush(QtGui.QBrush(self._color_for_key(spec)))
            painter.drawRoundedRect(rect, 4.0, 4.0)
            painter.setPen(QtGui.QColor(44, 48, 56))
            painter.setFont(QtGui.QFont('Segoe UI', max(6, int(round(8 * self._key_scale))), QtGui.QFont.Weight.Medium))
            painter.drawText(rect.adjusted(0, rect.height() - (34 * self._key_scale), 0, -(16 * self._key_scale)), QtCore.Qt.AlignmentFlag.AlignCenter, self._note_name(int(spec['pitch'])))
            painter.setPen(QtGui.QColor(96, 104, 118))
            painter.setFont(QtGui.QFont('Segoe UI', max(6, int(round(7 * self._key_scale)))))
            painter.drawText(rect.adjusted(0, rect.height() - (18 * self._key_scale), 0, -4), QtCore.Qt.AlignmentFlag.AlignCenter, str(spec['primary']))

        for spec in self._black_keys:
            rect = QtCore.QRectF(spec['rect'])
            painter.setPen(QtGui.QPen(QtGui.QColor(12, 14, 18), 1.0))
            painter.setBrush(QtGui.QBrush(self._color_for_key(spec)))
            painter.drawRoundedRect(rect, 4.0, 4.0)
            painter.setPen(QtGui.QColor(232, 240, 248))
            painter.setFont(QtGui.QFont('Segoe UI', max(6, int(round(7 * self._key_scale))), QtGui.QFont.Weight.Medium))
            painter.drawText(rect.adjusted(0, rect.height() - (26 * self._key_scale), 0, -(12 * self._key_scale)), QtCore.Qt.AlignmentFlag.AlignCenter, self._note_name(int(spec['pitch'])))
            painter.setPen(QtGui.QColor(176, 188, 204))
            painter.setFont(QtGui.QFont('Segoe UI', max(6, int(round(6 * self._key_scale)))))
            painter.drawText(rect.adjusted(0, rect.height() - (14 * self._key_scale), 0, -3), QtCore.Qt.AlignmentFlag.AlignCenter, str(spec['primary']))

    def _hit_test(self, pos: QtCore.QPointF) -> dict[str, object] | None:
        for spec in self._black_keys:
            if QtCore.QRectF(spec['rect']).contains(pos):
                return spec
        for spec in self._white_keys:
            if QtCore.QRectF(spec['rect']).contains(pos):
                return spec
        return None

    def _clear_pressed_pitch(self) -> None:
        if self._pressed_pitch is None:
            return
        self._pressed_pitch = None
        self.update()

    def flash_pitch(self, pitch: int) -> None:
        self._pressed_pitch = int(pitch)
        self._pressed_timer.start(140)
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        spec = self._hit_test(event.position())
        if spec is None:
            super().mousePressEvent(event)
            return
        pitch = int(spec['pitch'])
        self.flash_pitch(pitch)
        self.noteTriggered.emit(pitch)
        event.accept()


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
        self.vsti_description_cache: dict[str, tuple[str, bool, bool, str, bool, str]] = {}
        self._vsti_worker_pool = QtCore.QThreadPool(self)
        self._vsti_worker_pool.setMaxThreadCount(2)
        self._vsti_background_loads_inflight: set[str] = set()
        self._active_vsti_workers: dict[str, VSTLoadWorker] = {}
        self.vsti_directory = BUNDLED_VSTI_DIR
        self.user_vsti_directory = USER_VSTI_DIR
        self.user_vsti_directory.mkdir(parents=True, exist_ok=True)
        if not getattr(sys, "frozen", False):
            self.vsti_directory.mkdir(parents=True, exist_ok=True)
        self.selected_audio_output_id = ''
        self.audio_buffer_ms = 80
        self.playback_ui_refresh_ms = 16
        self.prefer_gpu_rendering = True
        self._main_splitter_sizes = [320, 1180]
        self._note_editor_inner_sizes = [640, 160]
        self._tools_window_visible = True
        self._tools_window_geometry_b64 = ''
        self._mixer_window_visible = True
        self._mixer_window_geometry_b64 = ''
        self._transport_window_visible = True
        self._transport_window_geometry_b64 = ''
        self._virtual_piano_window_visible = False
        self._virtual_piano_window_geometry_b64 = ''
        self._virtual_piano_key_scale_percent = 50
        self._virtual_piano_shortcuts: list[QtGui.QShortcut] = []
        self._layout_save_timer = QtCore.QTimer(self)
        self._layout_save_timer.setSingleShot(True)
        self._layout_save_timer.timeout.connect(self._save_preferences)
        self._load_preferences()
        self._sync_bundled_vsti_directory()
        self._apply_selected_audio_output()
        self._apply_audio_buffer_preference()
        self.playback_mix_path = RENDER_DIR / "_playback_mix.wav"
        self.note_preview_path = RENDER_DIR / "_preview" / "note_preview.wav"
        self._playback_sink: QtMultimedia.QAudioSink | None = None
        self._playback_sink_device: QtCore.QIODevice | None = None
        self._preview_sink: QtMultimedia.QAudioSink | None = None
        self._preview_buffer_device: QtCore.QBuffer | None = None
        self._preview_resources: list[tuple[QtMultimedia.QAudioSink, QtCore.QBuffer]] = []
        self._playback_mix_wav_bytes = b''
        self._playback_frame_position = 0
        self._playback_generated_total_frames = 0
        self._playback_logical_origin_frame = 0
        self._playback_chunk_frames = 1024
        self._playback_sample_rate = 44100
        self._playback_channel_count = 2
        self._playback_active = False
        self._realtime_mix_cache: object | None = None
        self._realtime_mix_cache_start_frame = 0
        self._realtime_mix_cache_frame_count = 0
        self._realtime_track_states: dict[int, RealtimeTrackPlaybackState] = {}
        self._track_meter_levels: dict[int, float] = {}
        self._sample_audio_cache: dict[str, tuple[object, int, int]] = {}
        self._realtime_reset_pending = False
        self._playback_mix_cache_key = ''
        self._playback_mix_duration_sec = 0.0
        self._track_playback_audio_cache: dict[tuple[int, str], tuple[object, int]] = {}
        self._cleanup_legacy_playback_files()
        self._playback_loop_ms = 0
        self._deferred_note_refresh_timer = QtCore.QTimer(self)
        self._deferred_note_refresh_timer.setSingleShot(True)
        self._deferred_note_refresh_timer.timeout.connect(self._flush_deferred_note_refresh)
        self._deferred_refresh_velocity = False
        self._deferred_refresh_timeline = False
        self._deferred_rebuild_sections = False
        self._deferred_refresh_arrangement = False
        self._deferred_reload_mix = False
        self._tempo_ui_refresh_timer = QtCore.QTimer(self)
        self._tempo_ui_refresh_timer.setSingleShot(True)
        self._tempo_ui_refresh_timer.timeout.connect(self._flush_tempo_ui_refresh)
        self._tempo_refresh_seconds_layout = False
        self._tempo_refresh_arrangement = False
        self._tempo_refresh_timeline = False
        self._audio_pump_timer = QtCore.QTimer(self)
        self._audio_pump_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._audio_pump_timer.setInterval(5)
        self._audio_pump_timer.timeout.connect(self._pump_realtime_audio)
        self.current_project_path: Path | None = None
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 900)

        self.track_list = QtWidgets.QListWidget()
        self._selected_track_index = 0
        self._track_list_rebuilding = False
        self.track_list.currentRowChanged.connect(self._track_changed)
        self.track_list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.track_list.customContextMenuRequested.connect(self._show_track_context_menu_from_list)
        self.track_list.viewport().installEventFilter(self)
        self.last_added_track_type = 'instrument'

        self.timeline = TimelineWidget(self.project)
        self.piano_roll = PianoRollWidget(self.project, self.current_track_index, self.set_playhead_position, self.set_left_locator_position, self.set_right_locator_position)
        self.piano_roll.noteChanged.connect(self.on_piano_roll_notes_committed)
        self.piano_roll.notePreviewRequested.connect(self.preview_current_track_note)
        self.piano_roll.horizontalZoomChanged.connect(self.set_note_editor_zoom)
        self.velocity_editor = VelocityEditorWidget(self.project, self.current_track_index, lambda: self.piano_roll.ruler_display_mode)
        self.piano_roll.selectionChanged.connect(self.velocity_editor.refresh)
        self.piano_roll.rulerDisplayModeChanged.connect(lambda _mode: self.velocity_editor.refresh())
        self.velocity_editor.velocityChanged.connect(self.on_velocity_editor_changed)
        self.velocity_editor.horizontalZoomChanged.connect(self.set_note_editor_zoom)

        self.mixer = MixerWidget(
            self.project,
            self.current_track,
            self.available_fx_plugin_names,
            self.on_mixer_track_changed,
            self.select_track_by_index,
            self.track_meter_levels,
        )
        self.instruments = InstrumentFxWidget(self.project, self.current_track, self.refresh_vsti_rack_ui, self.on_track_instrument_changed, self.load_vsti_binary_by_name, self.open_vsti_gui_by_name, self.vsti_parameter_names_for_rack)
        self.sample_timeline = SampleTimelineWidget(self.project, self.sample_track_indices, self.place_sample_asset_on_track, self.set_playhead_position)
        self.arrangement_overview = ArrangementOverviewWidget(self.project, self.set_playhead_position, self.set_left_locator_position, self.set_right_locator_position, self.apply_arrangement_section_move, lambda: self.project.bpm)
        self.sample_library = SampleLibraryWidget()

        self.quantize_box = QtWidgets.QComboBox()
        quantize_values = [
            "Off", "1/1", "1/2", "1/2T", "1/4", "1/4T", "1/8", "1/8T", "1/16", "1/16T", "1/32", "1/32T", "1/64", "1/64T"
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
        right_tabs.setMinimumHeight(0)
        right_tabs.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        right_tabs.addTab(self.timeline, "Timeline")
        right_tabs.addTab(self.arrangement_overview, "Arrangement Overview")
        right_tabs.addTab(self.sample_timeline, "Sample Timeline")
        right_tabs.addTab(self.instruments, "Instruments / FX")
        self.right_tabs = right_tabs
        self.tools_window = FloatingPanelWindow('Panels', self)
        self.tools_window.setCentralWidget(self.right_tabs)
        self.tools_window.visibilityChanged.connect(self._on_tools_window_visibility_changed)

        self.mixer_window = FloatingPanelWindow('Mixer', self)
        self.mixer_window.setCentralWidget(self.mixer)
        self.mixer_window.resize(1100, 500)
        self.mixer_window.visibilityChanged.connect(self._on_mixer_window_visibility_changed)

        note_editor = QtWidgets.QWidget()
        note_editor.setMinimumHeight(0)
        note_editor_layout = QtWidgets.QVBoxLayout(note_editor)
        note_editor_layout.setContentsMargins(0, 0, 0, 0)
        note_editor_layout.setSpacing(0)

        velocity_panel = QtWidgets.QWidget()
        velocity_panel.setMinimumHeight(0)
        velocity_panel_layout = QtWidgets.QVBoxLayout(velocity_panel)
        velocity_panel_layout.setContentsMargins(0, 4, 0, 0)
        velocity_panel_layout.setSpacing(4)
        velocity_panel_layout.addWidget(QtWidgets.QLabel('Velocity Editor'))
        velocity_panel_layout.addWidget(self.velocity_editor)

        self.piano_roll.setMinimumHeight(0)
        self.velocity_editor.setMinimumHeight(0)
        self.piano_roll.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.velocity_editor.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        note_editor_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        note_editor_splitter.setHandleWidth(10)
        note_editor_splitter.addWidget(self.piano_roll)
        note_editor_splitter.addWidget(velocity_panel)
        note_editor_splitter.setChildrenCollapsible(False)
        note_editor_splitter.setStretchFactor(0, 4)
        note_editor_splitter.setStretchFactor(1, 1)
        note_editor_splitter.setSizes(self._note_editor_inner_sizes)
        note_editor_splitter.splitterMoved.connect(self._on_layout_splitter_moved)
        self.note_editor_splitter = note_editor_splitter
        note_editor_layout.addWidget(note_editor_splitter)

        splitter_main = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter_main.addWidget(left_panel)
        splitter_main.addWidget(note_editor)
        splitter_main.setSizes(self._main_splitter_sizes)
        splitter_main.splitterMoved.connect(self._on_layout_splitter_moved)
        self.splitter_main = splitter_main

        self.setCentralWidget(splitter_main)
        self._setup_menus()
        self._setup_floating_transport()
        self._setup_virtual_piano_window()
        self._setup_shortcuts()
        self._populate_track_list()
        self.track_list.setCurrentRow(0)
        self.refresh_sample_library()
        self.scan_sample_paths()
        self._apply_tools_window_preferences()
        self._apply_mixer_window_preferences()
        self._update_window_title()

    def _setup_menus(self) -> None:
        file_menu = self.menuBar().addMenu('File')
        new_project_action = QtGui.QAction('New Project', self)
        new_project_action.triggered.connect(self.new_project)
        open_project_action = QtGui.QAction('Open Project...', self)
        open_project_action.triggered.connect(self.load_project)
        save_project_action = QtGui.QAction('Save Project', self)
        save_project_action.triggered.connect(self.save_project)
        save_project_as_action = QtGui.QAction('Save Project As...', self)
        save_project_as_action.triggered.connect(self.save_project_as)
        import_midi_action = QtGui.QAction('Import MIDI + AI Instrument Render', self)
        import_midi_action.triggered.connect(self.import_midi)
        export_midi_action = QtGui.QAction('Export MIDI', self)
        export_midi_action.triggered.connect(self.export_midi)
        export_sequence_wav_action = QtGui.QAction('Export Sequence as WAV...', self)
        export_sequence_wav_action.triggered.connect(self.export_sequence_wav)
        export_audio_action = QtGui.QAction('Export Sample Timeline Audio (WAV/MP3)', self)
        export_audio_action.triggered.connect(self.export_sample_timeline_audio)
        file_menu.addAction(new_project_action)
        file_menu.addAction(open_project_action)
        file_menu.addAction(save_project_action)
        file_menu.addAction(save_project_as_action)
        file_menu.addSeparator()
        file_menu.addAction(import_midi_action)
        file_menu.addSeparator()
        file_menu.addAction(export_sequence_wav_action)
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
        manage_vsti_folders = QtGui.QAction('Manage VSTI Folders', self)
        manage_vsti_folders.triggered.connect(self.manage_vsti_folders)
        instruments_menu.addAction(manage_vsti_folders)
        add_vsti_to_rack = QtGui.QAction('Add Discovered VSTI To Rack', self)
        add_vsti_to_rack.triggered.connect(self.add_discovered_vsti_to_rack)
        instruments_menu.addAction(add_vsti_to_rack)
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

        windows_menu = self.menuBar().addMenu('Windows')
        self.show_panels_window_action = QtGui.QAction('Show Panels Window', self)
        self.show_panels_window_action.setCheckable(True)
        self.show_panels_window_action.setChecked(self._tools_window_visible)
        self.show_panels_window_action.toggled.connect(self.toggle_tools_window)
        windows_menu.addAction(self.show_panels_window_action)
        self.show_mixer_window_action = QtGui.QAction('Show Mixer Window', self)
        self.show_mixer_window_action.setCheckable(True)
        self.show_mixer_window_action.setChecked(self._mixer_window_visible)
        self.show_mixer_window_action.toggled.connect(self.toggle_mixer_window)
        windows_menu.addAction(self.show_mixer_window_action)
        self.show_transport_window_action = QtGui.QAction('Show Transport Window', self)
        self.show_transport_window_action.setCheckable(True)
        self.show_transport_window_action.setChecked(self._transport_window_visible)
        self.show_transport_window_action.toggled.connect(self.toggle_transport_window)
        windows_menu.addAction(self.show_transport_window_action)
        self.show_virtual_piano_window_action = QtGui.QAction('Show Virtual Piano Window', self)
        self.show_virtual_piano_window_action.setCheckable(True)
        self.show_virtual_piano_window_action.setChecked(self._virtual_piano_window_visible)
        self.show_virtual_piano_window_action.toggled.connect(self.toggle_virtual_piano_window)
        windows_menu.addAction(self.show_virtual_piano_window_action)
        self.toggle_transport_shortcut = QtGui.QShortcut(QtGui.QKeySequence('F2'), self)
        self.toggle_transport_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self.toggle_transport_shortcut.activated.connect(self._toggle_transport_window_shortcut)
        windows_menu.addSeparator()
        tile_windows_action = QtGui.QAction('Tile Windows', self)
        tile_windows_action.triggered.connect(self.tile_floating_windows)
        windows_menu.addAction(tile_windows_action)

        performance_menu = settings.addMenu('Playback Performance')
        self.gpu_rendering_action = QtGui.QAction('Prefer GPU Rendering (restart required)', self)
        self.gpu_rendering_action.setCheckable(True)
        self.gpu_rendering_action.setChecked(self.prefer_gpu_rendering)
        self.gpu_rendering_action.triggered.connect(self.set_prefer_gpu_rendering)
        performance_menu.addAction(self.gpu_rendering_action)

        buffer_menu = performance_menu.addMenu('Audio Buffer')
        self.audio_buffer_group = QtGui.QActionGroup(buffer_menu)
        self.audio_buffer_group.setExclusive(True)
        for value in (20, 40, 80, 120, 200):
            action = buffer_menu.addAction(f'{value} ms')
            action.setCheckable(True)
            action.setChecked(value == self.audio_buffer_ms)
            action.triggered.connect(lambda _checked=False, v=value: self.set_audio_buffer_ms(v))
            self.audio_buffer_group.addAction(action)

        refresh_menu = performance_menu.addMenu('Playhead Refresh')
        self.playhead_refresh_group = QtGui.QActionGroup(refresh_menu)
        self.playhead_refresh_group.setExclusive(True)
        for value in (16, 33, 50, 66):
            action = refresh_menu.addAction(f'{value} ms')
            action.setCheckable(True)
            action.setChecked(value == self.playback_ui_refresh_ms)
            action.triggered.connect(lambda _checked=False, v=value: self.set_playback_ui_refresh_ms(v))
            self.playhead_refresh_group.addAction(action)

        self.refresh_vsti_rack_ui()
        self.refresh_openai_status()
        QtCore.QTimer.singleShot(350, self._start_vsti_background_warmup)

    def _setup_floating_transport(self) -> None:
        self.playback_timer = QtCore.QTimer(self)
        self.playback_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.playback_timer.setInterval(self.playback_ui_refresh_ms)
        self.playback_timer.timeout.connect(self._tick_playback)
        self._playback_started_at = 0.0
        self._playback_origin_sec = 0.0
        self._playback_rate = 1.0
        self._last_playhead_ui_refresh = 0.0
        self._playhead_ui_refresh_interval = max(0.001, self.playback_ui_refresh_ms / 1000.0)

        self.transport_window = FloatingPanelWindow('Transport', self, always_on_top=True)
        self.transport_window.resize(980, 92)
        self.transport_window.visibilityChanged.connect(self._on_transport_window_visibility_changed)
        transport_widget = QtWidgets.QWidget()
        transport_layout = QtWidgets.QHBoxLayout(transport_widget)
        transport_layout.setContentsMargins(10, 10, 10, 10)
        transport_layout.setSpacing(8)

        icon_size = QtCore.QSize(20, 20)
        style = self.style()

        def configure_transport_button(
            button: QtWidgets.QToolButton,
            *,
            tooltip: str,
            icon: QtGui.QIcon | None = None,
            text: str = '',
            checkable: bool = False,
        ) -> None:
            button.setToolTip(tooltip)
            button.setCheckable(checkable)
            button.setAutoRaise(False)
            button.setFixedSize(36 if not text else 56, 30)
            button.setIconSize(icon_size)
            if icon is not None:
                button.setIcon(icon)
            if text:
                button.setText(text)
                button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly if icon is None else QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            else:
                button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)

        self.transport_home_btn = QtWidgets.QToolButton()
        configure_transport_button(
            self.transport_home_btn,
            tooltip='Jump to project start',
            icon=style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaSkipBackward),
        )
        self.transport_home_btn.clicked.connect(self.jump_playhead_to_start)

        self.transport_back_btn = QtWidgets.QToolButton()
        configure_transport_button(
            self.transport_back_btn,
            tooltip='Move playhead to the previous bar',
            icon=style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaSeekBackward),
        )
        self.transport_back_btn.clicked.connect(self.skip_to_previous_bar)

        self.transport_play_btn = QtWidgets.QToolButton()
        configure_transport_button(
            self.transport_play_btn,
            tooltip='Start playback',
            icon=style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay),
        )

        self.transport_stop_btn = QtWidgets.QToolButton()
        configure_transport_button(
            self.transport_stop_btn,
            tooltip='Stop playback',
            icon=style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaStop),
        )

        self.transport_forward_btn = QtWidgets.QToolButton()
        configure_transport_button(
            self.transport_forward_btn,
            tooltip='Move playhead to the next bar',
            icon=style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaSeekForward),
        )
        self.transport_forward_btn.clicked.connect(self.skip_to_next_bar)

        self.transport_loop_btn = QtWidgets.QToolButton()
        configure_transport_button(
            self.transport_loop_btn,
            tooltip='Toggle looping between the locators',
            text='Loop',
            checkable=True,
        )
        self.transport_loop_btn.toggled.connect(self.set_loop_enabled)

        self.transport_metronome_btn = QtWidgets.QToolButton()
        configure_transport_button(
            self.transport_metronome_btn,
            tooltip='Toggle metronome click on every beat',
            text='Click',
            checkable=True,
        )
        self.transport_metronome_btn.toggled.connect(self.set_metronome_enabled)

        self.transport_play_btn.clicked.connect(self.start_playback)
        self.transport_stop_btn.clicked.connect(self.stop_playback)
        transport_layout.addWidget(self.transport_home_btn)
        transport_layout.addWidget(self.transport_back_btn)
        transport_layout.addWidget(self.transport_play_btn)
        transport_layout.addWidget(self.transport_stop_btn)
        transport_layout.addWidget(self.transport_forward_btn)
        transport_layout.addWidget(self.transport_loop_btn)
        transport_layout.addWidget(self.transport_metronome_btn)

        self.playhead_spin = QtWidgets.QDoubleSpinBox()
        self.playhead_spin.setRange(0.0, 3600.0)
        self.playhead_spin.setDecimals(2)
        self.playhead_spin.setSingleStep(0.1)
        self.playhead_spin.setValue(self.project.playhead_sec)
        self.playhead_spin.valueChanged.connect(self.set_playhead_position)
        transport_layout.addWidget(QtWidgets.QLabel('Playhead'))
        transport_layout.addWidget(self.playhead_spin)
        self.left_locator = QtWidgets.QDoubleSpinBox()
        self.left_locator.setRange(0.0, 9999.0)
        self.left_locator.setDecimals(2)
        self.right_locator = QtWidgets.QDoubleSpinBox()
        self.right_locator.setRange(0.0, 9999.0)
        self.right_locator.setDecimals(2)
        self.left_locator.valueChanged.connect(self.update_locators)
        self.right_locator.valueChanged.connect(self.update_locators)
        transport_layout.addWidget(QtWidgets.QLabel('L'))
        transport_layout.addWidget(self.left_locator)
        transport_layout.addWidget(QtWidgets.QLabel('R'))
        transport_layout.addWidget(self.right_locator)
        self._refresh_locator_spin_configuration()
        self._set_locator_spin_values(self.project.left_locator_sec, self.project.right_locator_sec)
        self.tempo_spin = QtWidgets.QSpinBox()
        self.tempo_spin.setRange(20, 300)
        self.tempo_spin.setValue(self.project.bpm)
        self.tempo_spin.valueChanged.connect(self.update_tempo)
        transport_layout.addWidget(QtWidgets.QLabel('Tempo'))
        transport_layout.addWidget(self.tempo_spin)
        transport_layout.addStretch(1)
        self.transport_window.setCentralWidget(transport_widget)
        self._refresh_transport_controls()
        self._apply_transport_window_preferences()

    def _update_window_title(self) -> None:
        if self.current_project_path is not None:
            self.setWindowTitle(f"{APP_NAME} - {self.current_project_path.name}")
            return
        self.setWindowTitle(APP_NAME)

    @staticmethod
    def _ensure_project_file_suffix(path: str | Path) -> Path:
        target = Path(path).expanduser()
        if target.suffix:
            return target
        return target.with_suffix(PROJECT_FILE_EXTENSION)

    @staticmethod
    def _coerce_int(value: object, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
        try:
            result = int(value)
        except Exception:
            result = int(default)
        if minimum is not None:
            result = max(minimum, result)
        if maximum is not None:
            result = min(maximum, result)
        return result

    @staticmethod
    def _coerce_float(value: object, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
        try:
            result = float(value)
        except Exception:
            result = float(default)
        if minimum is not None:
            result = max(minimum, result)
        if maximum is not None:
            result = min(maximum, result)
        return result

    def _project_quantize_text(self) -> str:
        if not getattr(self.project, 'quantize_enabled', True):
            return 'Off'
        div = max(1, int(self.project.quantize_div))
        return f"1/{div}{'T' if self.project.quantize_triplet else ''}"

    def _seconds_to_locator_bars(self, sec: float) -> float:
        return max(0.0, float(sec) / max(1e-9, self._bar_duration_seconds()))

    def _locator_bars_to_seconds(self, bars: float) -> float:
        return max(0.0, float(bars)) * self._bar_duration_seconds()

    def _locator_step_bars(self) -> float:
        if not getattr(self.project, 'quantize_enabled', True):
            return 0.01
        beats = 4.0 / max(1, int(self.project.quantize_div))
        if getattr(self.project, 'quantize_triplet', False):
            beats *= 2.0 / 3.0
        return max(0.01, beats / 4.0)

    def _refresh_locator_spin_configuration(self) -> None:
        if hasattr(self, 'left_locator'):
            self.left_locator.setSingleStep(self._locator_step_bars())
            self.left_locator.setToolTip('Left locator position in bars')
        if hasattr(self, 'right_locator'):
            self.right_locator.setSingleStep(self._locator_step_bars())
            self.right_locator.setToolTip('Right locator position in bars')

    @staticmethod
    def _decode_project_blob(value: object) -> bytes:
        if not value:
            return b''
        try:
            return base64.b64decode(str(value).encode('ascii'))
        except Exception:
            return b''

    @staticmethod
    def _encode_track_vsti_state(track: TrackState) -> str:
        if not track.vsti_state_path:
            return ''
        state_path = Path(track.vsti_state_path)
        if not state_path.exists():
            return ''
        try:
            raw_state = state_path.read_bytes()
        except Exception:
            return ''
        if not raw_state:
            return ''
        return base64.b64encode(raw_state).decode('ascii')

    def _project_ui_payload(self) -> dict[str, object]:
        return {
            'selected_track_index': self.current_track_index(),
            'quantize_text': self.quantize_box.currentText().strip() or self._project_quantize_text(),
            'piano_roll_tool': self.piano_roll.tool,
            'piano_roll_note_length_div': int(self.piano_roll.note_length_div),
            'piano_roll_ruler_mode': str(self.piano_roll.ruler_display_mode),
            'piano_roll_cell_w': int(self.piano_roll.cell_w),
            'piano_roll_cell_h': int(self.piano_roll.cell_h),
            'velocity_cell_w': int(self.velocity_editor.cell_w),
            'sample_timeline_pixels_per_second': int(self.sample_timeline.pixels_per_second),
            'arrangement_pixels_per_second': int(self.arrangement_overview.pixels_per_second),
            'panels_tab_index': int(self.right_tabs.currentIndex()),
            'panels_visible': bool(self.tools_window.isVisible()),
            'mixer_visible': bool(self.mixer_window.isVisible()) if hasattr(self, 'mixer_window') else True,
            'transport_visible': bool(self.transport_window.isVisible()),
            'virtual_piano_visible': bool(self.virtual_piano_window.isVisible()) if hasattr(self, 'virtual_piano_window') else True,
        }

    def _project_payload(self) -> dict[str, object]:
        track_payload: list[dict[str, object]] = []
        for track in self.project.tracks:
            serialized = dataclasses.asdict(track)
            serialized['vsti_state_b64'] = self._encode_track_vsti_state(track)
            track_payload.append(serialized)

        return {
            'format': 'ai_music_studio_project',
            'version': PROJECT_FILE_VERSION,
            'saved_at_unix': int(time.time()),
            'project': {
                'bpm': int(self.project.bpm),
                'quantize_enabled': bool(self.project.quantize_enabled),
                'quantize_div': int(self.project.quantize_div),
                'quantize_triplet': bool(self.project.quantize_triplet),
                'loop_enabled': bool(self.project.loop_enabled),
                'metronome_enabled': bool(self.project.metronome_enabled),
                'left_locator_sec': float(self.project.left_locator_sec),
                'right_locator_sec': float(self.project.right_locator_sec),
                'playhead_sec': float(self.project.playhead_sec),
                'tracks': track_payload,
                'vsti_paths': list(self.project.vsti_paths),
                'vsti_folder_paths': list(self.project.vsti_folder_paths),
                'sample_paths': list(self.project.sample_paths),
                'vsti_rack': [dataclasses.asdict(vst) for vst in self.project.vsti_rack],
                'sample_assets': [dataclasses.asdict(asset) for asset in self.project.sample_assets],
                'sample_clips': [dataclasses.asdict(clip) for clip in self.project.sample_clips],
                'midi_sections': [dataclasses.asdict(section) for section in self.project.midi_sections],
            },
            'ui': self._project_ui_payload(),
        }

    def _project_from_payload(self, payload: dict[str, object]) -> tuple[ProjectState, dict[int, bytes], dict[str, object]]:
        project_root = payload.get('project', payload)
        if not isinstance(project_root, dict):
            raise ValueError('The selected file does not contain a valid project payload.')

        project = ProjectState()
        project.bpm = self._coerce_int(project_root.get('bpm'), DEFAULT_BPM, 20, 300)
        project.quantize_enabled = bool(project_root.get('quantize_enabled', True))
        project.quantize_div = self._coerce_int(project_root.get('quantize_div'), 16, 1, 64)
        project.quantize_triplet = bool(project_root.get('quantize_triplet', False))
        project.loop_enabled = bool(project_root.get('loop_enabled', True))
        project.metronome_enabled = bool(project_root.get('metronome_enabled', False))
        default_bar_sec = 4.0 * (60.0 / max(1, project.bpm))
        project.left_locator_sec = self._coerce_float(project_root.get('left_locator_sec'), default_bar_sec, 0.0, 3600.0)
        project.right_locator_sec = self._coerce_float(project_root.get('right_locator_sec'), default_bar_sec * 2.0, 0.0, 3600.0)
        if project.right_locator_sec <= project.left_locator_sec:
            project.right_locator_sec = project.left_locator_sec + default_bar_sec
        project.playhead_sec = self._coerce_float(project_root.get('playhead_sec'), 0.0, 0.0, project.right_locator_sec)

        track_state_blobs: dict[int, bytes] = {}
        tracks: list[TrackState] = []
        for index, raw_track in enumerate(project_root.get('tracks', [])):
            if not isinstance(raw_track, dict):
                continue
            track = TrackState(name=str(raw_track.get('name') or f'Track {index + 1}'))
            track.track_type = 'sample' if str(raw_track.get('track_type') or '').lower() == 'sample' else 'instrument'
            track.notes = []
            for raw_note in raw_track.get('notes', []):
                if not isinstance(raw_note, dict):
                    continue
                track.notes.append(
                    MidiNote(
                        start_tick=self._coerce_int(raw_note.get('start_tick'), 0, 0),
                        duration_tick=self._coerce_int(raw_note.get('duration_tick'), TICKS_PER_BEAT // 2, 1),
                        pitch=self._coerce_int(raw_note.get('pitch'), 60, 0, 127),
                        velocity=self._coerce_int(raw_note.get('velocity'), 100, 1, 127),
                        selected=bool(raw_note.get('selected', False)),
                    )
                )
            track.volume = self._coerce_float(raw_track.get('volume'), 0.8, 0.0, 2.0)
            track.pan = self._coerce_float(raw_track.get('pan'), 0.0, -1.0, 1.0)
            track.instrument = str(raw_track.get('instrument') or track.instrument)
            track.instrument_mode = str(raw_track.get('instrument_mode') or track.instrument_mode)
            track.rack_vsti = str(raw_track.get('rack_vsti') or '')
            track.plugins = [str(item) for item in raw_track.get('plugins', []) if isinstance(item, str)]
            if isinstance(raw_track.get('vsti_parameters'), dict):
                track.vsti_parameters = {
                    str(key): self._coerce_float(value, 0.0, 0.0, 100.0)
                    for key, value in raw_track.get('vsti_parameters', {}).items()
                }
            track.vsti_state_path = str(raw_track.get('vsti_state_path') or '')
            track.vsti_output_gain_db = self._coerce_float(raw_track.get('vsti_output_gain_db'), 0.0, -48.0, 24.0)
            track.vsti_wet_mix = self._coerce_float(raw_track.get('vsti_wet_mix'), 100.0, 0.0, 100.0)
            track.vst_fx_chain = [str(item) for item in raw_track.get('vst_fx_chain', []) if isinstance(item, str)]
            track.midi_program = self._coerce_int(raw_track.get('midi_program'), 0, 0, 127)
            track.midi_channel = self._coerce_int(raw_track.get('midi_channel'), index % 16, 0, 15)
            track.synth_profile = str(raw_track.get('synth_profile') or track.synth_profile)
            track.rendered_audio_path = str(raw_track.get('rendered_audio_path') or '')
            track.mute = bool(raw_track.get('mute', False))
            track.solo = bool(raw_track.get('solo', False))
            track.color_hex = str(raw_track.get('color_hex') or '')
            tracks.append(track)
            blob = self._decode_project_blob(raw_track.get('vsti_state_b64'))
            if blob:
                track_state_blobs[index] = blob
        project.tracks = tracks or [TrackState(name='Track 1')]

        project.vsti_paths = [str(item) for item in project_root.get('vsti_paths', []) if isinstance(item, str)]
        project.vsti_folder_paths = [str(item) for item in project_root.get('vsti_folder_paths', []) if isinstance(item, str)]
        project.sample_paths = [str(item) for item in project_root.get('sample_paths', []) if isinstance(item, str)]

        project.vsti_rack = []
        for raw_vst in project_root.get('vsti_rack', []):
            if not isinstance(raw_vst, dict):
                continue
            project.vsti_rack.append(
                VSTInstrument(
                    name=str(raw_vst.get('name') or Path(str(raw_vst.get('path') or '')).stem or 'VSTI'),
                    path=str(raw_vst.get('path') or ''),
                    plugin_name=str(raw_vst.get('plugin_name') or raw_vst.get('name') or ''),
                    is_instrument=bool(raw_vst.get('is_instrument', False)),
                    is_effect=bool(raw_vst.get('is_effect', False)),
                    category=str(raw_vst.get('category') or ''),
                    host_supported=bool(raw_vst.get('host_supported', True)),
                    host_error=str(raw_vst.get('host_error') or ''),
                )
            )

        project.sample_assets = []
        for raw_asset in project_root.get('sample_assets', []):
            if not isinstance(raw_asset, dict):
                continue
            project.sample_assets.append(
                SampleAsset(
                    path=str(raw_asset.get('path') or ''),
                    duration_sec=self._coerce_float(raw_asset.get('duration_sec'), 0.0, 0.0),
                    sample_rate=self._coerce_int(raw_asset.get('sample_rate'), 44100, 1000, 384000),
                    waveform_preview=[
                        self._coerce_float(value, 0.0, -1.0, 1.0)
                        for value in raw_asset.get('waveform_preview', [])
                        if isinstance(value, (int, float))
                    ],
                )
            )

        project.sample_clips = []
        for raw_clip in project_root.get('sample_clips', []):
            if not isinstance(raw_clip, dict):
                continue
            project.sample_clips.append(
                SampleClip(
                    path=str(raw_clip.get('path') or ''),
                    track_index=self._coerce_int(raw_clip.get('track_index'), 0, 0),
                    start_sec=self._coerce_float(raw_clip.get('start_sec'), 0.0, 0.0),
                    duration_sec=self._coerce_float(raw_clip.get('duration_sec'), 0.0, 0.0),
                    sample_rate=self._coerce_int(raw_clip.get('sample_rate'), 44100, 1000, 384000),
                    waveform_preview=[
                        self._coerce_float(value, 0.0, -1.0, 1.0)
                        for value in raw_clip.get('waveform_preview', [])
                        if isinstance(value, (int, float))
                    ],
                )
            )

        project.midi_sections = []
        for raw_section in project_root.get('midi_sections', []):
            if not isinstance(raw_section, dict):
                continue
            project.midi_sections.append(
                MidiSection(
                    track_index=self._coerce_int(raw_section.get('track_index'), 0, 0),
                    start_sec=self._coerce_float(raw_section.get('start_sec'), 0.0, 0.0),
                    duration_sec=self._coerce_float(raw_section.get('duration_sec'), 0.0, 0.0),
                    name=str(raw_section.get('name') or 'MIDI Part'),
                )
            )

        ui_state = payload.get('ui', {})
        return project, track_state_blobs, ui_state if isinstance(ui_state, dict) else {}

    def _materialize_project_vsti_states(self, project: ProjectState, track_state_blobs: dict[int, bytes]) -> None:
        if not track_state_blobs:
            return
        for index, raw_state in track_state_blobs.items():
            if not raw_state or not (0 <= index < len(project.tracks)):
                continue
            track = project.tracks[index]
            entry = next((item for item in project.vsti_rack if item.name == track.rack_vsti), None)
            if entry is not None:
                target_path = self._default_vsti_state_cache_path(track, entry)
            else:
                track_key = hashlib.sha1(f'{track.name}:{track.rack_vsti}:{index}'.encode('utf-8')).hexdigest()[:16]
                target_path = RENDER_DIR / '_vsti_state' / f'{track_key}.bin'
            target_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                target_path.write_bytes(raw_state)
                track.vsti_state_path = str(target_path)
            except Exception:
                continue

    def _set_project_references(self, project: ProjectState) -> None:
        self.project = project
        self.timeline.project = project
        self.piano_roll.project = project
        self.velocity_editor.project = project
        self.sample_timeline.project = project
        self.arrangement_overview.project = project
        self.mixer.project = project
        self.instruments.project = project
        self._track_meter_levels = {}

    def _reset_project_runtime_state(self) -> None:
        self._deferred_note_refresh_timer.stop()
        self._tempo_ui_refresh_timer.stop()
        self._deferred_refresh_velocity = False
        self._deferred_refresh_timeline = False
        self._deferred_rebuild_sections = False
        self._deferred_refresh_arrangement = False
        self._deferred_reload_mix = False
        self._tempo_refresh_seconds_layout = False
        self._tempo_refresh_arrangement = False
        self._tempo_refresh_timeline = False
        self.stop_playback()
        self._close_preview_audio()
        self._track_playback_audio_cache = {}
        self._sample_audio_cache = {}
        self._realtime_track_states = {}
        self._playback_mix_wav_bytes = b''
        self._playback_mix_cache_key = ''
        self._playback_mix_duration_sec = 0.0
        self._cleanup_legacy_playback_files()

    def _apply_project_to_ui(self, project: ProjectState, ui_state: dict[str, object] | None = None, project_path: Path | None = None) -> None:
        self._reset_project_runtime_state()
        self._set_project_references(project)
        self._dedupe_and_filter_vsti_state()
        self._sync_bundled_vsti_directory()
        self.current_project_path = project_path.resolve() if project_path is not None else None

        quantize_text = str((ui_state or {}).get('quantize_text') or self._project_quantize_text()).strip()
        if self.quantize_box.findText(quantize_text) < 0:
            quantize_text = self._project_quantize_text()
        self.quantize_box.blockSignals(True)
        self.quantize_box.setCurrentText(quantize_text)
        self.quantize_box.blockSignals(False)
        self.project.quantize_enabled = quantize_text.strip().upper() != 'OFF'

        self.tempo_spin.blockSignals(True)
        self.tempo_spin.setValue(int(self.project.bpm))
        self.tempo_spin.blockSignals(False)
        self._refresh_locator_spin_configuration()
        self._set_locator_spin_values(self.project.left_locator_sec, self.project.right_locator_sec)
        self._refresh_transport_controls()

        self.piano_roll.tool = str((ui_state or {}).get('piano_roll_tool') or self.piano_roll.tool)
        self.piano_roll.note_length_div = self._coerce_int((ui_state or {}).get('piano_roll_note_length_div'), self.piano_roll.note_length_div, 1, 64)
        ruler_mode = str((ui_state or {}).get('piano_roll_ruler_mode') or self.piano_roll.ruler_display_mode).strip().lower()
        self.piano_roll.ruler_display_mode = ruler_mode if ruler_mode in {'seconds', 'bars'} else 'bars'
        self.piano_roll.cell_w = self._coerce_int((ui_state or {}).get('piano_roll_cell_w'), self.piano_roll.cell_w, 8, 160)
        self.piano_roll.cell_h = self._coerce_int((ui_state or {}).get('piano_roll_cell_h'), self.piano_roll.cell_h, 10, 28)
        self.velocity_editor.cell_w = self._coerce_int((ui_state or {}).get('velocity_cell_w'), self.velocity_editor.cell_w, 8, 160)
        self.sample_timeline.pixels_per_second = self._coerce_int((ui_state or {}).get('sample_timeline_pixels_per_second'), self.sample_timeline.pixels_per_second, 20, 320)
        self.arrangement_overview.pixels_per_second = self._coerce_int((ui_state or {}).get('arrangement_pixels_per_second'), self.arrangement_overview.pixels_per_second, 20, 320)
        self.right_tabs.setCurrentIndex(self._coerce_int((ui_state or {}).get('panels_tab_index'), self.right_tabs.currentIndex(), 0, max(0, self.right_tabs.count() - 1)))

        self.refresh_sample_library()
        self.sample_timeline.refresh()
        self.arrangement_overview.refresh()
        self.piano_roll.refresh()
        self.velocity_editor.refresh()
        self.refresh_vsti_rack_ui()

        selected_track = self._coerce_int((ui_state or {}).get('selected_track_index'), 0, 0, max(0, len(self.project.tracks) - 1))
        self._populate_track_list()
        if self.project.tracks:
            self.track_list.setCurrentRow(selected_track)
        self.timeline.refresh()
        self.set_playhead_position(float(self.project.playhead_sec))

        if hasattr(self, 'show_panels_window_action'):
            panels_visible = bool((ui_state or {}).get('panels_visible', self._tools_window_visible))
            self.show_panels_window_action.blockSignals(True)
            self.show_panels_window_action.setChecked(panels_visible)
            self.show_panels_window_action.blockSignals(False)
            self.toggle_tools_window(panels_visible)
        if hasattr(self, 'show_mixer_window_action'):
            mixer_visible = bool((ui_state or {}).get('mixer_visible', self._mixer_window_visible))
            self.show_mixer_window_action.blockSignals(True)
            self.show_mixer_window_action.setChecked(mixer_visible)
            self.show_mixer_window_action.blockSignals(False)
            self.toggle_mixer_window(mixer_visible)
        if hasattr(self, 'show_transport_window_action'):
            transport_visible = bool((ui_state or {}).get('transport_visible', self._transport_window_visible))
            self.show_transport_window_action.blockSignals(True)
            self.show_transport_window_action.setChecked(transport_visible)
            self.show_transport_window_action.blockSignals(False)
            self.toggle_transport_window(transport_visible)
        if hasattr(self, 'show_virtual_piano_window_action'):
            piano_visible = bool((ui_state or {}).get('virtual_piano_visible', self._virtual_piano_window_visible))
            self.show_virtual_piano_window_action.blockSignals(True)
            self.show_virtual_piano_window_action.setChecked(piano_visible)
            self.show_virtual_piano_window_action.blockSignals(False)
            self.toggle_virtual_piano_window(piano_visible)

        self._update_window_title()

    def on_quantize_changed(self, text: str) -> None:
        value = text.strip().upper()
        if value == 'OFF':
            self.project.quantize_enabled = False
            self._refresh_locator_spin_configuration()
            self.piano_roll.refresh()
            self.statusBar().showMessage('Quantize disabled')
            return
        self.project.quantize_enabled = True
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
        self._refresh_locator_spin_configuration()
        self.piano_roll.quantize_selected()
        self.piano_roll.refresh()

    def _locator_quantize_step_seconds(self) -> float | None:
        if not getattr(self.project, 'quantize_enabled', True):
            return None
        beats = 4.0 / max(1, int(self.project.quantize_div))
        if getattr(self.project, 'quantize_triplet', False):
            beats *= 2.0 / 3.0
        step = beats * (60.0 / max(1, self.project.bpm))
        if step <= 0.0:
            return None
        return float(step)

    def _snap_locator_seconds(self, sec: float) -> float:
        snapped = max(0.0, float(sec))
        step = self._locator_quantize_step_seconds()
        if step is None:
            return snapped
        return max(0.0, round(snapped / step) * step)

    def _set_locator_spin_values(self, left: float, right: float) -> None:
        self.left_locator.blockSignals(True)
        self.left_locator.setValue(self._seconds_to_locator_bars(left))
        self.left_locator.blockSignals(False)
        self.right_locator.blockSignals(True)
        self.right_locator.setValue(self._seconds_to_locator_bars(right))
        self.right_locator.blockSignals(False)

    def _scene_pos_from_global_cursor(self, view: QtWidgets.QGraphicsView, global_pos: QtCore.QPoint) -> QtCore.QPointF | None:
        if view is None or not view.isVisible():
            return None
        viewport = view.viewport()
        if viewport is None or not viewport.isVisible():
            return None
        local_pos = viewport.mapFromGlobal(global_pos)
        if not viewport.rect().contains(local_pos):
            return None
        return view.mapToScene(local_pos)

    def _locator_seconds_from_global_pos(self, global_pos: QtCore.QPoint | None = None) -> float | None:
        cursor_pos = global_pos or QtGui.QCursor.pos()

        scene_pos = self._scene_pos_from_global_cursor(self.piano_roll, cursor_pos)
        if scene_pos is not None:
            return self.piano_roll._x_to_sec(scene_pos.x())

        scene_pos = self._scene_pos_from_global_cursor(self.velocity_editor, cursor_pos)
        if scene_pos is not None:
            return self.velocity_editor._x_to_sec(scene_pos.x())

        scene_pos = self._scene_pos_from_global_cursor(self.arrangement_overview, cursor_pos)
        if scene_pos is not None:
            return max(0.0, scene_pos.x() / max(1e-9, float(self.arrangement_overview.pixels_per_second)))

        scene_pos = self._scene_pos_from_global_cursor(self.sample_timeline, cursor_pos)
        if scene_pos is not None:
            return max(0.0, scene_pos.x() / max(1e-9, float(self.sample_timeline.pixels_per_second)))

        return None

    def set_left_locator_from_mouse(self) -> None:
        sec = self._locator_seconds_from_global_pos()
        if sec is None:
            self.statusBar().showMessage('Hover over the piano roll or timeline to set the left locator')
            return
        self.set_left_locator_position(sec)

    def set_right_locator_from_mouse(self) -> None:
        sec = self._locator_seconds_from_global_pos()
        if sec is None:
            self.statusBar().showMessage('Hover over the piano roll or timeline to set the right locator')
            return
        self.set_right_locator_position(sec)

    def set_left_locator_position(self, sec: float) -> None:
        snapped = self._snap_locator_seconds(sec)
        self.project.left_locator_sec = min(max(0.0, snapped), self.project.right_locator_sec)
        self._set_locator_spin_values(self.project.left_locator_sec, self.project.right_locator_sec)
        self.update_locators()

    def set_right_locator_position(self, sec: float) -> None:
        snapped = self._snap_locator_seconds(sec)
        self.project.right_locator_sec = max(max(0.0, snapped), self.project.left_locator_sec)
        self._set_locator_spin_values(self.project.left_locator_sec, self.project.right_locator_sec)
        self.update_locators()

    def update_locators(self) -> None:
        snapped_left = self._snap_locator_seconds(self._locator_bars_to_seconds(self.left_locator.value()))
        snapped_right = self._snap_locator_seconds(self._locator_bars_to_seconds(self.right_locator.value()))
        self.project.left_locator_sec = min(snapped_left, snapped_right)
        self.project.right_locator_sec = max(snapped_left, snapped_right)
        self._set_locator_spin_values(self.project.left_locator_sec, self.project.right_locator_sec)
        self._refresh_locator_bound_views_if_needed()
        self._update_locator_overlays()
        self._sync_playback_loop_state()
        self._update_locator_playback_state()

    def _refresh_locator_bound_views_if_needed(self) -> None:
        sample_timeline_width = max(0.0, self.project.right_locator_sec * self.sample_timeline.pixels_per_second)
        if self.sample_timeline.sceneRect().width() < sample_timeline_width + 4.0:
            self.sample_timeline.refresh()

        arrangement_width = max(0.0, self.project.right_locator_sec * self.arrangement_overview.pixels_per_second)
        if self.arrangement_overview.sceneRect().width() < arrangement_width + 4.0:
            self.arrangement_overview.refresh()

        piano_width = max(0.0, self.piano_roll._locator_x(self.project.right_locator_sec))
        if self.piano_roll.sceneRect().width() < piano_width + 4.0:
            self.piano_roll.refresh()

        velocity_width = max(0.0, self.velocity_editor._locator_x(self.project.right_locator_sec))
        if self.velocity_editor.sceneRect().width() < velocity_width + 4.0:
            self.velocity_editor.refresh()

    def _update_locator_overlays(self) -> None:
        self.sample_timeline.update_overlay_items()
        self.arrangement_overview.update_overlay_items()
        self.piano_roll.update_overlay_items()
        self.velocity_editor.update_overlay_items()
        self.timeline.update_locator_header()

    def _invalidate_playback_caches(self, clear_track_audio: bool = True, *, reset_realtime: bool = True) -> None:
        self._playback_mix_cache_key = ''
        self._playback_mix_duration_sec = 0.0
        if clear_track_audio:
            self._track_playback_audio_cache.clear()
        if reset_realtime:
            self._realtime_reset_pending = True
            self._clear_realtime_mix_cache()

    def _cleanup_legacy_playback_files(self) -> None:
        transient_files = [
            self.playback_mix_path,
            self.note_preview_path,
        ]
        transient_globs = [
            (RENDER_DIR, '_play_track_*.wav'),
            (RENDER_DIR / '_render_work', '*.wav'),
        ]
        for candidate in transient_files:
            try:
                candidate.unlink(missing_ok=True)
            except Exception:
                pass
        for root, pattern in transient_globs:
            if not root.exists():
                continue
            for candidate in root.glob(pattern):
                try:
                    candidate.unlink(missing_ok=True)
                except Exception:
                    continue

    def _clear_realtime_mix_cache(self) -> None:
        self._realtime_mix_cache = None
        self._realtime_mix_cache_start_frame = 0
        self._realtime_mix_cache_frame_count = 0

    def _selected_audio_device(self) -> QtMultimedia.QAudioDevice:
        if self.selected_audio_output_id:
            for device in QtMultimedia.QMediaDevices.audioOutputs():
                if bytes(device.id()).hex() == self.selected_audio_output_id:
                    return device
        return QtMultimedia.QMediaDevices.defaultAudioOutput()

    def _playback_audio_format(self) -> QtMultimedia.QAudioFormat:
        fmt = QtMultimedia.QAudioFormat()
        fmt.setSampleRate(self._playback_sample_rate)
        fmt.setChannelCount(self._playback_channel_count)
        fmt.setSampleFormat(QtMultimedia.QAudioFormat.SampleFormat.Int16)
        return fmt

    def _create_audio_sink(self) -> QtMultimedia.QAudioSink:
        sink = QtMultimedia.QAudioSink(self._selected_audio_device(), self._playback_audio_format(), self)
        sink.setBufferFrameCount(max(self._playback_chunk_frames * 4, int((self._playback_sample_rate * self.audio_buffer_ms) / 1000)))
        sink.setVolume(1.0)
        return sink

    def _finalize_preview_resource(self, sink: QtMultimedia.QAudioSink, buffer: QtCore.QBuffer) -> None:
        try:
            self._preview_resources.remove((sink, buffer))
        except ValueError:
            pass
        try:
            sink.stop()
        except Exception:
            pass
        sink.deleteLater()
        buffer.deleteLater()

    def _cleanup_finished_preview_resources(self) -> None:
        stopped_state = QtMultimedia.QtAudio.State.StoppedState
        active_resources = list(self._preview_resources)
        for sink, buffer in active_resources:
            try:
                state = sink.state()
            except RuntimeError:
                state = stopped_state
            except Exception:
                state = stopped_state
            if state == stopped_state:
                self._finalize_preview_resource(sink, buffer)

    def _close_preview_audio(self) -> None:
        sink = self._preview_sink
        buffer = self._preview_buffer_device
        self._preview_sink = None
        self._preview_buffer_device = None
        if sink is None or buffer is None:
            self._cleanup_finished_preview_resources()
            return
        if (sink, buffer) not in self._preview_resources:
            self._preview_resources.append((sink, buffer))
        try:
            sink.stop()
        except Exception:
            self._finalize_preview_resource(sink, buffer)
            return
        QtCore.QTimer.singleShot(250, self._cleanup_finished_preview_resources)

    def _on_preview_sink_state_changed(self, sink: QtMultimedia.QAudioSink, buffer: QtCore.QBuffer, state) -> None:
        if state == QtMultimedia.QtAudio.State.IdleState and self._preview_sink is sink:
            self._preview_sink = None
            self._preview_buffer_device = None
            if (sink, buffer) not in self._preview_resources:
                self._preview_resources.append((sink, buffer))
            try:
                sink.stop()
            except Exception:
                self._finalize_preview_resource(sink, buffer)
                return
            QtCore.QTimer.singleShot(0, self._cleanup_finished_preview_resources)
            return
        if state == QtMultimedia.QtAudio.State.StoppedState:
            self._finalize_preview_resource(sink, buffer)

    def _play_pcm_preview(self, samples: object, sample_rate: int) -> None:
        if sample_rate != self._playback_sample_rate:
            samples = resample_samples(samples, sample_rate, self._playback_sample_rate)
        pcm_bytes = encode_pcm16_stereo_samples(samples)
        frame_count = max(1, len(pcm_bytes) // max(1, self._playback_channel_count * 2))
        preview_duration_ms = int((frame_count / float(self._playback_sample_rate)) * 1000.0) + 80
        self._close_preview_audio()
        buffer = QtCore.QBuffer(self)
        buffer.setData(QtCore.QByteArray(pcm_bytes))
        buffer.open(QtCore.QIODevice.OpenModeFlag.ReadOnly)
        sink = self._create_audio_sink()
        sink.start(buffer)
        self._preview_buffer_device = buffer
        self._preview_sink = sink
        QtCore.QTimer.singleShot(preview_duration_ms, lambda current_sink=sink: self._close_preview_audio() if self._preview_sink is current_sink else None)

    def _sync_playback_loop_state(self) -> None:
        left = float(self.project.left_locator_sec)
        right = max(left + 0.001, float(self.project.right_locator_sec))
        self._playback_loop_ms = max(1, int((right - left) * 1000)) if self.project.loop_enabled else 0

    def _loop_frame_bounds(self) -> tuple[int, int]:
        left_frame = seconds_to_sample_frame(self.project.left_locator_sec, self._playback_sample_rate)
        right_frame = max(left_frame + 1, seconds_to_sample_frame(self.project.right_locator_sec, self._playback_sample_rate))
        return left_frame, right_frame

    def _beat_sample_frame(self, beat_index: int) -> int:
        return tick_to_sample_frame(max(0, int(beat_index)) * TICKS_PER_BEAT, self._playback_sample_rate, self.project.bpm)

    def _mix_metronome_click(self, buffer: object, offset_frame: int, *, accented: bool) -> None:
        sample_rate = max(1, int(self._playback_sample_rate))
        click_frames = max(8, int(sample_rate * (0.032 if accented else 0.022)))
        available = 0
        if np is not None and isinstance(buffer, np.ndarray):
            available = max(0, int(buffer.shape[-1]) - int(offset_frame))
        else:
            available = max(0, len(buffer[0]) - int(offset_frame))
        count = min(click_frames, available)
        if count <= 0:
            return

        freq = 1760.0 if accented else 1320.0
        gain = 0.22 if accented else 0.14
        decay_rate = 78.0 if accented else 96.0
        harmonic = 0.28 if accented else 0.18
        start = int(offset_frame)

        if np is not None and isinstance(buffer, np.ndarray):
            t = np.arange(count, dtype=np.float32) / float(sample_rate)
            env = np.exp(-t * decay_rate).astype(np.float32, copy=False)
            wave = np.sin(2.0 * math.pi * freq * t) + (harmonic * np.sin(2.0 * math.pi * freq * 2.0 * t))
            click = (wave * env * gain).astype(np.float32, copy=False)
            buffer[0, start:start + count] += click
            buffer[1, start:start + count] += click
            return

        for index in range(count):
            t = index / float(sample_rate)
            env = math.exp(-t * decay_rate)
            value = (math.sin(2.0 * math.pi * freq * t) + (harmonic * math.sin(2.0 * math.pi * freq * 2.0 * t))) * env * gain
            buffer[0][start + index] += value
            buffer[1][start + index] += value

    def _render_metronome_segment(self, start_frame: int, frame_count: int) -> object:
        if np is not None:
            click_mix: object = np.zeros((2, frame_count), dtype=np.float32)
        else:
            click_mix = [[0.0] * frame_count, [0.0] * frame_count]

        if frame_count <= 0:
            return click_mix

        sample_rate = max(1, int(self._playback_sample_rate))
        bpm = max(1, int(self.project.bpm))
        samples_per_beat_numerator = sample_rate * 60
        start = max(0, int(start_frame))
        segment_end = start + max(1, int(frame_count))
        beat_index = max(0, ((start * bpm) // samples_per_beat_numerator) - 1)

        while self._beat_sample_frame(beat_index + 1) <= start:
            beat_index += 1
        while beat_index > 0 and self._beat_sample_frame(beat_index) > start:
            beat_index -= 1

        while True:
            beat_frame = self._beat_sample_frame(beat_index)
            if beat_frame >= segment_end:
                break
            if beat_frame >= start:
                self._mix_metronome_click(click_mix, beat_frame - start, accented=(beat_index % 4) == 0)
            beat_index += 1

        return click_mix

    def _bar_duration_seconds(self) -> float:
        return 4.0 * (60.0 / max(1, self.project.bpm))

    @staticmethod
    def _seconds_to_beats_at_bpm(sec: float, bpm: int) -> float:
        return max(0.0, float(sec)) * (max(1, int(bpm)) / 60.0)

    @staticmethod
    def _beats_to_seconds_at_bpm(beats: float, bpm: int) -> float:
        return max(0.0, float(beats)) * (60.0 / max(1, int(bpm)))

    def _rescale_transport_for_tempo_change(self, old_bpm: int, new_bpm: int) -> None:
        if max(1, int(old_bpm)) == max(1, int(new_bpm)):
            return

        left_beats = self._seconds_to_beats_at_bpm(self.project.left_locator_sec, old_bpm)
        right_beats = self._seconds_to_beats_at_bpm(self.project.right_locator_sec, old_bpm)
        playhead_beats = self._seconds_to_beats_at_bpm(self.project.playhead_sec, old_bpm)

        minimum_gap_sec = max(0.001, 1.0 / max(1, self._playback_sample_rate))
        left_sec = self._beats_to_seconds_at_bpm(left_beats, new_bpm)
        right_sec = self._beats_to_seconds_at_bpm(right_beats, new_bpm)
        playhead_sec = self._beats_to_seconds_at_bpm(playhead_beats, new_bpm)

        self.project.left_locator_sec = max(0.0, float(left_sec))
        self.project.right_locator_sec = max(self.project.left_locator_sec + minimum_gap_sec, float(right_sec))
        self._set_locator_spin_values(self.project.left_locator_sec, self.project.right_locator_sec)
        self.project.playhead_sec = max(0.0, float(playhead_sec))
        if hasattr(self, 'playhead_spin'):
            self.playhead_spin.blockSignals(True)
            self.playhead_spin.setValue(self.project.playhead_sec)
            self.playhead_spin.blockSignals(False)

    def _ruler_uses_bar_positions(self) -> bool:
        return str(getattr(self.piano_roll, 'ruler_display_mode', 'bars')).strip().lower() == 'bars'

    def _mark_realtime_track_states_for_reset(self) -> None:
        for state in self._realtime_track_states.values():
            state.reset_pending = True
        self._clear_realtime_mix_cache()

    def _schedule_tempo_ui_refresh(self, *, seconds_layout_changed: bool, arrangement_changed: bool = True, timeline_changed: bool = True) -> None:
        self._tempo_refresh_seconds_layout = self._tempo_refresh_seconds_layout or bool(seconds_layout_changed)
        self._tempo_refresh_arrangement = self._tempo_refresh_arrangement or bool(arrangement_changed)
        self._tempo_refresh_timeline = self._tempo_refresh_timeline or bool(timeline_changed)
        interval_ms = 90 if (hasattr(self, 'playback_timer') and self.playback_timer.isActive()) else 0
        self._tempo_ui_refresh_timer.start(interval_ms)

    def _flush_tempo_ui_refresh(self) -> None:
        refresh_seconds_layout = self._tempo_refresh_seconds_layout
        refresh_arrangement = self._tempo_refresh_arrangement
        refresh_timeline = self._tempo_refresh_timeline
        self._tempo_refresh_seconds_layout = False
        self._tempo_refresh_arrangement = False
        self._tempo_refresh_timeline = False

        self.rebuild_midi_sections()
        if refresh_seconds_layout:
            self._refresh_locator_bound_views_if_needed()
            self.piano_roll.refresh()
            self.velocity_editor.refresh()
        else:
            self._update_locator_overlays()
        if refresh_arrangement:
            self.arrangement_overview.refresh()
        if refresh_timeline:
            self.timeline.refresh()

    def _previous_bar_start(self, sec: float) -> float:
        bar = self._bar_duration_seconds()
        if bar <= 0:
            return 0.0
        current_bar = math.floor(max(0.0, sec) / bar)
        current_bar_start = current_bar * bar
        if sec - current_bar_start > 0.05:
            return current_bar_start
        return max(0.0, (current_bar - 1) * bar)

    def _next_bar_start(self, sec: float) -> float:
        bar = self._bar_duration_seconds()
        if bar <= 0:
            return 0.0
        current_bar = math.floor(max(0.0, sec) / bar)
        current_bar += 1
        return max(0.0, current_bar * bar)

    def _clamp_transport_target(self, sec: float) -> float:
        target_sec = max(0.0, float(sec))
        if self.project.loop_enabled:
            left = float(self.project.left_locator_sec)
            right = max(left + (1.0 / self._playback_sample_rate), float(self.project.right_locator_sec))
            target_sec = min(target_sec, right - (1.0 / self._playback_sample_rate))
            target_sec = max(left, target_sec)
        return target_sec

    def _refresh_transport_controls(self) -> None:
        if hasattr(self, 'transport_loop_btn'):
            self.transport_loop_btn.blockSignals(True)
            self.transport_loop_btn.setChecked(bool(self.project.loop_enabled))
            self.transport_loop_btn.blockSignals(False)
        if hasattr(self, 'transport_metronome_btn'):
            self.transport_metronome_btn.blockSignals(True)
            self.transport_metronome_btn.setChecked(bool(getattr(self.project, 'metronome_enabled', False)))
            self.transport_metronome_btn.blockSignals(False)
        is_playing = bool(getattr(self, '_playback_active', False))
        if hasattr(self, 'transport_play_btn'):
            self.transport_play_btn.setEnabled(not is_playing)
        if hasattr(self, 'transport_stop_btn'):
            self.transport_stop_btn.setEnabled(is_playing or float(self.project.playhead_sec) > 0.0)

    def jump_playhead_to_start(self) -> None:
        self.set_playhead_position(0.0)
        self.statusBar().showMessage('Playhead moved to project start')

    def skip_to_previous_bar(self) -> None:
        target = self._clamp_transport_target(self._previous_bar_start(float(self.project.playhead_sec)))
        self.set_playhead_position(target)
        self.statusBar().showMessage(f'Playhead moved to {target:.2f}s')

    def skip_to_next_bar(self) -> None:
        target = self._clamp_transport_target(self._next_bar_start(float(self.project.playhead_sec)))
        self.set_playhead_position(target)
        self.statusBar().showMessage(f'Playhead moved to {target:.2f}s')

    def set_loop_enabled(self, enabled: bool) -> None:
        self.project.loop_enabled = bool(enabled)
        self._sync_playback_loop_state()
        if self.playback_timer.isActive():
            self._update_locator_playback_state()
        else:
            self._realtime_reset_pending = True
        self._refresh_transport_controls()
        self.statusBar().showMessage('Looping between locators enabled' if self.project.loop_enabled else 'Looping between locators disabled')

    def set_metronome_enabled(self, enabled: bool) -> None:
        self.project.metronome_enabled = bool(enabled)
        self._clear_realtime_mix_cache()
        self._refresh_transport_controls()
        self.statusBar().showMessage('Metronome enabled' if self.project.metronome_enabled else 'Metronome disabled')

    def _seek_media_to_project_time(self, sec: float) -> None:
        target_sec = max(0.0, float(sec))
        if self.project.loop_enabled and self.project.right_locator_sec > self.project.left_locator_sec:
            target_sec = min(target_sec, max(self.project.left_locator_sec, self.project.right_locator_sec - (1.0 / self._playback_sample_rate)))
        self._playback_frame_position = seconds_to_sample_frame(target_sec, self._playback_sample_rate)
        self._playback_logical_origin_frame = self._playback_frame_position
        self._playback_generated_total_frames = 0
        self._realtime_reset_pending = True
        self._clear_realtime_mix_cache()

    def _update_locator_playback_state(self) -> None:
        if not hasattr(self, 'playback_timer') or not self.playback_timer.isActive():
            return

        target_pos = self.project.playhead_sec
        if self.project.loop_enabled:
            loop_start = self.project.left_locator_sec
            loop_end = self.project.right_locator_sec
            if target_pos < loop_start or target_pos > loop_end:
                target_pos = loop_start
                self.set_playhead_position(target_pos)
            else:
                self._seek_media_to_project_time(target_pos)
        else:
            self._seek_media_to_project_time(target_pos)
        self._realtime_reset_pending = True

    def set_playhead_position(self, sec: float, playback_tick: bool = False) -> None:
        self.project.playhead_sec = max(0.0, float(sec))
        if hasattr(self, 'playhead_spin'):
            self.playhead_spin.blockSignals(True)
            self.playhead_spin.setValue(self.project.playhead_sec)
            self.playhead_spin.blockSignals(False)

        if not playback_tick and hasattr(self, 'playback_timer') and self.playback_timer.isActive():
            if self.project.loop_enabled:
                loop_start = self.project.left_locator_sec
                loop_end = self.project.right_locator_sec
                if loop_end > loop_start and (self.project.playhead_sec < loop_start or self.project.playhead_sec > loop_end):
                    self.project.playhead_sec = loop_start
                    if hasattr(self, 'playhead_spin'):
                        self.playhead_spin.blockSignals(True)
                        self.playhead_spin.setValue(self.project.playhead_sec)
                        self.playhead_spin.blockSignals(False)
            self._seek_media_to_project_time(self.project.playhead_sec)
        self._refresh_transport_controls()

        if playback_tick:
            now = time.time()
            if now - self._last_playhead_ui_refresh < self._playhead_ui_refresh_interval:
                return
            self._last_playhead_ui_refresh = now

        self.sample_timeline.update_overlay_items()
        self.arrangement_overview.update_overlay_items()
        self.piano_roll.update_overlay_items()
        self.velocity_editor.update_overlay_items()

    def _reload_playback_mix_if_running(self) -> None:
        if not hasattr(self, 'playback_timer') or not self.playback_timer.isActive():
            return
        self._realtime_reset_pending = True
        self._clear_realtime_mix_cache()

    def _schedule_deferred_note_refresh(
        self,
        *,
        refresh_velocity: bool = False,
        refresh_timeline: bool = False,
        rebuild_sections: bool = False,
        refresh_arrangement: bool = False,
        reload_mix: bool = False,
    ) -> None:
        self._deferred_refresh_velocity = self._deferred_refresh_velocity or refresh_velocity
        self._deferred_refresh_timeline = self._deferred_refresh_timeline or refresh_timeline
        self._deferred_rebuild_sections = self._deferred_rebuild_sections or rebuild_sections
        self._deferred_refresh_arrangement = self._deferred_refresh_arrangement or refresh_arrangement
        self._deferred_reload_mix = self._deferred_reload_mix or reload_mix
        if not self._deferred_note_refresh_timer.isActive():
            self._deferred_note_refresh_timer.start(0)

    def _flush_deferred_note_refresh(self) -> None:
        refresh_velocity = self._deferred_refresh_velocity
        refresh_timeline = self._deferred_refresh_timeline
        rebuild_sections = self._deferred_rebuild_sections
        refresh_arrangement = self._deferred_refresh_arrangement
        reload_mix = self._deferred_reload_mix

        self._deferred_refresh_velocity = False
        self._deferred_refresh_timeline = False
        self._deferred_rebuild_sections = False
        self._deferred_refresh_arrangement = False
        self._deferred_reload_mix = False

        if refresh_velocity:
            self.velocity_editor.refresh()
        if refresh_timeline:
            self.timeline.refresh()
        if rebuild_sections:
            self.rebuild_midi_sections()
        if refresh_arrangement:
            self.arrangement_overview.refresh()
        if reload_mix:
            self._reload_playback_mix_if_running()

    def on_piano_roll_notes_committed(self) -> None:
        self._schedule_deferred_note_refresh(
            refresh_velocity=True,
            refresh_timeline=True,
            rebuild_sections=True,
            refresh_arrangement=True,
            reload_mix=True,
        )

    def on_velocity_editor_changed(self) -> None:
        self._schedule_deferred_note_refresh(reload_mix=True)

    def _has_realtime_playable_audio(self) -> bool:
        if any(track.track_type == 'instrument' and track.notes for track in self.project.tracks):
            return True
        return any(0 <= clip.track_index < len(self.project.tracks) for clip in self.project.sample_clips)

    def _active_solo_track_indices(self) -> set[int]:
        return {idx for idx, track in enumerate(self.project.tracks) if track.solo}

    def _track_is_audible(self, idx: int, solo_tracks: set[int]) -> bool:
        if idx < 0 or idx >= len(self.project.tracks):
            return False
        track = self.project.tracks[idx]
        if track.mute:
            return False
        if solo_tracks and idx not in solo_tracks:
            return False
        return True

    def _stop_realtime_audio_sink(self) -> None:
        self._audio_pump_timer.stop()
        if self._playback_sink is not None:
            try:
                self._playback_sink.stop()
            except Exception:
                pass
        self._playback_sink = None
        self._playback_sink_device = None
        self._playback_active = False
        self._track_meter_levels = {idx: 0.0 for idx in range(len(self.project.tracks))}
        if hasattr(self, 'mixer'):
            self.mixer.refresh_meters()
        self._clear_realtime_mix_cache()

    def _reset_realtime_track_states(self, *, clear_plugins: bool = False) -> None:
        for state in self._realtime_track_states.values():
            state.reset_pending = True
            if clear_plugins:
                state.instrument_plugin = None
                state.fx_plugins = []
        self._clear_realtime_mix_cache()

    def _prepare_realtime_playback(self, start_sec: float) -> bool:
        self._stop_realtime_audio_sink()
        sink = self._create_audio_sink()
        sink_device = sink.start()
        if sink_device is None:
            return False
        self._playback_sink = sink
        self._playback_sink_device = sink_device
        self._playback_active = True
        self._realtime_reset_pending = True
        self._reset_realtime_track_states()
        self._seek_media_to_project_time(start_sec)
        self._audio_pump_timer.start()
        self._pump_realtime_audio()
        return True

    def _sample_audio_cache_key(self, path: str) -> str:
        clip_path = Path(path)
        return f'{str(clip_path.resolve()) if clip_path.exists() else path}:{self._path_mtime_ns(clip_path)}:{self._playback_sample_rate}'

    def _sample_audio_data(self, path: str) -> tuple[object, int]:
        cache_key = self._sample_audio_cache_key(path)
        cached = self._sample_audio_cache.get(cache_key)
        if cached is not None:
            data, sample_rate, _mtime = cached
            return data, sample_rate

        wav_path = Path(path)
        if wav_path.suffix.lower() == '.mp3':
            converted = RENDER_DIR / f'{wav_path.stem}_play.wav'
            convert_audio(wav_path, converted)
            wav_path = converted
        data, sample_rate = load_wav_samples(wav_path)
        if sample_rate != self._playback_sample_rate:
            data = resample_samples(data, sample_rate, self._playback_sample_rate)
            sample_rate = self._playback_sample_rate
        if np is not None and isinstance(data, np.ndarray):
            stored_data: object = np.asarray(data, dtype=np.float32).copy()
        else:
            stored_data = list(data)
        self._sample_audio_cache = {key: value for key, value in self._sample_audio_cache.items() if key == cache_key}
        self._sample_audio_cache[cache_key] = (stored_data, sample_rate, self._path_mtime_ns(Path(path)))
        return stored_data, sample_rate

    def _collect_chunk_midi_messages(
        self,
        track: TrackState,
        start_frame: int,
        frame_count: int,
        *,
        bootstrap_active: bool,
    ) -> list[mido.Message]:
        chunk_start_frame = max(0, int(start_frame))
        chunk_frame_count = max(1, int(frame_count))
        chunk_end_frame = chunk_start_frame + chunk_frame_count
        sample_rate = max(1, int(self._playback_sample_rate))
        events: list[tuple[int, int, mido.Message]] = []
        if bootstrap_active:
            events.append((
                0,
                0,
                mido.Message(
                    'program_change',
                    channel=int(clamp(track.midi_channel, 0, 15)),
                    program=int(clamp(track.midi_program, 0, 127)),
                    time=0.0,
                )
            ))

        for note in track.notes:
            note_start_frame = tick_to_sample_frame(note.start_tick, sample_rate, self.project.bpm)
            note_end_frame = max(
                note_start_frame + 1,
                tick_to_sample_frame(note.start_tick + note.duration_tick, sample_rate, self.project.bpm),
            )
            if note_end_frame <= chunk_start_frame or note_start_frame >= chunk_end_frame:
                continue

            if bootstrap_active and note_start_frame < chunk_start_frame < note_end_frame:
                events.append((
                    0,
                    2,
                    mido.Message(
                        'note_on',
                        channel=int(clamp(track.midi_channel, 0, 15)),
                        note=int(clamp(note.pitch, 0, 127)),
                        velocity=int(clamp(note.velocity, 1, 127)),
                        time=0.0,
                    )
                ))
            elif chunk_start_frame <= note_start_frame < chunk_end_frame:
                note_on_offset = note_start_frame - chunk_start_frame
                events.append((
                    note_on_offset,
                    2,
                    mido.Message(
                        'note_on',
                        channel=int(clamp(track.midi_channel, 0, 15)),
                        note=int(clamp(note.pitch, 0, 127)),
                        velocity=int(clamp(note.velocity, 1, 127)),
                        time=sample_frame_to_seconds(note_on_offset, sample_rate),
                    )
                ))
            if chunk_start_frame < note_end_frame <= chunk_end_frame:
                note_off_offset = note_end_frame - chunk_start_frame
                events.append((
                    note_off_offset,
                    1,
                    mido.Message(
                        'note_off',
                        channel=int(clamp(track.midi_channel, 0, 15)),
                        note=int(clamp(note.pitch, 0, 127)),
                        velocity=0,
                        time=sample_frame_to_seconds(note_off_offset, sample_rate),
                    )
                ))

        events.sort(key=lambda item: (item[0], item[1]))
        return [msg for _offset, _order, msg in events]

    def _apply_track_pan(self, data: object, pan: float) -> object:
        pan_value = float(clamp(float(pan), -1.0, 1.0))
        angle = (pan_value + 1.0) * math.pi * 0.25
        left_gain = math.cos(angle)
        right_gain = math.sin(angle)
        if np is not None:
            mono = np.asarray(data, dtype=np.float32)
            return np.vstack((mono * left_gain, mono * right_gain)).astype(np.float32, copy=False)
        left = [float(sample) * left_gain for sample in data]
        right = [float(sample) * right_gain for sample in data]
        return [left, right]

    def _audio_peak_level(self, data: object) -> float:
        if np is not None:
            try:
                audio = np.asarray(data, dtype=np.float32)
                if audio.size == 0:
                    return 0.0
                return max(0.0, min(1.0, float(np.max(np.abs(audio)))))
            except Exception:
                return 0.0
        if isinstance(data, (list, tuple)) and data and isinstance(data[0], (list, tuple)):
            maximum = 0.0
            for channel in data:
                for sample in channel:
                    maximum = max(maximum, abs(float(sample)))
            return max(0.0, min(1.0, maximum))
        if isinstance(data, (list, tuple)):
            maximum = max((abs(float(sample)) for sample in data), default=0.0)
            return max(0.0, min(1.0, maximum))
        return 0.0

    def _realtime_track_state(self, idx: int, track: TrackState) -> RealtimeTrackPlaybackState:
        state = self._realtime_track_states.get(idx)
        if state is None:
            state = RealtimeTrackPlaybackState()
            self._realtime_track_states[idx] = state
        track_key = self._track_audio_cache_key(track, idx)
        if state.key != track_key:
            state.key = track_key
            state.instrument_plugin = None
            state.fx_plugins = []
            state.reset_pending = True
            state.last_error = ""
        elif self._realtime_reset_pending:
            state.reset_pending = True
        return state

    def _effect_chain_entries(self, track: TrackState) -> list[VSTInstrument]:
        entries: list[VSTInstrument] = []
        for fx_name in track.vst_fx_chain:
            entry = self._rack_vsti_entry(fx_name)
            if entry is None or not entry.is_effect:
                continue
            entries.append(entry)
        return entries

    def _apply_realtime_vst_fx_chain(
        self,
        idx: int,
        track: TrackState,
        data: object,
        sample_rate: int,
        state: RealtimeTrackPlaybackState,
    ) -> object:
        if not PEDALBOARD_AVAILABLE or np is None or Pedalboard is None:
            return data
        effect_entries = self._effect_chain_entries(track)
        if not effect_entries:
            return data

        if not state.fx_plugins:
            plugins: list[object] = []
            for entry in effect_entries:
                try:
                    plugin = self._load_rack_plugin(entry.name)
                except Exception:
                    plugin = None
                if plugin is not None and bool(getattr(plugin, 'is_effect', False)):
                    plugins.append(plugin)
            state.fx_plugins = plugins
        if not state.fx_plugins:
            return data

        audio = np.asarray(data, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[None, :]
        reset_flag = state.reset_pending
        for plugin in state.fx_plugins:
            audio = np.asarray(plugin(audio, sample_rate, buffer_size=max(64, audio.shape[-1]), reset=reset_flag), dtype=np.float32)
            if audio.ndim == 1:
                audio = audio[None, :]
        return self._as_mono_audio(audio)

    def _render_sample_track_chunk(self, idx: int, track: TrackState, start_frame: int, frame_count: int) -> tuple[object, int]:
        total_samples = max(1, int(frame_count))
        start_sec = sample_frame_to_seconds(start_frame, self._playback_sample_rate)
        duration_sec = total_samples / float(self._playback_sample_rate)
        if np is not None:
            data: object = np.zeros(total_samples, dtype=np.float32)
        else:
            data = [0.0] * total_samples

        for clip in self.project.sample_clips:
            if clip.track_index != idx:
                continue
            clip_data, sample_rate = self._sample_audio_data(clip.path)
            clip_length = clip_data.shape[0] if np is not None and isinstance(clip_data, np.ndarray) else len(clip_data)
            clip_start = float(clip.start_sec)
            clip_end = clip_start + (clip_length / max(1, sample_rate))
            if clip_end <= start_sec or clip_start >= start_sec + duration_sec:
                continue

            overlap_start = max(start_sec, clip_start)
            overlap_end = min(start_sec + duration_sec, clip_end)
            src_start = max(0, int(round((overlap_start - clip_start) * sample_rate)))
            dst_start = max(0, int(round((overlap_start - start_sec) * sample_rate)))
            count = max(0, int(round((overlap_end - overlap_start) * sample_rate)))
            if count <= 0:
                continue

            if np is not None and isinstance(data, np.ndarray) and isinstance(clip_data, np.ndarray):
                data[dst_start:dst_start + count] += clip_data[src_start:src_start + count] * 0.7 * float(track.volume)
            else:
                source = list(clip_data)
                for offset in range(count):
                    data[dst_start + offset] += source[src_start + offset] * 0.7 * float(track.volume)

        state = self._realtime_track_state(idx, track)
        processed = self._apply_realtime_vst_fx_chain(idx, track, data, self._playback_sample_rate, state)
        state.reset_pending = False
        return processed, self._playback_sample_rate

    def _render_instrument_track_chunk(self, idx: int, track: TrackState, start_frame: int, frame_count: int) -> tuple[object, int]:
        state = self._realtime_track_state(idx, track)
        entry = self._rack_vsti_entry(track.rack_vsti) if track.rack_vsti else None
        sample_rate = self._playback_sample_rate
        start_sec = sample_frame_to_seconds(start_frame, sample_rate)
        duration_sec = max(frame_count, 1) / float(sample_rate)

        if track.instrument_mode == 'VSTI Rack' and entry is not None and entry.is_instrument and entry.host_supported and PEDALBOARD_AVAILABLE and np is not None:
            try:
                if state.instrument_plugin is None:
                    plugin = self._load_rack_plugin(entry.name)
                    if plugin is None or not bool(getattr(plugin, 'is_instrument', False)):
                        raise RuntimeError(f'Rack instrument could not be created: {entry.name}')
                    self._load_saved_vsti_plugin_state(plugin, track, entry)
                    self._apply_saved_plugin_parameters(plugin, track.vsti_parameters)
                    state.instrument_plugin = plugin
                midi_messages = self._collect_chunk_midi_messages(track, start_frame, frame_count, bootstrap_active=state.reset_pending)
                rendered = state.instrument_plugin(
                    midi_messages,
                    duration=max(duration_sec, 1.0 / sample_rate),
                    sample_rate=sample_rate,
                    num_channels=2,
                    buffer_size=max(64, int(frame_count)),
                    reset=state.reset_pending,
                )
                audio = self._as_mono_audio(rendered)
                gain_linear = max(0.0, float(track.volume)) * (10.0 ** (float(track.vsti_output_gain_db) / 20.0))
                data = np.clip(np.asarray(audio, dtype=np.float32) * gain_linear, -1.0, 1.0)
                data = self._apply_realtime_vst_fx_chain(idx, track, data, sample_rate, state)
                state.last_error = ""
                state.reset_pending = False
                return data, sample_rate
            except Exception as exc:
                state.instrument_plugin = None
                state.reset_pending = True
                error_text = str(exc)
                if state.last_error != error_text:
                    self.statusBar().showMessage(f'VST instrument fallback to synth for {track.name}: {error_text}')
                    state.last_error = error_text

        data, sample_rate = self.renderer.render_track_chunk(track, self.project.bpm, start_sec, duration_sec)
        data = self._apply_realtime_vst_fx_chain(idx, track, data, sample_rate, state)
        state.last_error = ""
        state.reset_pending = False
        return data, sample_rate

    def _render_track_chunk_realtime(self, idx: int, track: TrackState, start_frame: int, frame_count: int) -> object:
        expected_samples = max(1, int(frame_count))
        if track.track_type == 'sample':
            data, _sample_rate = self._render_sample_track_chunk(idx, track, start_frame, expected_samples)
        else:
            data, _sample_rate = self._render_instrument_track_chunk(idx, track, start_frame, expected_samples)
        mono = self._ensure_mono_sample_count(data, expected_samples)
        stereo = self._apply_track_pan(mono, track.pan)
        return self._ensure_stereo_sample_count(stereo, expected_samples)

    def _active_realtime_vsti_track_count(self) -> int:
        solo_tracks = self._active_solo_track_indices()
        count = 0
        for idx, track in enumerate(self.project.tracks):
            if not self._track_is_audible(idx, solo_tracks):
                continue
            if track.track_type != 'instrument' or not track.notes:
                continue
            entry = self._rack_vsti_entry(track.rack_vsti) if track.rack_vsti else None
            if track.instrument_mode == 'VSTI Rack' and entry is not None and entry.is_instrument and entry.host_supported:
                count += 1
        return count

    def _realtime_render_ahead_frames(self) -> int:
        base = max(self._playback_chunk_frames * 4, 4096)
        extra = max(0, self._active_realtime_vsti_track_count() - 1) * 1024
        return min(16384, base + extra)

    def _estimated_queued_output_frames(self) -> int:
        sink = getattr(self, '_playback_sink', None)
        if sink is None:
            return 0
        bytes_per_frame = max(1, self._playback_channel_count * 2)
        try:
            buffer_frames = int(max(0, sink.bufferFrameCount()))
        except Exception:
            buffer_frames = 0
        try:
            free_frames = int(max(0, sink.bytesFree()) // bytes_per_frame)
        except Exception:
            free_frames = 0
        if buffer_frames <= 0:
            return 0
        return max(0, buffer_frames - free_frames)

    def _current_audible_playback_sec(self) -> float:
        if not getattr(self, '_playback_active', False):
            return float(self.project.playhead_sec)
        audible_generated = max(0, int(self._playback_generated_total_frames) - self._estimated_queued_output_frames())
        if not self.project.loop_enabled:
            logical_frame = int(self._playback_logical_origin_frame) + audible_generated
            return sample_frame_to_seconds(logical_frame, self._playback_sample_rate)

        loop_start_frame, loop_end_frame = self._loop_frame_bounds()
        loop_length = max(1, loop_end_frame - loop_start_frame)
        origin_frame = int(self._playback_logical_origin_frame)
        origin_frame = min(max(origin_frame, loop_start_frame), loop_end_frame - 1)
        origin_offset = origin_frame - loop_start_frame
        logical_frame = loop_start_frame + ((origin_offset + audible_generated) % loop_length)
        return sample_frame_to_seconds(logical_frame, self._playback_sample_rate)

    def _slice_stereo_audio(self, data: object, start_frame: int, frame_count: int) -> object:
        start = max(0, int(start_frame))
        count = max(1, int(frame_count))
        if np is not None and isinstance(data, np.ndarray):
            audio = self._ensure_stereo_sample_count(data, max(start + count, data.shape[-1] if data.ndim >= 2 else count))
            return audio[:, start:start + count].astype(np.float32, copy=False)
        stereo = self._ensure_stereo_sample_count(data, start + count)
        return [stereo[0][start:start + count], stereo[1][start:start + count]]

    def _render_realtime_segment_cached(self, start_frame: int, frame_count: int, *, available_frames: int | None = None) -> object:
        requested_start = max(0, int(start_frame))
        requested_count = max(1, int(frame_count))
        cache = self._realtime_mix_cache
        cache_start = int(self._realtime_mix_cache_start_frame)
        cache_end = cache_start + int(self._realtime_mix_cache_frame_count)
        if cache is not None and requested_start >= cache_start and (requested_start + requested_count) <= cache_end:
            return self._slice_stereo_audio(cache, requested_start - cache_start, requested_count)

        render_count = max(requested_count, self._realtime_render_ahead_frames())
        if available_frames is not None:
            render_count = min(render_count, max(requested_count, int(available_frames)))
        segment = self._render_realtime_segment(requested_start, render_count)
        segment = self._ensure_stereo_sample_count(segment, render_count)
        self._realtime_mix_cache = segment
        self._realtime_mix_cache_start_frame = requested_start
        self._realtime_mix_cache_frame_count = render_count
        return self._slice_stereo_audio(segment, 0, requested_count)

    def _render_realtime_segment(self, start_frame: int, frame_count: int) -> object:
        if np is not None:
            mix = np.zeros((2, frame_count), dtype=np.float32)
        else:
            mix = [[0.0] * frame_count, [0.0] * frame_count]
        meter_levels: dict[int, float] = {}

        if bool(getattr(self.project, 'metronome_enabled', False)):
            metronome = self._render_metronome_segment(start_frame, frame_count)
            if np is not None and isinstance(mix, np.ndarray) and isinstance(metronome, np.ndarray):
                mix += metronome
            else:
                for channel in range(2):
                    for pos in range(frame_count):
                        mix[channel][pos] += metronome[channel][pos]

        solo_tracks = self._active_solo_track_indices()
        for idx, track in enumerate(self.project.tracks):
            if not self._track_is_audible(idx, solo_tracks):
                continue
            if track.track_type == 'instrument' and not track.notes:
                continue
            if track.track_type == 'sample' and not any(clip.track_index == idx for clip in self.project.sample_clips):
                continue
            stereo = self._render_track_chunk_realtime(idx, track, start_frame, frame_count)
            stereo = self._ensure_stereo_sample_count(stereo, frame_count)
            meter_levels[idx] = self._audio_peak_level(stereo)
            if np is not None and isinstance(mix, np.ndarray) and isinstance(stereo, np.ndarray):
                mix += stereo
            else:
                left, right = stereo
                for pos in range(frame_count):
                    mix[0][pos] += left[pos]
                    mix[1][pos] += right[pos]
        if meter_levels or self._track_meter_levels:
            decayed_levels: dict[int, float] = {}
            for idx in range(len(self.project.tracks)):
                previous = float(self._track_meter_levels.get(idx, 0.0)) * 0.72
                current = float(meter_levels.get(idx, 0.0))
                level = max(previous, current)
                if level > 0.001:
                    decayed_levels[idx] = level
            self._track_meter_levels = decayed_levels
        if np is not None and isinstance(mix, np.ndarray):
            return np.clip(mix, -1.0, 1.0).astype(np.float32, copy=False)
        return [
            [clamp(value, -1.0, 1.0) for value in mix[0]],
            [clamp(value, -1.0, 1.0) for value in mix[1]],
        ]

    def _generate_realtime_chunk_bytes(self, frame_count: int) -> bytes:
        if frame_count <= 0:
            return b''
        if np is not None:
            mix = np.zeros((2, frame_count), dtype=np.float32)
        else:
            mix = [[0.0] * frame_count, [0.0] * frame_count]

        if not self.project.loop_enabled:
            if self._realtime_reset_pending:
                self._reset_realtime_track_states()
            segment = self._render_realtime_segment_cached(self._playback_frame_position, frame_count)
            if np is not None and isinstance(mix, np.ndarray) and isinstance(segment, np.ndarray):
                mix[:, :frame_count] = segment
            else:
                for channel in range(2):
                    for i in range(frame_count):
                        mix[channel][i] = segment[channel][i]
            self._playback_frame_position += frame_count
            self._realtime_reset_pending = False
        else:
            remaining = frame_count
            offset = 0
            loop_start_frame, loop_end_frame = self._loop_frame_bounds()
            if self._playback_frame_position < loop_start_frame or self._playback_frame_position >= loop_end_frame:
                self._playback_frame_position = loop_start_frame
                self._realtime_reset_pending = True

            while remaining > 0:
                if self._playback_frame_position < loop_start_frame or self._playback_frame_position >= loop_end_frame:
                    self._playback_frame_position = loop_start_frame
                    self._realtime_reset_pending = True
                segment_frames = min(remaining, max(1, loop_end_frame - self._playback_frame_position))
                if self._realtime_reset_pending:
                    self._reset_realtime_track_states()
                segment = self._render_realtime_segment_cached(
                    self._playback_frame_position,
                    segment_frames,
                    available_frames=max(1, loop_end_frame - self._playback_frame_position),
                )
                if np is not None and isinstance(mix, np.ndarray) and isinstance(segment, np.ndarray):
                    mix[:, offset:offset + segment_frames] = segment
                else:
                    for channel in range(2):
                        for i in range(segment_frames):
                            mix[channel][offset + i] = segment[channel][i]
                self._playback_frame_position += segment_frames
                remaining -= segment_frames
                offset += segment_frames
                self._realtime_reset_pending = False
                if self._playback_frame_position >= loop_end_frame:
                    self._playback_frame_position = loop_start_frame
                    self._realtime_reset_pending = True

        self._playback_generated_total_frames += frame_count
        self.project.playhead_sec = sample_frame_to_seconds(self._playback_frame_position, self._playback_sample_rate)
        return encode_pcm16_stereo_samples(mix)

    def _pump_realtime_audio(self) -> None:
        if not self._playback_active or self._playback_sink is None or self._playback_sink_device is None:
            return
        bytes_per_frame = self._playback_channel_count * 2
        minimum_bytes = max(bytes_per_frame, self._playback_chunk_frames * bytes_per_frame)
        try:
            buffer_frames = int(max(self._playback_chunk_frames, self._playback_sink.bufferFrameCount()))
        except Exception:
            buffer_frames = self._playback_chunk_frames * 4
        max_writes = max(8, int(math.ceil(buffer_frames / max(1, self._playback_chunk_frames))) + 2)
        writes = 0
        try:
            while self._playback_sink.bytesFree() >= minimum_bytes and writes < max_writes:
                chunk = self._generate_realtime_chunk_bytes(self._playback_chunk_frames)
                if not chunk:
                    break
                written = self._playback_sink_device.write(chunk)
                if written <= 0:
                    break
                writes += 1
        except Exception as exc:
            self.stop_playback()
            self.statusBar().showMessage(f'Realtime playback stopped: {exc}')

    def _on_playback_sink_state_changed(self, state) -> None:
        if state == QtMultimedia.QtAudio.State.StoppedState and self._playback_sink is not None:
            error = self._playback_sink.error()
            if error != QtMultimedia.QtAudio.Error.NoError:
                self.statusBar().showMessage(f'Playback audio stopped: {error}')

    def start_playback(self) -> None:
        self._sync_playback_loop_state()
        start_sec = self.project.playhead_sec
        if self.project.loop_enabled and (start_sec < self.project.left_locator_sec or start_sec > self.project.right_locator_sec):
            start_sec = self.project.left_locator_sec
        if not self._prepare_realtime_playback(start_sec):
            QtWidgets.QMessageBox.warning(self, 'Playback unavailable', 'Could not start the realtime audio output stream.')
            return
        self.set_playhead_position(start_sec)
        self.playback_timer.start()
        self._refresh_transport_controls()
        self.statusBar().showMessage(f'Playback started at {self.project.bpm} BPM (realtime)')

    def stop_playback(self) -> None:
        should_reset = hasattr(self, 'playback_timer') and (not self.playback_timer.isActive())
        if hasattr(self, 'playback_timer'):
            self.playback_timer.stop()
        self._stop_realtime_audio_sink()
        self._refresh_transport_controls()
        if should_reset:
            self.set_playhead_position(0.0)
            self.statusBar().showMessage('Playback reset to 0.00s')
            return
        self.statusBar().showMessage('Playback stopped')

    def _tick_playback(self) -> None:
        if not self._playback_active:
            return
        self.set_playhead_position(self._current_audible_playback_sec(), playback_tick=True)

    def update_tempo(self, bpm: int) -> None:
        new_bpm = int(clamp(int(bpm), 20, 300))
        old_bpm = int(self.project.bpm)
        if new_bpm == old_bpm:
            self.project.bpm = new_bpm
            return

        was_playing = bool(hasattr(self, 'playback_timer') and self.playback_timer.isActive())
        keep_musical_transport = True
        if keep_musical_transport:
            self._rescale_transport_for_tempo_change(old_bpm, new_bpm)
        self.project.bpm = new_bpm
        self._invalidate_playback_caches(reset_realtime=False)
        self._sync_playback_loop_state()
        self._update_locator_overlays()
        self._refresh_locator_spin_configuration()
        self._clear_realtime_mix_cache()
        if hasattr(self, 'tempo_spin') and self.tempo_spin.value() != new_bpm:
            self.tempo_spin.blockSignals(True)
            self.tempo_spin.setValue(new_bpm)
            self.tempo_spin.blockSignals(False)
        if was_playing:
            if keep_musical_transport:
                self._seek_media_to_project_time(self.project.playhead_sec)
            self._mark_realtime_track_states_for_reset()
        self._schedule_tempo_ui_refresh(seconds_layout_changed=not self._ruler_uses_bar_positions())
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
        device_name = 'system default soundcard'
        for device in QtMultimedia.QMediaDevices.audioOutputs():
            if bytes(device.id()).hex() == device_id:
                device_name = device.description()
                break
        if device_id and device_name == 'system default soundcard':
            self.selected_audio_output_id = ''
        self._apply_audio_buffer_preference()
        self._save_preferences()
        if self.playback_timer.isActive():
            current_sec = self.project.playhead_sec
            self.stop_playback()
            self.set_playhead_position(current_sec)
            self.start_playback()
        self.refresh_audio_output_menu()
        if self.selected_audio_output_id:
            self.statusBar().showMessage(f'Audio output set to {device_name}')
        else:
            self.statusBar().showMessage('Audio output set to system default soundcard')

    def set_audio_buffer_ms(self, value: int) -> None:
        self.audio_buffer_ms = int(clamp(value, 20, 500))
        self._apply_audio_buffer_preference()
        self._save_preferences()
        self.statusBar().showMessage(f'Playback audio buffer set to {self.audio_buffer_ms} ms')

    def set_playback_ui_refresh_ms(self, value: int) -> None:
        self.playback_ui_refresh_ms = int(clamp(value, 16, 200))
        self._playhead_ui_refresh_interval = max(0.001, self.playback_ui_refresh_ms / 1000.0)
        if hasattr(self, 'playback_timer'):
            self.playback_timer.setInterval(self.playback_ui_refresh_ms)
        self._save_preferences()
        self.statusBar().showMessage(f'Playhead refresh set to every {self.playback_ui_refresh_ms} ms')

    def set_prefer_gpu_rendering(self, enabled: bool) -> None:
        self.prefer_gpu_rendering = bool(enabled)
        self._save_preferences()
        state = 'enabled' if self.prefer_gpu_rendering else 'disabled'
        self.statusBar().showMessage(f'Prefer GPU rendering {state}. Restart the app to apply.')

    def _apply_selected_audio_output(self) -> None:
        return

    def _apply_audio_buffer_preference(self) -> None:
        if getattr(self, '_playback_sink', None) is not None:
            try:
                self._playback_sink.setBufferFrameCount(max(self._playback_chunk_frames * 4, int((self._playback_sample_rate * self.audio_buffer_ms) / 1000)))
            except Exception:
                pass

    def _rack_vsti_path(self, rack_name: str) -> str:
        for vst in self.project.vsti_rack:
            if vst.name == rack_name:
                return vst.path
        return ''

    def _rack_vsti_entry(self, rack_name: str) -> VSTInstrument | None:
        for vst in self.project.vsti_rack:
            if vst.name == rack_name:
                return vst
        return None

    def available_fx_plugin_names(self) -> list[str]:
        return [entry.name for entry in self.project.vsti_rack if entry.host_supported and entry.is_effect]

    def available_instrument_plugin_names(self) -> list[str]:
        return [entry.name for entry in self.project.vsti_rack if entry.host_supported and entry.is_instrument]

    def _describe_plugin_path(self, plugin_path: str) -> tuple[str, bool, bool, str, bool, str]:
        normalized_path = self._normalized_vsti_path(plugin_path)
        cached = self.vsti_description_cache.get(normalized_path)
        if cached is not None:
            return cached

        default_name = Path(normalized_path).stem
        suffix = Path(normalized_path).suffix.lower()
        if suffix != '.vst3':
            result = (
                default_name,
                False,
                False,
                '',
                False,
                'This backend currently hosts VST3 plugins only. VST2-style .dll plugins will fall back to the built-in synth.',
            )
            self.vsti_description_cache[normalized_path] = result
            return result
        if not PEDALBOARD_AVAILABLE:
            result = (default_name, True, False, '', True, '')
            self.vsti_description_cache[normalized_path] = result
            return result
        try:
            plugin = self.vsti_binary_loader.handle(normalized_path)
            if plugin is None:
                plugin, _resolved_path = load_vst_plugin_with_fallback(normalized_path)
            name = str(getattr(plugin, 'name', '') or default_name)
            is_instrument = bool(getattr(plugin, 'is_instrument', False))
            is_effect = bool(getattr(plugin, 'is_effect', False))
            category = str(getattr(plugin, 'category', '') or '')
            result = (name, is_instrument, is_effect, category, True, '')
        except Exception as exc:
            result = (default_name, False, False, '', False, str(exc))
        self.vsti_description_cache[normalized_path] = result
        return result

    def _normalized_vsti_path(self, path: str) -> str:
        return str(Path(path).expanduser().resolve())

    def _is_valid_vsti_plugin_path(self, path: str) -> bool:
        try:
            resolved = Path(path).expanduser().resolve()
        except Exception:
            return False
        if not resolved.exists():
            return False
        for parent in resolved.parents:
            if parent.suffix.lower() == '.vst3':
                return False
        if resolved.is_dir():
            return resolved.suffix.lower() == '.vst3'
        return resolved.suffix.lower() in {'.dll', '.so', '.vst3'}

    def _add_vsti_to_rack(self, plugin_path: str, show_status: bool = True, eager_load: bool = True) -> bool:
        normalized_path = self._normalized_vsti_path(plugin_path)
        if not self._is_valid_vsti_plugin_path(normalized_path):
            if show_status:
                QtWidgets.QMessageBox.warning(
                    self,
                    'Invalid VSTI location',
                    'Choose a valid plugin file (`.dll`, `.so`) or a `.vst3` bundle.',
                )
            return False

        existing_paths = {self._normalized_vsti_path(v.path) for v in self.project.vsti_rack}
        if normalized_path in existing_paths:
            if show_status:
                self.statusBar().showMessage(f'VSTI already in rack: {Path(normalized_path).name}')
            return False

        name, is_instrument, is_effect, category, host_supported, host_error = self._describe_plugin_path(normalized_path)
        if not host_supported:
            if show_status:
                self.statusBar().showMessage(f'Skipped unsupported VST: {name}')
            return False
        if any(v.name == name for v in self.project.vsti_rack):
            name = f'{name} ({hashlib.sha1(normalized_path.encode("utf-8")).hexdigest()[:6]})'
        self.vsti_description_cache[normalized_path] = (name, is_instrument, is_effect, category, host_supported, host_error)

        self.project.vsti_rack.append(
            VSTInstrument(
                name=name,
                path=normalized_path,
                plugin_name=name,
                is_instrument=is_instrument,
                is_effect=is_effect,
                category=category,
                host_supported=host_supported,
                host_error=host_error,
            )
        )
        if normalized_path not in self.project.vsti_paths:
            self.project.vsti_paths.append(normalized_path)
        if eager_load:
            self._load_vsti_binary_path(normalized_path, show_message=False)
        return True

    def _sync_discovered_vstis_to_rack(self, paths: list[str] | None = None, eager_load: bool = False) -> int:
        added = 0
        existing_rack_paths = {self._normalized_vsti_path(v.path) for v in self.project.vsti_rack}
        source_paths = list(paths) if paths is not None else list(self.project.vsti_paths)
        for path in source_paths:
            normalized = self._normalized_vsti_path(path)
            if not self._is_valid_vsti_plugin_path(normalized):
                continue
            if normalized in existing_rack_paths:
                continue
            if self._add_vsti_to_rack(normalized, show_status=False, eager_load=eager_load):
                existing_rack_paths.add(normalized)
                added += 1
        return added

    def _dedupe_and_filter_vsti_state(self) -> None:
        unique_paths: list[str] = []
        seen_paths: set[str] = set()
        for path in self.project.vsti_paths:
            normalized = self._normalized_vsti_path(path)
            if normalized in seen_paths:
                continue
            if not self._is_valid_vsti_plugin_path(normalized):
                continue
            seen_paths.add(normalized)
            unique_paths.append(normalized)
        self.project.vsti_paths = unique_paths

        rack: list[VSTInstrument] = []
        rack_seen: set[str] = set()
        for vst in self.project.vsti_rack:
            normalized = self._normalized_vsti_path(vst.path)
            if normalized in rack_seen:
                continue
            if normalized not in seen_paths:
                continue
            rack_seen.add(normalized)
            preserved = VSTInstrument(
                name=vst.name or Path(normalized).stem,
                path=normalized,
                plugin_name=vst.plugin_name or vst.name or Path(normalized).stem,
                is_instrument=bool(vst.is_instrument),
                is_effect=bool(vst.is_effect),
                category=vst.category,
                host_supported=bool(vst.host_supported),
                host_error=vst.host_error,
            )
            self.vsti_description_cache[normalized] = (
                preserved.name,
                preserved.is_instrument,
                preserved.is_effect,
                preserved.category,
                preserved.host_supported,
                preserved.host_error,
            )
            if not preserved.host_supported:
                continue
            rack.append(preserved)
        self.project.vsti_rack = rack

    def vsti_parameter_names_for_rack(self, rack_name: str) -> list[str]:
        plugin_path = self._rack_vsti_path(rack_name)
        if not plugin_path:
            return []
        return list(self.vsti_plugin_metadata.get(self._normalized_vsti_path(plugin_path), []))

    def _capture_vsti_metadata(self, plugin_path: str, plugin_instance=None) -> None:
        normalized_path = self._normalized_vsti_path(plugin_path)
        if not PEDALBOARD_AVAILABLE or normalized_path in self.vsti_plugin_metadata:
            return
        try:
            plugin = plugin_instance
            if plugin is None:
                plugin, _resolved_path = load_vst_plugin_with_fallback(normalized_path)
            names = [str(name) for name in plugin.parameters.keys()]
            self.vsti_plugin_metadata[normalized_path] = names
        except Exception:
            self.vsti_plugin_metadata[normalized_path] = []

    def _load_rack_plugin(self, rack_name: str):
        if not PEDALBOARD_AVAILABLE or not load_plugin:
            return None
        entry = self._rack_vsti_entry(rack_name)
        if entry is None or not Path(entry.path).exists():
            return None
        plugin, _resolved_path = load_vst_plugin_with_fallback(entry.path)
        self._capture_vsti_metadata(entry.path, plugin)
        return plugin

    @staticmethod
    def _plugin_parameter_snapshot(plugin) -> dict[str, float]:
        snapshot: dict[str, float] = {}
        try:
            for idx, (name, param) in enumerate(plugin.parameters.items(), start=1):
                raw_value = getattr(param, 'raw_value', None)
                if raw_value is None:
                    continue
                key = str(name) if name else f'Param {idx}'
                snapshot[key] = float(clamp(float(raw_value) * 100.0, 0.0, 100.0))
        except Exception:
            return {}
        return snapshot

    @staticmethod
    def _default_vsti_state_cache_path(track: TrackState, entry: VSTInstrument) -> Path:
        track_key = hashlib.sha1(f'{track.name}:{track.track_type}:{track.midi_channel}'.encode('utf-8')).hexdigest()[:12]
        plugin_key = hashlib.sha1(entry.path.encode('utf-8')).hexdigest()[:12]
        return RENDER_DIR / '_vsti_state' / f'{track_key}_{plugin_key}.bin'

    def _load_saved_vsti_plugin_state(self, plugin, track: TrackState, entry: VSTInstrument) -> bool:
        state_path = Path(track.vsti_state_path) if track.vsti_state_path else self._default_vsti_state_cache_path(track, entry)
        if not state_path.exists():
            return False
        try:
            raw_state = state_path.read_bytes()
        except Exception:
            return False
        if not raw_state:
            return False
        for attr in ('raw_state', 'preset_data'):
            try:
                setattr(plugin, attr, raw_state)
                track.vsti_state_path = str(state_path)
                return True
            except Exception:
                continue
        return False

    def _save_vsti_plugin_state(self, plugin, track: TrackState, entry: VSTInstrument) -> bool:
        raw_state = b''
        for attr in ('raw_state', 'preset_data'):
            try:
                raw_state = bytes(getattr(plugin, attr) or b'')
            except Exception:
                raw_state = b''
            if raw_state:
                break
        if not raw_state:
            return False

        state_path = self._default_vsti_state_cache_path(track, entry)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            state_path.write_bytes(raw_state)
        except Exception:
            return False
        track.vsti_state_path = str(state_path)
        return True

    @staticmethod
    def _as_mono_audio(data: object) -> object:
        if np is None:
            return data
        audio = np.asarray(data, dtype=np.float32)
        if audio.ndim == 1:
            return audio
        if audio.ndim == 2:
            if audio.shape[0] <= audio.shape[1]:
                return audio.mean(axis=0).astype(np.float32, copy=False)
            return audio.mean(axis=1).astype(np.float32, copy=False)
        return audio.reshape(-1).astype(np.float32, copy=False)

    @staticmethod
    def _ensure_mono_sample_count(data: object, sample_count: int) -> object:
        target = max(1, int(sample_count))
        if np is not None:
            audio = np.asarray(MainWindow._as_mono_audio(data), dtype=np.float32).reshape(-1)
            if audio.shape[0] == target:
                return audio.astype(np.float32, copy=False)
            if audio.shape[0] > target:
                return audio[:target].astype(np.float32, copy=False)
            padded = np.zeros(target, dtype=np.float32)
            padded[:audio.shape[0]] = audio
            return padded

        if isinstance(data, (list, tuple)) and data and isinstance(data[0], (list, tuple)):
            channels = [list(channel) for channel in data if isinstance(channel, (list, tuple))]
            length = max((len(channel) for channel in channels), default=0)
            mono = []
            for idx in range(length):
                total = 0.0
                count = 0
                for channel in channels:
                    if idx < len(channel):
                        total += float(channel[idx])
                        count += 1
                mono.append(total / max(1, count))
        else:
            try:
                mono = [float(value) for value in data]
            except Exception:
                mono = [0.0]
        if len(mono) > target:
            return mono[:target]
        if len(mono) < target:
            mono.extend([0.0] * (target - len(mono)))
        return mono

    @staticmethod
    def _ensure_stereo_sample_count(data: object, sample_count: int) -> object:
        target = max(1, int(sample_count))
        if np is not None:
            audio = np.asarray(data, dtype=np.float32)
            if audio.ndim == 1:
                mono = np.asarray(MainWindow._ensure_mono_sample_count(audio, target), dtype=np.float32)
                return np.vstack((mono, mono)).astype(np.float32, copy=False)
            if audio.ndim == 2:
                if audio.shape[0] == 2:
                    channels = audio.astype(np.float32, copy=False)
                elif audio.shape[1] == 2:
                    channels = audio.T.astype(np.float32, copy=False)
                else:
                    mono = np.asarray(MainWindow._ensure_mono_sample_count(audio, target), dtype=np.float32)
                    return np.vstack((mono, mono)).astype(np.float32, copy=False)
                if channels.shape[1] == target:
                    return channels
                if channels.shape[1] > target:
                    return channels[:, :target].astype(np.float32, copy=False)
                padded = np.zeros((2, target), dtype=np.float32)
                padded[:, :channels.shape[1]] = channels[:, :channels.shape[1]]
                return padded
            mono = np.asarray(MainWindow._ensure_mono_sample_count(audio, target), dtype=np.float32)
            return np.vstack((mono, mono)).astype(np.float32, copy=False)

        if isinstance(data, (list, tuple)) and len(data) >= 2 and all(isinstance(channel, (list, tuple)) for channel in data[:2]):
            left = [float(value) for value in data[0]]
            right = [float(value) for value in data[1]]
        else:
            mono = list(MainWindow._ensure_mono_sample_count(data, target))
            return [mono[:], mono[:]]
        if len(left) > target:
            left = left[:target]
        if len(right) > target:
            right = right[:target]
        if len(left) < target:
            left.extend([0.0] * (target - len(left)))
        if len(right) < target:
            right.extend([0.0] * (target - len(right)))
        return [left, right]

    def _apply_saved_plugin_parameters(self, plugin, values: dict[str, float]) -> None:
        if not PEDALBOARD_AVAILABLE or not values:
            return
        try:
            param_names = [str(name) for name in plugin.parameters.keys()]
            for idx, param in enumerate(plugin.parameters.values()):
                key = param_names[idx] if idx < len(param_names) else f'Param {idx + 1}'
                legacy_key = f'Param {idx + 1}'
                if key in values:
                    param_value = values[key]
                elif legacy_key in values:
                    param_value = values[legacy_key]
                else:
                    continue
                try:
                    param.raw_value = max(0.0, min(1.0, float(param_value) / 100.0))
                except Exception:
                    continue
        except Exception:
            return

    def _is_bundled_vsti(self, vst: VSTInstrument) -> bool:
        try:
            return Path(vst.path).resolve().is_relative_to(self.vsti_directory.resolve())
        except Exception:
            return False

    def _screen_available_geometry(self) -> QtCore.QRect | None:
        screen = None
        if self.windowHandle() is not None:
            screen = self.windowHandle().screen()
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return None
        return screen.availableGeometry()

    def _center_widget_on_screen(self, widget: QtWidgets.QWidget, *, width: int | None = None, height: int | None = None) -> None:
        available = self._screen_available_geometry()
        if available is None:
            return
        if width is not None or height is not None:
            target_width = width if width is not None else widget.width()
            target_height = height if height is not None else widget.height()
            widget.resize(target_width, target_height)
        frame = widget.frameGeometry()
        frame.moveCenter(available.center())
        widget.move(frame.topLeft())

    def _center_dialog(self, dialog: QtWidgets.QDialog) -> None:
        available = self._screen_available_geometry()
        if available is None:
            return
        width = min(max(dialog.width(), 760), max(640, available.width() - 80))
        height = min(max(dialog.height(), 620), max(480, available.height() - 80))
        self._center_widget_on_screen(dialog, width=width, height=height)

    def _center_foreground_native_window_async(self, *, delay_sec: float = 0.12, retries: int = 12) -> None:
        if os.name != 'nt':
            return

        def worker() -> None:
            user32 = ctypes.windll.user32

            class RECT(ctypes.Structure):
                _fields_ = [('left', ctypes.c_long), ('top', ctypes.c_long), ('right', ctypes.c_long), ('bottom', ctypes.c_long)]

            class MONITORINFO(ctypes.Structure):
                _fields_ = [('cbSize', ctypes.c_ulong), ('rcMonitor', RECT), ('rcWork', RECT), ('dwFlags', ctypes.c_ulong)]

            try:
                main_hwnd = int(self.winId()) if self.winId() else 0
            except Exception:
                main_hwnd = 0

            time.sleep(max(0.0, float(delay_sec)))
            for _attempt in range(max(1, int(retries))):
                try:
                    hwnd = user32.GetForegroundWindow()
                    if not hwnd or hwnd == main_hwnd:
                        time.sleep(0.08)
                        continue
                    rect = RECT()
                    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                        time.sleep(0.08)
                        continue
                    width = max(1, rect.right - rect.left)
                    height = max(1, rect.bottom - rect.top)
                    monitor = user32.MonitorFromWindow(hwnd, 2)
                    monitor_info = MONITORINFO()
                    monitor_info.cbSize = ctypes.sizeof(MONITORINFO)
                    if monitor and user32.GetMonitorInfoW(monitor, ctypes.byref(monitor_info)):
                        work = monitor_info.rcWork
                    else:
                        work = RECT()
                        user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(work), 0)
                    x = int(work.left + max(0, ((work.right - work.left) - width) // 2))
                    y = int(work.top + max(0, ((work.bottom - work.top) - height) // 2))
                    user32.SetWindowPos(hwnd, None, x, y, 0, 0, 0x0001 | 0x0004 | 0x0010)
                    return
                except Exception:
                    time.sleep(0.08)

        threading.Thread(target=worker, name='center-native-vst-window', daemon=True).start()

    def _bundled_vsti_theme(self, vst_name: str) -> dict[str, str]:
        normalized = vst_name.strip().lower()
        if '808' in normalized:
            return {
                'title': '808 Circuit',
                'subtitle': 'Longer subs, softer metallic hats, and warmer analogue body for classic 808-style drum patterns.',
                'accent': '#FFAA52',
                'accent_soft': '#5A2E11',
                'panel': '#171513',
                'hero_a': '#4A2A12',
                'hero_b': '#11161C',
            }
        if '303' in normalized:
            return {
                'title': 'Acid Lane',
                'subtitle': 'Resonant cutoff sweeps, squelch, and tighter decay in a more polished acid-bass control room.',
                'accent': '#74F0A8',
                'accent_soft': '#143826',
                'panel': '#121A16',
                'hero_a': '#103222',
                'hero_b': '#10161C',
            }
        if 'drum' in normalized:
            return {
                'title': '909 Lab',
                'subtitle': 'Punchier 909-style hits, sharper hats, and quicker transient shaping for classic drum-machine grooves.',
                'accent': '#FF8A3D',
                'accent_soft': '#5E2B12',
                'panel': '#151A20',
                'hero_a': '#3F1D10',
                'hero_b': '#10161C',
            }
        if 'bass' in normalized:
            return {
                'title': 'Bass Forge',
                'subtitle': 'Sub weight, edge, and motion for a firmer low-end voice.',
                'accent': '#65E0A3',
                'accent_soft': '#123829',
                'panel': '#121920',
                'hero_a': '#113224',
                'hero_b': '#0F171E',
            }
        if 'lead' in normalized:
            return {
                'title': 'Lead Arc',
                'subtitle': 'Sharper attack, brighter harmonics, and more forward motion for hook lines.',
                'accent': '#FF7676',
                'accent_soft': '#4C1D20',
                'panel': '#18171B',
                'hero_a': '#4A1E26',
                'hero_b': '#11151C',
            }
        if 'pad' in normalized:
            return {
                'title': 'Pad Atlas',
                'subtitle': 'Slow blooms, wide stereo drift, and softer harmonic clouds for beds and lifts.',
                'accent': '#8DD7FF',
                'accent_soft': '#193B4D',
                'panel': '#131A20',
                'hero_a': '#173B4C',
                'hero_b': '#10171E',
            }
        if 'pluck' in normalized:
            return {
                'title': 'Pluck Deck',
                'subtitle': 'Tight transients, short tails, and clear bite for rhythmic hooks.',
                'accent': '#FFB26B',
                'accent_soft': '#4B2D14',
                'panel': '#191813',
                'hero_a': '#4A2F16',
                'hero_b': '#12171D',
            }
        if 'string' in normalized:
            return {
                'title': 'String Bloom',
                'subtitle': 'Ensemble width, movement, and shimmer in a softer performance view.',
                'accent': '#F5C76D',
                'accent_soft': '#4B3915',
                'panel': '#171922',
                'hero_a': '#46371A',
                'hero_b': '#11161D',
            }
        if 'sampler' in normalized:
            return {
                'title': 'Sampler Deck',
                'subtitle': 'Switch sample banks, trim the playback window, and shape a more sample-forward voice.',
                'accent': '#C694FF',
                'accent_soft': '#3B2352',
                'panel': '#171422',
                'hero_a': '#322044',
                'hero_b': '#10141B',
            }
        return {
            'title': 'Control Room',
            'subtitle': 'Fine tune the bundled instrument and save the result back to the track.',
            'accent': '#6CB8FF',
            'accent_soft': '#153552',
            'panel': '#141A20',
            'hero_a': '#173552',
            'hero_b': '#10161D',
        }

    def _open_vsti_wrapper_dialog(self, vst: VSTInstrument, track: TrackState) -> None:
        plugin = None
        parameter_items: list[tuple[str, object]] = []
        if PEDALBOARD_AVAILABLE and vst.host_supported:
            try:
                plugin = self._load_rack_plugin(vst.name)
                if plugin is not None:
                    self._load_saved_vsti_plugin_state(plugin, track, vst)
                    self._apply_saved_plugin_parameters(plugin, track.vsti_parameters)
                    parameter_items = [(str(key), param) for key, param in plugin.parameters.items()]
            except Exception:
                plugin = None

        self._capture_vsti_metadata(vst.path)
        param_names = [key for key, _param in parameter_items] if parameter_items else self.vsti_parameter_names_for_rack(vst.name)
        if not param_names:
            param_names = [f'Param {i}' for i in range(1, 9)]

        bundled_editor = self._is_bundled_vsti(vst)
        theme = self._bundled_vsti_theme(vst.name)
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f'VST Instrument Editor - {vst.name}')
        dialog.setModal(True)
        dialog.resize(980, 720)
        dialog.setSizeGripEnabled(True)
        if bundled_editor:
            dialog.setStyleSheet(
                f"""
                QDialog {{
                    background-color: #0E1218;
                    color: #F2F5F8;
                }}
                QFrame#vstiHero {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {theme['hero_a']}, stop:1 {theme['hero_b']});
                    border: 1px solid {theme['accent_soft']};
                    border-radius: 20px;
                }}
                QLabel#vstiHeroTitle {{
                    font-size: 28px;
                    font-weight: 700;
                    color: #FFFFFF;
                }}
                QLabel#vstiHeroSubtitle {{
                    font-size: 14px;
                    color: #DDE7EF;
                }}
                QLabel#vstiHeroBadge {{
                    background-color: {theme['accent_soft']};
                    color: {theme['accent']};
                    border: 1px solid {theme['accent']};
                    border-radius: 11px;
                    padding: 4px 10px;
                    font-weight: 700;
                }}
                QFrame#vstiCard {{
                    background-color: {theme['panel']};
                    border: 1px solid {theme['accent_soft']};
                    border-radius: 18px;
                }}
                QLabel#vstiCardTitle {{
                    font-size: 13px;
                    font-weight: 700;
                    color: #F7F9FB;
                }}
                QLabel#vstiCardHint {{
                    font-size: 11px;
                    color: #9FB0C0;
                }}
                QDial {{
                    background: transparent;
                }}
                QComboBox, QDoubleSpinBox {{
                    background-color: #0C1015;
                    border: 1px solid {theme['accent_soft']};
                    border-radius: 9px;
                    padding: 6px 8px;
                    min-height: 28px;
                }}
                QDialogButtonBox QPushButton {{
                    background-color: {theme['accent_soft']};
                    border: 1px solid {theme['accent']};
                    border-radius: 10px;
                    padding: 8px 14px;
                    min-width: 110px;
                    color: #FFFFFF;
                }}
                QScrollArea {{
                    border: none;
                    background: transparent;
                }}
                """
            )
        layout = QtWidgets.QVBoxLayout(dialog)
        if bundled_editor:
            hero = QtWidgets.QFrame()
            hero.setObjectName('vstiHero')
            hero_layout = QtWidgets.QVBoxLayout(hero)
            hero_layout.setContentsMargins(22, 20, 22, 20)
            hero_layout.setSpacing(8)
            badge = QtWidgets.QLabel(f"{theme['title']}  |  {vst.name}")
            badge.setObjectName('vstiHeroBadge')
            badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
            hero_layout.addWidget(badge, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
            title = QtWidgets.QLabel(theme['title'])
            title.setObjectName('vstiHeroTitle')
            hero_layout.addWidget(title)
            subtitle = QtWidgets.QLabel(theme['subtitle'])
            subtitle.setObjectName('vstiHeroSubtitle')
            subtitle.setWordWrap(True)
            hero_layout.addWidget(subtitle)
            layout.addWidget(hero)
        else:
            info = QtWidgets.QLabel(
                'This VST editor uses the in-app wrapper because a native plugin window is unavailable here. Changes are saved back to the track and applied during playback/render.'
            )
            info.setWordWrap(True)
            layout.addWidget(info)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        apply_changes: list[callable] = []

        if bundled_editor:
            body_layout = QtWidgets.QVBoxLayout(body)
            body_layout.setContentsMargins(8, 8, 8, 8)
            body_layout.setSpacing(14)
            bundled_knob_size = 64
            intro = QtWidgets.QLabel('Turn the knobs, shape the patch, and press OK to commit the sound to the current track.')
            intro.setObjectName('vstiCardHint')
            intro.setWordWrap(True)
            body_layout.addWidget(intro)
            knob_grid = QtWidgets.QGridLayout()
            knob_grid.setHorizontalSpacing(16)
            knob_grid.setVerticalSpacing(16)
            body_layout.addLayout(knob_grid)
            card_index = 0

            def add_card(title: str, control_widget: QtWidgets.QWidget, hint_text: str = '') -> None:
                nonlocal card_index
                card = QtWidgets.QFrame()
                card.setObjectName('vstiCard')
                card_layout = QtWidgets.QVBoxLayout(card)
                card_layout.setContentsMargins(14, 14, 14, 14)
                card_layout.setSpacing(10)
                title_label = QtWidgets.QLabel(title)
                title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                title_label.setObjectName('vstiCardTitle')
                card_layout.addWidget(title_label)
                card_layout.addWidget(control_widget, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
                if hint_text:
                    hint_label = QtWidgets.QLabel(hint_text)
                    hint_label.setObjectName('vstiCardHint')
                    hint_label.setWordWrap(True)
                    hint_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    card_layout.addWidget(hint_label)
                row, col = divmod(card_index, 4)
                knob_grid.addWidget(card, row, col)
                card_index += 1

            if parameter_items:
                for key, param in parameter_items[:24]:
                    display_name = str(getattr(param, 'name', None) or key.replace('_', ' ').title())
                    valid_values = list(getattr(param, 'valid_values', []) or [])
                    min_value = getattr(param, 'min_value', None)
                    max_value = getattr(param, 'max_value', None)
                    raw_value = float(getattr(param, 'raw_value', 0.0) or 0.0)
                    string_value = str(getattr(param, 'string_value', '') or '')

                    if valid_values and all(isinstance(v, str) for v in valid_values):
                        combo = QtWidgets.QComboBox()
                        for value in valid_values:
                            combo.addItem(str(value), value)
                        current_idx = combo.findText(string_value)
                        if current_idx < 0:
                            current_idx = max(0, min(combo.count() - 1, int(round(raw_value * max(0, combo.count() - 1)))))
                        combo.setCurrentIndex(current_idx)
                        add_card(display_name, combo, 'Mode')

                        def apply_choice(combo=combo, param=param) -> None:
                            count = max(1, combo.count() - 1)
                            index = max(0, combo.currentIndex())
                            param.raw_value = 0.0 if count == 0 else float(index) / float(count)

                        apply_changes.append(apply_choice)
                        continue

                    if min_value is not None and max_value is not None:
                        min_float = float(min_value)
                        max_float = float(max_value)
                        current_value = min_float + (max_float - min_float) * raw_value
                        step = 0.01
                        if valid_values and all(isinstance(v, (int, float)) for v in valid_values) and len(valid_values) >= 2:
                            step = abs(float(valid_values[1]) - float(valid_values[0])) or step
                        elif max_float - min_float >= 100.0:
                            step = 1.0
                        elif max_float - min_float >= 10.0:
                            step = 0.1

                        dial = QtWidgets.QDial()
                        dial.setRange(0, 1000)
                        dial.setNotchesVisible(True)
                        dial.setWrapping(False)
                        dial.setFixedSize(bundled_knob_size, bundled_knob_size)
                        dial.setValue(int(round(raw_value * 1000.0)))

                        spin = QtWidgets.QDoubleSpinBox()
                        spin.setRange(min_float, max_float)
                        spin.setSingleStep(step)
                        spin.setDecimals(3 if step < 1.0 else 2)
                        spin.setValue(current_value)
                        spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

                        def dial_to_value(dial_value: int, lo=min_float, hi=max_float) -> float:
                            return lo + (hi - lo) * (float(dial_value) / 1000.0)

                        def value_to_dial(value: float, lo=min_float, hi=max_float) -> int:
                            if hi <= lo:
                                return 0
                            return int(round(((value - lo) / (hi - lo)) * 1000.0))

                        dial.valueChanged.connect(lambda v, spinbox=spin, convert=dial_to_value: spinbox.setValue(convert(v)))
                        spin.valueChanged.connect(lambda v, knob=dial, convert=value_to_dial: knob.setValue(convert(float(v))))

                        control_widget = QtWidgets.QWidget()
                        control_layout = QtWidgets.QVBoxLayout(control_widget)
                        control_layout.setContentsMargins(0, 0, 0, 0)
                        control_layout.setSpacing(6)
                        control_layout.addWidget(dial, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
                        control_layout.addWidget(spin)
                        add_card(display_name, control_widget, 'Knob')

                        def apply_numeric(spin=spin, param=param, lo=min_float, hi=max_float) -> None:
                            if hi <= lo:
                                param.raw_value = 0.0
                            else:
                                param.raw_value = max(0.0, min(1.0, (float(spin.value()) - lo) / (hi - lo)))

                        apply_changes.append(apply_numeric)
                        continue

                    dial = QtWidgets.QDial()
                    dial.setRange(0, 100)
                    dial.setValue(int(round(raw_value * 100.0)))
                    dial.setNotchesVisible(True)
                    dial.setWrapping(False)
                    dial.setFixedSize(bundled_knob_size, bundled_knob_size)
                    value_label = QtWidgets.QLabel(f'{dial.value()}%')
                    value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    dial.valueChanged.connect(lambda v, label=value_label: label.setText(f'{v}%'))
                    control_widget = QtWidgets.QWidget()
                    control_layout = QtWidgets.QVBoxLayout(control_widget)
                    control_layout.setContentsMargins(0, 0, 0, 0)
                    control_layout.setSpacing(6)
                    control_layout.addWidget(dial, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
                    control_layout.addWidget(value_label)
                    add_card(display_name, control_widget, 'Amount')

                    def apply_percent(dial=dial, param=param) -> None:
                        param.raw_value = max(0.0, min(1.0, float(dial.value()) / 100.0))

                    apply_changes.append(apply_percent)
            else:
                sliders: dict[str, QtWidgets.QDial] = {}
                for key in param_names[:16]:
                    dial = QtWidgets.QDial()
                    dial.setRange(0, 100)
                    dial.setValue(int(track.vsti_parameters.get(key, 50)))
                    dial.setNotchesVisible(True)
                    dial.setWrapping(False)
                    dial.setFixedSize(bundled_knob_size, bundled_knob_size)
                    value_label = QtWidgets.QLabel(f'{dial.value()}%')
                    value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    dial.valueChanged.connect(lambda v, label=value_label: label.setText(f'{v}%'))
                    control_widget = QtWidgets.QWidget()
                    control_layout = QtWidgets.QVBoxLayout(control_widget)
                    control_layout.setContentsMargins(0, 0, 0, 0)
                    control_layout.setSpacing(6)
                    control_layout.addWidget(dial, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
                    control_layout.addWidget(value_label)
                    add_card(key.replace('_', ' ').title(), control_widget, 'Saved control')
                    sliders[key] = dial
                apply_changes.append(lambda sliders=sliders: track.vsti_parameters.update({key: float(dial.value()) for key, dial in sliders.items()}))
        else:
            form = QtWidgets.QFormLayout(body)
            form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            if parameter_items:
                for key, param in parameter_items[:48]:
                    display_name = str(getattr(param, 'name', None) or key.replace('_', ' ').title())
                    valid_values = list(getattr(param, 'valid_values', []) or [])
                    min_value = getattr(param, 'min_value', None)
                    max_value = getattr(param, 'max_value', None)
                    raw_value = float(getattr(param, 'raw_value', 0.0) or 0.0)
                    string_value = str(getattr(param, 'string_value', '') or '')

                    if valid_values and all(isinstance(v, str) for v in valid_values):
                        combo = QtWidgets.QComboBox()
                        for value in valid_values:
                            combo.addItem(str(value), value)
                        current_idx = combo.findText(string_value)
                        if current_idx < 0:
                            current_idx = max(0, min(combo.count() - 1, int(round(raw_value * max(0, combo.count() - 1)))))
                        combo.setCurrentIndex(current_idx)
                        form.addRow(display_name, combo)

                        def apply_choice(combo=combo, param=param) -> None:
                            count = max(1, combo.count() - 1)
                            index = max(0, combo.currentIndex())
                            param.raw_value = 0.0 if count == 0 else float(index) / float(count)

                        apply_changes.append(apply_choice)
                        continue

                    if min_value is not None and max_value is not None:
                        min_float = float(min_value)
                        max_float = float(max_value)
                        current_value = min_float + (max_float - min_float) * raw_value
                        step = 0.01
                        if valid_values and all(isinstance(v, (int, float)) for v in valid_values) and len(valid_values) >= 2:
                            step = abs(float(valid_values[1]) - float(valid_values[0])) or step
                        elif max_float - min_float >= 100.0:
                            step = 1.0
                        elif max_float - min_float >= 10.0:
                            step = 0.1

                        knob = QtWidgets.QDial()
                        knob.setRange(0, 1000)
                        knob.setValue(int(round(raw_value * 1000.0)))
                        knob.setNotchesVisible(True)
                        knob.setWrapping(False)
                        knob.setFixedSize(76, 76)

                        spin = QtWidgets.QDoubleSpinBox()
                        spin.setRange(min_float, max_float)
                        spin.setSingleStep(step)
                        spin.setDecimals(3 if step < 1.0 else 2)
                        spin.setValue(current_value)

                        def knob_to_value(knob_value: int, lo=min_float, hi=max_float) -> float:
                            return lo + (hi - lo) * (float(knob_value) / 1000.0)

                        def value_to_knob(value: float, lo=min_float, hi=max_float) -> int:
                            if hi <= lo:
                                return 0
                            return int(round(((value - lo) / (hi - lo)) * 1000.0))

                        knob.valueChanged.connect(lambda v, spinbox=spin, convert=knob_to_value: spinbox.setValue(convert(v)))
                        spin.valueChanged.connect(lambda v, target=knob, convert=value_to_knob: target.setValue(convert(float(v))))

                        row = QtWidgets.QHBoxLayout()
                        row.addWidget(knob)
                        row.addWidget(spin)
                        row_widget = QtWidgets.QWidget()
                        row_widget.setLayout(row)
                        form.addRow(display_name, row_widget)

                        def apply_numeric(spin=spin, param=param, lo=min_float, hi=max_float) -> None:
                            if hi <= lo:
                                param.raw_value = 0.0
                            else:
                                param.raw_value = max(0.0, min(1.0, (float(spin.value()) - lo) / (hi - lo)))

                        apply_changes.append(apply_numeric)
                        continue

                    knob = KnobInput(0, 100, int(round(raw_value * 100.0)), '%')
                    form.addRow(display_name, knob)

                    def apply_percent(knob=knob, param=param) -> None:
                        param.raw_value = max(0.0, min(1.0, float(knob.value()) / 100.0))

                    apply_changes.append(apply_percent)
            else:
                sliders: dict[str, KnobInput] = {}
                for key in param_names[:24]:
                    knob = KnobInput(0, 100, int(track.vsti_parameters.get(key, 50)), '%')
                    form.addRow(key, knob)
                    sliders[key] = knob
                apply_changes.append(lambda sliders=sliders: track.vsti_parameters.update({key: float(knob.value()) for key, knob in sliders.items()}))

        scroll.setWidget(body)
        layout.addWidget(scroll)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        self._center_dialog(dialog)

        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        for apply_change in apply_changes:
            apply_change()
        if plugin is not None:
            snapshot = self._plugin_parameter_snapshot(plugin)
            if snapshot:
                track.vsti_parameters = snapshot
            self._save_vsti_plugin_state(plugin, track, vst)
        self.statusBar().showMessage(f'Updated VSTI wrapper controls: {vst.name}')
        self.on_track_instrument_changed()

    def _track_midi_messages(self, track: TrackState) -> list[mido.Message]:
        messages: list[mido.Message] = [
            mido.Message(
                'program_change',
                channel=int(clamp(track.midi_channel, 0, 15)),
                program=int(clamp(track.midi_program, 0, 127)),
                time=0.0,
            )
        ]
        for note in track.notes:
            start_frame = tick_to_sample_frame(note.start_tick, self._playback_sample_rate, self.project.bpm)
            end_frame = max(
                start_frame + 1,
                tick_to_sample_frame(note.start_tick + note.duration_tick, self._playback_sample_rate, self.project.bpm),
            )
            messages.append(
                mido.Message(
                    'note_on',
                    channel=int(clamp(track.midi_channel, 0, 15)),
                    note=int(clamp(note.pitch, 0, 127)),
                    velocity=int(clamp(note.velocity, 1, 127)),
                    time=sample_frame_to_seconds(start_frame, self._playback_sample_rate),
                )
            )
            messages.append(
                mido.Message(
                    'note_off',
                    channel=int(clamp(track.midi_channel, 0, 15)),
                    note=int(clamp(note.pitch, 0, 127)),
                    velocity=0,
                    time=sample_frame_to_seconds(end_frame, self._playback_sample_rate),
                )
            )
        order = {'program_change': 0, 'note_off': 1, 'note_on': 2}
        messages.sort(key=lambda msg: (float(msg.time), order.get(msg.type, 3)))
        return messages

    def _apply_vst_fx_chain(self, track: TrackState, data: object, sample_rate: int) -> object:
        if not PEDALBOARD_AVAILABLE or np is None or Pedalboard is None or not track.vst_fx_chain:
            return data

        effects = []
        for fx_name in track.vst_fx_chain:
            entry = self._rack_vsti_entry(fx_name)
            if entry is None or not entry.is_effect:
                continue
            try:
                plugin = self._load_rack_plugin(entry.name)
            except Exception:
                plugin = None
            if plugin is not None and bool(getattr(plugin, 'is_effect', False)):
                effects.append(plugin)
        if not effects:
            return data

        audio = np.asarray(data, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[None, :]
        processed = Pedalboard(effects)(audio, sample_rate)
        return self._as_mono_audio(processed)

    def _render_track_audio(self, track: TrackState) -> tuple[object, int]:
        sample_rate = 44100
        if np is not None:
            silent = np.zeros(1, dtype=np.float32)
        else:
            silent = [0.0]
        if track.track_type != 'instrument' or not track.notes:
            return silent, sample_rate

        entry = self._rack_vsti_entry(track.rack_vsti) if track.rack_vsti else None
        if track.instrument_mode == 'VSTI Rack' and entry is not None and not entry.host_supported:
            self.statusBar().showMessage(f'Unsupported rack VST for {track.name}: {entry.name}. Falling back to the built-in synth.')
        if track.instrument_mode == 'VSTI Rack' and entry is not None and entry.is_instrument and not PEDALBOARD_AVAILABLE:
            self.statusBar().showMessage(f'VST instrument hosting unavailable for {track.name}: {pedalboard_runtime_hint() or "install pedalboard and restart the app."}')
        if track.instrument_mode == 'VSTI Rack' and entry is not None and entry.is_instrument and PEDALBOARD_AVAILABLE and np is not None:
            try:
                plugin = self._load_rack_plugin(entry.name)
                if plugin is not None and bool(getattr(plugin, 'is_instrument', False)):
                    self._load_saved_vsti_plugin_state(plugin, track, entry)
                    self._apply_saved_plugin_parameters(plugin, track.vsti_parameters)
                    midi_messages = self._track_midi_messages(track)
                    max_tick = max((note.start_tick + note.duration_tick for note in track.notes), default=0)
                    duration = (max_tick / TICKS_PER_BEAT) * (60.0 / max(1, self.project.bpm)) + 1.0
                    rendered = plugin(
                        midi_messages,
                        sample_rate=sample_rate,
                        duration=max(1.0, duration),
                        num_channels=2,
                    )
                    audio = self._as_mono_audio(rendered)
                    gain_linear = max(0.0, float(track.volume)) * (10.0 ** (float(track.vsti_output_gain_db) / 20.0))
                    data = np.clip(np.asarray(audio, dtype=np.float32) * gain_linear, -1.0, 1.0)
                    data = self._apply_vst_fx_chain(track, data, sample_rate)
                    return data, sample_rate
            except Exception as exc:
                self.statusBar().showMessage(f'VST instrument fallback to synth for {track.name}: {exc}')

        data, sr = self.renderer.render_track_audio(track, self.project.bpm)

        if track.instrument_mode == 'VSTI Rack' and entry is not None and entry.is_effect:
            try:
                plugin = self._load_rack_plugin(entry.name)
                if plugin is not None and bool(getattr(plugin, 'is_effect', False)) and np is not None:
                    self._load_saved_vsti_plugin_state(plugin, track, entry)
                    self._apply_saved_plugin_parameters(plugin, track.vsti_parameters)
                    dry_audio = np.asarray(data, dtype=np.float32)
                    if dry_audio.ndim == 1:
                        dry_audio = dry_audio[None, :]
                    wet_audio = plugin(dry_audio, sr)
                    wet_audio = np.asarray(wet_audio, dtype=np.float32)
                    if wet_audio.ndim == 1:
                        wet_audio = wet_audio[None, :]
                    wet_mix = max(0.0, min(1.0, float(track.vsti_wet_mix) / 100.0))
                    blended = (wet_audio * wet_mix) + (dry_audio * (1.0 - wet_mix))
                    gain_linear = 10.0 ** (float(track.vsti_output_gain_db) / 20.0)
                    data = np.clip(self._as_mono_audio(blended) * gain_linear, -1.0, 1.0)
            except Exception as exc:
                self.statusBar().showMessage(f'VST effect fallback to synth for {track.name}: {exc}')

        data = self._apply_vst_fx_chain(track, data, sr)
        return data, sr

    @staticmethod
    def _cache_payload_hash(payload: object) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
        return hashlib.sha1(raw.encode('utf-8')).hexdigest()

    @staticmethod
    def _path_mtime_ns(path: Path) -> int:
        try:
            return path.stat().st_mtime_ns
        except Exception:
            return 0

    def _track_audio_cache_key(self, track: TrackState, idx: int) -> str:
        instrument_entry = self._rack_vsti_entry(track.rack_vsti) if track.rack_vsti else None
        payload = {
            'idx': idx,
            'bpm': self.project.bpm,
            'vst_host_backend': 'pedalboard' if PEDALBOARD_AVAILABLE else 'builtin',
            'track_type': track.track_type,
            'instrument': track.instrument,
            'instrument_mode': track.instrument_mode,
            'rack_vsti': track.rack_vsti,
            'rack_vsti_path': instrument_entry.path if instrument_entry else '',
            'midi_program': track.midi_program,
            'synth_profile': track.synth_profile,
            'volume': round(float(track.volume), 6),
            'pan': round(float(track.pan), 6),
            'vsti_parameters': sorted((key, round(float(value), 6)) for key, value in track.vsti_parameters.items()),
            'vsti_state_path': track.vsti_state_path,
            'vsti_state_mtime_ns': self._path_mtime_ns(Path(track.vsti_state_path)) if track.vsti_state_path else 0,
            'vsti_output_gain_db': round(float(track.vsti_output_gain_db), 6),
            'vsti_wet_mix': round(float(track.vsti_wet_mix), 6),
            'vst_fx_chain': list(track.vst_fx_chain),
            'vst_fx_paths': [self._rack_vsti_path(name) for name in track.vst_fx_chain],
            'notes': [(note.start_tick, note.duration_tick, note.pitch, note.velocity) for note in track.notes],
        }
        return self._cache_payload_hash(payload)

    def _get_track_playback_audio(self, idx: int, track: TrackState) -> tuple[object, int]:
        cache_key = self._track_audio_cache_key(track, idx)
        cache_id = (idx, cache_key)
        cached = self._track_playback_audio_cache.get(cache_id)
        if cached is not None:
            return cached

        stale_keys = [key for key in self._track_playback_audio_cache if key[0] == idx and key[1] != cache_key]
        for stale_key in stale_keys:
            self._track_playback_audio_cache.pop(stale_key, None)

        data, sr = self._render_track_audio(track)
        if sr != 44100:
            data = resample_samples(data, sr, 44100)
            sr = 44100
        if np is not None and isinstance(data, np.ndarray):
            cached_data: object = np.asarray(data, dtype=np.float32).copy()
        else:
            cached_data = list(data)
        self._track_playback_audio_cache[cache_id] = (cached_data, sr)
        return cached_data, sr

    def _project_content_end_seconds(self) -> float:
        duration_sec = max(float(self.project.right_locator_sec), float(self.project.playhead_sec))
        sec_per_tick = 60.0 / max(1, self.project.bpm) / TICKS_PER_BEAT
        for track in self.project.tracks:
            if track.track_type != 'instrument' or not track.notes:
                continue
            tail_sec = 1.5 if track.instrument_mode == 'VSTI Rack' else 0.5
            max_tick = max((note.start_tick + note.duration_tick for note in track.notes), default=0)
            duration_sec = max(duration_sec, (max_tick * sec_per_tick) + tail_sec)
        for clip in self.project.sample_clips:
            duration_sec = max(duration_sec, float(clip.start_sec) + float(clip.duration_sec))
        return max(duration_sec, 0.0)

    def _playback_mix_duration_seconds(self) -> float:
        return max(60.0, self._project_content_end_seconds())

    def _playback_mix_cache_signature(self, duration_sec: float) -> str:
        solo_tracks = {idx for idx, track in enumerate(self.project.tracks) if track.solo}
        active_tracks = []
        for idx, track in enumerate(self.project.tracks):
            if track.track_type != 'instrument' or not track.notes:
                continue
            if solo_tracks and idx not in solo_tracks:
                continue
            if track.mute:
                continue
            active_tracks.append({'idx': idx, 'audio': self._track_audio_cache_key(track, idx)})

        clips = []
        for clip in self.project.sample_clips:
            clip_path = Path(clip.path)
            resolved = str(clip_path.resolve()) if clip_path.exists() else clip.path
            track = self.project.tracks[clip.track_index] if 0 <= clip.track_index < len(self.project.tracks) else None
            clips.append(
                {
                    'path': resolved,
                    'mtime_ns': self._path_mtime_ns(clip_path),
                    'start_sec': round(float(clip.start_sec), 6),
                    'duration_sec': round(float(clip.duration_sec), 6),
                    'sample_rate': clip.sample_rate,
                    'track_index': clip.track_index,
                    'track_volume': round(float(track.volume), 6) if track else 0.0,
                    'track_mute': bool(track.mute) if track else False,
                    'track_solo': bool(track.solo) if track else False,
                    'track_fx': list(track.vst_fx_chain) if track else [],
                    'track_fx_paths': [self._rack_vsti_path(name) for name in track.vst_fx_chain] if track else [],
                }
            )

        return self._cache_payload_hash(
            {
                'duration_sec': round(float(duration_sec), 6),
                'bpm': self.project.bpm,
                'tracks': active_tracks,
                'clips': clips,
            }
        )

    def _build_playback_mix(self) -> bool:
        left = self.project.left_locator_sec
        right = self.project.right_locator_sec
        if right <= left:
            return False

        mix_duration_sec = self._playback_mix_duration_seconds()
        cache_key = self._playback_mix_cache_signature(mix_duration_sec)
        if self._playback_mix_cache_key == cache_key and self._playback_mix_wav_bytes:
            self._playback_mix_duration_sec = mix_duration_sec
            return True

        sample_rate = 44100
        mix_length = max(1, int(mix_duration_sec * sample_rate))
        if np is not None:
            mix: object = np.zeros(mix_length, dtype=np.float32)
        else:
            mix = [0.0] * mix_length
        has_audio = False

        solo_tracks = {idx for idx, t in enumerate(self.project.tracks) if t.solo}
        for idx, track in enumerate(self.project.tracks):
            if track.track_type != 'instrument' or not track.notes:
                continue
            if solo_tracks and idx not in solo_tracks:
                continue
            if track.mute:
                continue

            data, sr = self._get_track_playback_audio(idx, track)
            if sr != sample_rate:
                data = resample_samples(data, sr, sample_rate)

            if np is not None and isinstance(mix, np.ndarray) and isinstance(data, np.ndarray):
                count = min(mix.shape[0], data.shape[0])
                if count <= 0:
                    continue
                mix[:count] += data[:count]
            else:
                source = list(data)
                count = min(len(mix), len(source))
                if count <= 0:
                    continue
                for i in range(count):
                    mix[i] += source[i]
            has_audio = True

        for clip in self.project.sample_clips:
            if clip.track_index < 0 or clip.track_index >= len(self.project.tracks):
                continue
            clip_track = self.project.tracks[clip.track_index]
            if solo_tracks and clip.track_index not in solo_tracks:
                continue
            if clip_track.mute:
                continue
            wav_path = Path(clip.path)
            if wav_path.suffix.lower() == '.mp3':
                converted = RENDER_DIR / f'{wav_path.stem}_play.wav'
                convert_audio(wav_path, converted)
                wav_path = converted
            data, sr = load_wav_samples(wav_path)
            if sr != sample_rate:
                data = resample_samples(data, sr, sample_rate)
                sr = sample_rate

            data = self._apply_vst_fx_chain(clip_track, data, sr)

            clip_start = clip.start_sec
            clip_length = data.shape[0] if np is not None and isinstance(data, np.ndarray) else len(data)
            clip_end = clip.start_sec + (clip_length / sample_rate)
            if clip_end <= 0.0 or clip_start >= mix_duration_sec:
                continue
            src_start = 0
            dst_start = max(0, int(clip_start * sample_rate))
            if np is not None and isinstance(mix, np.ndarray) and isinstance(data, np.ndarray):
                count = min(data.shape[0] - src_start, mix.shape[0] - dst_start)
                if count <= 0:
                    continue
                mix[dst_start : dst_start + count] += data[src_start : src_start + count] * 0.7 * float(clip_track.volume)
            else:
                source = list(data)
                count = min(len(source) - src_start, len(mix) - dst_start)
                if count <= 0:
                    continue
                for i in range(count):
                    mix[dst_start + i] += source[src_start + i] * 0.7 * float(clip_track.volume)
            has_audio = True

        if not has_audio:
            self._playback_mix_cache_key = ''
            self._playback_mix_duration_sec = 0.0
            self._playback_mix_wav_bytes = b''
            return False

        if np is not None and isinstance(mix, np.ndarray):
            mix_for_output: object = np.clip(mix, -1.0, 1.0).astype(np.float32, copy=False)
        else:
            mix_for_output = [clamp(v, -1.0, 1.0) for v in mix]
        self._playback_mix_wav_bytes = encode_wav_samples(mix_for_output, sample_rate)
        self._playback_mix_cache_key = cache_key
        self._playback_mix_duration_sec = mix_length / sample_rate
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
        payload = load_startup_preferences()
        if not payload:
            return

        main_sizes = payload.get('main_splitter_sizes', self._main_splitter_sizes)
        note_editor_sizes = payload.get('note_editor_inner_sizes', self._note_editor_inner_sizes)
        self._tools_window_visible = bool(payload.get('tools_window_visible', True))
        self._tools_window_geometry_b64 = str(payload.get('tools_window_geometry_b64', '') or '')
        self._mixer_window_visible = bool(payload.get('mixer_window_visible', True))
        self._mixer_window_geometry_b64 = str(payload.get('mixer_window_geometry_b64', '') or '')
        self._transport_window_visible = bool(payload.get('transport_window_visible', True))
        self._transport_window_geometry_b64 = str(payload.get('transport_window_geometry_b64', '') or '')
        self._virtual_piano_window_visible = bool(payload.get('virtual_piano_window_visible', False))
        self._virtual_piano_window_geometry_b64 = str(payload.get('virtual_piano_window_geometry_b64', '') or '')
        self._virtual_piano_key_scale_percent = self._coerce_int(payload.get('virtual_piano_key_scale_percent', 50), 50, 35, 175)
        if isinstance(main_sizes, list) and len(main_sizes) == 2:
            try:
                self._main_splitter_sizes = [max(120, int(main_sizes[0])), max(240, int(main_sizes[1]))]
            except Exception:
                pass
        if isinstance(note_editor_sizes, list) and len(note_editor_sizes) == 2:
            try:
                self._note_editor_inner_sizes = [max(180, int(note_editor_sizes[0])), max(60, int(note_editor_sizes[1]))]
            except Exception:
                pass

        self.project.vsti_paths = [p for p in payload.get('vsti_paths', []) if isinstance(p, str)]
        self.project.vsti_folder_paths = [p for p in payload.get('vsti_folder_paths', []) if isinstance(p, str)]
        self.project.sample_paths = [p for p in payload.get('sample_paths', []) if isinstance(p, str)]
        self.selected_audio_output_id = str(payload.get('selected_audio_output_id', '') or '')
        self.audio_buffer_ms = int(clamp(float(payload.get('audio_buffer_ms', 80)), 20, 500))
        refresh_pref = payload.get('playback_ui_refresh_ms', 16)
        try:
            refresh_value = float(refresh_pref)
        except Exception:
            refresh_value = 16.0
        self.playback_ui_refresh_ms = int(clamp(min(refresh_value, 16.0), 16, 200))
        self.prefer_gpu_rendering = bool(payload.get('prefer_gpu_rendering', True))
        rack_paths = [p for p in payload.get('vsti_rack_paths', []) if isinstance(p, str)]
        saved_rack_state: dict[str, dict[str, object]] = {}
        for raw_entry in payload.get('vsti_rack_state', []):
            if not isinstance(raw_entry, dict):
                continue
            raw_path = raw_entry.get('path')
            if not isinstance(raw_path, str):
                continue
            saved_rack_state[self._normalized_vsti_path(raw_path)] = raw_entry
        self.project.vsti_paths = [p for p in self.project.vsti_paths if Path(p).exists()]
        self.project.vsti_folder_paths = [p for p in self.project.vsti_folder_paths if Path(p).exists() and Path(p).is_dir()]
        original_vsti_paths = list(self.project.vsti_paths)
        rack = []
        for path in rack_paths:
            normalized = self._normalized_vsti_path(path)
            if not Path(normalized).exists():
                continue
            saved_entry = saved_rack_state.get(normalized)
            if saved_entry is not None:
                entry = VSTInstrument(
                    name=str(saved_entry.get('name') or Path(normalized).stem),
                    path=normalized,
                    plugin_name=str(saved_entry.get('plugin_name') or saved_entry.get('name') or Path(normalized).stem),
                    is_instrument=bool(saved_entry.get('is_instrument', False)),
                    is_effect=bool(saved_entry.get('is_effect', False)),
                    category=str(saved_entry.get('category') or ''),
                    host_supported=bool(saved_entry.get('host_supported', True)),
                    host_error=str(saved_entry.get('host_error') or ''),
                )
                self.vsti_description_cache[normalized] = (
                    entry.name,
                    entry.is_instrument,
                    entry.is_effect,
                    entry.category,
                    entry.host_supported,
                    entry.host_error,
                )
            else:
                name, is_instrument, is_effect, category, host_supported, host_error = self._describe_plugin_path(normalized)
                entry = VSTInstrument(
                    name=name,
                    path=normalized,
                    plugin_name=name,
                    is_instrument=is_instrument,
                    is_effect=is_effect,
                    category=category,
                    host_supported=host_supported,
                    host_error=host_error,
                )
            if not entry.host_supported:
                continue
            rack.append(entry)
        original_rack_paths = [v.path for v in rack]
        self.project.vsti_rack = rack
        migrated = self._sync_discovered_vstis_to_rack(eager_load=False)
        self._dedupe_and_filter_vsti_state()
        cleaned = self.project.vsti_paths != original_vsti_paths or [v.path for v in self.project.vsti_rack] != original_rack_paths
        refresh_migrated = payload.get('playback_ui_refresh_ms') != self.playback_ui_refresh_ms
        if migrated or cleaned or refresh_migrated:
            self._save_preferences()

    def _save_preferences(self) -> None:
        if hasattr(self, 'splitter_main'):
            self._main_splitter_sizes = [int(size) for size in self.splitter_main.sizes()]
        if hasattr(self, 'note_editor_splitter'):
            self._note_editor_inner_sizes = [int(size) for size in self.note_editor_splitter.sizes()]
        if hasattr(self, 'tools_window'):
            self._tools_window_visible = self.tools_window.isVisible()
            self._tools_window_geometry_b64 = bytes(self.tools_window.saveGeometry().toBase64()).decode('ascii')
        if hasattr(self, 'mixer_window'):
            self._mixer_window_visible = self.mixer_window.isVisible()
            self._mixer_window_geometry_b64 = bytes(self.mixer_window.saveGeometry().toBase64()).decode('ascii')
        if hasattr(self, 'transport_window'):
            self._transport_window_visible = self.transport_window.isVisible()
            self._transport_window_geometry_b64 = bytes(self.transport_window.saveGeometry().toBase64()).decode('ascii')
        if hasattr(self, 'virtual_piano_window'):
            self._virtual_piano_window_visible = self.virtual_piano_window.isVisible()
            self._virtual_piano_window_geometry_b64 = bytes(self.virtual_piano_window.saveGeometry().toBase64()).decode('ascii')
        if hasattr(self, 'virtual_piano_scale_combo'):
            self._virtual_piano_key_scale_percent = self._coerce_int(self.virtual_piano_scale_combo.currentData(), self._virtual_piano_key_scale_percent, 35, 175)
        payload = {
            'vsti_paths': self.project.vsti_paths,
            'vsti_folder_paths': self.project.vsti_folder_paths,
            'vsti_rack_paths': [v.path for v in self.project.vsti_rack],
            'vsti_rack_state': [
                {
                    'name': v.name,
                    'path': v.path,
                    'plugin_name': v.plugin_name,
                    'is_instrument': v.is_instrument,
                    'is_effect': v.is_effect,
                    'category': v.category,
                    'host_supported': v.host_supported,
                    'host_error': v.host_error,
                }
                for v in self.project.vsti_rack
            ],
            'sample_paths': self.project.sample_paths,
            'selected_audio_output_id': self.selected_audio_output_id,
            'audio_buffer_ms': self.audio_buffer_ms,
            'playback_ui_refresh_ms': self.playback_ui_refresh_ms,
            'prefer_gpu_rendering': self.prefer_gpu_rendering,
            'main_splitter_sizes': self._main_splitter_sizes,
            'note_editor_inner_sizes': self._note_editor_inner_sizes,
            'tools_window_visible': self._tools_window_visible,
            'tools_window_geometry_b64': self._tools_window_geometry_b64,
            'mixer_window_visible': self._mixer_window_visible,
            'mixer_window_geometry_b64': self._mixer_window_geometry_b64,
            'transport_window_visible': self._transport_window_visible,
            'transport_window_geometry_b64': self._transport_window_geometry_b64,
            'virtual_piano_window_visible': self._virtual_piano_window_visible,
            'virtual_piano_window_geometry_b64': self._virtual_piano_window_geometry_b64,
            'virtual_piano_key_scale_percent': self._virtual_piano_key_scale_percent,
        }
        APP_PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        APP_PREFS_PATH.write_text(json.dumps(payload, indent=2))

    def _sync_bundled_vsti_directory(self) -> None:
        discovered_roots = [folder for folder in (self.vsti_directory, self.user_vsti_directory) if folder.exists()]
        if not discovered_roots:
            return

        bundled_paths: list[str] = []
        seen_paths: set[str] = set()
        for folder in discovered_roots:
            for path in self._discover_vstis_in_folder(folder):
                normalized = self._normalized_vsti_path(str(path))
                if normalized in seen_paths:
                    continue
                seen_paths.add(normalized)
                bundled_paths.append(normalized)
        if not bundled_paths:
            return

        changed = False
        for path in bundled_paths:
            if path not in self.project.vsti_paths:
                self.project.vsti_paths.append(path)
                changed = True

        added_to_rack = self._sync_discovered_vstis_to_rack(bundled_paths, eager_load=False)
        self._dedupe_and_filter_vsti_state()
        if changed or added_to_rack:
            self._save_preferences()

    def _preferred_vsti_browser_directory(self) -> Path:
        candidates: list[Path] = []
        if sys.platform == "darwin":
            candidates.extend(
                [
                    Path.home() / "Library" / "Audio" / "Plug-Ins" / "VST3",
                    Path("/Library/Audio/Plug-Ins/VST3"),
                ]
            )
        elif os.name == "nt":
            candidates.extend(
                [
                    Path("C:/Program Files/Common Files/VST3"),
                    Path("C:/Program Files/Steinberg/VstPlugins"),
                    Path("C:/Program Files (x86)/Steinberg/VstPlugins"),
                ]
            )
        else:
            candidates.extend(
                [
                    Path.home() / ".vst3",
                    Path("/usr/lib/vst3"),
                    Path("/usr/local/lib/vst3"),
                ]
            )
        candidates.extend([self.user_vsti_directory, self.vsti_directory])
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return DEFAULT_USER_FILES_DIR

    def _discover_vstis_in_folder(self, folder: Path) -> list[Path]:
        discovered: list[Path] = []
        for root, dirs, files in os.walk(folder):
            root_path = Path(root)
            vst3_dirs = [name for name in dirs if name.lower().endswith('.vst3')]
            for name in vst3_dirs:
                discovered.append(root_path / name)
            dirs[:] = [name for name in dirs if not name.lower().endswith('.vst3')]
            for name in files:
                if Path(name).suffix.lower() in {'.dll', '.so', '.vst3'}:
                    discovered.append(root_path / name)
        unique: list[Path] = []
        seen: set[str] = set()
        for path in discovered:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        return unique

    def _discover_vsti_plugin_paths(self, folder: Path) -> list[str]:
        return [self._normalized_vsti_path(str(path)) for path in self._discover_vstis_in_folder(folder)]

    def _sync_vsti_folder(self, folder: str, *, remember: bool = True) -> tuple[int, int]:
        folder_path = Path(self._normalized_vsti_path(folder))
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError('Choose an existing folder that contains plugins.')

        normalized_folder = str(folder_path)
        if remember and normalized_folder not in self.project.vsti_folder_paths:
            self.project.vsti_folder_paths.append(normalized_folder)

        found_paths = self._discover_vsti_plugin_paths(folder_path)
        added = 0
        for plugin_path in found_paths:
            if plugin_path in self.project.vsti_paths:
                continue
            self.project.vsti_paths.append(plugin_path)
            added += 1
        added_to_rack = self._sync_discovered_vstis_to_rack(found_paths, eager_load=False)
        self._dedupe_and_filter_vsti_state()
        self._save_preferences()
        self.refresh_vsti_rack_ui()
        self.instruments.load_track()
        return added, added_to_rack

    def _remove_vsti_folder(self, folder: str, *, save: bool = True) -> int:
        normalized_folder = self._normalized_vsti_path(folder)
        if normalized_folder not in self.project.vsti_folder_paths:
            return 0

        remaining_folders = [path for path in self.project.vsti_folder_paths if self._normalized_vsti_path(path) != normalized_folder]
        removable = set(self._discover_vsti_plugin_paths(Path(normalized_folder)))
        retained: set[str] = set()
        for folder_path in remaining_folders:
            retained.update(self._discover_vsti_plugin_paths(Path(folder_path)))
        removable -= retained

        self.project.vsti_folder_paths = remaining_folders
        if removable:
            self.project.vsti_paths = [path for path in self.project.vsti_paths if self._normalized_vsti_path(path) not in removable]
            self.project.vsti_rack = [entry for entry in self.project.vsti_rack if self._normalized_vsti_path(entry.path) not in removable]
            for plugin_path in removable:
                self.vsti_description_cache.pop(plugin_path, None)
                self.vsti_plugin_metadata.pop(plugin_path, None)
        self._dedupe_and_filter_vsti_state()
        if save:
            self._save_preferences()
            self.refresh_vsti_rack_ui()
            self.instruments.load_track()
        return len(removable)

    def add_vsti_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose VST folder', str(self._preferred_vsti_browser_directory()))
        if not folder:
            return
        try:
            added, added_to_rack = self._sync_vsti_folder(folder, remember=True)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, 'Invalid VST folder', str(exc))
            return
        self.statusBar().showMessage(f'Remembered VST folder and discovered {added} plugin(s); added {added_to_rack} supported plugin(s) to the rack.')

    def manage_vsti_folders(self) -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Manage VSTI Folders')
        dialog.resize(760, 420)
        layout = QtWidgets.QVBoxLayout(dialog)

        info = QtWidgets.QLabel('These folders are remembered and rescanned when you add or update them. Remove a folder here to stop showing plugins discovered only from that location.')
        info.setWordWrap(True)
        layout.addWidget(info)

        folder_list = QtWidgets.QListWidget()
        layout.addWidget(folder_list, 1)

        status_label = QtWidgets.QLabel('')
        status_label.setWordWrap(True)
        layout.addWidget(status_label)

        button_row = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton('Add Folder')
        change_btn = QtWidgets.QPushButton('Change Selected')
        rescan_btn = QtWidgets.QPushButton('Rescan Selected')
        remove_btn = QtWidgets.QPushButton('Remove Selected')
        close_btn = QtWidgets.QPushButton('Close')
        for button in (add_btn, change_btn, rescan_btn, remove_btn, close_btn):
            button_row.addWidget(button)
        layout.addLayout(button_row)

        def update_buttons() -> None:
            has_selection = folder_list.currentRow() >= 0
            change_btn.setEnabled(has_selection)
            rescan_btn.setEnabled(has_selection)
            remove_btn.setEnabled(has_selection)

        def refresh_list() -> None:
            current = selected_folder()
            folder_list.clear()
            for folder_path in self.project.vsti_folder_paths:
                folder_list.addItem(folder_path)
            if current:
                matches = folder_list.findItems(current, QtCore.Qt.MatchFlag.MatchExactly)
                if matches:
                    folder_list.setCurrentItem(matches[0])
            if folder_list.currentRow() < 0 and folder_list.count() > 0:
                folder_list.setCurrentRow(0)
            update_buttons()

        def selected_folder() -> str:
            item = folder_list.currentItem()
            return item.text() if item is not None else ''

        def add_folder() -> None:
            chosen = QtWidgets.QFileDialog.getExistingDirectory(dialog, 'Choose VST folder', str(self._preferred_vsti_browser_directory()))
            if not chosen:
                return
            try:
                added, added_to_rack = self._sync_vsti_folder(chosen, remember=True)
                status_label.setText(f'Added folder. Discovered {added} plugin(s) and added {added_to_rack} supported plugin(s) to the rack.')
                refresh_list()
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(dialog, 'Invalid VST folder', str(exc))

        def change_folder() -> None:
            current = selected_folder()
            if not current:
                return
            chosen = QtWidgets.QFileDialog.getExistingDirectory(dialog, 'Choose replacement VST folder', current)
            if not chosen:
                return
            self._remove_vsti_folder(current, save=False)
            try:
                added, added_to_rack = self._sync_vsti_folder(chosen, remember=True)
                status_label.setText(f'Replaced folder. Discovered {added} plugin(s) and added {added_to_rack} supported plugin(s) to the rack.')
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(dialog, 'Invalid VST folder', str(exc))
            refresh_list()

        def rescan_folder() -> None:
            current = selected_folder()
            if not current:
                return
            try:
                added, added_to_rack = self._sync_vsti_folder(current, remember=True)
                status_label.setText(f'Rescanned folder. Discovered {added} new plugin(s) and added {added_to_rack} supported plugin(s) to the rack.')
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(dialog, 'Invalid VST folder', str(exc))
            refresh_list()

        def remove_folder() -> None:
            current = selected_folder()
            if not current:
                return
            removed = self._remove_vsti_folder(current, save=True)
            status_label.setText(f'Removed folder. Hid {removed} plugin(s) that only came from that location.')
            refresh_list()

        folder_list.currentRowChanged.connect(lambda _row: update_buttons())
        add_btn.clicked.connect(add_folder)
        change_btn.clicked.connect(change_folder)
        rescan_btn.clicked.connect(rescan_folder)
        remove_btn.clicked.connect(remove_folder)
        close_btn.clicked.connect(dialog.accept)
        refresh_list()
        self._center_dialog(dialog)
        dialog.exec()

    def add_sample_path(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose sample folder', str(DEFAULT_USER_FILES_DIR))
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
                            converted = RENDER_DIR / f'{src.stem}_import.wav'
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

    def _queue_vsti_background_load(self, path: str, *, show_message: bool = False, reason: str = 'manual') -> bool:
        normalized = self._normalized_vsti_path(path)
        if show_message:
            self.statusBar().showMessage(f'VST3 plugins now load on the main thread when first used: {Path(normalized).name}')
        return False

    def _on_vsti_background_load_finished(
        self,
        path: str,
        ok: bool,
        detail: str,
        param_names: list[str],
        reason: str,
        show_message: bool,
    ) -> None:
        normalized = self._normalized_vsti_path(path)
        self._vsti_background_loads_inflight.discard(normalized)
        self._active_vsti_workers.pop(normalized, None)
        if ok:
            self.vsti_plugin_metadata[normalized] = list(param_names)
            if show_message:
                self.statusBar().showMessage(f'Loaded VST3 plugin in background: {Path(normalized).name}')
        elif show_message:
            QtWidgets.QMessageBox.warning(self, 'VSTI load failed', f'Could not load {Path(normalized).name}\n\n{detail}')

        self.refresh_vsti_rack_ui()

    def _start_vsti_background_warmup(self) -> None:
        return

    def _temporarily_stop_playback_for_vsti_load(self) -> tuple[bool, float]:
        was_playing = bool(hasattr(self, 'playback_timer') and self.playback_timer.isActive())
        resume_sec = float(self.project.playhead_sec)
        if not was_playing:
            return False, resume_sec
        self.stop_playback()
        self._reset_realtime_track_states(clear_plugins=True)
        self._realtime_reset_pending = True
        return True, resume_sec

    def _resume_playback_after_vsti_load(self, was_playing: bool, resume_sec: float) -> None:
        if not was_playing:
            return
        self.set_playhead_position(resume_sec)
        QtCore.QTimer.singleShot(0, self.start_playback)

    def _load_vsti_binary_path(self, path: str, show_message: bool = True) -> bool:
        normalized = self._normalized_vsti_path(path)
        paused_playback, resume_sec = self._temporarily_stop_playback_for_vsti_load()
        try:
            ok, detail = self.vsti_binary_loader.load(normalized)
            if ok:
                self._capture_vsti_metadata(normalized, self.vsti_binary_loader.handle(normalized))
        finally:
            self._resume_playback_after_vsti_load(paused_playback, resume_sec)
        if show_message:
            name = Path(normalized).name
            if ok:
                self.statusBar().showMessage(f'Loaded VST3 plugin: {name} (pedalboard host)')
            else:
                QtWidgets.QMessageBox.warning(self, 'VSTI load failed', f'Could not load {name}\n\n{detail}')
        return ok

    def load_vsti_binary_by_name(self, vsti_name: str) -> None:
        for vst in self.project.vsti_rack:
            if vst.name == vsti_name:
                self._load_vsti_binary_path(vst.path, show_message=True)
                return
        QtWidgets.QMessageBox.information(self, 'VSTI not found', f'No rack VSTI named {vsti_name}.')

    def open_vsti_gui_by_name(self, vsti_name: str) -> None:
        for vst in self.project.vsti_rack:
            if vst.name != vsti_name:
                continue

            track = self.current_track()
            if not vst.host_supported:
                detail = vst.host_error or 'This plugin cannot be hosted by the current VST backend.'
                QtWidgets.QMessageBox.information(self, 'Unsupported VSTI', f'{vst.name} does not have a usable native editor here.\n\n{detail}')
                return
            if self._is_bundled_vsti(vst):
                self._open_vsti_wrapper_dialog(vst, track)
                return
            if PEDALBOARD_AVAILABLE and load_plugin:
                try:
                    plugin = self._load_rack_plugin(vsti_name)
                    if plugin is not None and hasattr(plugin, 'show_editor'):
                        self._load_saved_vsti_plugin_state(plugin, track, vst)
                        self._apply_saved_plugin_parameters(plugin, track.vsti_parameters)
                        self.statusBar().showMessage(f'Opening native VST editor: {vst.name}')
                        self._center_foreground_native_window_async()
                        plugin.show_editor()
                        snapshot = self._plugin_parameter_snapshot(plugin)
                        if snapshot:
                            track.vsti_parameters = snapshot
                        self._save_vsti_plugin_state(plugin, track, vst)
                        self.statusBar().showMessage(f'Updated native VST editor state: {vst.name}')
                        self.on_track_instrument_changed()
                        return
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(
                        self,
                        'Native VST editor unavailable',
                        f'Could not open the native editor for {vst.name}.\n\n{exc}\n\nFalling back to the built-in wrapper controls instead.',
                    )

            self._open_vsti_wrapper_dialog(vst, track)
            return
        QtWidgets.QMessageBox.information(self, 'VSTI not found', f'No rack VSTI named {vsti_name}.')

    def add_discovered_vsti_to_rack(self) -> None:
        available: list[str] = []
        existing_paths = {self._normalized_vsti_path(v.path) for v in self.project.vsti_rack}
        for path in self.project.vsti_paths:
            if not Path(path).exists() or not self._is_valid_vsti_plugin_path(path):
                continue
            normalized = self._normalized_vsti_path(path)
            if normalized in existing_paths:
                continue
            _name, _is_instrument, _is_effect, _category, host_supported, _host_error = self._describe_plugin_path(normalized)
            if host_supported:
                available.append(normalized)
        if not available:
            QtWidgets.QMessageBox.information(self, 'No discovered VSTI', 'No supported discovered VST3 plugins are available to add to the rack.')
            return

        labels = []
        for path in available:
            name, is_instrument, is_effect, _category, _host_supported, _host_error = self._describe_plugin_path(path)
            roles = []
            if is_instrument:
                roles.append('INST')
            if is_effect:
                roles.append('FX')
            role_text = '/'.join(roles) if roles else 'UNK'
            labels.append(f'[{role_text}] {name}')
        selected, ok = QtWidgets.QInputDialog.getItem(self, 'Add VSTI To Rack', 'Choose instrument:', labels, 0, False)
        if not ok:
            return
        idx = labels.index(selected)
        chosen_path = available[idx]
        self._add_vsti_to_rack(chosen_path, show_status=False)
        self._dedupe_and_filter_vsti_state()
        self._save_preferences()
        self.refresh_vsti_rack_ui()
        self.statusBar().showMessage(f'Added to rack: {Path(chosen_path).stem}')

    def add_vsti_path(self) -> None:
        initial_dir = str(self._preferred_vsti_browser_directory())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose VST instrument', initial_dir, 'VST Plugins (*.dll *.vst3 *.so);;All files (*)')
        if not path:
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose VST3 bundle', initial_dir)
            if not folder:
                return
            path = folder
        if not self._add_vsti_to_rack(path, show_status=True):
            return
        self._dedupe_and_filter_vsti_state()
        self._save_preferences()
        self.refresh_vsti_rack_ui()
        self.statusBar().showMessage(f'Added VSTI to rack: {Path(path).name}')

    def _on_layout_splitter_moved(self, _pos: int, _index: int) -> None:
        if hasattr(self, 'splitter_main'):
            self._main_splitter_sizes = [int(size) for size in self.splitter_main.sizes()]
        if hasattr(self, 'note_editor_splitter'):
            self._note_editor_inner_sizes = [int(size) for size in self.note_editor_splitter.sizes()]
        self._layout_save_timer.start(150)

    def _apply_tools_window_preferences(self) -> None:
        if not hasattr(self, 'tools_window'):
            return
        if self._tools_window_geometry_b64:
            try:
                geometry = QtCore.QByteArray.fromBase64(self._tools_window_geometry_b64.encode('ascii'))
                if not geometry.isEmpty():
                    self.tools_window.restoreGeometry(geometry)
            except Exception:
                pass
        else:
            self._position_tools_window_default()
        if hasattr(self, 'show_panels_window_action'):
            self.show_panels_window_action.blockSignals(True)
            self.show_panels_window_action.setChecked(self._tools_window_visible)
            self.show_panels_window_action.blockSignals(False)
        if self._tools_window_visible:
            self.tools_window.show()
            self.tools_window.raise_()
        else:
            self.tools_window.hide()

    def _apply_mixer_window_preferences(self) -> None:
        if not hasattr(self, 'mixer_window'):
            return
        if self._mixer_window_geometry_b64:
            try:
                geometry = QtCore.QByteArray.fromBase64(self._mixer_window_geometry_b64.encode('ascii'))
                if not geometry.isEmpty():
                    self.mixer_window.restoreGeometry(geometry)
            except Exception:
                pass
        else:
            self._position_mixer_window_default()
        if hasattr(self, 'show_mixer_window_action'):
            self.show_mixer_window_action.blockSignals(True)
            self.show_mixer_window_action.setChecked(self._mixer_window_visible)
            self.show_mixer_window_action.blockSignals(False)
        if self._mixer_window_visible:
            self.mixer_window.show()
            self.mixer_window.raise_()
        else:
            self.mixer_window.hide()

    def _apply_transport_window_preferences(self) -> None:
        if not hasattr(self, 'transport_window'):
            return
        if self._transport_window_geometry_b64:
            try:
                geometry = QtCore.QByteArray.fromBase64(self._transport_window_geometry_b64.encode('ascii'))
                if not geometry.isEmpty():
                    self.transport_window.restoreGeometry(geometry)
            except Exception:
                pass
        else:
            self._position_transport_window_default()
        if hasattr(self, 'show_transport_window_action'):
            self.show_transport_window_action.blockSignals(True)
            self.show_transport_window_action.setChecked(self._transport_window_visible)
            self.show_transport_window_action.blockSignals(False)
        if self._transport_window_visible:
            self.transport_window.show()
            self.transport_window.raise_()
        else:
            self.transport_window.hide()

    def _apply_virtual_piano_window_preferences(self) -> None:
        if not hasattr(self, 'virtual_piano_window'):
            return
        if hasattr(self, 'virtual_piano_scale_combo'):
            self._set_virtual_piano_key_scale_percent(self._virtual_piano_key_scale_percent)
        if self._virtual_piano_window_geometry_b64:
            try:
                geometry = QtCore.QByteArray.fromBase64(self._virtual_piano_window_geometry_b64.encode('ascii'))
                if not geometry.isEmpty():
                    self.virtual_piano_window.restoreGeometry(geometry)
            except Exception:
                pass
        else:
            self._position_virtual_piano_window_default()
        if hasattr(self, 'show_virtual_piano_window_action'):
            self.show_virtual_piano_window_action.blockSignals(True)
            self.show_virtual_piano_window_action.setChecked(self._virtual_piano_window_visible)
            self.show_virtual_piano_window_action.blockSignals(False)
        if self._virtual_piano_window_visible:
            self.virtual_piano_window.show()
            self.virtual_piano_window.raise_()
        else:
            self.virtual_piano_window.hide()

    def _on_tools_window_visibility_changed(self, visible: bool) -> None:
        self._tools_window_visible = bool(visible)
        if hasattr(self, 'show_panels_window_action'):
            self.show_panels_window_action.blockSignals(True)
            self.show_panels_window_action.setChecked(self._tools_window_visible)
            self.show_panels_window_action.blockSignals(False)
        self._save_preferences()

    def _on_mixer_window_visibility_changed(self, visible: bool) -> None:
        self._mixer_window_visible = bool(visible)
        if hasattr(self, 'show_mixer_window_action'):
            self.show_mixer_window_action.blockSignals(True)
            self.show_mixer_window_action.setChecked(self._mixer_window_visible)
            self.show_mixer_window_action.blockSignals(False)
        self._save_preferences()

    def _on_transport_window_visibility_changed(self, visible: bool) -> None:
        self._transport_window_visible = bool(visible)
        if hasattr(self, 'show_transport_window_action'):
            self.show_transport_window_action.blockSignals(True)
            self.show_transport_window_action.setChecked(self._transport_window_visible)
            self.show_transport_window_action.blockSignals(False)
        self._save_preferences()

    def _on_virtual_piano_window_visibility_changed(self, visible: bool) -> None:
        self._virtual_piano_window_visible = bool(visible)
        if hasattr(self, 'show_virtual_piano_window_action'):
            self.show_virtual_piano_window_action.blockSignals(True)
            self.show_virtual_piano_window_action.setChecked(self._virtual_piano_window_visible)
            self.show_virtual_piano_window_action.blockSignals(False)
        self._save_preferences()

    def toggle_tools_window(self, checked: bool) -> None:
        self._tools_window_visible = bool(checked)
        if not hasattr(self, 'tools_window'):
            return
        if checked:
            self.tools_window.show()
            self.tools_window.raise_()
            self.tools_window.activateWindow()
        else:
            self.tools_window.hide()

    def toggle_mixer_window(self, checked: bool) -> None:
        self._mixer_window_visible = bool(checked)
        if not hasattr(self, 'mixer_window'):
            return
        if checked:
            self.mixer.load_track()
            self.mixer_window.show()
            self.mixer_window.raise_()
            self.mixer_window.activateWindow()
        else:
            self.mixer_window.hide()

    def toggle_transport_window(self, checked: bool) -> None:
        self._transport_window_visible = bool(checked)
        if not hasattr(self, 'transport_window'):
            return
        if checked:
            self.transport_window.show()
            self.transport_window.raise_()
            self.transport_window.activateWindow()
        else:
            self.transport_window.hide()

    def toggle_virtual_piano_window(self, checked: bool) -> None:
        self._virtual_piano_window_visible = bool(checked)
        if not hasattr(self, 'virtual_piano_window'):
            return
        if checked:
            self.virtual_piano_window.show()
            self.virtual_piano_window.raise_()
            self.virtual_piano_window.activateWindow()
        else:
            self.virtual_piano_window.hide()

    def _set_virtual_piano_key_scale_percent(self, percent: int) -> None:
        value = self._coerce_int(percent, self._virtual_piano_key_scale_percent, 35, 175)
        self._virtual_piano_key_scale_percent = value
        if hasattr(self, 'virtual_piano_scale_combo'):
            index = self.virtual_piano_scale_combo.findData(value)
            if index >= 0 and self.virtual_piano_scale_combo.currentIndex() != index:
                self.virtual_piano_scale_combo.blockSignals(True)
                self.virtual_piano_scale_combo.setCurrentIndex(index)
                self.virtual_piano_scale_combo.blockSignals(False)
        if hasattr(self, 'virtual_piano_keyboard'):
            self.virtual_piano_keyboard.set_key_scale(value / 100.0)
        self._save_preferences()

    def _toggle_transport_window_shortcut(self) -> None:
        next_visible = not bool(self._transport_window_visible)
        if hasattr(self, 'show_transport_window_action'):
            self.show_transport_window_action.setChecked(next_visible)
            return
        self.toggle_transport_window(next_visible)

    def tile_floating_windows(self) -> None:
        screen = None
        if self.windowHandle() is not None:
            screen = self.windowHandle().screen()
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        spacing = 16

        if hasattr(self, 'transport_window'):
            transport_width = min(900, max(620, available.width() - 120))
            transport_height = max(88, min(120, available.height() // 8))
            transport_x = available.x() + max(0, (available.width() - transport_width) // 2)
            transport_y = available.y() + spacing
            self.transport_window.setGeometry(transport_x, transport_y, transport_width, transport_height)
            self.transport_window.show()
            self.transport_window.raise_()
            self._transport_window_visible = True

        if hasattr(self, 'tools_window'):
            tools_width = min(available.width() - 80, max(960, available.width() // 2))
            tools_height = min(available.height() - 180, max(460, (available.height() * 2) // 5))
            tools_x = available.x() + max(0, available.width() - tools_width - spacing)
            tools_y = available.y() + spacing + 96
            self.tools_window.setGeometry(tools_x, tools_y, tools_width, tools_height)
            self.tools_window.show()
            self.tools_window.raise_()
            self._tools_window_visible = True

        if hasattr(self, 'mixer_window'):
            mixer_width = min(available.width() - 120, max(980, (available.width() * 3) // 5))
            mixer_height = min(available.height() - 220, max(420, available.height() // 2))
            mixer_x = available.x() + spacing
            mixer_y = available.y() + spacing + 96
            self.mixer_window.setGeometry(mixer_x, mixer_y, mixer_width, mixer_height)
            self.mixer.load_track()
            self.mixer_window.show()
            self.mixer_window.raise_()
            self._mixer_window_visible = True

        if hasattr(self, 'virtual_piano_window'):
            piano_width = min(760, max(520, available.width() - 280))
            piano_height = min(220, max(160, available.height() // 5))
            piano_x = available.x() + max(0, (available.width() - piano_width) // 2)
            piano_y = available.y() + max(available.height() - piano_height - spacing, spacing + 120)
            self.virtual_piano_window.setGeometry(piano_x, piano_y, piano_width, piano_height)
            self.virtual_piano_window.show()
            self.virtual_piano_window.raise_()
            self._virtual_piano_window_visible = True

        if hasattr(self, 'show_panels_window_action'):
            self.show_panels_window_action.blockSignals(True)
            self.show_panels_window_action.setChecked(True)
            self.show_panels_window_action.blockSignals(False)
        if hasattr(self, 'show_mixer_window_action'):
            self.show_mixer_window_action.blockSignals(True)
            self.show_mixer_window_action.setChecked(True)
            self.show_mixer_window_action.blockSignals(False)
        if hasattr(self, 'show_transport_window_action'):
            self.show_transport_window_action.blockSignals(True)
            self.show_transport_window_action.setChecked(True)
            self.show_transport_window_action.blockSignals(False)
        if hasattr(self, 'show_virtual_piano_window_action'):
            self.show_virtual_piano_window_action.blockSignals(True)
            self.show_virtual_piano_window_action.setChecked(True)
            self.show_virtual_piano_window_action.blockSignals(False)
        self._save_preferences()

    def _position_transport_window_default(self) -> None:
        if not hasattr(self, 'transport_window'):
            return
        available = self._screen_available_geometry()
        if available is None:
            return
        width = min(900, max(620, available.width() - 120))
        height = 88
        self._center_widget_on_screen(self.transport_window, width=width, height=height)
        self.transport_window.move(self.transport_window.x(), available.y() + 16)

    def _position_tools_window_default(self) -> None:
        if not hasattr(self, 'tools_window'):
            return
        available = self._screen_available_geometry()
        if available is None:
            return
        width = min(available.width() - 80, max(980, available.width() // 2))
        height = min(available.height() - 160, max(480, (available.height() * 2) // 5))
        self._center_widget_on_screen(self.tools_window, width=width, height=height)
        self.tools_window.move(available.x() + max(0, available.width() - width - 24), available.y() + 120)

    def _position_mixer_window_default(self) -> None:
        if not hasattr(self, 'mixer_window'):
            return
        available = self._screen_available_geometry()
        if available is None:
            return
        width = min(available.width() - 120, max(980, (available.width() * 3) // 5))
        height = min(available.height() - 220, max(420, available.height() // 2))
        self._center_widget_on_screen(self.mixer_window, width=width, height=height)
        self.mixer_window.move(available.x() + 24, available.y() + 120)

    def _position_virtual_piano_window_default(self) -> None:
        if not hasattr(self, 'virtual_piano_window'):
            return
        available = self._screen_available_geometry()
        if available is None:
            return
        width = min(760, max(520, available.width() - 280))
        height = min(220, max(160, available.height() // 5))
        self._center_widget_on_screen(self.virtual_piano_window, width=width, height=height)
        self.virtual_piano_window.move(
            available.x() + max(0, (available.width() - width) // 2),
            available.y() + max(available.height() - height - 28, 120),
        )

    def refresh_vsti_rack_ui(self) -> None:
        if hasattr(self, 'vsti_menu'):
            existing = [a for a in self.vsti_menu.actions() if a.property('rack_item')]
            for action in existing:
                self.vsti_menu.removeAction(action)
            separator = self.vsti_menu.addSeparator()
            separator.setProperty('rack_item', True)
            supported_rack = [vst for vst in self.project.vsti_rack if vst.host_supported]
            if supported_rack:
                for vst in supported_rack:
                    loaded_flag = 'OK' if self.vsti_binary_loader.is_loaded(vst.path) else '...'
                    roles = []
                    if vst.is_instrument:
                        roles.append('INST')
                    if vst.is_effect:
                        roles.append('FX')
                    role_text = '/'.join(roles) if roles else 'UNK'
                    action = QtGui.QAction(f'Rack: {loaded_flag} [{role_text}] {vst.name}', self)
                    action.setProperty('rack_item', True)
                    action.setEnabled(False)
                    self.vsti_menu.addAction(action)
        self.instruments.reload_vsti_choices()
        if hasattr(self, 'mixer'):
            self.mixer.load_track()
        self._populate_track_list()

    def refresh_openai_status(self) -> None:
        if hasattr(self, 'openai_status_action'):
            self.openai_status_action.setText(self.ai_client.auth_status())

    def on_track_instrument_changed(self) -> None:
        track = self.current_track()
        if track.instrument_mode == 'VSTI Rack' and track.rack_vsti:
            plugin_path = self._rack_vsti_path(track.rack_vsti)
            if plugin_path:
                self._load_vsti_binary_path(plugin_path, show_message=False)
        self._update_selected_track_list_item()
        self.timeline.refresh()
        self._invalidate_playback_caches()
        self._reload_playback_mix_if_running()

    def on_mixer_track_changed(self, row: int | None = None) -> None:
        if row is not None:
            self._update_track_list_item(int(row))
        self._update_selected_track_list_item()
        self._invalidate_playback_caches()
        self._reload_playback_mix_if_running()

    def _apply_track_sound_assignment(self, row: int) -> None:
        self._update_track_list_item(row)
        self.timeline.refresh()
        if row == self.current_track_index():
            self.mixer.load_track()
            self.instruments.load_track()
        self._invalidate_playback_caches()
        self._reload_playback_mix_if_running()

    def _assign_general_midi_to_track(self, row: int, instrument_name: str) -> bool:
        if row < 0 or row >= len(self.project.tracks):
            return False
        track = self.project.tracks[row]
        if track.track_type != 'instrument':
            QtWidgets.QMessageBox.information(self, 'Instrument track required', 'General MIDI instruments can only be assigned to instrument tracks.')
            return False
        track.instrument_mode = 'General MIDI'
        track.rack_vsti = ''
        track.vsti_parameters = {}
        track.vsti_state_path = ''
        track.instrument = instrument_name
        track.midi_program = self.instruments._default_gm_program(track.instrument)
        track.synth_profile = self.instruments._infer_synth_profile(track.instrument, track.midi_program)
        self._apply_track_sound_assignment(row)
        return True

    def _assign_rack_vsti_to_track(self, row: int, rack_name: str) -> bool:
        if row < 0 or row >= len(self.project.tracks):
            return False
        track = self.project.tracks[row]
        if track.track_type != 'instrument':
            QtWidgets.QMessageBox.information(self, 'Instrument track required', 'Rack VST instruments can only be assigned to instrument tracks.')
            return False
        entry = self._rack_vsti_entry(rack_name)
        if entry is None or not entry.is_instrument:
            QtWidgets.QMessageBox.information(self, 'VSTI not found', f'No rack instrument named {rack_name}.')
            return False
        if not entry.host_supported:
            detail = entry.host_error or 'This plugin cannot be hosted by the current VST backend.'
            QtWidgets.QMessageBox.information(self, 'Unsupported VSTI', f'{entry.name} cannot be used as a playable rack instrument.\n\n{detail}')
            return False
        if track.rack_vsti != entry.name:
            track.vsti_parameters = {}
            track.vsti_state_path = ''
        track.instrument_mode = 'VSTI Rack'
        track.rack_vsti = entry.name
        track.instrument = entry.name
        track.synth_profile = 'vst_instrument'
        self._load_vsti_binary_path(entry.path, show_message=False)
        self._apply_track_sound_assignment(row)
        self.statusBar().showMessage(f'Assigned rack VSTI to {track.name}: {entry.name}')
        return True

    def assign_instrument_to_track(self, row: int) -> None:
        if not self.project.tracks:
            return
        if row < 0:
            row = 0
        if row >= len(self.project.tracks):
            return
        track = self.project.tracks[row]
        if track.track_type != 'instrument':
            QtWidgets.QMessageBox.information(self, 'Instrument track required', 'Choose an instrument track to assign a General MIDI sound.')
            return

        options = [self.instruments.instrument.itemText(i) for i in range(self.instruments.instrument.count())]
        current_index = max(0, options.index(track.instrument)) if track.instrument in options else 0
        chosen, ok = QtWidgets.QInputDialog.getItem(self, 'Assign Instrument', 'General MIDI instrument:', options, current_index, False)
        if not ok:
            return
        self._assign_general_midi_to_track(row, str(chosen))

    def assign_instrument_to_selected_track(self) -> None:
        if not self.project.tracks:
            return
        row = self.track_list.currentRow()
        if row < 0:
            row = 0
        self.assign_instrument_to_track(row)

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
                if payload.get('access_token'):
                    self.ai_client.set_oauth_tokens(access_token=payload['access_token'], expires_in=3600 * 24)
                else:
                    self._exchange_oauth_code(payload)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, 'OpenAI connection failed', str(exc))
            return
        self.refresh_openai_status()
        self.statusBar().showMessage('OpenAI connected successfully')

    def _exchange_oauth_code(self, payload: dict) -> None:
        if not payload['token_url'] or not payload['auth_code'] or not payload.get('client_id'):
            raise RuntimeError('Advanced OAuth code exchange requires client id, token URL, and authorization code. Otherwise paste an access token directly.')
        if not payload['code_verifier']:
            raise RuntimeError('Click "Open OAuth Login (Advanced)" first so a PKCE code verifier is generated.')

        request_data = {
            'grant_type': 'authorization_code',
            'client_id': payload['client_id'],
            'code': payload['auth_code'],
            'redirect_uri': payload['redirect_uri'],
            'code_verifier': payload['code_verifier'],
        }

        req_body = urllib.parse.urlencode(request_data).encode('utf-8')
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
            if isinstance(mode, str) and mode in {'General MIDI', 'Sample'}:
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
        self.sample_timeline.refresh()
        self._invalidate_playback_caches(clear_track_audio=False)
        self._reload_playback_mix_if_running()

    def _setup_shortcuts(self) -> None:
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+N"), self, self.new_project)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+O"), self, self.load_project)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, self.save_project)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+S"), self, self.save_project_as)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+I"), self, self.import_midi)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+O"), self, self.import_sample)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+E"), self, self.export_sequence_wav)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Alt+E"), self, self.export_sample_timeline_audio)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self, self.render_all_tracks)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Q"), self, self.piano_roll.quantize_selected)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+D"), self, self.piano_roll.duplicate_selected_by_grid)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+G"), self, self.compose_with_ai)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Delete), self, self.piano_roll.delete_selected)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Space), self, lambda: self.stop_playback() if self.playback_timer.isActive() else self.start_playback())
        left_locator_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+1"), self)
        left_locator_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        left_locator_shortcut.activated.connect(self.set_left_locator_from_mouse)
        right_locator_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+2"), self)
        right_locator_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        right_locator_shortcut.activated.connect(self.set_right_locator_from_mouse)

    def _virtual_piano_key_specs(self) -> list[tuple[int, str, list[str]]]:
        return [
            (60, 'Z', []),
            (61, 'S', []),
            (62, 'X', []),
            (63, 'D', []),
            (64, 'C', []),
            (65, 'V', []),
            (66, 'G', []),
            (67, 'B', []),
            (68, 'H', []),
            (69, 'N', []),
            (70, 'J', []),
            (71, 'M', []),
            (72, 'Q', [',']),
            (73, '2', ['L']),
            (74, 'W', ['.']),
            (75, '3', [';']),
            (76, 'E', ['/']),
            (77, 'R', []),
            (78, '5', []),
            (79, 'T', []),
            (80, '6', []),
            (81, 'Y', []),
            (82, '7', []),
            (83, 'U', []),
        ]

    def _virtual_piano_shortcuts_enabled(self) -> bool:
        widget = QtWidgets.QApplication.focusWidget()
        while isinstance(widget, QtWidgets.QWidget):
            if isinstance(widget, (QtWidgets.QLineEdit, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit, QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)):
                return False
            widget = widget.parentWidget()
        return True

    def _trigger_virtual_piano_pitch(self, pitch: int, *, from_shortcut: bool = False) -> None:
        if from_shortcut and not self._virtual_piano_shortcuts_enabled():
            return
        self.insert_live_note(int(pitch))
        if hasattr(self, 'virtual_piano_keyboard'):
            self.virtual_piano_keyboard.flash_pitch(int(pitch))

    def _register_virtual_piano_shortcuts(self) -> None:
        self._virtual_piano_shortcuts.clear()
        key_map: dict[str, int] = {}
        for pitch, primary, aliases in self._virtual_piano_key_specs():
            key_map[primary] = int(pitch)
            for alias in aliases:
                key_map[str(alias)] = int(pitch)

        for key, pitch in key_map.items():
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(lambda p=pitch: self._trigger_virtual_piano_pitch(p, from_shortcut=True))
            self._virtual_piano_shortcuts.append(shortcut)

    def _setup_virtual_piano_window(self) -> None:
        self.virtual_piano_window = FloatingPanelWindow('Virtual Piano', self)
        self.virtual_piano_window.resize(760, 220)
        self.virtual_piano_window.visibilityChanged.connect(self._on_virtual_piano_window_visibility_changed)

        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        control_row = QtWidgets.QHBoxLayout()
        control_row.setContentsMargins(0, 0, 0, 0)
        control_row.setSpacing(8)
        control_row.addWidget(QtWidgets.QLabel('Key Scale'))
        self.virtual_piano_scale_combo = QtWidgets.QComboBox()
        for percent in (50, 75, 100, 125, 150):
            self.virtual_piano_scale_combo.addItem(f'{percent}%', percent)
        self.virtual_piano_scale_combo.currentIndexChanged.connect(
            lambda _index: self._set_virtual_piano_key_scale_percent(self.virtual_piano_scale_combo.currentData())
        )
        control_row.addWidget(self.virtual_piano_scale_combo)
        control_row.addStretch(1)
        layout.addLayout(control_row)

        hint = QtWidgets.QLabel(
            'Click the keys or play from the computer keyboard. '
            'Lower octave: Z S X D C V G B H N J M. '
            'Upper octave: Q 2 W 3 E R 5 T 6 Y 7 U. '
            'Aliases: , L . ; /.'
        )
        hint.setWordWrap(True)
        hint.setStyleSheet('color: #C7D0DC; font-size: 11px;')
        layout.addWidget(hint)

        self.virtual_piano_keyboard = VirtualPianoKeyboardWidget(self._virtual_piano_key_specs())
        self.virtual_piano_keyboard.noteTriggered.connect(self._trigger_virtual_piano_pitch)
        keyboard_scroll = QtWidgets.QScrollArea()
        keyboard_scroll.setWidgetResizable(False)
        keyboard_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        keyboard_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        keyboard_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        keyboard_scroll.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        keyboard_scroll.setWidget(self.virtual_piano_keyboard)
        layout.addWidget(keyboard_scroll, 1)

        self.virtual_piano_window.setCentralWidget(root)
        self._register_virtual_piano_shortcuts()
        self._apply_virtual_piano_window_preferences()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            self._save_preferences()
        except Exception:
            pass
        super().closeEvent(event)

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
            state.synth_profile = self.instruments._infer_synth_profile(state.instrument, state.midi_program)
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
        track.instrument_mode = 'General MIDI'
        track.rack_vsti = ''
        track.synth_profile = profile

    def render_all_tracks(self) -> None:
        if not self.project.tracks:
            return
        stem_paths: list[str] = []
        for index, track in enumerate(self.project.tracks, start=1):
            if track.track_type != 'instrument':
                continue
            stem_name = f"track_{index:02d}_{track.name.replace(' ', '_')}.wav"
            stem_path = RENDER_DIR / stem_name
            data, sr = self._render_track_audio(track)
            write_wav_samples(stem_path, data, sr)
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

    def import_sample(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import sample",
            str(DEFAULT_USER_FILES_DIR),
            "Audio files (*.wav *.mp3)",
        )
        if not path:
            return

        src = Path(path)
        sample_wav = src
        if src.suffix.lower() == ".mp3":
            converted = RENDER_DIR / f"{src.stem}_import.wav"
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
            str(DEFAULT_USER_FILES_DIR / "sample_timeline.wav"),
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
        solo_tracks = {idx for idx, t in enumerate(self.project.tracks) if t.solo}
        loaded: list[tuple[SampleClip, TrackState, object, int]] = []
        for clip in self.project.sample_clips:
            if clip.track_index < 0 or clip.track_index >= len(self.project.tracks):
                continue
            clip_track = self.project.tracks[clip.track_index]
            if solo_tracks and clip.track_index not in solo_tracks:
                continue
            if clip_track.mute:
                continue
            wav_path = Path(clip.path)
            if wav_path.suffix.lower() == ".mp3":
                converted = RENDER_DIR / f"{wav_path.stem}_mix.wav"
                convert_audio(wav_path, converted)
                wav_path = converted
            data, sr = load_wav_samples(wav_path)
            if sr != sample_rate:
                data = resample_samples(data, sr, sample_rate)
                sr = sample_rate
            data = self._apply_vst_fx_chain(clip_track, data, sr)
            loaded.append((clip, clip_track, data, sr))

        mix_length = int((right - left) * sample_rate)
        mix = [0.0] * max(1, mix_length)
        for clip, clip_track, data, sr in loaded:
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
                mix[idx] += v * 0.7 * float(clip_track.volume)

        mix = [clamp(v, -1.0, 1.0) for v in mix]
        out = Path(path)
        if out.suffix.lower() == ".mp3":
            temp_wav = RENDER_DIR / "sample_timeline_export.wav"
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

    def set_note_editor_zoom(self, cell_w: int) -> None:
        clamped = max(8, min(160, int(cell_w)))
        if self.piano_roll.cell_w != clamped:
            self.piano_roll.cell_w = clamped
            self.piano_roll.refresh()
        if self.velocity_editor.cell_w != clamped:
            self.velocity_editor.cell_w = clamped
            self.velocity_editor.refresh()

    def preview_current_track_note(self, pitch: int, velocity: int = 100, duration_tick: int = TICKS_PER_BEAT // 2) -> None:
        if not self.project.tracks:
            return
        track = self.current_track()
        if track.track_type != 'instrument':
            return
        try:
            preview_note = MidiNote(start_tick=0, duration_tick=max(1, int(duration_tick)), pitch=int(pitch), velocity=int(clamp(velocity, 1, 127)))
            preview_track = dataclasses.replace(
                track,
                notes=[preview_note],
                rendered_audio_path='',
                mute=False,
                solo=False,
            )
            data, sr = self._render_track_audio(preview_track)
            preview_seconds = max(0.3, min(1.5, (preview_note.duration_tick / TICKS_PER_BEAT) * (60.0 / max(1, self.project.bpm)) + 0.35))
            preview_samples = max(1, int(preview_seconds * sr))
            if np is not None and isinstance(data, np.ndarray):
                clip = np.asarray(data[:preview_samples], dtype=np.float32)
            else:
                clip = list(data[:preview_samples])
            self._play_pcm_preview(clip, sr)
        except Exception:
            return

    def insert_live_note(self, pitch: int) -> None:
        track = self.current_track()
        cursor_tick = max((n.start_tick + n.duration_tick for n in track.notes), default=0)
        duration_tick = TICKS_PER_BEAT // 2
        track.notes.append(MidiNote(start_tick=cursor_tick, duration_tick=duration_tick, pitch=pitch))
        self.preview_current_track_note(pitch, 100, duration_tick)
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
        row = self.track_list.currentRow()
        if row < 0:
            return max(0, min(self._selected_track_index, len(self.project.tracks) - 1))
        self._selected_track_index = row
        return row

    def current_track(self) -> TrackState:
        return self.project.tracks[self.current_track_index()]

    def select_track_by_index(self, row: int) -> None:
        if row < 0 or row >= len(self.project.tracks):
            return
        self.track_list.setCurrentRow(int(row))

    def track_meter_levels(self) -> dict[int, float]:
        return dict(self._track_meter_levels)

    def _track_display_text(self, track: TrackState) -> str:
        extra = track.instrument
        if track.instrument_mode == 'VSTI Rack' and track.rack_vsti:
            entry = self._rack_vsti_entry(track.rack_vsti)
            if entry is not None and not entry.host_supported:
                extra = f"{track.instrument} (unsupported)"
        ch = f"Ch {track.midi_channel + 1}" if track.track_type == 'instrument' else 'Sample'
        return f"{track.name} • {track.track_type} • {ch} • {track.instrument_mode} • {extra}"

    def _track_color_icon(self, color: QtGui.QColor) -> QtGui.QIcon:
        pixmap = QtGui.QPixmap(14, 14)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QtGui.QPen(color.darker(180), 1))
        painter.setBrush(QtGui.QBrush(color))
        painter.drawRoundedRect(QtCore.QRectF(1, 1, 12, 12), 2, 2)
        painter.end()
        return QtGui.QIcon(pixmap)

    def _apply_track_row_visuals(self, widget: QtWidgets.QWidget, track: TrackState, row: int) -> None:
        color = track_display_color(track, row)
        swatch = widget.findChild(QtWidgets.QFrame, 'track_row_color')
        if swatch is not None:
            swatch.setStyleSheet(
                f"background-color: {color.name()}; border: 1px solid {color.darker(180).name()}; border-radius: 5px;"
            )
        label = widget.findChild(QtWidgets.QLabel, 'track_row_label')
        if label is not None:
            label.setStyleSheet(f"color: {color.lighter(135).name()};")

    def _set_track_color(self, row: int, color: QtGui.QColor | str | None) -> None:
        if row < 0 or row >= len(self.project.tracks):
            return
        if isinstance(color, str):
            chosen = QtGui.QColor(color)
        elif isinstance(color, QtGui.QColor):
            chosen = QtGui.QColor(color)
        else:
            chosen = QtGui.QColor()
        self.project.tracks[row].color_hex = chosen.name(QtGui.QColor.NameFormat.HexRgb) if chosen.isValid() else ""
        self._update_track_list_item(row)
        self.timeline.refresh()
        self.piano_roll.refresh()
        self.velocity_editor.refresh()
        self.sample_timeline.refresh()
        self.arrangement_overview.refresh()

    def _choose_track_color(self, row: int) -> None:
        if row < 0 or row >= len(self.project.tracks):
            return
        current = track_display_color(self.project.tracks[row], row)
        color = QtWidgets.QColorDialog.getColor(current, self, 'Choose Track Color')
        if color.isValid():
            self._set_track_color(row, color)

    def _install_track_row_context_menu(self, widget: QtWidgets.QWidget, row: int) -> None:
        widget.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        widget.customContextMenuRequested.connect(
            lambda pos, idx=row, source=widget: self._show_track_context_menu(idx, source.mapToGlobal(pos))
        )

    def _show_track_context_menu_from_list(self, pos: QtCore.QPoint) -> None:
        item = self.track_list.itemAt(pos)
        if item is None:
            return
        row = self.track_list.row(item)
        self._show_track_context_menu(row, self.track_list.viewport().mapToGlobal(pos))

    def _show_track_context_menu(self, row: int, global_pos: QtCore.QPoint) -> None:
        if row < 0 or row >= len(self.project.tracks):
            return
        if self.track_list.currentRow() != row:
            self.track_list.setCurrentRow(row)

        track = self.project.tracks[row]
        menu = QtWidgets.QMenu(self)

        rack_menu = menu.addMenu('Use Rack VSTI')
        rack_instruments = [entry for entry in self.project.vsti_rack if entry.host_supported and entry.is_instrument]
        if track.track_type != 'instrument':
            unavailable = rack_menu.addAction('Instrument tracks only')
            unavailable.setEnabled(False)
        elif not rack_instruments:
            unavailable = rack_menu.addAction('No supported rack VST3 instruments available')
            unavailable.setEnabled(False)
        else:
            for entry in rack_instruments:
                action = rack_menu.addAction(entry.name)
                action.setCheckable(True)
                action.setChecked(track.instrument_mode == 'VSTI Rack' and track.rack_vsti == entry.name)
                action.triggered.connect(lambda _checked=False, idx=row, name=entry.name: self._assign_rack_vsti_to_track(idx, name))

        gm_action = menu.addAction('Assign General MIDI Instrument...')
        gm_action.setEnabled(track.track_type == 'instrument')
        gm_action.triggered.connect(lambda _checked=False, idx=row: self.assign_instrument_to_track(idx))

        color_menu = menu.addMenu('Track Color')
        current_color = track_display_color(track, row)
        for color_hex in TRACK_COLOR_PALETTE:
            color = QtGui.QColor(color_hex)
            action = color_menu.addAction(self._track_color_icon(color), color.name().upper())
            action.setCheckable(True)
            action.setChecked(track.color_hex.lower() == color.name().lower() or (not track.color_hex and current_color.name().lower() == color.name().lower()))
            action.triggered.connect(lambda _checked=False, idx=row, value=color.name(): self._set_track_color(idx, value))
        color_menu.addSeparator()
        custom_color = color_menu.addAction('Custom...')
        custom_color.triggered.connect(lambda _checked=False, idx=row: self._choose_track_color(idx))
        reset_color = color_menu.addAction('Reset To Default')
        reset_color.triggered.connect(lambda _checked=False, idx=row: self._set_track_color(idx, None))

        if track.track_type == 'instrument' and track.instrument_mode == 'VSTI Rack' and track.rack_vsti:
            edit_action = menu.addAction(f'Edit "{track.rack_vsti}" Parameters...')
            edit_action.triggered.connect(lambda _checked=False, name=track.rack_vsti: self.open_vsti_gui_by_name(name))

        menu.exec(global_pos)

    def _track_row_widget(self, track: TrackState, row: int) -> QtWidgets.QWidget:
        row_widget = QtWidgets.QWidget()
        row_widget.setObjectName('track_row_widget')
        layout = QtWidgets.QHBoxLayout(row_widget)
        layout.setContentsMargins(4, 1, 4, 1)
        layout.setSpacing(6)

        swatch = QtWidgets.QFrame()
        swatch.setObjectName('track_row_color')
        swatch.setFixedSize(12, 12)
        layout.addWidget(swatch)

        label = QtWidgets.QLabel(self._track_display_text(track))
        label.setObjectName('track_row_label')
        label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        layout.addWidget(label)

        mute_btn = QtWidgets.QToolButton()
        mute_btn.setText('M')
        mute_btn.setCheckable(True)
        mute_btn.setChecked(track.mute)
        mute_btn.setToolTip('Mute track')
        mute_btn.toggled.connect(lambda checked, idx=row: self._on_track_mute_toggled(idx, checked))
        layout.addWidget(mute_btn)

        solo_btn = QtWidgets.QToolButton()
        solo_btn.setText('S')
        solo_btn.setCheckable(True)
        solo_btn.setChecked(track.solo)
        solo_btn.setToolTip('Solo track')
        solo_btn.toggled.connect(lambda checked, idx=row: self._on_track_solo_toggled(idx, checked))
        layout.addWidget(solo_btn)

        self._install_track_row_context_menu(row_widget, row)
        self._install_track_row_context_menu(label, row)
        self._install_track_row_context_menu(mute_btn, row)
        self._install_track_row_context_menu(solo_btn, row)
        self._apply_track_row_visuals(row_widget, track, row)

        return row_widget

    def _update_track_list_item(self, row: int) -> None:
        if row < 0 or row >= len(self.project.tracks):
            return
        item = self.track_list.item(row)
        if item is None:
            return
        widget = self.track_list.itemWidget(item)
        if widget is None:
            item.setText(self._track_display_text(self.project.tracks[row]))
            return
        label = widget.findChild(QtWidgets.QLabel, 'track_row_label')
        if label is not None:
            label.setText(self._track_display_text(self.project.tracks[row]))
        self._apply_track_row_visuals(widget, self.project.tracks[row], row)

    def _on_track_mute_toggled(self, row: int, checked: bool) -> None:
        if self._track_list_rebuilding or row < 0 or row >= len(self.project.tracks):
            return
        self.project.tracks[row].mute = bool(checked)
        if row == self.current_track_index():
            self.mixer.load_track()
        self._invalidate_playback_caches(clear_track_audio=False)
        self._reload_playback_mix_if_running()

    def _on_track_solo_toggled(self, row: int, checked: bool) -> None:
        if self._track_list_rebuilding or row < 0 or row >= len(self.project.tracks):
            return
        self.project.tracks[row].solo = bool(checked)
        if row == self.current_track_index():
            self.mixer.load_track()
        self._invalidate_playback_caches(clear_track_audio=False)
        self._reload_playback_mix_if_running()

    def _update_selected_track_list_item(self) -> None:
        if not self.project.tracks:
            return
        self._update_track_list_item(self.current_track_index())

    def _populate_track_list(self) -> None:
        selected = self.current_track_index() if self.project.tracks else 0
        self._track_list_rebuilding = True
        self.track_list.blockSignals(True)
        self.track_list.clear()
        for row, track in enumerate(self.project.tracks):
            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(QtCore.QSize(0, 28))
            self.track_list.addItem(item)
            self.track_list.setItemWidget(item, self._track_row_widget(track, row))
        if self.project.tracks:
            safe_index = max(0, min(selected, len(self.project.tracks) - 1))
            self._selected_track_index = safe_index
            self.track_list.setCurrentRow(safe_index)
        self.track_list.blockSignals(False)
        self._track_list_rebuilding = False
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
        self._invalidate_playback_caches()
        self._populate_track_list()
        self.track_list.setCurrentRow(idx - 1)
        self.timeline.refresh()
        self.sample_timeline.refresh()
        self.rebuild_midi_sections()
        self.arrangement_overview.refresh()

    def new_project(self) -> None:
        self._reset_project_runtime_state()
        self.project = ProjectState()
        self._load_preferences()
        self._sync_bundled_vsti_directory()
        self._set_project_references(self.project)
        self.current_project_path = None
        if hasattr(self, 'quantize_box'):
            self.quantize_box.blockSignals(True)
            self.quantize_box.setCurrentText(self._project_quantize_text())
            self.quantize_box.blockSignals(False)
        if hasattr(self, 'tempo_spin'):
            self.tempo_spin.setValue(self.project.bpm)
        if hasattr(self, 'left_locator'):
            self._set_locator_spin_values(self.project.left_locator_sec, self.project.right_locator_sec)
        self._refresh_transport_controls()
        self.set_playhead_position(self.project.playhead_sec)
        self.sample_timeline.refresh()
        self._populate_track_list()
        self.track_list.setCurrentRow(0)
        self.refresh_vsti_rack_ui()
        self.scan_sample_paths()
        self.on_notes_changed()
        self._update_window_title()

    def on_notes_changed(self) -> None:
        self._deferred_note_refresh_timer.stop()
        self._deferred_refresh_velocity = False
        self._deferred_refresh_timeline = False
        self._deferred_rebuild_sections = False
        self._deferred_refresh_arrangement = False
        self._deferred_reload_mix = False
        self._invalidate_playback_caches()
        self.piano_roll.refresh()
        self.velocity_editor.refresh()
        self.timeline.refresh()
        self.rebuild_midi_sections()
        self.arrangement_overview.refresh()
        self._reload_playback_mix_if_running()

    def save_project(self) -> None:
        if self.current_project_path is None:
            self.save_project_as()
            return
        self._save_project_to_path(self.current_project_path)

    def save_project_as(self) -> None:
        default_name = self.current_project_path.name if self.current_project_path is not None else f"project{PROJECT_FILE_EXTENSION}"
        default_dir = self.current_project_path.parent if self.current_project_path is not None else DEFAULT_USER_FILES_DIR
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            'Save project',
            str(default_dir / default_name),
            PROJECT_FILE_FILTER,
        )
        if not path:
            return
        self._save_project_to_path(self._ensure_project_file_suffix(path))

    def _save_project_to_path(self, path: Path) -> None:
        payload = self._project_payload()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        self.current_project_path = path.resolve()
        self._update_window_title()
        self.statusBar().showMessage(f'Saved project: {self.current_project_path.name}')

    def load_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            'Open project',
            str(self.current_project_path.parent if self.current_project_path is not None else DEFAULT_USER_FILES_DIR),
            PROJECT_FILE_FILTER,
        )
        if not path:
            return

        project_path = Path(path).expanduser()
        try:
            payload = json.loads(project_path.read_text(encoding='utf-8'))
            project, track_state_blobs, ui_state = self._project_from_payload(payload)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, 'Project load failed', str(exc))
            return

        self._materialize_project_vsti_states(project, track_state_blobs)
        self._apply_project_to_ui(project, ui_state, project_path)
        self.statusBar().showMessage(f'Loaded project: {project_path.name}')

    def _render_sequence_mix(self, left_sec: float, right_sec: float) -> tuple[object, int]:
        sample_rate = 44100
        mix_length = max(1, int(max(0.0, right_sec - left_sec) * sample_rate))
        if np is not None:
            mix: object = np.zeros(mix_length, dtype=np.float32)
        else:
            mix = [0.0] * mix_length

        solo_tracks = {idx for idx, track in enumerate(self.project.tracks) if track.solo}

        for idx, track in enumerate(self.project.tracks):
            if track.track_type != 'instrument':
                continue
            if solo_tracks and idx not in solo_tracks:
                continue
            if track.mute:
                continue

            data, sr = self._get_track_playback_audio(idx, track)
            if sr != sample_rate:
                data = resample_samples(data, sr, sample_rate)
            source_start = max(0, int(left_sec * sample_rate))
            source_end = min(source_start + mix_length, data.shape[0] if np is not None and isinstance(data, np.ndarray) else len(data))
            if source_end <= source_start:
                continue

            if np is not None and isinstance(mix, np.ndarray) and isinstance(data, np.ndarray):
                count = min(mix.shape[0], source_end - source_start)
                if count > 0:
                    mix[:count] += data[source_start:source_start + count]
            else:
                source = list(data)[source_start:source_end]
                count = min(len(mix), len(source))
                for i in range(count):
                    mix[i] += source[i]

        for clip in self.project.sample_clips:
            if clip.track_index < 0 or clip.track_index >= len(self.project.tracks):
                continue
            clip_track = self.project.tracks[clip.track_index]
            if solo_tracks and clip.track_index not in solo_tracks:
                continue
            if clip_track.mute:
                continue

            wav_path = Path(clip.path)
            if wav_path.suffix.lower() == '.mp3':
                converted = RENDER_DIR / f'{wav_path.stem}_sequence_export.wav'
                convert_audio(wav_path, converted)
                wav_path = converted
            data, sr = load_wav_samples(wav_path)
            if sr != sample_rate:
                data = resample_samples(data, sr, sample_rate)
                sr = sample_rate
            data = self._apply_vst_fx_chain(clip_track, data, sr)

            clip_start = float(clip.start_sec)
            clip_end = clip_start + ((data.shape[0] if np is not None and isinstance(data, np.ndarray) else len(data)) / sample_rate)
            if clip_end <= left_sec or clip_start >= right_sec:
                continue

            overlap_start = max(left_sec, clip_start)
            overlap_end = min(right_sec, clip_end)
            src_start = int((overlap_start - clip_start) * sample_rate)
            dst_start = int((overlap_start - left_sec) * sample_rate)
            count = int((overlap_end - overlap_start) * sample_rate)
            if count <= 0:
                continue

            if np is not None and isinstance(mix, np.ndarray) and isinstance(data, np.ndarray):
                mix[dst_start:dst_start + count] += data[src_start:src_start + count] * 0.7 * float(clip_track.volume)
            else:
                source = list(data)[src_start:src_start + count]
                for i, sample in enumerate(source):
                    target = dst_start + i
                    if target >= len(mix):
                        break
                    mix[target] += sample * 0.7 * float(clip_track.volume)

        if np is not None and isinstance(mix, np.ndarray):
            return np.clip(mix, -1.0, 1.0).astype(np.float32, copy=False), sample_rate
        return [clamp(value, -1.0, 1.0) for value in mix], sample_rate

    def export_sequence_wav(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            'Export sequence as WAV',
            str(DEFAULT_USER_FILES_DIR / 'sequence.wav'),
            'WAV files (*.wav)',
        )
        if not path:
            return

        left_sec = float(self.project.left_locator_sec)
        right_sec = float(self.project.right_locator_sec)
        if right_sec <= left_sec:
            QtWidgets.QMessageBox.warning(self, 'Invalid locators', 'Right locator must be greater than left locator for export.')
            return

        data, sample_rate = self._render_sequence_mix(left_sec, right_sec)
        output_path = Path(path).expanduser().with_suffix('.wav')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_wav_samples(output_path, data, sample_rate)
        self.statusBar().showMessage(f'Exported sequence WAV: {output_path.name}')

    def import_midi(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import MIDI", str(DEFAULT_USER_FILES_DIR), "MIDI files (*.mid *.midi)")
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
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export MIDI", str(DEFAULT_USER_FILES_DIR / "project.mid"), "MIDI files (*.mid)")
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
    startup_prefs = load_startup_preferences()
    if startup_prefs.get('prefer_gpu_rendering', True):
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_UseDesktopOpenGL, True)
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
