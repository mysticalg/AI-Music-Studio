#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


AUDIO_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".ogg", ".mp3"}
INSTALL_HINT = "py -3.10 -m pip install basic-pitch librosa pretty_midi soundfile numpy scipy"
TRANSCRIPTION_PROFILES = {
    "balanced": {
        "onset_threshold": 0.50,
        "frame_threshold": 0.30,
        "minimum_note_length": 127.7,
        "drum_min_gap_seconds": 0.08,
        "drum_energy_quantile": 0.55,
        "melodia_trick": True,
        "note_velocity_floor": 34,
        "isolated_short_note_seconds": 0.11,
        "polyphony_limit": 6,
    },
    "cleaner": {
        "onset_threshold": 0.62,
        "frame_threshold": 0.42,
        "minimum_note_length": 220.0,
        "drum_min_gap_seconds": 0.12,
        "drum_energy_quantile": 0.72,
        "melodia_trick": False,
        "note_velocity_floor": 42,
        "isolated_short_note_seconds": 0.16,
        "polyphony_limit": 5,
    },
    "detail": {
        "onset_threshold": 0.42,
        "frame_threshold": 0.24,
        "minimum_note_length": 90.0,
        "drum_min_gap_seconds": 0.05,
        "drum_energy_quantile": 0.40,
        "melodia_trick": True,
        "note_velocity_floor": 28,
        "isolated_short_note_seconds": 0.08,
        "polyphony_limit": 8,
    },
}
ROLE_ORDER = {
    "vocals": 0,
    "drums": 1,
    "bass": 2,
    "piano": 3,
    "guitar": 4,
    "organ": 5,
    "strings": 6,
    "brass": 7,
    "sax": 8,
    "flute": 9,
    "choir": 10,
    "lead": 11,
    "pad": 12,
    "pluck": 13,
    "other": 14,
}
ROLE_LABELS = {
    "vocals": "Vocals",
    "drums": "Drums",
    "bass": "Bass",
    "piano": "Piano",
    "guitar": "Guitar",
    "organ": "Organ",
    "strings": "Strings",
    "brass": "Brass",
    "sax": "Saxophone",
    "flute": "Flute",
    "choir": "Choir",
    "lead": "Lead",
    "pad": "Pad",
    "pluck": "Pluck",
    "other": "Other",
}


def require_backends():
    missing = []

    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        missing.append(("numpy", exc))
        np = None

    try:
        import librosa
    except Exception as exc:  # pragma: no cover
        missing.append(("librosa", exc))
        librosa = None

    try:
        import pretty_midi
    except Exception as exc:  # pragma: no cover
        missing.append(("pretty_midi", exc))
        pretty_midi = None

    try:
        from basic_pitch.inference import predict
    except Exception as exc:  # pragma: no cover
        missing.append(("basic-pitch", exc))
        predict = None

    if missing:
        details = "\n".join(f"- {name}: {exc}" for name, exc in missing)
        raise RuntimeError(
            "Missing audio-to-MIDI backend packages.\n"
            f"Install with:\n{INSTALL_HINT}\n\n"
            f"Details:\n{details}"
        )

    return np, librosa, pretty_midi, predict


def discover_audio_files(root: Path):
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def infer_stem_role(path: Path) -> str:
    name = path.stem.lower()
    if "drum" in name or "perc" in name or "beat" in name:
        return "drums"
    if "bass" in name or "sub" in name:
        return "bass"
    if "vocal" in name or "vox" in name or "voice" in name:
        return "vocals"
    if "guitar" in name:
        return "guitar"
    if "piano" in name or "keys" in name or "key" in name:
        return "piano"
    if "organ" in name:
        return "organ"
    if "string" in name or "violin" in name or "cello" in name or "orch" in name:
        return "strings"
    if "brass" in name or "trumpet" in name or "trombone" in name or "horn" in name:
        return "brass"
    if "sax" in name:
        return "sax"
    if "flute" in name or "pipe" in name or "wind" in name:
        return "flute"
    if "choir" in name or "aah" in name or "ooh" in name:
        return "choir"
    if "pad" in name:
        return "pad"
    if "lead" in name or "synth" in name:
        return "lead"
    if "pluck" in name or "harp" in name or "mallet" in name:
        return "pluck"
    return "other"


def stem_sort_key(path: Path):
    role = infer_stem_role(path)
    rank = ROLE_ORDER.get(role, 100)
    return (rank, path.stem.lower())


def display_name_for_stem(path: Path) -> str:
    stem = path.stem.replace("_", " ").replace("-", " ").strip()
    if not stem:
        return "Stem"
    return " ".join(part[:1].upper() + part[1:] for part in stem.split())


def display_name_for_role(role: str) -> str:
    return ROLE_LABELS.get(role, "Other")


def gm_program_for_role(role: str) -> int:
    return {
        "bass": 33,
        "vocals": 54,
        "guitar": 27,
        "piano": 0,
        "organ": 19,
        "strings": 48,
        "brass": 61,
        "sax": 65,
        "flute": 73,
        "choir": 52,
        "lead": 80,
        "pad": 88,
        "pluck": 24,
        "other": 48,
    }.get(role, 0)


def role_frequency_range(role: str):
    return {
        "bass": (32.0, 420.0),
        "vocals": (80.0, 1400.0),
        "guitar": (70.0, 1800.0),
        "piano": (45.0, 4200.0),
        "organ": (55.0, 2600.0),
        "strings": (110.0, 3200.0),
        "brass": (70.0, 2200.0),
        "sax": (110.0, 1800.0),
        "flute": (220.0, 3200.0),
        "choir": (140.0, 1800.0),
        "lead": (80.0, 3200.0),
        "pad": (60.0, 2600.0),
        "pluck": (100.0, 2800.0),
        "other": (45.0, 3500.0),
    }.get(role, (45.0, 3500.0))


def bounded_score(value: float, minimum: float, maximum: float, weight: float) -> float:
    if minimum > maximum:
        minimum, maximum = maximum, minimum
    if minimum <= value <= maximum:
        return weight
    width = max(maximum - minimum, 1.0)
    distance = minimum - value if value < minimum else value - maximum
    return max(0.0, weight * (1.0 - (distance / (width * 1.5))))


def upper_score(value: float, maximum: float, weight: float, softness: float = 1.0) -> float:
    if value <= maximum:
        return weight
    falloff = max(softness, 1.0)
    return max(0.0, weight * (1.0 - ((value - maximum) / falloff)))


def lower_score(value: float, minimum: float, weight: float, softness: float = 1.0) -> float:
    if value >= minimum:
        return weight
    falloff = max(softness, 1.0)
    return max(0.0, weight * (1.0 - ((minimum - value) / falloff)))


def pick_analysis_segments(audio, sample_rate, np):
    total_samples = int(audio.size)
    if total_samples <= 0:
        return []

    segment_seconds = 8.0
    hop_seconds = 4.0
    segment_samples = max(int(round(segment_seconds * sample_rate)), sample_rate)
    hop_samples = max(int(round(hop_seconds * sample_rate)), sample_rate // 2)

    if total_samples <= segment_samples:
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        return [(audio, rms)]

    candidates = []
    for start in range(0, total_samples - segment_samples + 1, hop_samples):
        segment = audio[start:start + segment_samples]
        if segment.size < sample_rate:
            continue
        rms = float(np.sqrt(np.mean(np.square(segment))))
        if rms < 1.0e-3:
            continue
        candidates.append((segment.copy(), rms))

    if not candidates:
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        return [(audio[:segment_samples], rms)]

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[:6]


def analyse_instrument_segment(audio, sample_rate, librosa, np):
    harmonic, percussive = librosa.effects.hpss(audio)
    harmonic_energy = float(np.mean(np.abs(harmonic))) + 1.0e-8
    percussive_energy = float(np.mean(np.abs(percussive))) + 1.0e-8
    harmonic_ratio = harmonic_energy / percussive_energy

    centroid = librosa.feature.spectral_centroid(y=harmonic, sr=sample_rate)
    rolloff = librosa.feature.spectral_rolloff(y=harmonic, sr=sample_rate)
    flatness = librosa.feature.spectral_flatness(y=harmonic)
    zcr = librosa.feature.zero_crossing_rate(harmonic)
    onset_env = librosa.onset.onset_strength(y=harmonic, sr=sample_rate)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sample_rate)
    duration_seconds = max(1.0, float(audio.size) / float(sample_rate))
    onset_density = float(len(onset_frames)) / duration_seconds
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0

    centroid_median = float(np.median(centroid)) if centroid.size else 0.0
    rolloff_median = float(np.median(rolloff)) if rolloff.size else 0.0
    flatness_median = float(np.median(flatness)) if flatness.size else 0.0
    zcr_median = float(np.median(zcr)) if zcr.size else 0.0

    try:
        f0 = librosa.yin(harmonic, fmin=55.0, fmax=1760.0, sr=sample_rate)
        voiced = f0[np.isfinite(f0)]
        pitch_median = float(np.median(voiced)) if voiced.size else 0.0
        pitch_spread = float(np.percentile(voiced, 75) - np.percentile(voiced, 25)) if voiced.size > 8 else 0.0
        voiced_ratio = float(voiced.size) / float(max(1, f0.size))
    except Exception:
        pitch_median = 0.0
        pitch_spread = 0.0
        voiced_ratio = 0.0

    return {
        "harmonic_ratio": harmonic_ratio,
        "centroid": centroid_median,
        "rolloff": rolloff_median,
        "flatness": flatness_median,
        "zcr": zcr_median,
        "onset_density": onset_density,
        "pitch": pitch_median,
        "pitch_spread": pitch_spread,
        "voiced_ratio": voiced_ratio,
        "rms": rms,
    }


def score_instrument_role(features: dict, role_hint: str):
    pitch = features["pitch"]
    centroid = features["centroid"]
    rolloff = features["rolloff"]
    flatness = features["flatness"]
    zcr = features["zcr"]
    onset_density = features["onset_density"]
    harmonic_ratio = features["harmonic_ratio"]
    pitch_spread = features["pitch_spread"]
    voiced_ratio = features["voiced_ratio"]

    scores = {role: 0.0 for role in ("bass", "piano", "guitar", "organ", "strings", "brass", "sax", "flute", "choir", "lead", "pad", "pluck", "other")}

    scores["bass"] += bounded_score(pitch, 55.0, 180.0, 3.0)
    scores["bass"] += upper_score(centroid, 1100.0, 2.0, 1200.0)
    scores["bass"] += lower_score(harmonic_ratio, 3.0, 0.0, 1.0) if harmonic_ratio < 3.0 else 1.2
    scores["bass"] += upper_score(onset_density, 2.2, 1.0, 2.0)

    scores["piano"] += bounded_score(onset_density, 1.6, 5.5, 2.6)
    scores["piano"] += bounded_score(centroid, 700.0, 2400.0, 2.1)
    scores["piano"] += bounded_score(pitch_spread, 80.0, 520.0, 1.8)
    scores["piano"] += bounded_score(harmonic_ratio, 1.8, 10.0, 1.2)

    scores["guitar"] += bounded_score(onset_density, 1.2, 4.0, 2.2)
    scores["guitar"] += bounded_score(centroid, 1000.0, 2600.0, 2.2)
    scores["guitar"] += bounded_score(zcr, 0.035, 0.12, 1.6)
    scores["guitar"] += bounded_score(pitch, 90.0, 900.0, 1.0)

    scores["organ"] += lower_score(harmonic_ratio, 10.0, 0.0, 1.0) if harmonic_ratio < 10.0 else 2.6
    scores["organ"] += upper_score(onset_density, 1.2, 2.1, 1.6)
    scores["organ"] += upper_score(centroid, 1200.0, 1.8, 1600.0)
    scores["organ"] += upper_score(flatness, 0.03, 0.8, 0.08)

    scores["strings"] += bounded_score(pitch, 160.0, 900.0, 2.0)
    scores["strings"] += upper_score(onset_density, 1.5, 2.0, 2.0)
    scores["strings"] += bounded_score(harmonic_ratio, 4.0, 18.0, 1.8)
    scores["strings"] += upper_score(zcr, 0.08, 1.2, 0.10)

    scores["brass"] += bounded_score(centroid, 700.0, 1800.0, 2.2)
    scores["brass"] += bounded_score(rolloff, 1800.0, 4200.0, 1.7)
    scores["brass"] += bounded_score(flatness, 0.008, 0.08, 1.0)
    scores["brass"] += upper_score(onset_density, 2.2, 1.0, 2.0)

    scores["sax"] += bounded_score(centroid, 1100.0, 2300.0, 2.3)
    scores["sax"] += bounded_score(rolloff, 1400.0, 3200.0, 1.3)
    scores["sax"] += bounded_score(zcr, 0.01, 0.09, 1.1)
    scores["sax"] += bounded_score(pitch, 120.0, 720.0, 1.2)

    scores["flute"] += bounded_score(pitch, 260.0, 1500.0, 2.5)
    scores["flute"] += lower_score(harmonic_ratio, 6.0, 0.0, 1.0) if harmonic_ratio < 6.0 else 1.8
    scores["flute"] += bounded_score(centroid, 1700.0, 3200.0, 1.8)
    scores["flute"] += upper_score(zcr, 0.08, 1.2, 0.10)

    scores["choir"] += bounded_score(pitch, 140.0, 900.0, 1.3)
    scores["choir"] += upper_score(onset_density, 0.9, 2.2, 1.4)
    scores["choir"] += upper_score(flatness, 0.03, 1.4, 0.06)
    scores["choir"] += bounded_score(voiced_ratio, 0.45, 1.0, 1.6)

    scores["lead"] += bounded_score(centroid, 1700.0, 4200.0, 2.1)
    scores["lead"] += lower_score(flatness, 0.035, 0.0, 0.05) if flatness < 0.035 else 1.4
    scores["lead"] += lower_score(zcr, 0.055, 0.0, 0.08) if zcr < 0.055 else 1.3
    scores["lead"] += bounded_score(onset_density, 0.8, 3.5, 1.0)

    scores["pad"] += lower_score(harmonic_ratio, 8.0, 0.0, 1.0) if harmonic_ratio < 8.0 else 2.1
    scores["pad"] += upper_score(onset_density, 0.8, 2.0, 1.6)
    scores["pad"] += bounded_score(centroid, 300.0, 1800.0, 1.5)
    scores["pad"] += upper_score(flatness, 0.05, 1.0, 0.08)

    scores["pluck"] += lower_score(onset_density, 2.4, 2.1, 2.0) if onset_density >= 2.4 else 0.8
    scores["pluck"] += bounded_score(centroid, 1300.0, 3200.0, 1.9)
    scores["pluck"] += bounded_score(zcr, 0.045, 0.14, 1.4)
    scores["pluck"] += bounded_score(pitch_spread, 20.0, 240.0, 0.9)

    scores["other"] += 1.0
    scores["other"] += bounded_score(centroid, 600.0, 2600.0, 0.6)

    if role_hint in scores:
        scores[role_hint] += 0.75
    return scores


def detect_instrument_role(stem_file, role_hint, librosa, np):
    if role_hint not in {"other", "piano", "guitar"}:
        return role_hint

    try:
        audio, sample_rate = librosa.load(str(stem_file), sr=22050, mono=True, duration=120.0)
    except Exception:
        return role_hint

    if audio.size < sample_rate * 2 or float(np.max(np.abs(audio))) < 1.0e-4:
        return role_hint

    segment_scores = []
    total_weight = 0.0
    aggregate_scores = {}

    for segment, rms in pick_analysis_segments(audio, sample_rate, np):
        features = analyse_instrument_segment(segment, sample_rate, librosa, np)
        scores = score_instrument_role(features, role_hint)
        weight = max(0.2, min(2.0, rms * 20.0))
        segment_scores.append((scores, weight))
        total_weight += weight
        for role, score in scores.items():
            aggregate_scores[role] = aggregate_scores.get(role, 0.0) + (score * weight)

    if not aggregate_scores or total_weight <= 0.0:
        return role_hint

    ranked = sorted(aggregate_scores.items(), key=lambda item: item[1], reverse=True)
    best_role, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    average_best_score = best_score / total_weight
    confidence_margin = (best_score - second_score) / total_weight

    if average_best_score < 2.2:
        return role_hint
    if best_role != role_hint and confidence_margin < 0.25:
        return role_hint
    return best_role


def first_float(value, fallback: float) -> float:
    try:
        if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            return float(value[0]) if len(value) > 0 else fallback
        return float(value)
    except Exception:
        return fallback


def estimate_tempo(stem_files, librosa, np) -> float:
    ordered = sorted(stem_files, key=lambda path: (0 if infer_stem_role(path) == "drums" else 1, stem_sort_key(path)))
    for stem_file in ordered:
        try:
            audio, sample_rate = librosa.load(str(stem_file), sr=22050, mono=True, duration=120.0)
        except Exception:
            continue

        if audio.size < sample_rate * 2:
            continue
        if float(np.max(np.abs(audio))) < 1.0e-4:
            continue

        onset_envelope = librosa.onset.onset_strength(y=audio, sr=sample_rate)
        if onset_envelope.size == 0:
            continue

        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_envelope, sr=sample_rate)
        tempo_value = first_float(tempo, 120.0)
        if 40.0 <= tempo_value <= 220.0:
            return tempo_value

    return 120.0


def detect_drum_band_onsets(spectrum, sample_rate, hop_length, band_mask, quantile, min_gap_seconds, librosa, np):
    band_spectrum = spectrum[band_mask, :]
    if band_spectrum.size == 0:
        return []

    power = np.maximum(band_spectrum ** 2, 1.0e-10)
    onset_envelope = librosa.onset.onset_strength(
        S=librosa.power_to_db(power, ref=np.max),
        sr=sample_rate,
        hop_length=hop_length,
    )
    if onset_envelope.size == 0:
        return []

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        hop_length=hop_length,
        backtrack=False,
        units="frames",
    )
    if len(onset_frames) == 0:
        return []

    strengths = onset_envelope[np.clip(onset_frames, 0, len(onset_envelope) - 1)]
    threshold = float(np.quantile(strengths, quantile))
    minimum_gap_frames = max(1, int(round((min_gap_seconds * sample_rate) / hop_length)))

    filtered = []
    last_kept_frame = -minimum_gap_frames
    for frame, strength in zip(onset_frames, strengths):
        if float(strength) < threshold:
            continue
        frame = int(frame)
        if frame - last_kept_frame < minimum_gap_frames:
            continue
        filtered.append(frame)
        last_kept_frame = frame
    return filtered


def mean_band_energy(slice_spectrum, band_mask, np):
    band = slice_spectrum[band_mask]
    if band.size == 0:
        return 0.0
    return float(np.mean(band))


def mean_band_window_energy(spectrum, band_mask, start_frame, end_frame, np):
    if end_frame <= start_frame:
        return 0.0
    band_window = spectrum[band_mask, start_frame:end_frame]
    if band_window.size == 0:
        return 0.0
    return float(np.mean(band_window))


def spectral_centroid_from_slice(slice_spectrum, frequencies, np):
    total = float(slice_spectrum.sum())
    if total <= 1.0e-9:
        return 0.0
    return float(np.sum(slice_spectrum * frequencies) / total)


def transcribe_drum_stem(stem_file, track_name, pretty_midi, librosa, np, profile):
    audio, sample_rate = librosa.load(str(stem_file), sr=22050, mono=True)
    if audio.size == 0 or float(np.max(np.abs(audio))) < 1.0e-4:
        return None, 0

    hop_length = 512
    spectrum = np.abs(librosa.stft(audio, n_fft=2048, hop_length=hop_length))
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
    low_mask = frequencies < 180.0
    tom_mask = (frequencies >= 90.0) & (frequencies < 900.0)
    snare_body_mask = (frequencies >= 180.0) & (frequencies < 2400.0)
    snare_crack_mask = (frequencies >= 2400.0) & (frequencies < 5200.0)
    clap_mask = (frequencies >= 1800.0) & (frequencies < 5200.0)
    hat_presence_mask = frequencies >= 3200.0
    closed_hat_mask = frequencies >= 5200.0
    cymbal_mask = frequencies >= 6000.0
    upper_mid_mask = (frequencies >= 180.0) & (frequencies < 2800.0)
    high_mask = frequencies >= 2800.0

    kick_frames = detect_drum_band_onsets(
        spectrum,
        sample_rate,
        hop_length,
        low_mask,
        quantile=max(0.35, profile["drum_energy_quantile"] - 0.12),
        min_gap_seconds=max(0.10, profile["drum_min_gap_seconds"]),
        librosa=librosa,
        np=np,
    )
    snare_frames = detect_drum_band_onsets(
        spectrum,
        sample_rate,
        hop_length,
        snare_body_mask,
        quantile=max(0.24, profile["drum_energy_quantile"] - 0.24),
        min_gap_seconds=max(0.09, profile["drum_min_gap_seconds"] * 0.9),
        librosa=librosa,
        np=np,
    )
    clap_frames = detect_drum_band_onsets(
        spectrum,
        sample_rate,
        hop_length,
        clap_mask,
        quantile=max(0.22, profile["drum_energy_quantile"] - 0.18),
        min_gap_seconds=max(0.07, profile["drum_min_gap_seconds"] * 0.8),
        librosa=librosa,
        np=np,
    )
    hat_frames = detect_drum_band_onsets(
        spectrum,
        sample_rate,
        hop_length,
        hat_presence_mask,
        quantile=max(0.16, profile["drum_energy_quantile"] - 0.30),
        min_gap_seconds=max(0.03, profile["drum_min_gap_seconds"] * 0.4),
        librosa=librosa,
        np=np,
    )
    cymbal_frames = detect_drum_band_onsets(
        spectrum,
        sample_rate,
        hop_length,
        cymbal_mask,
        quantile=max(0.18, profile["drum_energy_quantile"] - 0.24),
        min_gap_seconds=max(0.18, profile["drum_min_gap_seconds"] * 1.8),
        librosa=librosa,
        np=np,
    )
    tom_frames = detect_drum_band_onsets(
        spectrum,
        sample_rate,
        hop_length,
        tom_mask,
        quantile=max(0.24, profile["drum_energy_quantile"] - 0.20),
        min_gap_seconds=max(0.08, profile["drum_min_gap_seconds"] * 0.9),
        librosa=librosa,
        np=np,
    )

    if not kick_frames and not snare_frames and not clap_frames and not hat_frames and not cymbal_frames and not tom_frames:
        return None, 0

    instrument = pretty_midi.Instrument(program=0, is_drum=True, name=track_name)
    note_count = 0
    seen_events = set()

    candidate_events = (
        [(frame, "kick") for frame in kick_frames]
        + [(frame, "snare") for frame in snare_frames]
        + [(frame, "clap") for frame in clap_frames]
        + [(frame, "hat") for frame in hat_frames]
        + [(frame, "cymbal") for frame in cymbal_frames]
        + [(frame, "tom") for frame in tom_frames]
    )

    for frame, event_kind in sorted(candidate_events, key=lambda item: (item[0], item[1])):
        start_frame = max(int(frame) - 1, 0)
        end_frame = min(int(frame) + 4, spectrum.shape[1])
        if end_frame <= start_frame:
            continue

        slice_spectrum = spectrum[:, start_frame:end_frame].mean(axis=1)
        total_energy = float(slice_spectrum.sum())
        if total_energy <= 1.0e-9:
            continue

        low_energy = float(slice_spectrum[low_mask].sum())
        mid_energy = float(slice_spectrum[upper_mid_mask].sum())
        high_energy = float(slice_spectrum[high_mask].sum())
        tom_energy = float(slice_spectrum[tom_mask].sum())
        snare_body_energy = float(slice_spectrum[snare_body_mask].sum())
        snare_crack_energy = float(slice_spectrum[snare_crack_mask].sum())
        closed_hat_energy = float(slice_spectrum[closed_hat_mask].sum())
        cymbal_energy = float(slice_spectrum[cymbal_mask].sum())
        event_centroid = spectral_centroid_from_slice(slice_spectrum, frequencies, np)

        low_ratio = low_energy / total_energy
        high_ratio = high_energy / total_energy
        mid_ratio = mid_energy / total_energy
        tom_ratio = tom_energy / total_energy
        snare_body_ratio = snare_body_energy / total_energy
        snare_crack_ratio = snare_crack_energy / total_energy
        closed_hat_ratio = closed_hat_energy / total_energy
        cymbal_ratio = cymbal_energy / total_energy

        pitch = 0
        duration = 0.08

        if event_kind == "kick":
            if low_ratio < 0.28:
                continue
            pitch = 36
            duration = 0.12
        elif event_kind == "snare":
            if snare_body_ratio < 0.17:
                continue
            if low_ratio > 0.42 or closed_hat_ratio > 0.34:
                continue
            if (snare_body_ratio + (snare_crack_ratio * 0.6)) < 0.24:
                continue
            pitch = 38
            duration = 0.10
        elif event_kind == "clap":
            if snare_crack_ratio < 0.16:
                continue
            if low_ratio > 0.18 or snare_body_ratio > 0.26:
                continue
            pitch = 39 if high_ratio >= 0.20 else 37
            duration = 0.08
        elif event_kind == "tom":
            if tom_ratio < 0.22 or snare_body_ratio > 0.34 or high_ratio > 0.42:
                continue
            if low_ratio > 0.40 and event_centroid < 150.0:
                continue
            if event_centroid < 220.0:
                pitch = 45
            elif event_centroid < 420.0:
                pitch = 47
            else:
                pitch = 50
            duration = 0.14
        elif event_kind == "cymbal":
            immediate_energy = mean_band_window_energy(
                spectrum,
                cymbal_mask,
                int(frame),
                min(int(frame) + 2, spectrum.shape[1]),
                np,
            )
            sustain_energy = mean_band_window_energy(
                spectrum,
                cymbal_mask,
                min(int(frame) + 2, spectrum.shape[1]),
                min(int(frame) + 16, spectrum.shape[1]),
                np,
            )
            sustain_ratio = sustain_energy / max(immediate_energy, 1.0e-9)
            if cymbal_ratio < 0.12 or high_ratio < 0.20:
                continue
            if sustain_ratio >= 0.95:
                pitch = 49
                duration = 0.55
            elif sustain_ratio >= 0.52:
                pitch = 51
                duration = 0.30
            else:
                continue
        else:
            immediate_energy = mean_band_window_energy(
                spectrum,
                hat_presence_mask,
                int(frame),
                min(int(frame) + 2, spectrum.shape[1]),
                np,
            )
            sustain_energy = mean_band_window_energy(
                spectrum,
                hat_presence_mask,
                min(int(frame) + 2, spectrum.shape[1]),
                min(int(frame) + 10, spectrum.shape[1]),
                np,
            )
            sustain_ratio = sustain_energy / max(immediate_energy, 1.0e-9)
            if high_ratio < 0.16:
                continue
            if mid_ratio > 0.68 and closed_hat_ratio < 0.08:
                continue
            if sustain_ratio >= 0.58 and high_ratio >= 0.18:
                pitch = 46
                duration = 0.18
            else:
                if closed_hat_ratio < 0.10 and high_ratio < 0.24:
                    continue
                pitch = 42
                duration = 0.06

        start = float(librosa.frames_to_time(frame, sr=sample_rate, hop_length=hop_length))
        quantized_start_ms = int(round(start * 1000.0))
        event_key = (quantized_start_ms, pitch)
        if event_key in seen_events:
            continue
        seen_events.add(event_key)

        if pitch in {49, 51}:
            velocity_weight = cymbal_ratio * 1.6
        elif pitch in {42, 46}:
            velocity_weight = high_ratio * 1.4
        elif pitch in {37, 39}:
            velocity_weight = (snare_crack_ratio + (high_ratio * 0.3)) * 1.1
        elif pitch == 38:
            velocity_weight = (snare_body_ratio + (snare_crack_ratio * 0.5)) * 1.2
        elif pitch in {45, 47, 50}:
            velocity_weight = (tom_ratio + (low_ratio * 0.4)) * 1.0
        else:
            velocity_weight = low_ratio * 1.1
        velocity = int(max(28, min(127, round(total_energy * (1.5 + (velocity_weight * 3.0))))))
        instrument.notes.append(
            pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start,
                end=start + duration,
            )
        )
        note_count += 1

    return (instrument if instrument.notes else None), note_count


def clone_note(note, pretty_midi):
    return pretty_midi.Note(
        velocity=max(1, min(127, int(note.velocity))),
        pitch=max(0, min(127, int(note.pitch))),
        start=float(note.start),
        end=max(float(note.end), float(note.start) + 0.05),
    )


def rebuild_instrument_notes(instrument, notes, pretty_midi):
    instrument.notes = sorted(
        [
            pretty_midi.Note(
                velocity=max(1, min(127, int(note.velocity))),
                pitch=max(0, min(127, int(note.pitch))),
                start=float(note.start),
                end=max(float(note.end), float(note.start) + 0.05),
            )
            for note in notes
            if float(note.end) > float(note.start)
        ],
        key=lambda note: (note.start, note.pitch, note.end),
    )
    return instrument


def cluster_notes_by_start(notes, threshold_seconds):
    if not notes:
        return []

    ordered = sorted(notes, key=lambda note: (note.start, note.pitch, note.end))
    clusters = [[ordered[0]]]
    for note in ordered[1:]:
        cluster = clusters[-1]
        cluster_start = min(item.start for item in cluster)
        if float(note.start) - float(cluster_start) <= threshold_seconds:
            cluster.append(note)
        else:
            clusters.append([note])
    return clusters


def normalize_instrument_notes(instrument, pretty_midi):
    if not instrument.notes:
        return instrument

    notes = sorted(instrument.notes, key=lambda note: (note.pitch, note.start, note.end))
    merged = []

    for note in notes:
        start = float(note.start)
        end = max(float(note.end), start + 0.05)
        velocity = max(1, min(127, int(note.velocity)))

        if merged:
            previous = merged[-1]
            if note.pitch == previous.pitch and start <= previous.end + 0.05:
                previous.end = max(previous.end, end)
                previous.velocity = max(previous.velocity, velocity)
                continue
            if note.pitch == previous.pitch and abs(start - previous.start) <= 0.03:
                previous.end = max(previous.end, end)
                previous.velocity = max(previous.velocity, velocity)
                continue

        merged.append(
            pretty_midi.Note(
                velocity=velocity,
                pitch=max(0, min(127, int(note.pitch))),
                start=start,
                end=end,
            )
        )

    instrument.notes = sorted(merged, key=lambda note: (note.start, note.pitch))
    return instrument


def filter_low_confidence_pitched_notes(notes, role, profile, pretty_midi):
    if not notes:
        return []

    velocity_floor = int(profile.get("note_velocity_floor", 34))
    isolated_floor = float(profile.get("isolated_short_note_seconds", 0.11))
    role_velocity_bonus = {
        "bass": 8,
        "vocals": 7,
        "strings": 5,
        "pad": 5,
        "organ": 4,
    }.get(role, 0)

    minimum_velocity = velocity_floor + role_velocity_bonus
    ordered = sorted(notes, key=lambda note: (note.start, note.pitch, note.end))
    kept = []

    for index, note in enumerate(ordered):
        duration = float(note.end) - float(note.start)
        previous_gap = float(note.start) - float(ordered[index - 1].end) if index > 0 else 999.0
        next_gap = float(ordered[index + 1].start) - float(note.end) if index + 1 < len(ordered) else 999.0
        isolated = previous_gap > 0.24 and next_gap > 0.24
        velocity = int(note.velocity)

        if velocity < minimum_velocity and duration < (isolated_floor * 0.8) and isolated:
            continue
        if role in {"bass", "vocals"} and velocity < (minimum_velocity + 5) and duration < isolated_floor:
            continue
        if role in {"pad", "strings", "organ"} and duration < 0.09 and velocity < (minimum_velocity + 3):
            continue
        kept.append(clone_note(note, pretty_midi))

    return kept if kept else [clone_note(max(ordered, key=lambda note: (note.velocity, note.end - note.start)), pretty_midi)]


def cleanup_bass_notes(notes, pretty_midi):
    if not notes:
        return []

    clusters = cluster_notes_by_start(notes, 0.055)
    selected = []
    for cluster in clusters:
        best = min(cluster, key=lambda note: (note.pitch, -note.velocity, note.start))
        note = clone_note(best, pretty_midi)
        note.start = min(item.start for item in cluster)
        note.end = max(item.end for item in cluster if item.pitch == best.pitch)
        selected.append(note)

    selected = sorted(selected, key=lambda note: (note.start, note.pitch))
    monophonic = []
    for note in selected:
        if monophonic:
            previous = monophonic[-1]
            if note.start < previous.end - 0.02:
                previous.end = max(previous.start + 0.05, note.start - 0.01)
            if note.pitch == previous.pitch and note.start - previous.end < 0.08:
                previous.end = max(previous.end, note.end)
                previous.velocity = max(previous.velocity, note.velocity)
                continue
        monophonic.append(note)
    return monophonic


def cleanup_vocal_notes(notes, pretty_midi):
    if not notes:
        return []

    ordered = sorted(notes, key=lambda note: (note.start, note.pitch))
    cleaned = []
    for note in ordered:
        current = clone_note(note, pretty_midi)
        if not cleaned:
            cleaned.append(current)
            continue

        previous = cleaned[-1]
        if current.start <= previous.end - 0.02:
            previous_strength = previous.velocity + ((previous.end - previous.start) * 60.0)
            current_strength = current.velocity + ((current.end - current.start) * 60.0)
            if abs(current.pitch - previous.pitch) <= 1:
                previous.end = max(previous.end, current.end)
                previous.velocity = max(previous.velocity, current.velocity)
                if current_strength > previous_strength:
                    previous.pitch = current.pitch
                continue

            if current_strength > previous_strength:
                previous.end = max(previous.start + 0.05, current.start - 0.01)
                cleaned.append(current)
            else:
                previous.end = max(previous.start + 0.05, min(previous.end, current.start))
            continue

        if abs(current.pitch - previous.pitch) <= 1 and current.start - previous.end < 0.08:
            previous.end = max(previous.end, current.end)
            previous.velocity = max(previous.velocity, current.velocity)
            continue
        cleaned.append(current)

    return cleaned


def cleanup_piano_notes(notes, pretty_midi, profile):
    if not notes:
        return []

    clusters = cluster_notes_by_start(notes, 0.045)
    cleaned = []
    max_polyphony = int(profile.get("polyphony_limit", 6))
    for cluster in clusters:
        deduped = {}
        for note in cluster:
            existing = deduped.get(int(note.pitch))
            if existing is None or note.velocity > existing.velocity or note.end > existing.end:
                deduped[int(note.pitch)] = clone_note(note, pretty_midi)

        cluster_notes = list(deduped.values())
        cluster_notes.sort(key=lambda note: (-note.velocity, note.pitch))
        cluster_notes = cluster_notes[:max_polyphony]
        anchor = min(note.start for note in cluster_notes)
        for note in cluster_notes:
            note.start = anchor
            note.end = max(note.end, note.start + 0.08)
        cleaned.extend(sorted(cluster_notes, key=lambda note: note.pitch))
    return sorted(cleaned, key=lambda note: (note.start, note.pitch))


def cleanup_guitar_notes(notes, pretty_midi, profile):
    if not notes:
        return []

    clusters = cluster_notes_by_start(notes, 0.09)
    cleaned = []
    max_polyphony = min(6, int(profile.get("polyphony_limit", 6)))
    for cluster in clusters:
        deduped = {}
        for note in cluster:
            existing = deduped.get(int(note.pitch))
            if existing is None or note.velocity > existing.velocity:
                deduped[int(note.pitch)] = clone_note(note, pretty_midi)

        ordered = sorted(deduped.values(), key=lambda note: (note.start, note.pitch))
        ordered = sorted(ordered, key=lambda note: (-note.velocity, note.pitch))[:max_polyphony]
        ordered = sorted(ordered, key=lambda note: (note.start, note.pitch))
        anchor = min(note.start for note in ordered)
        for index, note in enumerate(ordered):
            note.start = anchor + min(0.012 * index, 0.05)
            note.end = max(note.end, note.start + 0.11)
        cleaned.extend(ordered)
    return sorted(cleaned, key=lambda note: (note.start, note.pitch))


def cleanup_polyphonic_sustain_notes(notes, pretty_midi, profile):
    if not notes:
        return []

    clusters = cluster_notes_by_start(notes, 0.05)
    cleaned = []
    max_polyphony = int(profile.get("polyphony_limit", 6))
    for cluster in clusters:
        deduped = {}
        for note in cluster:
            existing = deduped.get(int(note.pitch))
            if existing is None or note.velocity > existing.velocity:
                deduped[int(note.pitch)] = clone_note(note, pretty_midi)
        cluster_notes = sorted(deduped.values(), key=lambda note: (-note.velocity, note.pitch))[:max_polyphony]
        cleaned.extend(sorted(cluster_notes, key=lambda note: note.pitch))
    return sorted(cleaned, key=lambda note: (note.start, note.pitch))


def apply_role_specific_cleanup(instrument, role, pretty_midi, profile):
    notes = sorted(instrument.notes, key=lambda note: (note.start, note.pitch, note.end))
    if role == "bass":
        notes = cleanup_bass_notes(notes, pretty_midi)
    elif role == "vocals":
        notes = cleanup_vocal_notes(notes, pretty_midi)
    elif role == "piano":
        notes = cleanup_piano_notes(notes, pretty_midi, profile)
    elif role == "guitar":
        notes = cleanup_guitar_notes(notes, pretty_midi, profile)
    elif role in {"organ", "strings", "pad", "brass", "sax", "flute", "choir", "lead", "pluck", "other"}:
        notes = cleanup_polyphonic_sustain_notes(notes, pretty_midi, profile)
    return rebuild_instrument_notes(instrument, notes, pretty_midi)


def transcribe_pitched_stem(stem_file, track_name, role, pretty_midi, predict, tempo_bpm, profile):
    minimum_frequency, maximum_frequency = role_frequency_range(role)
    _, midi_data, _ = predict(
        str(stem_file),
        onset_threshold=profile["onset_threshold"],
        frame_threshold=profile["frame_threshold"],
        minimum_note_length=profile["minimum_note_length"],
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        melodia_trick=profile["melodia_trick"],
        midi_tempo=tempo_bpm,
    )
    notes = []
    for source_instrument in midi_data.instruments:
        for note in source_instrument.notes:
            if note.end <= note.start:
                continue
            notes.append(note)

    if not notes:
        return None, 0

    instrument = pretty_midi.Instrument(
        program=gm_program_for_role(role),
        is_drum=False,
        name=track_name,
    )

    for note in notes:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=max(1, min(127, int(note.velocity))),
                pitch=max(0, min(127, int(note.pitch))),
                start=float(note.start),
                end=max(float(note.end), float(note.start) + 0.05),
            )
        )

    normalize_instrument_notes(instrument, pretty_midi)
    filtered_notes = filter_low_confidence_pitched_notes(instrument.notes, role, profile, pretty_midi)
    rebuild_instrument_notes(instrument, filtered_notes, pretty_midi)
    apply_role_specific_cleanup(instrument, role, pretty_midi, profile)
    normalize_instrument_notes(instrument, pretty_midi)
    if not instrument.notes:
        return None, 0
    return instrument, len(instrument.notes)


def build_output(stem_files, output_midi, output_summary, profile_name, forced_tempo_bpm=None):
    np, librosa, pretty_midi, predict = require_backends()

    if not stem_files:
        raise RuntimeError("No separated stem audio files were found.")

    profile = TRANSCRIPTION_PROFILES.get(profile_name)
    if profile is None:
        raise RuntimeError(f"Unknown transcription profile: {profile_name}")

    tempo_bpm = float(forced_tempo_bpm) if forced_tempo_bpm is not None else estimate_tempo(stem_files, librosa, np)
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)
    midi.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))

    summary_tracks = []
    track_count = 0

    for stem_file in sorted(stem_files, key=stem_sort_key):
        role_hint = infer_stem_role(stem_file)
        role = role_hint if role_hint in {"drums", "bass", "vocals"} else detect_instrument_role(stem_file, role_hint, librosa, np)
        track_name = display_name_for_stem(stem_file)
        generic_names = {"other", "piano", "guitar", "keys", "stem"}
        if track_name.strip().lower() in generic_names or role != role_hint:
            track_name = display_name_for_role(role)

        if role == "drums":
            instrument, note_count = transcribe_drum_stem(stem_file, track_name, pretty_midi, librosa, np, profile)
        else:
            instrument, note_count = transcribe_pitched_stem(
                stem_file,
                track_name,
                role,
                pretty_midi,
                predict,
                tempo_bpm,
                profile,
            )

        if instrument is None or note_count <= 0:
            continue

        midi.instruments.append(instrument)
        summary_tracks.append(
            {
                "name": track_name,
                "role": role,
                "program": 0 if instrument.is_drum else int(instrument.program),
                "is_drum": bool(instrument.is_drum),
                "note_count": int(note_count),
                "source": str(stem_file),
            }
        )
        track_count += 1

    if track_count == 0:
        raise RuntimeError("The stems were separated, but no MIDI notes could be extracted.")

    output_midi.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_midi))

    if output_summary is not None:
        output_summary.parent.mkdir(parents=True, exist_ok=True)
        output_summary.write_text(
            json.dumps(
                {
                    "tempo_bpm": round(float(tempo_bpm), 2),
                    "tempo_source": "forced" if forced_tempo_bpm is not None else "detected",
                    "profile": profile_name,
                    "track_count": track_count,
                    "tracks": summary_tracks,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Convert separated audio stems into a multi-track MIDI file.")
    parser.add_argument("--check", action="store_true", help="Verify backend dependencies are installed.")
    parser.add_argument("--input-dir", type=Path, help="Directory that contains Demucs-separated stems.")
    parser.add_argument("--output-midi", type=Path, help="Destination MIDI file.")
    parser.add_argument("--output-summary", type=Path, help="Optional JSON summary file.")
    parser.add_argument("--profile", choices=sorted(TRANSCRIPTION_PROFILES.keys()), default="balanced",
                        help="Transcription profile that trades density for accuracy.")
    parser.add_argument("--tempo-bpm", type=float, help="Override the detected MIDI tempo with a fixed BPM.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.check:
        try:
            require_backends()
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print("audio-to-midi backend ready")
        return 0

    if args.input_dir is None or args.output_midi is None:
        print("Both --input-dir and --output-midi are required.", file=sys.stderr)
        return 1

    try:
        stem_files = discover_audio_files(args.input_dir)
        build_output(stem_files, args.output_midi, args.output_summary, args.profile, args.tempo_bpm)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
