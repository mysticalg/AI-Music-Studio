#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


INSTALL_HINT = "py -3.10 -m pip install demucs librosa soundfile numpy scipy"
AUDIO_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".ogg", ".mp3"}


def require_backends():
    missing = []

    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        missing.append(("numpy", exc))
        np = None

    try:
        import soundfile as sf
    except Exception as exc:  # pragma: no cover
        missing.append(("soundfile", exc))
        sf = None

    try:
        import librosa
    except Exception as exc:  # pragma: no cover
        missing.append(("librosa", exc))
        librosa = None

    try:
        import julius
    except Exception as exc:  # pragma: no cover
        missing.append(("julius", exc))
        julius = None

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        missing.append(("torch", exc))
        torch = None

    try:
        from demucs.apply import BagOfModels, apply_model
        from demucs.htdemucs import HTDemucs
        from demucs.pretrained import get_model
    except Exception as exc:  # pragma: no cover
        missing.append(("demucs", exc))
        BagOfModels = None
        HTDemucs = None
        apply_model = None
        get_model = None

    if missing:
        details = "\n".join(f"- {name}: {exc}" for name, exc in missing)
        raise RuntimeError(
            "Missing stem-separation backend packages.\n"
            f"Install with:\n{INSTALL_HINT}\n\n"
            f"Details:\n{details}"
        )

    return np, sf, librosa, julius, torch, BagOfModels, HTDemucs, apply_model, get_model


def convert_audio_channels(wav, channels: int):
    src_channels, length = wav.shape
    if src_channels == channels:
        return wav
    if channels == 1:
        return wav.mean(dim=0, keepdim=True)
    if src_channels == 1:
        return wav.expand(channels, length)
    if src_channels >= channels:
        return wav[:channels, :]
    raise ValueError("The audio file has fewer channels than requested.")


def load_audio_file(input_file, target_samplerate, target_channels, sf, librosa, julius, np, torch):
    errors = []

    try:
        audio, samplerate = sf.read(str(input_file), always_2d=True, dtype="float32")
        wav = torch.from_numpy(np.ascontiguousarray(audio.T))
    except Exception as exc:
        errors.append(("soundfile", exc))
    else:
        wav = convert_audio_channels(wav, target_channels)
        if samplerate != target_samplerate:
            wav = julius.resample_frac(wav, samplerate, target_samplerate)
        return wav.contiguous()

    try:
        audio, samplerate = librosa.load(str(input_file), sr=None, mono=False)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    except Exception as exc:
        errors.append(("librosa", exc))
    else:
        wav = convert_audio_channels(wav, target_channels)
        if samplerate != target_samplerate:
            wav = julius.resample_frac(wav, samplerate, target_samplerate)
        return wav.contiguous()

    details = "\n".join(f"- {backend}: {exc}" for backend, exc in errors)
    raise RuntimeError(
        "Could not load the input audio file.\n"
        f"Supported extensions: {', '.join(sorted(AUDIO_EXTENSIONS))}\n\n"
        f"Details:\n{details}"
    )


def write_stem(output_file, audio, samplerate, sf):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    data = audio.detach().cpu().transpose(0, 1).numpy()
    sf.write(str(output_file), data, samplerate, subtype="PCM_16")


def build_output_tree(output_dir: Path, model_name: str, input_file: Path) -> Path:
    return output_dir / model_name / input_file.stem


def separate(args):
    np, sf, librosa, julius, torch, BagOfModels, HTDemucs, apply_model, get_model = require_backends()

    if not args.input.exists():
        raise RuntimeError(f"Input audio file was not found: {args.input}")

    if args.input.suffix.lower() not in AUDIO_EXTENSIONS:
        raise RuntimeError(
            f"Unsupported audio file extension: {args.input.suffix}\n"
            f"Supported extensions: {', '.join(sorted(AUDIO_EXTENSIONS))}"
        )

    model = get_model(args.model)
    max_allowed_segment = float("inf")
    if isinstance(model, HTDemucs):
        max_allowed_segment = float(model.segment)
    elif isinstance(model, BagOfModels):
        max_allowed_segment = model.max_allowed_segment

    if args.segment is not None and args.segment > max_allowed_segment:
        raise RuntimeError(
            "The selected Demucs model does not support that segment length. "
            f"Maximum segment is {max_allowed_segment}."
        )

    model.cpu()
    model.eval()

    wav = load_audio_file(
        args.input,
        target_samplerate=model.samplerate,
        target_channels=model.audio_channels,
        sf=sf,
        librosa=librosa,
        julius=julius,
        np=np,
        torch=torch,
    )

    ref = wav.mean(0)
    mean = ref.mean()
    std = ref.std()
    if float(std) < 1.0e-8:
        std = torch.tensor(1.0, dtype=wav.dtype)

    wav = (wav - mean) / std
    sources = apply_model(
        model,
        wav[None],
        device=args.device,
        shifts=max(1, int(args.shifts)),
        split=True,
        overlap=float(args.overlap),
        progress=False,
        num_workers=0,
        segment=args.segment,
    )[0]
    sources = sources * std + mean

    output_root = build_output_tree(args.output_dir, args.model, args.input)
    output_root.mkdir(parents=True, exist_ok=True)

    for source, stem_name in zip(sources, model.sources):
        stem_file = output_root / f"{stem_name}.wav"
        write_stem(stem_file, source, model.samplerate, sf)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Separate audio stems with Demucs and write WAVs without torchaudio.")
    parser.add_argument("--check", action="store_true", help="Verify backend packages and exit.")
    parser.add_argument("--input", type=Path, help="Input WAV or MP3 file.")
    parser.add_argument("--output-dir", type=Path, help="Root output directory for separated stems.")
    parser.add_argument("--model", default="mdx_q", help="Demucs model name.")
    parser.add_argument("--device", default="cpu", help="Torch device to use.")
    parser.add_argument("--segment", type=float, default=8.0, help="Segment length in seconds.")
    parser.add_argument("--shifts", type=int, default=1, help="Number of shift-averaging passes.")
    parser.add_argument("--overlap", type=float, default=0.25, help="Chunk overlap ratio.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    if args.check:
        require_backends()
        print("ok")
        return 0

    if args.input is None or args.output_dir is None:
        raise RuntimeError("--input and --output-dir are required unless --check is used.")

    separate(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
