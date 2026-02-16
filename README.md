# AI Music Studio (MIDI + AI Instrument Rendering + Samples)

A desktop MIDI editor DAW prototype with a GUI built in **PySide6**, with:
- OpenAI-assisted composition,
- MIDI import split by channel/program with AI instrument assignment, and
- sample workflow with **WAV/MP3 import/export** plus waveform timeline display.

## Implemented features

- Track timeline panel for MIDI tracks
- Piano roll editor with mouse note drawing/selection
- Quantization (1/4, 1/8, 1/16, 1/32)
- MIDI import/export (`.mid`) via `mido`
- MIDI import by channel/program into separate tracks
- AI instrument assignment per MIDI track (horn/strings/bass/etc.)
- AI-style MIDI synthesis rendering to WAV stems per track (`renders/*.wav`)
- **Sample toolbox** for WAV/MP3 sample import
- **Sample timeline tab** that displays waveform blocks when samples are placed
- **Sample timeline audio export** to WAV or MP3
- Mixer board (volume + pan per track)
- Instrument board (instrument + synth profile + FX metadata)
- Built-in FX rack controls for EQ, Compression, Distortion, Phaser, Flanger, Delay, Reverb
- Virtual piano keyboard input (computer keyboard)
- Keyboard shortcuts for transport/editing
- OpenAI Codex composition from natural language prompts

## OpenAI integration setup

Set your API key before launching:

```bash
export OPENAI_API_KEY="your_api_key_here"
# optional (defaults to gpt-5-codex)
export OPENAI_MODEL="gpt-5-codex"
```

OpenAI is used for:
- AI composition (`Ctrl+G`), and
- optional instrument-family classification during MIDI import.

If no API key is present, classification falls back to deterministic GM/track-name heuristics.

## Audio format notes (WAV/MP3)

- WAV import/export is native.
- MP3 import/export requires `ffmpeg` available on your system PATH.
- Imported MP3 files are converted to WAV internally for waveform preview/rendering.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Usage flow (samples)

1. Click **Import Sample (WAV/MP3)** (or `Ctrl+Shift+O`)
2. Select a sample from **Samples Toolbox**
3. Click **Place Selected Sample On Timeline** and set start time
4. View waveform block in **Sample Timeline** tab
5. Export combined sample timeline audio via **Export Sample Timeline Audio (WAV/MP3)** (`Ctrl+E`)

## Keyboard shortcuts

- `Space` → Toggle Play/Stop
- `Ctrl+N` → New project
- `Ctrl+O` → Import MIDI + AI instrument assignment
- `Ctrl+Shift+O` → Import sample WAV/MP3
- `Ctrl+S` → Export MIDI
- `Ctrl+E` → Export sample timeline audio WAV/MP3
- `Ctrl+Q` → Quantize selected notes
- `Ctrl+R` → Render AI audio stems
- `Ctrl+G` → AI Compose
- `Delete` → Delete selected notes
- `Z/X/C/V/B/N/M/,` → Trigger virtual piano notes
