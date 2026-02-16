# AI Music Studio (MIDI + AI Instrument Rendering)

A desktop MIDI editor DAW prototype with a GUI built in **PySide6**, with:
- OpenAI-assisted composition, and
- MIDI import that separates channels/programs into individual tracks, assigns AI instrument families, and renders synthesized audio stems.

## Implemented features

- Track timeline panel with multiple tracks
- Piano roll editor with mouse note drawing/selection
- Quantization (1/4, 1/8, 1/16, 1/32)
- MIDI import/export (`.mid`) via `mido`
- **MIDI import by channel/program** into separate tracks
- **AI instrument assignment** per track (horn/strings/bass/etc.)
- **AI-style synthesis rendering** to WAV stems per track (`renders/*.wav`)
- Mixer board (volume + pan per track)
- Instrument board (instrument + synth profile + FX metadata)
- Built-in FX rack controls for EQ, Compression, Distortion, Phaser, Flanger, Delay, Reverb
- Virtual piano keyboard input (computer keyboard)
- Keyboard shortcuts for transport/editing
- Basic transport controls (play/stop simulation)
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
- optional instrument family classification during MIDI import.

If no API key is present, instrument classification falls back to deterministic GM/track-name heuristics.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Usage flow for your requested workflow

1. Click **Import MIDI + AI Instrument Render** (`Ctrl+O`)
2. The app splits imported material into channel-specific tracks.
3. Each track gets instrument + synth profile assignment (horn/strings/etc.).
4. Confirm render prompt to synthesize WAV stems, or press `Ctrl+R` later.
5. Rendered stem paths are shown in mixer panel and saved to `renders/`.

## Keyboard shortcuts

- `Space` → Toggle Play/Stop
- `Ctrl+N` → New project
- `Ctrl+O` → Import MIDI + AI instrument assignment
- `Ctrl+S` → Export MIDI
- `Ctrl+Q` → Quantize selected notes
- `Ctrl+R` → Render AI audio stems
- `Ctrl+G` → AI Compose
- `Delete` → Delete selected notes
- `Z/X/C/V/B/N/M/,` → Trigger virtual piano notes
