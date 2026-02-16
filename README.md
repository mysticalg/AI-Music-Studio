# AI Music Studio (MVP+AI)

A desktop MIDI editor DAW prototype with a GUI built in **PySide6**, now with **OpenAI Codex-assisted composition**.

## Implemented features

- Track timeline panel with multiple tracks
- Piano roll editor with mouse note drawing/selection
- Quantization (1/4, 1/8, 1/16, 1/32)
- MIDI import/export (`.mid`) via `mido`
- Mixer board (volume + pan per track)
- Instrument board (instrument + plugin chain metadata)
- Built-in FX rack controls for EQ, Compression, Distortion, Phaser, Flanger, Delay, Reverb
- Virtual piano keyboard input (computer keyboard)
- Keyboard shortcuts for transport/editing
- Basic transport controls (play/stop simulation)
- **OpenAI Codex composition**: generate multi-track MIDI arrangements from natural language prompts

## OpenAI integration setup

Set your API key before launching:

```bash
export OPENAI_API_KEY="your_api_key_here"
# optional (defaults to gpt-5-codex)
export OPENAI_MODEL="gpt-5-codex"
```

In the app:
- Click **AI Compose (OpenAI Codex)** (or press `Ctrl+G`)
- Enter a prompt and bar length
- Generated tracks/notes are loaded directly into the project

## Planned / architecture-ready placeholders

- VST instrument hosting
- ASIO driver configuration
- Real-time audio engine and mastering chain
- Sample browser/sampler engine
- Advanced MIDI transformations and automation lanes

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Keyboard shortcuts

- `Space` → Toggle Play/Stop
- `Ctrl+N` → New project
- `Ctrl+O` → Import MIDI
- `Ctrl+S` → Export MIDI
- `Ctrl+Q` → Quantize selected notes
- `Ctrl+G` → AI Compose
- `Delete` → Delete selected notes
- `Z/X/C/V/B/N/M/,` → Trigger virtual piano notes

## Notes

This is still an MVP foundation, but now includes AI-assisted composition to generate usable starting arrangements quickly.
