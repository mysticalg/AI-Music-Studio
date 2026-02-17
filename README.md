# AI Music Studio (MIDI + AI Instrument Rendering + Samples)

A desktop MIDI editor DAW prototype with a GUI built in **PySide6**, with:
- OpenAI-assisted composition,
- MIDI import split by channel/program with AI instrument assignment, and
- sample workflow with **WAV/MP3 import/export** plus waveform timeline display.

## Implemented features

- Track timeline panel for MIDI tracks
- Piano roll editor with right-click mini toolbar (selector, pencil, scissors, eraser, line tool)
- Note length selector in-editor for drawing tools
- Quantization (1/4, 1/8, 1/16, 1/32)
- MIDI import/export (`.mid`) via `mido`
- MIDI import by channel/program into separate tracks
- AI instrument assignment per MIDI track (horn/strings/bass/etc.)
- AI-style MIDI synthesis rendering to WAV stems per track (`renders/*.wav`)
- **Sample toolbox** for WAV/MP3 sample import
- **Sample timeline tab** that displays waveform blocks when samples are placed
- **Sample timeline audio export** to WAV or MP3
- Mixer board (volume + pan + mute/solo per track)
- Instrument board (instrument type + GM/VSTI rack selection + FX metadata)
- Built-in FX rack controls for EQ, Compression, Distortion, Phaser, Flanger, Delay, Reverb
- Virtual piano keyboard input (computer keyboard)
- Floating transport bar with playback controls, tempo setting, and loop locators
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
- AI composition (`Ctrl+G`),
- Codex track assistant actions from **Settings > OpenAI > Prompt Codex About Tracks**, and
- optional instrument-family classification during MIDI import.

You can connect OpenAI in-app via **Settings > OpenAI > Connect** using either:
- API key mode, or
- OAuth (PKCE) mode by opening browser login, then pasting the returned authorization code.

If OpenAI is not connected, classification falls back to deterministic GM/track-name heuristics.

## Audio format notes (WAV/MP3)

- WAV import/export is native.
- MP3 import/export requires `ffmpeg` available on your system PATH.
- Imported MP3 files are converted to WAV internally for waveform preview/rendering.

## Carla integration plan (external dependency, no fork)

We will integrate with [Carla](https://github.com/falkTX/Carla) as an **external dependency** and **not fork Carla initially**.

### Decision

- Keep AI Music Studio as the arranger/editor UI.
- Use Carla (`carla-single` / `carla`) as the plugin host for VST runtime and GUI.
- Track Carla as an upstream dependency and avoid maintaining a custom Carla branch in the first iteration.

### Why this approach

- Fastest path to production use with the least maintenance overhead.
- Preserves cross-platform host behavior already provided by Carla.
- Keeps this repo focused on composition, timeline, and workflow UX.

### Implementation phases

- **Phase 1: External host baseline** ✅
  - Detect Carla binaries in PATH (or configure/reset host detection in **Settings > Instruments** using **Set Carla Host Binary…** and **Use PATH Carla Detection**).
  - Launch selected rack plugin GUI in Carla.
  - Store per-track Carla state path references.

- **Phase 2: Session interoperability** ✅
  - Export/import Carla session snapshots from **Settings > Instruments** (host path, rack assignment, VST state path, and VST parameter values).
  - Re-link track Carla metadata across sessions.

- **Phase 3: Transport + automation bridge** ✅
  - Write live bridge state to `renders/carla_bridge_state.json` during playback (transport, locators, rack-track mappings).
  - Apply targeted parameter mapping for rack tracks (`Param 1` from volume, `Param 2` from pan) when bridge is enabled.

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
