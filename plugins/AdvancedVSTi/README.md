# AdvancedVSTi (JUCE VST3 Instrument)

This folder adds a JUCE-based VST instrument scaffold with these implemented blocks:

- Oscillator types: sine, saw, square, noise, sample (internal placeholder sample buffer).
- Unison voice stacking + detune.
- FM operator amount.
- Oscillator hard-sync style reset control.
- Amp and filter ADSR envelopes with logarithmic/exponential shaping.
- Filter modes: low-pass, high-pass, band-pass, notch.
- Two assignable LFOs (pitch and filter assignments in this version).
- Arpeggiator modes: up, down, up/down, random.
- Rhythm gate and per-note oscillator length gate.

## Build (Windows example)

1. Install JUCE and CMake.
2. Configure:

```bash
cmake -S plugins/AdvancedVSTi -B plugins/AdvancedVSTi/build -DJUCE_DIR=C:/dev/JUCE
```

3. Build:

```bash
cmake --build plugins/AdvancedVSTi/build --config Release
```

4. The VST3 output is created by JUCE in the build artifacts and can be loaded by VST3 hosts.

## About `.dll`

Modern plugin hosts typically use **VST3** (`.vst3`). Legacy VST2 `.dll` output is generally not enabled in current JUCE/public SDK workflows.

If you specifically need a `.dll`, use one of:
- a host that supports CLAP/LV2 with bridge tools,
- an older licensed VST2 pipeline,
- or deploy VST3 and rely on host-native support (recommended).
