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

## Automated GitHub build + download

The repository includes a GitHub Actions workflow at `.github/workflows/build-vsti.yml` that compiles the plugin on `windows-latest` and uploads a downloadable ZIP artifact.

### When it runs

- Pushes affecting `plugins/AdvancedVSTi/**`
- Pull requests affecting `plugins/AdvancedVSTi/**`
- Manual runs from **Actions > Build AdvancedVSTi > Run workflow**

### How to download

1. Open the **Actions** tab in GitHub.
2. Open a successful **Build AdvancedVSTi** run.
3. Download the artifact named **AdvancedVSTi-windows-release**.
4. Unzip it and load the contained `.vst3` plugin in your VST3 host.

## Build locally (Windows example)

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
