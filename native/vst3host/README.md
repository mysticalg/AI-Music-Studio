# Mutagen VST Host

This is the native JUCE host and shared live engine layer that powers Mutagen's playback and editor path.

Current scope:

- Windows JUCE GUI app
- Loads one VST3 at a time
- Opens the plugin's real native editor
- Hosts audio directly through JUCE's audio device manager
- Lets you choose the JUCE backend and output device in the host window
- Provides a built-in MIDI keyboard for live testing
- Remembers the last plugin path, host window bounds, and editor window bounds
- Direct in-process C exports for the Mutagen app runtime
- Optional legacy JSON command surface retained only for low-level diagnostics

Build:

```powershell
.\scripts\build_native_vst_host.ps1
```

The Windows build now enables:

- `Windows Audio`
- `Windows Audio (Exclusive Mode)`
- `Windows Audio (Low Latency Mode)`
- `DirectSound`
- `ASIO` when a usable ASIO driver is installed

Run:

```powershell
.\build\native-vst3host\AIMusicStudioVSTHost_artefacts\Release\AI Music Studio VST Host.exe
```

Run with a plugin preloaded:

```powershell
.\build\native-vst3host\AIMusicStudioVSTHost_artefacts\Release\AI Music Studio VST Host.exe --plugin "C:\Users\drhoo\AppData\Local\Programs\Common\VST3\Dexed.vst3"
```

Run with the native editor opened immediately:

```powershell
.\build\native-vst3host\AIMusicStudioVSTHost_artefacts\Release\AI Music Studio VST Host.exe --plugin "C:\Users\drhoo\AppData\Local\Programs\Common\VST3\Dexed.vst3" --open-editor
```

Run with a plugin preloaded and the editor opened immediately:

```powershell
.\build\native-vst3host\AIMusicStudioVSTHost_artefacts\Release\AI Music Studio VST Host.exe --plugin "C:\Users\drhoo\AppData\Local\Programs\Common\VST3\Dexed.vst3" --open-editor
```

You can also request a backend on startup:

```powershell
.\build\native-vst3host\AIMusicStudioVSTHost_artefacts\Release\AI Music Studio VST Host.exe --audio-device-type "Windows Audio (Exclusive Mode)"
```

Smoke test multiple installed synths:

```powershell
.\scripts\smoke_test_native_vst_host.ps1
.\scripts\smoke_test_native_vst_host.ps1 -OpenEditor
```

Output target:

- `build/native-vst3host/.../AIMusicStudioVSTHost.exe`

Smoke-tested locally against:

- `Dexed.vst3`
- `YMulator-Synth.vst3`
- `js80p.vst3`
- `TAL-NoiseMaker.vst3`
- `Helm`

Mutagen uses the host through direct in-process exports. The Python bridge helpers and Python shell are no longer part of the supported repo workflow.
