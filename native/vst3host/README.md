# AI Music Studio VST Host

This is the native JUCE host that powers AI Music Studio's VST playback and editor path.

Current scope:

- Windows JUCE GUI app
- Loads one VST3 at a time
- Opens the plugin's real native editor
- Hosts audio directly through JUCE's audio device manager
- Lets you choose the JUCE backend and output device in the host window
- Provides a built-in MIDI keyboard for live testing
- Remembers the last plugin path, host window bounds, and editor window bounds
- Exposes an optional localhost JSON command bridge for load/editor/MIDI control

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

Run with the localhost command bridge enabled:

```powershell
.\build\native-vst3host\AIMusicStudioVSTHost_artefacts\Release\AI Music Studio VST Host.exe --plugin "C:\Users\drhoo\AppData\Local\Programs\Common\VST3\Dexed.vst3" --port 47653
py .\scripts\native_vst_host_client.py --port 47653 --command status
py .\scripts\native_vst_host_client.py --port 47653 --command open_editor
py .\scripts\native_vst_host_client.py --port 47653 --command note_on --note 60 --velocity 0.8
py .\scripts\native_vst_host_client.py --port 47653 --command note_off --note 60
```

You can also request a backend on startup:

```powershell
.\build\native-vst3host\AIMusicStudioVSTHost_artefacts\Release\AI Music Studio VST Host.exe --audio-device-type "Windows Audio (Exclusive Mode)"
```

Or use the Python bridge helper directly:

```python
from scripts.native_vst_host_bridge import NativeVstHostBridge

bridge = NativeVstHostBridge(plugin_path=r"C:\Users\drhoo\AppData\Local\Programs\Common\VST3\Dexed.vst3")
bridge.start()
bridge.command("open_editor")
bridge.command("note_on", note=60, velocity=0.8)
bridge.command("note_off", note=60)
bridge.stop()
```

Smoke test multiple installed synths:

```powershell
.\scripts\smoke_test_native_vst_host.ps1
.\scripts\smoke_test_native_vst_host.ps1 -OpenEditor
.\scripts\smoke_test_native_vst_bridge.ps1
```

Output target:

- `build/native-vst3host/.../AIMusicStudioVSTHost.exe`

Smoke-tested locally against:

- `Dexed.vst3`
- `YMulator-Synth.vst3`
- `js80p.vst3`
- `TAL-NoiseMaker.vst3`
- `Helm`

Planned follow-up work:

- multi-plugin graph/rack
- per-track state and IPC bridge back to `app.py`
- real MIDI input devices
- plugin scan/cache database
- better audio device setup UI
