# Mutagen Native

This is the first standalone no-bridge C++ app for Mutagen.

Current scope:

- native JUCE desktop shell
- native menu bar with `File`, `Edit`, `Settings`, and `Windows` actions wired to C++ project editing
- main shell layout centered on the track list/inspector plus a large piano-roll workspace docked beside it
- floating native panels window for arrangement, automation, samples, and piano-roll workflows
- dedicated floating native arrangement, automation, samples, and piano-roll windows from the `Windows` menu
- dedicated floating native tracks workspace with shared selection, editable inspector controls, and rack/playback actions
- dedicated floating native audio workspace with transport, mixer access, export actions, and live device summary
- dedicated floating native rack browser for bundled/imported plugins, selected-track rack assignment, and rack state/editor actions
- dedicated floating native render manager for track bounce files, relinking render paths, and placing renders back into the project
- C++ project model for `.aims` files
- load/save compatibility with the existing Python project format
- native track list plus editable track inspector
- native arrangement overview with draggable MIDI sections and transport/locator clicks
- floating native transport window with project play, track play, stop, tempo, loop, metronome, and audio-settings access
- arrangement and mixer opened as separate native windows by default to keep the main shell focused on tracks plus piano roll
- native mixer strips with volume, pan, mute, solo, and live host meter updates
- floating native mixer window for track-strip playback monitoring outside the main shell
- native automation editor for volume, pan, VST output gain, and saved VST parameters
- native sample library import plus sample-timeline placement and clip drag/delete editing
- piano roll note creation, drag/resize, quantize, duplicate, delete, copy/cut/paste, and select-all
- ruler-based playhead and locator editing
- undo/redo for native project and note edits
- native MIDI import/export
- native locator-range WAV export with offline rack rendering plus JUCE-side sample mixing
- native selected-track WAV export plus batch stem export that can update per-track rendered-audio paths
- bundled VST rack discovery plus rack-name resolution to real plugin paths for native playback/editor workflows
- direct C++ link to the JUCE VST host library, including opening a selected track's rack editor without Python
- native selected-track rack preview playback through the host transport
- native full-project preview through the host audio engine, including automation-lane playback
- live native audio-engine refresh for project edits while full-project preview is running
- native rack parameter sync from the JUCE host back into `TrackState.vstiParameters`
- native rack autosave on editor-close into `TrackState.vstiStatePath` while still supporting manual save
- native AI provider settings plus AI composition flow for OpenAI API-key usage and local Ollama models
- native audio-settings window backed by JUCE host device queries and live driver backend, device, buffer, and sample-rate changes

Build:

```powershell
.\scripts\build_native_app.ps1
```

Run:

```powershell
.\build\native-app\AIMusicStudioNative_artefacts\Release\Mutagen.exe
```

Open a project on startup:

```powershell
.\build\native-app\AIMusicStudioNative_artefacts\Release\Mutagen.exe --project "C:\path\to\song.aims"
```

This app deliberately avoids the Python bridge. The biggest remaining native milestones are:

- full in-process playback, transport, and audio-engine ownership inside the native app
- deeper render/stem utilities plus fuller sample workflow parity
- deeper mixer/rack and project-management parity with the legacy app
- remaining AI/service features beyond compose/settings, including broader assistant workflows and intelligence tools
