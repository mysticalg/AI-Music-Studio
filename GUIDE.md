# Mutagen Guide

This guide covers the supported native workflow for Mutagen.

## Build and run

1. Initialize submodules:

```powershell
git submodule update --init --recursive
```

2. Build the bundled instruments:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_bundled_vsti_windows.ps1
```

3. Build the app:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_native_app.ps1 -Configuration Release
```

4. Launch:

```powershell
.\build\native-app\AIMusicStudioNative_artefacts\Release\Mutagen.exe
```

## Main workflow

1. Create or select a track.
2. Draw pattern blocks in the sequencer.
3. Double-click a pattern to edit it in the piano roll.
4. Use the lower controller pane for velocity, MIDI CC, or automation drawing.
5. Assign a bundled or third-party VST3 to the track and open `Edit VST`.
6. Play, loop, mix, and render directly from the native app.

## Sequencer

- New pattern size defaults to `1 bar`.
- Sequencer snap is independent from piano-roll quantize.
- Pattern clips can be drawn, moved, resized, duplicated, pasted, glued, or erased.
- Resizing a clip trims notes and controller data that fall outside the new bounds.
- Double-click opens the clip in the piano roll.

## Piano roll

- Tools are available from the right-click menu.
- Supported tools include `Pencil`, `Select`, `Glue`, and `Eraser`.
- The pencil supports note shapes, chord drops, and rapid brush-style entry.
- Box selection is available in `Select` mode.
- The lower pane is always visible for velocity and controller editing.
- Piano-roll zoom and row height are local to the piano-roll window.

## Controller lane

- The lower pane can target velocity, MIDI CC lanes, and track automation targets.
- Shape drawing is available for controller editing, including line and waveform styles.
- Velocity remains the fastest lane for direct note-expression editing.

## Tracks and mixer

- Mute and solo live in the first track columns.
- The `V` column shows or hides the live VST editor window.
- The `Vol` strip beside the track name shows level and opens the mixer on click.
- Mixer meters and transport meters come from the native shared engine.

## VST workflow

- Mutagen uses one live plugin instance per track for playback and editor access.
- Editor parameter moves apply directly to the running playback instance.
- Multiple VST editor windows can stay open at once.
- VST folders are managed from `Settings > VST Folder Manager...`.
- Bundled VST3 plugins live in `vsti/`.

## Audio and playback

- Audio device settings live in the separate audio window.
- The transport shows CPU plus master left and right levels.
- The playback locator is refresh-rate aware on the UI side.
- Stress profiling is available via:

```powershell
python .\scripts\profile_native_stress.py
```

## AI and logging

- `AI Compose` and provider settings are native C++ features.
- `Windows > Show Activity Log Window` shows request, response, and app events.
- AI HTTP diagnostics are written to `ai-http-debug.log` in the Mutagen app-data logs folder.

## Packaging

Portable ZIP:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_release.ps1 -ReleaseVersion v0.3.2
```

Installer:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_installer.ps1 -ReleaseVersion v0.3.2
```

## Scope note

The supported repo path is now the native Mutagen application. The old Python shell and Python bridge tooling have been removed from the supported tree.
