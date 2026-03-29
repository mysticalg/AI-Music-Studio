# Mutagen Native App

This directory contains the main native Mutagen desktop application.

Current scope:

- native JUCE desktop shell
- sequencer-first main workspace with separate piano-roll window
- track list, inspector, mixer, transport, automation, samples, render manager, and rack browser
- C++ project model for `.aims` files
- shared live audio engine and per-track live VST editor windows
- native AI settings, compose flow, activity log, and HTTP debug logging
- native Windows packaging path through the root `scripts/` release utilities

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

The supported workflow is documented in the repo guide:

- [GUIDE.md](../../GUIDE.md)
