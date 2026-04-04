# Mutagen

Mutagen is a native C++ desktop music workstation built with JUCE. This repository is now centered on the standalone native app, the shared live VST engine, the bundled VST3 instrument set, and the static documentation site.

The legacy Python/PySide shell, PyInstaller packaging path, and Python bridge helpers are no longer part of the supported workflow in this repo.

Live site: [mysticalg.github.io/AI-Music-Studio](https://mysticalg.github.io/AI-Music-Studio/)  
Releases: [github.com/mysticalg/AI-Music-Studio/releases](https://github.com/mysticalg/AI-Music-Studio/releases)  
Guide: [GUIDE.md](GUIDE.md)

## Current scope

- Native JUCE desktop app in `native/app`
- Shared live playback engine and VST host in `native/vst3host`
- Bundled VST3 instrument suite in the `plugins/AdvancedVSTi` submodule
- Pattern sequencer, piano roll, controller lane, mixer, automation, samples, transport, themes, and floating panels
- Shared live plugin instances for playback and editor windows, so parameter moves apply directly during playback
- Native AI compose/settings flow and on-disk activity or HTTP debug logs
- Static GitHub Pages site in `docs/`

## Repository layout

- [native/app](native/app): main Mutagen desktop application
- [native/vst3host](native/vst3host): JUCE host and shared live engine layer
- [plugins/AdvancedVSTi](plugins/AdvancedVSTi): bundled synths and instruments submodule
- [docs](docs): GitHub Pages site and product docs
- [scripts](scripts): build, packaging, smoke, and profiling utilities for the native app

## Build locally

1. Initialize submodules:

```powershell
git submodule update --init --recursive
```

2. Build bundled VST3 instruments into `vsti/`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_bundled_vsti_windows.ps1
```

3. Build the native app:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_native_app.ps1 -Configuration Release
```

4. Run Mutagen:

```powershell
.\build\native-app\AIMusicStudioNative_artefacts\Release\Mutagen.exe
```

## Package a Windows release

Portable ZIP:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_release.ps1 -ReleaseVersion v0.3.2
```

Installer:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_installer.ps1 -ReleaseVersion v0.3.2
```

The release pipeline currently publishes Windows native builds only. The packaging output is a `Mutagen.exe` portable bundle plus a Windows installer.

## GitHub release automation

Pushing a `v*` tag triggers [.github/workflows/release.yml](.github/workflows/release.yml), which:

- builds the bundled VST3 instruments
- packages the native Windows app
- builds the Windows installer
- publishes the release assets to GitHub Releases

## GitHub Pages

The project site is served from [docs](docs) via [.github/workflows/deploy-pages.yml](.github/workflows/deploy-pages.yml). Pushes that change `docs/`, `README.md`, or `GUIDE.md` republish the site.

## Bundled instruments

The bundled VST3 instruments live in the `plugins/AdvancedVSTi` submodule. That shared source currently builds:

- `Virus Synth`
- `AI Drum Machine`
- `AI 808 Machine`
- `AI Bass Synth`
- `AI TB303`
- `AI String Synth`
- `AI Lead Synth`
- `AI Pad Synth`
- `AI Pluck Synth`
- `AI Sampler`
- `AI VEC1 Drum Pads`
- `AI Piano`
- `AI Strings`
- `AI Violin`
- `AI Flute`
- `AI Saxophone`
- `AI Bass Guitar`
- `AI Organ`

Some acoustic instruments can use open SFZ-based sample libraries cached under the submodule. Populate those with:

```powershell
python .\plugins\AdvancedVSTi\scripts\fetch_open_instrument_samples.py
```

## Project data and logs

Mutagen stores user data under the app data directory:

- Windows: `%APPDATA%\Mutagen`
- macOS: `~/Library/Application Support/Mutagen`
- Linux: `$XDG_DATA_HOME/Mutagen` or `~/.local/share/Mutagen`

AI request and response diagnostics are written to the native log folder, including the session activity log and `ai-http-debug.log`.

## Development notes

- The native runtime path now prefers direct shared-engine exports over the older JSON command path.
- Remaining generic host-command usage is limited to offline helper surfaces, not normal playback or editor interaction.
- Temporary debug screenshots and scratch directories are ignored through [.gitignore](.gitignore).

For day-to-day workflow details, use the repo guide: [GUIDE.md](GUIDE.md).
