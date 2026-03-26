#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# build_macos_release.sh
# Packages a portable macOS release of AI Music Studio.
#
# Usage:
#   ./scripts/build_macos_release.sh [--repo-root DIR] [--python PYTHON]
#       [--release-version VERSION] [--skip-bundled-vst-build]
#
# Prints the path to the final .zip archive on the last line of stdout.
# ------------------------------------------------------------------

REPO_ROOT=""
PYTHON_EXECUTABLE=""
PYTHON_VERSION="3.13"
RELEASE_VERSION="dev"
SKIP_BUNDLED_VST_BUILD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-root)        REPO_ROOT="$2";              shift 2;;
        --python)           PYTHON_EXECUTABLE="$2";      shift 2;;
        --python-version)   PYTHON_VERSION="$2";         shift 2;;
        --release-version)  RELEASE_VERSION="$2";        shift 2;;
        --skip-bundled-vst-build) SKIP_BUNDLED_VST_BUILD=true; shift;;
        *) echo "Unknown option: $1" >&2; exit 1;;
    esac
done

if [[ -z "$REPO_ROOT" ]]; then
    REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"

# ---- Python environment ------------------------------------------
if [[ -z "$PYTHON_EXECUTABLE" ]]; then
    VENV_ROOT="$REPO_ROOT/.venv-release"
    if [[ ! -f "$VENV_ROOT/bin/python" ]]; then
        python${PYTHON_VERSION} -m venv "$VENV_ROOT"
    fi
    PYTHON_EXECUTABLE="$VENV_ROOT/bin/python"
fi

"$PYTHON_EXECUTABLE" -m pip install --upgrade pip
"$PYTHON_EXECUTABLE" -m pip install -r "$REPO_ROOT/requirements.txt" "pyinstaller>=6.11"

# ---- Build bundled VST3 instruments (optional) --------------------
BUNDLED_VST_DIR="$REPO_ROOT/vsti"
if [[ "$SKIP_BUNDLED_VST_BUILD" == false ]] && [[ ! -d "$BUNDLED_VST_DIR" ]]; then
    echo "Building bundled VST3 instruments for macOS..."
    JUCE_DIR="$REPO_ROOT/JUCE"
    if [[ ! -d "$JUCE_DIR" ]]; then
        git clone --depth 1 https://github.com/juce-framework/JUCE.git "$JUCE_DIR"
    fi
    cmake -S "$REPO_ROOT/plugins/AdvancedVSTi" \
          -B "$REPO_ROOT/plugins/AdvancedVSTi/build-macos" \
          -DJUCE_DIR="$JUCE_DIR" \
          -DCMAKE_BUILD_TYPE=Release
    cmake --build "$REPO_ROOT/plugins/AdvancedVSTi/build-macos" --config Release

    mkdir -p "$BUNDLED_VST_DIR"
    find "$REPO_ROOT/plugins/AdvancedVSTi/build-macos" -type d -name "*.vst3" -print0 \
        | while IFS= read -r -d '' plugin; do
            cp -R "$plugin" "$BUNDLED_VST_DIR/"
          done
fi

# ---- PyInstaller packaging ----------------------------------------
PYI_ROOT="$REPO_ROOT/build/pyinstaller"
PYI_BUILD="$PYI_ROOT/build"
PYI_DIST="$PYI_ROOT/dist"
RELEASE_ROOT="$REPO_ROOT/dist"
PORTABLE_ROOT="$RELEASE_ROOT/AI Music Studio"
RELEASE_LABEL="$(echo "$RELEASE_VERSION" | sed 's/[^A-Za-z0-9._-]/-/g')"
ARCHIVE_PATH="$RELEASE_ROOT/AI-Music-Studio-${RELEASE_LABEL}-macos-x64.zip"
VERSION_PATH="$PORTABLE_ROOT/VERSION.txt"

rm -rf "$PYI_BUILD" "$PYI_DIST" "$PORTABLE_ROOT" "$ARCHIVE_PATH"
mkdir -p "$RELEASE_ROOT"

"$PYTHON_EXECUTABLE" -m PyInstaller \
    --noconfirm \
    --clean \
    --distpath "$PYI_DIST" \
    --workpath "$PYI_BUILD" \
    "$REPO_ROOT/AI-Music-Studio.spec"

BUILT_APP_ROOT="$PYI_DIST/AI Music Studio"
if [[ ! -d "$BUILT_APP_ROOT" ]]; then
    echo "Expected PyInstaller output folder not found: $BUILT_APP_ROOT" >&2
    exit 1
fi

cp -R "$BUILT_APP_ROOT" "$PORTABLE_ROOT"

if [[ -d "$BUNDLED_VST_DIR" ]]; then
    cp -R "$BUNDLED_VST_DIR" "$PORTABLE_ROOT/vsti"
fi

cp "$REPO_ROOT/README.md" "$PORTABLE_ROOT/README.md"
echo "$RELEASE_VERSION" > "$VERSION_PATH"

# Use ditto for macOS-friendly zip that preserves resource forks/symlinks
ditto -c -k --sequesterRsrc --keepParent "$PORTABLE_ROOT" "$ARCHIVE_PATH"

echo "$ARCHIVE_PATH"
