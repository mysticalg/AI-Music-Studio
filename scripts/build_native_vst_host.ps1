param(
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$SourceDir = Join-Path $RepoRoot "native\vst3host"
$BuildDir = Join-Path $RepoRoot "build\native-vst3host"
$JuceDir = Join-Path $RepoRoot ".cache\JUCE"

if (-not (Test-Path $JuceDir)) {
    throw "JUCE not found at $JuceDir"
}

cmake -S $SourceDir -B $BuildDir -DJUCE_DIR="$JuceDir"
if ($LASTEXITCODE -ne 0) {
    throw "CMake configure failed with exit code $LASTEXITCODE"
}

cmake --build $BuildDir --config $Configuration --target AIMusicStudioVSTHost
if ($LASTEXITCODE -ne 0) {
    throw "CMake build failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Built native host in $BuildDir" -ForegroundColor Green
