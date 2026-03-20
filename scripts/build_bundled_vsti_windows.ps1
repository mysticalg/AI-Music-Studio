param(
    [string]$RepoRoot = "",
    [string]$JuceDir = "",
    [string]$BuildDir = "",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
} else {
    $RepoRoot = (Resolve-Path $RepoRoot).Path
}

if (-not $JuceDir) {
    $JuceDir = Join-Path $RepoRoot ".cache\\JUCE"
}
if (-not $BuildDir) {
    $BuildDir = Join-Path $RepoRoot "plugins\\AdvancedVSTi\\build-windows"
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $RepoRoot "vsti"
}

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "cmake is required to build the bundled VST3 instruments."
}

if (-not (Test-Path $JuceDir)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $JuceDir) | Out-Null
    git clone --depth 1 https://github.com/juce-framework/JUCE.git $JuceDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to clone JUCE."
    }
}

$pluginRoot = Join-Path $RepoRoot "plugins\\AdvancedVSTi"
cmake -S $pluginRoot -B $BuildDir -DJUCE_DIR=$JuceDir
if ($LASTEXITCODE -ne 0) {
    throw "CMake configure failed for bundled VST3 instruments."
}

cmake --build $BuildDir --config Release
if ($LASTEXITCODE -ne 0) {
    throw "CMake build failed for bundled VST3 instruments."
}

$pluginBundles = Get-ChildItem -Path $BuildDir -Recurse -Directory -Filter *.vst3
if (-not $pluginBundles) {
    throw "No .vst3 bundles were produced under $BuildDir."
}

if (Test-Path $OutputDir) {
    Remove-Item $OutputDir -Recurse -Force
}
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

foreach ($bundle in $pluginBundles) {
    Copy-Item $bundle.FullName -Destination (Join-Path $OutputDir $bundle.Name) -Recurse -Force
}

Write-Output $OutputDir
