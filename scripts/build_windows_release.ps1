param(
    [string]$RepoRoot = "",
    [string]$PythonExecutable = "",
    [string]$PythonVersion = "3.13",
    [string]$ReleaseVersion = "dev",
    [switch]$SkipBundledVstBuild
)

$ErrorActionPreference = "Stop"

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
} else {
    $RepoRoot = (Resolve-Path $RepoRoot).Path
}

if (-not $PythonExecutable) {
    $venvRoot = Join-Path $RepoRoot ".venv-release"
    $venvPython = Join-Path $venvRoot "Scripts\\python.exe"
    if (-not (Test-Path $venvPython)) {
        py -$PythonVersion -m venv $venvRoot
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create the release virtual environment with Python $PythonVersion."
        }
    }
    $PythonExecutable = $venvPython
}

& $PythonExecutable -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip in the release environment."
}

& $PythonExecutable -m pip install -r (Join-Path $RepoRoot "requirements.txt") "pyinstaller>=6.11"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install packaging dependencies."
}

$bundledVstDir = Join-Path $RepoRoot "vsti"
if (-not $SkipBundledVstBuild -and -not (Test-Path $bundledVstDir)) {
    & (Join-Path $RepoRoot "scripts\\build_bundled_vsti_windows.ps1") -RepoRoot $RepoRoot -OutputDir $bundledVstDir
}

$pyiRoot = Join-Path $RepoRoot "build\\pyinstaller"
$pyiBuild = Join-Path $pyiRoot "build"
$pyiDist = Join-Path $pyiRoot "dist"
$releaseRoot = Join-Path $RepoRoot "dist"
$portableRoot = Join-Path $releaseRoot "AI Music Studio"
$releaseLabel = ($ReleaseVersion -replace '[^A-Za-z0-9._-]', '-')
$archivePath = Join-Path $releaseRoot "AI-Music-Studio-$releaseLabel-windows-x64.zip"
$versionPath = Join-Path $portableRoot "VERSION.txt"

foreach ($path in @($pyiBuild, $pyiDist, $portableRoot, $archivePath)) {
    if (Test-Path $path) {
        Remove-Item $path -Recurse -Force
    }
}

New-Item -ItemType Directory -Path $releaseRoot -Force | Out-Null

& $PythonExecutable -m PyInstaller `
    --noconfirm `
    --clean `
    --distpath $pyiDist `
    --workpath $pyiBuild `
    (Join-Path $RepoRoot "AI-Music-Studio.spec")
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller packaging failed."
}

$builtAppRoot = Join-Path $pyiDist "AI Music Studio"
if (-not (Test-Path $builtAppRoot)) {
    throw "Expected PyInstaller output folder not found: $builtAppRoot"
}

Copy-Item $builtAppRoot -Destination $portableRoot -Recurse -Force

if (Test-Path $bundledVstDir) {
    Copy-Item $bundledVstDir -Destination (Join-Path $portableRoot "vsti") -Recurse -Force
}

Copy-Item (Join-Path $RepoRoot "README.md") -Destination (Join-Path $portableRoot "README.md") -Force
Set-Content -Path $versionPath -Value $ReleaseVersion -Encoding ASCII

Compress-Archive -Path (Join-Path $portableRoot "*") -DestinationPath $archivePath

Write-Output $archivePath
