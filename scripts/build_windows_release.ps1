param(
    [string]$RepoRoot = "",
    [string]$ReleaseVersion = "dev",
    [string]$Configuration = "Release",
    [string]$OutputDir = "",
    [switch]$SkipBundledVstBuild
)

$ErrorActionPreference = "Stop"

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
} else {
    $RepoRoot = (Resolve-Path $RepoRoot).Path
}

if (-not $OutputDir) {
    $OutputDir = Join-Path $RepoRoot "dist"
}
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$OutputDir = (Resolve-Path $OutputDir).Path

$bundledVstDir = Join-Path $RepoRoot "vsti"
if (-not $SkipBundledVstBuild) {
    & (Join-Path $RepoRoot "scripts\\build_bundled_vsti_windows.ps1") -RepoRoot $RepoRoot -OutputDir $bundledVstDir | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Bundled VST build failed."
    }
}

$releaseRoot = $OutputDir
$portableRoot = Join-Path $releaseRoot "Mutagen"
$releaseLabel = ($ReleaseVersion -replace '[^A-Za-z0-9._-]', '-')
$archivePath = Join-Path $releaseRoot "Mutagen-$releaseLabel-windows-x64.zip"
$versionPath = Join-Path $portableRoot "VERSION.txt"
$guidePath = Join-Path $portableRoot "GUIDE.md"
$readmePath = Join-Path $portableRoot "README.md"
$builtExe = Join-Path $RepoRoot ("build\\native-app\\AIMusicStudioNative_artefacts\\{0}\\Mutagen.exe" -f $Configuration)

foreach ($path in @($portableRoot, $archivePath)) {
    if (Test-Path $path) {
        Remove-Item $path -Recurse -Force
    }
}

New-Item -ItemType Directory -Path $releaseRoot -Force | Out-Null

& (Join-Path $RepoRoot "scripts\\build_native_app.ps1") -Configuration $Configuration
if ($LASTEXITCODE -ne 0) {
    throw "Native app build failed."
}

if (-not (Test-Path $builtExe)) {
    throw "Expected native build output not found: $builtExe"
}

New-Item -ItemType Directory -Path $portableRoot -Force | Out-Null
Copy-Item $builtExe -Destination (Join-Path $portableRoot "Mutagen.exe") -Force

if (Test-Path $bundledVstDir) {
    Copy-Item $bundledVstDir -Destination (Join-Path $portableRoot "vsti") -Recurse -Force
}

Copy-Item (Join-Path $RepoRoot "README.md") -Destination $readmePath -Force
if (Test-Path (Join-Path $RepoRoot "GUIDE.md")) {
    Copy-Item (Join-Path $RepoRoot "GUIDE.md") -Destination $guidePath -Force
}
Set-Content -Path $versionPath -Value $ReleaseVersion -Encoding ASCII

Compress-Archive -Path (Join-Path $portableRoot "*") -DestinationPath $archivePath

Write-Output $archivePath
