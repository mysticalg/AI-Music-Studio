param(
    [string]$RepoRoot = "",
    [string]$ReleaseVersion = "dev",
    [string]$SourceDir = "",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
} else {
    $RepoRoot = (Resolve-Path $RepoRoot).Path
}

if (-not $SourceDir) {
    $SourceDir = Join-Path $RepoRoot "dist\\Mutagen"
}
$SourceDir = (Resolve-Path $SourceDir).Path

if (-not $OutputDir) {
    $OutputDir = Join-Path $RepoRoot "dist"
}
$OutputDir = (Resolve-Path $OutputDir).Path

$candidateIsccPaths = @(
    "C:\\Program Files (x86)\\Inno Setup 6\\ISCC.exe",
    "C:\\Program Files\\Inno Setup 6\\ISCC.exe",
    (Join-Path $env:LOCALAPPDATA "Programs\\Inno Setup 6\\ISCC.exe")
) | Where-Object { $_ }

$isccPath = $candidateIsccPaths | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $isccPath) {
    throw "Inno Setup compiler not found in standard install locations."
}

$scriptPath = Join-Path $RepoRoot "scripts\\AI-Music-Studio.iss"
if (-not (Test-Path $scriptPath)) {
    throw "Installer script not found: $scriptPath"
}

& $isccPath `
    "/DMyAppVersion=$ReleaseVersion" `
    "/DSourceRoot=$SourceDir" `
    "/DOutputRoot=$OutputDir" `
    $scriptPath

if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup build failed."
}

$installerPath = Join-Path $OutputDir ("Mutagen-{0}-setup.exe" -f $ReleaseVersion)
if (-not (Test-Path $installerPath)) {
    throw "Expected installer not found: $installerPath"
}

Write-Output $installerPath
