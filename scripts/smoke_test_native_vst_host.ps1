param(
    [int]$LaunchSeconds = 6,
    [switch]$OpenEditor
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repoRoot "build\native-vst3host\AIMusicStudioVSTHost_artefacts\Release\AI Music Studio VST Host.exe"

if (-not (Test-Path $exe)) {
    throw "Host executable not found at $exe"
}

$pluginRoots = @(
    "$env:LOCALAPPDATA\Programs\Common\VST3",
    "$env:ProgramFiles\Common Files\VST3",
    "${env:ProgramFiles(x86)}\Common Files\VST3"
) | Where-Object { $_ -and (Test-Path $_) }

$patterns = @("Dexed", "js80p", "YMulator", "Helm", "TAL-NoiseMaker")
$pluginMatches = foreach ($root in $pluginRoots) {
    Get-ChildItem $root -Filter *.vst3 -Recurse -ErrorAction SilentlyContinue |
        Where-Object {
            $name = $_.FullName
            @($patterns | Where-Object { $name -match $_ }).Count -gt 0
        }
}

$plugins = $pluginMatches | Sort-Object FullName -Unique
$plugins = $plugins | Where-Object { $_.FullName -notmatch '\\Contents\\x86_64-win\\' }

if (-not $plugins) {
    throw "No known smoke-test plugins were found."
}

$results = @()

foreach ($plugin in $plugins) {
    $args = @("--plugin", $plugin.FullName)
    if ($OpenEditor) {
        $args += "--open-editor"
    }

    Write-Host "Launching host with $($plugin.FullName)" -ForegroundColor Cyan
    $process = Start-Process -FilePath $exe -ArgumentList $args -PassThru
    Start-Sleep -Seconds $LaunchSeconds

    $alive = Get-Process -Id $process.Id -ErrorAction SilentlyContinue
    $results += [pscustomobject]@{
        Plugin = $plugin.FullName
        OpenEditor = [bool]$OpenEditor
        AliveAfterDelay = [bool]$alive
    }

    if ($alive) {
        Stop-Process -Id $alive.Id -Force
    }
}

$results | Format-Table -AutoSize

$failed = $results | Where-Object { -not $_.AliveAfterDelay }
if ($failed) {
    throw "One or more native host smoke runs failed."
}
