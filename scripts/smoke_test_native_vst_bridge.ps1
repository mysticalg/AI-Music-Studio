param(
    [string]$PluginPath = "$env:LOCALAPPDATA\Programs\Common\VST3\Dexed.vst3",
    [int]$Port = 47653
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repoRoot "build\native-vst3host\AIMusicStudioVSTHost_artefacts\Release\AI Music Studio VST Host.exe"
$client = Join-Path $repoRoot "scripts\native_vst_host_client.py"

if (-not (Test-Path $exe)) {
    throw "Host executable not found at $exe"
}

if (-not (Test-Path $PluginPath)) {
    throw "Plugin path not found: $PluginPath"
}

$process = $null

try {
    $process = Start-Process -FilePath $exe -ArgumentList @("--plugin", $PluginPath, "--port", "$Port") -PassThru
    Start-Sleep -Seconds 3

    Write-Host "STATUS" -ForegroundColor Cyan
    py -3.14 $client --port $Port --command status

    Write-Host "NOTE ON" -ForegroundColor Cyan
    py -3.14 $client --port $Port --command note_on --note 60 --velocity 0.8
    Start-Sleep -Milliseconds 250

    Write-Host "NOTE OFF" -ForegroundColor Cyan
    py -3.14 $client --port $Port --command note_off --note 60

    Write-Host "OPEN EDITOR" -ForegroundColor Cyan
    py -3.14 $client --port $Port --command open_editor
    Start-Sleep -Seconds 2

    Write-Host "CLOSE EDITOR" -ForegroundColor Cyan
    py -3.14 $client --port $Port --command close_editor

    Write-Host "UNLOAD" -ForegroundColor Cyan
    py -3.14 $client --port $Port --command unload
}
finally {
    if ($process -and (Get-Process -Id $process.Id -ErrorAction SilentlyContinue)) {
        Stop-Process -Id $process.Id -Force
    }
}
