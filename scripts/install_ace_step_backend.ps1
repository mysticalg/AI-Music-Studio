param(
    [string]$InstallRoot = "",
    [string]$SettingsRoot = "",
    [string]$RepoZipUrl = "https://codeload.github.com/ace-step/ACE-Step-1.5/zip/refs/heads/main"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message"
}

function Resolve-UvExecutable {
    $command = Get-Command uv -ErrorAction SilentlyContinue
    if ($command -and $command.Source) {
        return $command.Source
    }

    $candidates = @()
    if ($env:USERPROFILE) {
        $candidates += (Join-Path $env:USERPROFILE ".local\bin\uv.exe")
    }
    if ($env:LOCALAPPDATA) {
        $candidates += (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\uv.exe")
    }

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }

    return $null
}

function Invoke-CheckedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [string]$WorkingDirectory = ""
    )

    $displayCommand = @($FilePath) + $ArgumentList
    Write-Host ("> " + ($displayCommand -join " "))

    $startProcessArgs = @{
        FilePath     = $FilePath
        ArgumentList = $ArgumentList
        Wait         = $true
        PassThru     = $true
        NoNewWindow  = $true
    }

    if ($WorkingDirectory) {
        $startProcessArgs["WorkingDirectory"] = $WorkingDirectory
    }

    $process = Start-Process @startProcessArgs
    if ($process.ExitCode -ne 0) {
        throw "Command failed with exit code $($process.ExitCode): $FilePath"
    }
}

function Ensure-Directory {
    param([Parameter(Mandatory = $true)][string]$Path)
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Ensure-AceStepRepoInstalled {
    param(
        [Parameter(Mandatory = $true)][string]$TargetRoot,
        [Parameter(Mandatory = $true)][string]$ZipUrl
    )

    $launchScript = Join-Path $TargetRoot "start_api_server.bat"
    if (Test-Path $launchScript) {
        Write-Step "ACE-Step repository already present at $TargetRoot"
        return
    }

    Write-Step "Downloading ACE-Step repository"
    $parentRoot = Split-Path -Parent $TargetRoot
    Ensure-Directory -Path $parentRoot

    $tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("mutagen-acestep-" + [guid]::NewGuid().ToString("N"))
    Ensure-Directory -Path $tempRoot

    try {
        $zipPath = Join-Path $tempRoot "ace-step.zip"
        $extractRoot = Join-Path $tempRoot "extract"
        Invoke-WebRequest -Uri $ZipUrl -OutFile $zipPath -UseBasicParsing
        Expand-Archive -Path $zipPath -DestinationPath $extractRoot -Force

        $sourceDir = Get-ChildItem -Path $extractRoot -Directory | Where-Object {
            $_.Name -like "ACE-Step-1.5*"
        } | Select-Object -First 1

        if (-not $sourceDir) {
            throw "Downloaded ACE-Step archive did not contain the expected repository folder."
        }

        if (Test-Path $TargetRoot) {
            Remove-Item -LiteralPath $TargetRoot -Recurse -Force
        }

        Move-Item -LiteralPath $sourceDir.FullName -Destination $TargetRoot
    }
    finally {
        if (Test-Path $tempRoot) {
            Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

function Ensure-UvInstalled {
    param([Parameter(Mandatory = $true)][string]$AceStepRoot)

    $uvPath = Resolve-UvExecutable
    if ($uvPath) {
        Write-Step "Using uv at $uvPath"
        return $uvPath
    }

    $installerPath = Join-Path $AceStepRoot "install_uv.bat"
    if (-not (Test-Path $installerPath)) {
        throw "ACE-Step install_uv.bat was not found at $installerPath"
    }

    Write-Step "Installing uv"
    Invoke-CheckedProcess -FilePath "cmd.exe" -ArgumentList @("/c", $installerPath, "--silent")

    $uvPath = Resolve-UvExecutable
    if (-not $uvPath) {
        throw "uv installation completed but Mutagen could not find uv.exe afterwards."
    }

    return $uvPath
}

function Ensure-AceStepEnvironment {
    param(
        [Parameter(Mandatory = $true)][string]$AceStepRoot,
        [Parameter(Mandatory = $true)][string]$UvPath
    )

    $venvRoot = Join-Path $AceStepRoot ".venv"
    if (Test-Path $venvRoot) {
        Write-Step "ACE-Step Python environment already present"
        return
    }

    Write-Step "Creating ACE-Step Python environment"
    Invoke-CheckedProcess -FilePath $UvPath -ArgumentList @("sync") -WorkingDirectory $AceStepRoot
}

function Write-MutagenLauncher {
    param(
        [Parameter(Mandatory = $true)][string]$AceStepRoot,
        [Parameter(Mandatory = $true)][string]$UvPath
    )

    $launcherPath = Join-Path $AceStepRoot "mutagen-start-api-server.bat"
    $escapedUvPath = $UvPath.Replace('"', '""')
    $launcherContent = @"
@echo off
setlocal
cd /d "%~dp0"
if not defined APPDATA exit /b 1
set "MUTAGEN_LOG_DIR=%APPDATA%\Mutagen\logs"
if not exist "%MUTAGEN_LOG_DIR%" mkdir "%MUTAGEN_LOG_DIR%"
set "MUTAGEN_LOG_FILE=%MUTAGEN_LOG_DIR%\ace-step-server.log"
if not exist ".venv" (
    call "$escapedUvPath" sync >> "%MUTAGEN_LOG_FILE%" 2>&1
    if errorlevel 1 exit /b %errorlevel%
)
call "$escapedUvPath" run --no-sync acestep-api --host 127.0.0.1 --port 8001 > "%MUTAGEN_LOG_FILE%" 2>&1
"@

    Set-Content -Path $launcherPath -Value $launcherContent -Encoding Ascii
}

function Write-MutagenAceStepSettings {
    param(
        [Parameter(Mandatory = $true)][string]$AceStepRoot,
        [Parameter(Mandatory = $true)][string]$ConfigRoot
    )

    Ensure-Directory -Path $ConfigRoot
    $settingsPath = Join-Path $ConfigRoot "ace_step_settings.json"

    $settings = @{}
    if (Test-Path $settingsPath) {
        try {
            $existing = Get-Content -Path $settingsPath -Raw | ConvertFrom-Json
            if ($existing) {
                foreach ($property in $existing.PSObject.Properties) {
                    $settings[$property.Name] = $property.Value
                }
            }
        }
        catch {
            Write-Host "Existing ACE-Step settings were invalid JSON. Overwriting with a clean configuration."
        }
    }

    $settings["base_url"] = "http://127.0.0.1:8001"
    $settings["install_directory"] = $AceStepRoot
    $settings["default_model"] = [string]($settings["default_model"])
    $settings["default_audio_format"] = if ($settings.ContainsKey("default_audio_format") -and $settings["default_audio_format"]) {
        [string]$settings["default_audio_format"]
    } else {
        "wav"
    }
    $settings["auto_start_server"] = $true
    $settings["startup_timeout_seconds"] = if ($settings.ContainsKey("startup_timeout_seconds") -and $settings["startup_timeout_seconds"]) {
        [int]$settings["startup_timeout_seconds"]
    } else {
        120
    }
    $settings["job_timeout_seconds"] = if ($settings.ContainsKey("job_timeout_seconds") -and $settings["job_timeout_seconds"]) {
        [int]$settings["job_timeout_seconds"]
    } else {
        1800
    }

    $settings | ConvertTo-Json -Depth 8 | Set-Content -Path $settingsPath -Encoding UTF8
}

if (-not $InstallRoot) {
    $localAppData = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { [Environment]::GetFolderPath("LocalApplicationData") }
    $InstallRoot = Join-Path $localAppData "Mutagen\ACE-Step-1.5"
}

if (-not $SettingsRoot) {
    $appData = if ($env:APPDATA) { $env:APPDATA } else { [Environment]::GetFolderPath("ApplicationData") }
    $SettingsRoot = Join-Path $appData "Mutagen"
}

$InstallRoot = [System.IO.Path]::GetFullPath($InstallRoot)
$SettingsRoot = [System.IO.Path]::GetFullPath($SettingsRoot)

Write-Step "Preparing ACE-Step for Mutagen"
Write-Host "Install root : $InstallRoot"
Write-Host "Settings root: $SettingsRoot"

Ensure-AceStepRepoInstalled -TargetRoot $InstallRoot -ZipUrl $RepoZipUrl
$uvPath = Ensure-UvInstalled -AceStepRoot $InstallRoot
Ensure-AceStepEnvironment -AceStepRoot $InstallRoot -UvPath $uvPath
Write-MutagenLauncher -AceStepRoot $InstallRoot -UvPath $uvPath
Write-MutagenAceStepSettings -AceStepRoot $InstallRoot -ConfigRoot $SettingsRoot

Write-Step "ACE-Step is ready for Mutagen"
Write-Host "Mutagen will use the local backend at http://127.0.0.1:8001"
