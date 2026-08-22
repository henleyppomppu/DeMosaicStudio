<#
.SYNOPSIS
    Reports whether this machine can build and run DeMosaic Studio. prd.md §4.5, AC-4.5.

.DESCRIPTION
    Every item prints PASS or FAIL with the exact command that fixes it. This is the first thing a
    new machine runs.

    Written for Windows PowerShell 5.1, which is what ships on the target machine:
      - no ternary operator, no null-coalescing, no pipeline chain operators
      - pipeline results are wrapped as @( ... ) in full, never as @($x) | Where-Object { }
      - this file is saved as UTF-8 *with BOM*; without it PS 5.1 reads it as ANSI (CP949 here)
        and multi-byte characters break parsing rather than execution. Guarded by
        T-SCRIPT-ENCODING-01.

.PARAMETER Quiet
    Suppress per-item output; set the exit code only.

.EXAMPLE
    .\scripts\check-environment.ps1
#>
[CmdletBinding()]
param(
    [switch] $Quiet
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:Results = @()

function Add-Check {
    param(
        [Parameter(Mandatory = $true)][string] $Name,
        [Parameter(Mandatory = $true)][bool]   $Ok,
        [string] $Detail = '',
        [string] $Remedy = '',
        [switch] $Optional
    )

    $script:Results += [pscustomobject]@{
        Name     = $Name
        Ok       = $Ok
        Detail   = $Detail
        Remedy   = $Remedy
        Optional = [bool] $Optional
    }
}

function Test-Command {
    param([Parameter(Mandatory = $true)][string] $Name)

    $found = @(Get-Command -Name $Name -ErrorAction SilentlyContinue)
    return ($found.Count -gt 0)
}

$repoRoot = Split-Path -Parent $PSScriptRoot

# --- repository ---------------------------------------------------------------------------------
Add-Check -Name 'Git repository' -Ok (Test-Path (Join-Path $repoRoot '.git')) `
    -Detail $repoRoot -Remedy 'git init'

# --- .NET ---------------------------------------------------------------------------------------
if (Test-Command 'dotnet') {
    $sdks = @(& dotnet --list-sdks)
    $has10 = @($sdks | Where-Object { $_ -like '10.*' })
    Add-Check -Name '.NET SDK 10' -Ok ($has10.Count -gt 0) `
        -Detail (@($sdks | Select-Object -Last 1) -join '') `
        -Remedy 'Install the .NET 10 SDK: https://dotnet.microsoft.com/download'
}
else {
    Add-Check -Name '.NET SDK 10' -Ok $false -Remedy 'Install the .NET 10 SDK'
}

# --- Python worker runtime (D-01, §14.4) --------------------------------------------------------
$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
Add-Check -Name 'Worker virtual environment' -Ok (Test-Path $venvPython) `
    -Detail $venvPython -Remedy '.\scripts\setup-worker.ps1'

if (Test-Path $venvPython) {
    $pyVersion = (& $venvPython --version) -join ''
    Add-Check -Name 'Python 3.12+' -Ok ($pyVersion -match 'Python 3\.(1[2-9]|[2-9][0-9])') `
        -Detail $pyVersion -Remedy 'Recreate the venv from a Python 3.12 interpreter'
}
else {
    Add-Check -Name 'Python 3.12+' -Ok $false -Remedy '.\scripts\setup-worker.ps1'
}

# --- FFmpeg (D-12: GPL build with x265) ---------------------------------------------------------
$ffmpeg = Join-Path $repoRoot 'tools\ffmpeg\bin\ffmpeg.exe'
if (-not (Test-Path $ffmpeg)) {
    $onPath = @(Get-Command -Name 'ffmpeg' -ErrorAction SilentlyContinue)
    if ($onPath.Count -gt 0) { $ffmpeg = $onPath[0].Source }
}

if (Test-Path $ffmpeg) {
    $banner = @(& $ffmpeg -hide_banner -version 2>&1)
    $bannerText = ($banner -join ' ')
    Add-Check -Name 'FFmpeg' -Ok $true -Detail (@($banner | Select-Object -First 1) -join '')
    Add-Check -Name 'FFmpeg x265 (D-12)' -Ok ($bannerText -match 'enable-libx265') `
        -Detail 'Quality profile needs a GPL build with libx265' `
        -Remedy '.\scripts\setup-worker.ps1 -FfmpegVariant gpl'
}
else {
    Add-Check -Name 'FFmpeg' -Ok $false -Remedy '.\scripts\setup-worker.ps1'
    Add-Check -Name 'FFmpeg x265 (D-12)' -Ok $false -Remedy '.\scripts\setup-worker.ps1 -FfmpegVariant gpl'
}

# --- GPU ----------------------------------------------------------------------------------------
if (Test-Command 'nvidia-smi') {
    $gpu = @(& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)
    Add-Check -Name 'NVIDIA GPU' -Ok ($gpu.Count -gt 0) -Detail (@($gpu | Select-Object -First 1) -join '')
}
else {
    Add-Check -Name 'NVIDIA GPU' -Ok $false `
        -Detail 'CPU only: expect 10x or worse (prd.md §5.17c)' `
        -Remedy 'Install an NVIDIA driver, or accept the CPU path' -Optional
}

# --- optional, only needed if the native engine is revived (§3.4, Phase 5) -----------------------
Add-Check -Name 'CMake (native engine only)' -Ok (Test-Command 'cmake') `
    -Detail 'Not needed under D-01' -Remedy 'winget install Kitware.CMake' -Optional

# --- report -------------------------------------------------------------------------------------
$required = @($script:Results | Where-Object { -not $_.Optional })
$failed = @($required | Where-Object { -not $_.Ok })

if (-not $Quiet) {
    Write-Host ''
    Write-Host 'DeMosaic Studio - environment check (prd.md section 4.5)'
    Write-Host '-------------------------------------------------'

    foreach ($result in $script:Results) {
        if ($result.Ok) {
            $status = 'PASS'
            $colour = 'Green'
        }
        elseif ($result.Optional) {
            $status = 'SKIP'
            $colour = 'Yellow'
        }
        else {
            $status = 'FAIL'
            $colour = 'Red'
        }

        Write-Host ('{0,-4} {1}' -f $status, $result.Name) -ForegroundColor $colour

        if ($result.Detail) { Write-Host ('       {0}' -f $result.Detail) -ForegroundColor DarkGray }
        if ((-not $result.Ok) -and $result.Remedy) {
            Write-Host ('       fix: {0}' -f $result.Remedy) -ForegroundColor DarkYellow
        }
    }

    Write-Host ''
    Write-Host ('{0} of {1} required checks passed.' -f ($required.Count - $failed.Count), $required.Count)
    Write-Host ''
}

if ($failed.Count -gt 0) { exit 1 }
exit 0
