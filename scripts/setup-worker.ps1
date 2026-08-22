<#
.SYNOPSIS
    Builds the worker runtime: a Python virtual environment and a GPL FFmpeg. prd.md §14.4, D-12.

.DESCRIPTION
    This script is the source of truth for the contents of .venv\ and tools\, both of which are
    git-ignored. If it does not install something, that something is not part of the environment.

    Written for Windows PowerShell 5.1 (see check-environment.ps1 for the dialect rules).
    Saved as UTF-8 with BOM; console output stays ASCII because PS 5.1 writes to a CP949 console
    here and mangles anything else.

.PARAMETER FfmpegVariant
    'gpl' (default) pulls a build with libx264 and libx265, which the Quality encoder profile needs
    (D-12). 'lgpl' pulls a build without them: NVENC only, and no x265. Note that a GPL FFmpeg makes
    the application GPL if it is ever distributed -- see prd.md §2.4 and docs/DECISIONS.md D-11.

.PARAMETER SkipFfmpeg
    Only build the virtual environment.

.PARAMETER Force
    Recreate the virtual environment from scratch.

.EXAMPLE
    .\scripts\setup-worker.ps1
#>
[CmdletBinding()]
param(
    [ValidateSet('gpl', 'lgpl')]
    [string] $FfmpegVariant = 'gpl',

    [switch] $SkipFfmpeg,
    [switch] $Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $repoRoot '.venv'
$venvPython = Join-Path $venvPath 'Scripts\python.exe'
$requirements = Join-Path $repoRoot 'worker\requirements.lock'
$toolsPath = Join-Path $repoRoot 'tools'
$ffmpegPath = Join-Path $toolsPath 'ffmpeg'

function Write-Step {
    param([Parameter(Mandatory = $true)][string] $Message)
    Write-Host ''
    Write-Host ('==> {0}' -f $Message) -ForegroundColor Cyan
}

function Find-HostPython {
    <#
        Returns the interpreter to build the venv from. The venv itself is what the worker runs,
        so this only needs to exist once.
    #>
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe')
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }

    $launcher = @(Get-Command -Name 'py' -ErrorAction SilentlyContinue)
    if ($launcher.Count -gt 0) {
        $resolved = @(& py -3.12 -c "import sys; print(sys.executable)" 2>$null)
        if ($resolved.Count -gt 0 -and (Test-Path $resolved[0])) { return $resolved[0] }
    }

    $onPath = @(Get-Command -Name 'python' -ErrorAction SilentlyContinue)
    if ($onPath.Count -gt 0) { return $onPath[0].Source }

    return $null
}

# --- Python virtual environment ------------------------------------------------------------------
Write-Step 'Python virtual environment'

if ($Force -and (Test-Path $venvPath)) {
    Write-Host 'Removing the existing environment (-Force).'
    Remove-Item -Recurse -Force $venvPath
}

if (-not (Test-Path $venvPython)) {
    $hostPython = Find-HostPython

    if (-not $hostPython) {
        Write-Host 'No Python 3.12 interpreter found.' -ForegroundColor Red
        Write-Host 'Install one, then re-run this script:' -ForegroundColor Yellow
        Write-Host '    winget install -e --id Python.Python.3.12 --scope user' -ForegroundColor Yellow
        exit 1
    }

    Write-Host ('Creating .venv from {0}' -f $hostPython)
    & $hostPython -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
else {
    Write-Host '.venv already exists (pass -Force to recreate).'
}

Write-Host 'Installing pinned dependencies from worker\requirements.lock'
& $venvPython -m pip install --upgrade pip --quiet
& $venvPython -m pip install -r $requirements --quiet
if ($LASTEXITCODE -ne 0) { throw 'dependency installation failed' }

$pythonVersion = (& $venvPython --version) -join ''
Write-Host ('OK: {0}' -f $pythonVersion) -ForegroundColor Green

# --- FFmpeg --------------------------------------------------------------------------------------
if ($SkipFfmpeg) {
    Write-Step 'FFmpeg: skipped (-SkipFfmpeg)'
}
else {
    Write-Step ('FFmpeg ({0} build)' -f $FfmpegVariant)

    $needsFfmpeg = $true
    $ffmpegExe = Join-Path $ffmpegPath 'bin\ffmpeg.exe'

    if ((-not $Force) -and (Test-Path $ffmpegExe)) {
        $banner = (@(& $ffmpegExe -hide_banner -version 2>&1)) -join ' '
        if ($FfmpegVariant -eq 'lgpl' -or $banner -match 'enable-libx265') {
            Write-Host 'FFmpeg already present and satisfies the requested variant.'
            $needsFfmpeg = $false
        }
        else {
            Write-Host 'FFmpeg present but without libx265; replacing it.' -ForegroundColor Yellow
        }
    }

    if ($needsFfmpeg) {
        # BtbN publishes reproducible Windows builds of upstream FFmpeg. The 'gpl' variant is the
        # one carrying libx264/libx265; 'lgpl' omits them (D-12).
        $archiveName = 'ffmpeg-master-latest-win64-{0}.zip' -f $FfmpegVariant
        $url = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/{0}' -f $archiveName
        $archivePath = Join-Path $toolsPath $archiveName
        $stagingPath = Join-Path $toolsPath '_ffmpeg_staging'

        if (-not (Test-Path $toolsPath)) { New-Item -ItemType Directory -Path $toolsPath | Out-Null }

        Write-Host ('Downloading {0}' -f $url)
        $previousProgress = $ProgressPreference
        $ProgressPreference = 'SilentlyContinue'   # the progress bar makes this ~10x slower
        try {
            Invoke-WebRequest -Uri $url -OutFile $archivePath -UseBasicParsing
        }
        finally {
            $ProgressPreference = $previousProgress
        }

        $sizeMb = [math]::Round((Get-Item $archivePath).Length / 1MB, 1)
        Write-Host ('Downloaded {0} MB; extracting' -f $sizeMb)

        if (Test-Path $stagingPath) { Remove-Item -Recurse -Force $stagingPath }
        Expand-Archive -Path $archivePath -DestinationPath $stagingPath -Force

        $extracted = @(Get-ChildItem -Path $stagingPath -Directory)
        if ($extracted.Count -ne 1) { throw ('unexpected archive layout: {0} top-level directories' -f $extracted.Count) }

        if (Test-Path $ffmpegPath) { Remove-Item -Recurse -Force $ffmpegPath }
        Move-Item -Path $extracted[0].FullName -Destination $ffmpegPath

        Remove-Item -Recurse -Force $stagingPath
        Remove-Item -Force $archivePath
    }

    $banner = (@(& $ffmpegExe -hide_banner -version 2>&1)) -join ' '

    if ($FfmpegVariant -eq 'gpl' -and $banner -notmatch 'enable-libx265') {
        throw 'the downloaded FFmpeg has no libx265; the Quality encoder profile (D-12) needs it'
    }

    $encoders = @()
    if ($banner -match 'enable-libx265') { $encoders += 'x265' }
    if ($banner -match 'enable-libx264') { $encoders += 'x264' }
    if ($banner -match 'enable-ffnvcodec') { $encoders += 'nvenc/nvdec' }

    Write-Host ('OK: FFmpeg with {0}' -f ($encoders -join ', ')) -ForegroundColor Green
}

# --- done ----------------------------------------------------------------------------------------
Write-Step 'Done'
Write-Host 'Verify with:'
Write-Host '    .\scripts\check-environment.ps1'
Write-Host '    .\.venv\Scripts\python.exe -m pytest'
Write-Host ''
Write-Host 'Training (training\, not part of the shipped engine) needs torch as well:'
Write-Host '    .\.venv\Scripts\python.exe -m pip install -r training\requirements.lock --index-url https://download.pytorch.org/whl/cu124'
Write-Host ''
