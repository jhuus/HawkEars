[CmdletBinding()]
param(
    [string]$BriteKitPath = "",
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$PackagingRoot = $PSScriptRoot
$RepositoryRoot = (Resolve-Path (Join-Path $PackagingRoot "../..")).Path
$VirtualEnvironment = Join-Path $PackagingRoot ".venv"
$Python = Join-Path $VirtualEnvironment "Scripts/python.exe"
$BuildRoot = Join-Path $PackagingRoot "build"
$Launcher = Join-Path $PackagingRoot "HawkEars.py"
$Icon = Join-Path $PackagingRoot "assets/hawkears.ico"

function Assert-LastExitCode([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

if (-not (Test-Path $Python)) {
    py -3.12 -m venv $VirtualEnvironment
    Assert-LastExitCode "Creating the Python environment"
}

& $Python -m pip install --upgrade pip
Assert-LastExitCode "Upgrading pip"
& $Python -m pip install `
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 `
    --index-url https://download.pytorch.org/whl/cu126
Assert-LastExitCode "Installing CUDA 12.6 PyTorch"

if ($BriteKitPath) {
    & $Python -m pip install $BriteKitPath
    Assert-LastExitCode "Installing BriteKit"
}

& $Python -m pip install $RepositoryRoot
Assert-LastExitCode "Installing HawkEars"
& $Python -m pip install -r (Join-Path $PackagingRoot "requirements-build.txt")
Assert-LastExitCode "Installing Windows build tools"

& $Python -c @"
import torch
assert torch.__version__.startswith("2.8.0"), torch.__version__
assert torch.version.cuda == "12.6", torch.version.cuda
print(f"Building with torch {torch.__version__}, CUDA runtime {torch.version.cuda}")
"@
Assert-LastExitCode "Validating PyTorch"

$AppVersion = (& $Python -c "from hawkears import __version__; print(__version__)").Trim()
if ($AppVersion -notmatch "^(\d+)\.(\d+)\.(\d+)") {
    throw "Cannot convert HawkEars version '$AppVersion' to Windows version metadata."
}
$WindowsVersion = "$($Matches[1]).$($Matches[2]).$($Matches[3]).0"

New-Item -ItemType Directory -Force $BuildRoot | Out-Null
$EnvironmentLock = Join-Path $BuildRoot "requirements-windows-cu126.txt"
(& $Python -m pip freeze) | Set-Content -Encoding UTF8 $EnvironmentLock
Assert-LastExitCode "Recording the Windows build environment"

& $Python -m nuitka `
    --mode=standalone `
    --enable-plugin=pyside6 `
    --include-package-data=hawkears `
    --windows-console-mode=disable `
    --windows-icon-from-ico=$Icon `
    --output-dir=$BuildRoot `
    --output-filename=HawkEars.exe `
    --company-name=HawkEars `
    --product-name=HawkEars `
    --file-description="HawkEars Bioacoustic Classifier" `
    --file-version=$WindowsVersion `
    --product-version=$WindowsVersion `
    --copyright="Copyright (c) 2025-present Jan Huus" `
    --assume-yes-for-downloads `
    $Launcher
Assert-LastExitCode "Building HawkEars with Nuitka"

$Executable = Join-Path $BuildRoot "HawkEars.dist/HawkEars.exe"
if (-not (Test-Path $Executable)) {
    throw "Nuitka did not create the expected executable: $Executable"
}

& $Executable --packaging-smoke-test
Assert-LastExitCode "Running the packaged HawkEars smoke test"

if (-not $SkipInstaller) {
    & (Join-Path $PackagingRoot "build-installer.ps1") -AppVersion $AppVersion
}
