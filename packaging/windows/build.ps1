[CmdletBinding()]
param(
    [string]$BriteKitPath = "",
    [ValidateRange(1, 64)]
    [int]$Jobs = 2,
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
$NuitkaReport = Join-Path $BuildRoot "nuitka-report.xml"

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

& $Python (Join-Path $PackagingRoot "validate_build_environment.py")
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
    --include-qt-plugins=imageformats,iconengines `
    --low-memory `
    --jobs=$Jobs `
    --lto=no `
    --no-deployment-flag=excluded-module-usage `
    --module-parameter=torch-disable-jit=yes `
    --module-parameter=numba-disable-jit=yes `
    --nofollow-import-to=faiss `
    --nofollow-import-to=pyinaturalist `
    --nofollow-import-to=lightning `
    --nofollow-import-to=torchmetrics `
    --nofollow-import-to=britekit.commands `
    --nofollow-import-to=britekit.core.augmentation `
    --nofollow-import-to=britekit.core.data_module `
    --nofollow-import-to=britekit.core.dataset `
    --nofollow-import-to=britekit.core.pickler `
    --nofollow-import-to=britekit.core.reextractor `
    --nofollow-import-to=britekit.core.trainer `
    --nofollow-import-to=britekit.core.tuner `
    --nofollow-import-to=britekit.testing `
    --nofollow-import-to=britekit.training_db `
    --nofollow-import-to=sklearn `
    --nofollow-import-to=timm.optim `
    --include-package=hawkears.heuristics.canada `
    --include-package=scipy._external.array_api_compat `
    --include-package=librosa.util.example_data `
    --include-package-data=librosa.util.example_data `
    --include-package-data=hawkears `
    --windows-console-mode=disable `
    --windows-icon-from-ico=$Icon `
    --output-dir=$BuildRoot `
    --report=$NuitkaReport `
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

if (Select-String `
    -Path $NuitkaReport `
    -Pattern '<module name="(lightning|torchmetrics)(\.|")' `
    -Quiet) {
    throw "Nuitka included Lightning or TorchMetrics in the inference bundle."
}

$Executable = Join-Path $BuildRoot "HawkEars.dist/HawkEars.exe"
if (-not (Test-Path $Executable)) {
    throw "Nuitka did not create the expected executable: $Executable"
}

$SmokeLog = Join-Path $BuildRoot "HawkEars.dist/hawkears-packaging-smoke-test.log"
if (Test-Path $SmokeLog) {
    Remove-Item $SmokeLog
}
$SmokeTest = Start-Process `
    -FilePath $Executable `
    -ArgumentList "--packaging-smoke-test" `
    -Wait `
    -PassThru `
    -WindowStyle Hidden
if ($SmokeTest.ExitCode -ne 0) {
    if (Test-Path $SmokeLog) {
        Write-Host (Get-Content $SmokeLog -Raw)
    }
    throw "Packaged HawkEars smoke test failed with exit code $($SmokeTest.ExitCode)."
}

if (-not $SkipInstaller) {
    & (Join-Path $PackagingRoot "build-installer.ps1") -AppVersion $AppVersion
}
