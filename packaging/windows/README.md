# Windows native build

HawkEars is built on Windows because Nuitka standalone applications are
platform-specific. The build uses 64-bit Python 3.12, PyTorch 2.8.0 with its
CUDA 12.6 runtime, Nuitka 4.1, and Inno Setup 6.

## Prerequisites

- 64-bit Python 3.12, available through the Windows `py` launcher
- Inno Setup 6
- Git checkout of HawkEars
- Optional adjacent BriteKit checkout when building an unreleased BriteKit

No CUDA Toolkit is required on the build computer. The script installs the
official PyTorch CUDA 12.6 wheels and verifies the selected Torch runtime before
compilation.

## Build

From PowerShell at the repository root:

```powershell
.\packaging\windows\build.ps1
```

The build uses two parallel C compilation jobs by default. If that exceeds the
available memory, retry with one job:

```powershell
.\packaging\windows\build.ps1 -Jobs 1
```

For an unreleased local BriteKit:

```powershell
.\packaging\windows\build.ps1 -BriteKitPath ..\BriteKit
```

The standalone directory is written to
`packaging\windows\build\HawkEars.dist`. The installer is written to
`packaging\windows\build\installer`. The exact resolved environment is captured
as `build\requirements-windows-cu126.txt`; after the first validated build, that
file should be reviewed and promoted to the release build lock.

Use `-SkipInstaller` to stop after the standalone build. To compile only the
installer:

```powershell
.\packaging\windows\build-installer.ps1 -AppVersion 2.3.0b1
```

## Validation order

1. Run `HawkEars.exe` directly from `HawkEars.dist`.
2. Complete first-run setup in a new data directory.
3. Analyze a short recording with CUDA.
4. Temporarily force CPU use and repeat the analysis.
5. Build and install with Inno Setup.
6. Verify Start menu and optional desktop shortcuts.
7. Double-click a `.hawkears` project.
8. Upgrade over the same AppId and confirm projects and model data remain.
9. Uninstall and confirm user projects and downloaded models remain.

The application and installer must be Authenticode-signed before a public
release. Signing commands are intentionally not embedded until the certificate
and secret-storage approach are selected.
