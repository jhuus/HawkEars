[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$AppVersion,
    [string]$IsccPath = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $IsccPath) {
    $Candidates = @(
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6/ISCC.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs/Inno Setup 6/ISCC.exe")
    )
    $IsccPath = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $IsccPath -or -not (Test-Path $IsccPath)) {
    throw "Inno Setup 6 compiler not found. Pass its location with -IsccPath."
}

$Script = Join-Path $PSScriptRoot "hawkears.iss"
& $IsccPath "/DMyAppVersion=$AppVersion" $Script
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup failed with exit code $LASTEXITCODE."
}
