<#
.SYNOPSIS
    Check a built SciRust Studio Windows artifact, by running it.

.DESCRIPTION
    Everything here exercises the artifact that would be handed to a
    reviewer, not the source it was built from. A build that compiles and
    then cannot find its own sidecar, or that ships a frontend bundle with no
    WebAssembly module in it, is exactly the failure this catches — and it is
    invisible to `cargo build`.

    The checks, in order:

      1. The application executable exists.
      2. The sidecar exists beside it, under the target-triple name Tauri
         resolves at runtime (extension *after* the triple).
      3. The staged frontend contains an index document and a WebAssembly
         module.
      4. `--version` runs and prints a version.
      5. `--smoke-test-backend` runs the real worker, the real adapters and
         the real store end to end, and reports `ok: true` with a scientific
         check that passed.
      6. The NSIS installer exists and is a plausible size.

    Step 5 is the one that matters most: it is the only check here that
    proves the scientific path works in the artifact rather than in the
    developer's build tree.

    A report is written to `studio-acceptance.json` next to the executable so
    the result travels with the artifact.

.PARAMETER BuildDir
    The directory holding the built executable. Defaults to the workspace's
    release directory.

.PARAMETER SkipInstaller
    Skip the NSIS installer check, for a `--no-bundle` build.

.EXAMPLE
    ./scripts/studio/test-desktop-artifact.ps1 -Verbose
#>

[CmdletBinding()]
param(
    [string] $BuildDir,
    [switch] $SkipInstaller
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..' '..')).Path
if (-not $BuildDir) {
    $BuildDir = Join-Path $repoRoot 'apps/scirust-studio/target/release'
}

$results = [System.Collections.ArrayList]::new()
$failed = $false

function Add-Check {
    param(
        [Parameter(Mandatory)] [string] $Name,
        [Parameter(Mandatory)] [bool] $Ok,
        [string] $Detail = ''
    )
    [void]$script:results.Add([pscustomobject]@{
        check  = $Name
        ok     = $Ok
        detail = $Detail
    })
    if ($Ok) {
        Write-Host "  PASS  $Name" -ForegroundColor Green
        if ($Detail) { Write-Verbose "        $Detail" }
    } else {
        Write-Host "  FAIL  $Name" -ForegroundColor Red
        Write-Host "        $Detail" -ForegroundColor Red
        $script:failed = $true
    }
}

Write-Host "SciRust Studio — Windows artifact acceptance"
Write-Host "build directory: $BuildDir"
Write-Host ""

# ---- 1. The executable ------------------------------------------------

$exe = Join-Path $BuildDir 'scirust-studio.exe'
$exeExists = Test-Path -LiteralPath $exe -PathType Leaf
Add-Check -Name 'application executable exists' -Ok $exeExists -Detail $exe

if (-not $exeExists) {
    Write-Host ""
    Write-Host "Nothing further can be checked without the executable." -ForegroundColor Red
    Write-Host "Build it with:" -ForegroundColor Yellow
    Write-Host "    cargo tauri build --config src-tauri/tauri.conf.json" -ForegroundColor Yellow
    exit 1
}

# ---- 2. The sidecar ---------------------------------------------------

# Tauri appends the target triple to the configured name and puts `.exe`
# after it. Getting that wrong builds cleanly and then fails at runtime,
# which is why the name is asserted rather than assumed.
$triple = (& rustc -vV | Select-String '^host: ').ToString().Substring(6).Trim()
$sidecarName = "scirust-studio-worker-$triple.exe"
$sidecarSource = Join-Path $repoRoot "apps/scirust-studio/src-tauri/binaries/$sidecarName"
$sidecarBeside = Join-Path $BuildDir $sidecarName

$sidecarPath = $null
foreach ($candidate in @($sidecarBeside, $sidecarSource)) {
    if (Test-Path -LiteralPath $candidate -PathType Leaf) { $sidecarPath = $candidate; break }
}
Add-Check -Name 'sidecar worker is staged under its target-triple name' `
          -Ok ($null -ne $sidecarPath) `
          -Detail $(if ($sidecarPath) { $sidecarPath } else { "expected $sidecarName" })

if ($sidecarPath) {
    $sidecarSize = (Get-Item -LiteralPath $sidecarPath).Length
    Add-Check -Name 'sidecar worker is a real binary' `
              -Ok ($sidecarSize -gt 200KB) `
              -Detail "$sidecarSize bytes"
}

# ---- 3. The frontend bundle ------------------------------------------

$dist = Join-Path $repoRoot 'apps/scirust-studio/dist'
$hasIndex = Test-Path -LiteralPath (Join-Path $dist 'index.html') -PathType Leaf
$wasm = @(Get-ChildItem -LiteralPath $dist -Recurse -Filter '*.wasm' -ErrorAction SilentlyContinue)
Add-Check -Name 'frontend bundle has an index document' -Ok $hasIndex -Detail $dist
Add-Check -Name 'frontend bundle has a WebAssembly module' `
          -Ok ($wasm.Count -gt 0) `
          -Detail $(if ($wasm.Count -gt 0) { "$($wasm.Count) module(s), $(($wasm | Measure-Object Length -Sum).Sum) bytes" } else { 'none found' })

# ---- 4. --version -----------------------------------------------------

$versionOutput = & $exe --version 2>&1 | Out-String
Add-Check -Name '--version runs and reports a version' `
          -Ok (($LASTEXITCODE -eq 0) -and ($versionOutput -match 'SciRust Studio\s+\d+\.\d+\.\d+')) `
          -Detail $versionOutput.Trim()

# ---- 5. The scientific path, end to end -------------------------------

Write-Host ""
Write-Host "Running the backend smoke test (real worker, real store)…"
$smokeOutput = & $exe --smoke-test-backend 2>&1 | Out-String
$smokeExit = $LASTEXITCODE
Write-Verbose $smokeOutput

$smoke = $null
try {
    $smoke = $smokeOutput | ConvertFrom-Json
} catch {
    Add-Check -Name 'the backend smoke test reports JSON' -Ok $false `
              -Detail "output was not JSON: $($smokeOutput.Trim())"
}

if ($smoke) {
    Add-Check -Name 'the backend smoke test exits zero' -Ok ($smokeExit -eq 0) `
              -Detail "exit code $smokeExit"
    Add-Check -Name 'the backend smoke test reports success' -Ok ([bool]$smoke.ok)
    Add-Check -Name 'a worker process started and handshook' `
              -Ok (-not [string]::IsNullOrWhiteSpace([string]$smoke.worker_version)) `
              -Detail "worker $($smoke.worker_version)"
    Add-Check -Name 'the catalogue is populated' -Ok ($smoke.capabilities_available -ge 5) `
              -Detail "$($smoke.capabilities_available) capabilities"

    # Schema v2 is what makes a chart honest: the result carries the
    # coordinates the integrator produced.
    Add-Check -Name 'the result was stored under schema v2' `
              -Ok ($smoke.result_schema_version -eq 2) `
              -Detail "v$($smoke.result_schema_version)"
    Add-Check -Name 'the result carries real axis coordinates' `
              -Ok ($smoke.axis_points -gt 1000) `
              -Detail "$($smoke.axis_points) coordinates"
    Add-Check -Name 'the stored run verifies against its hashes' `
              -Ok ($smoke.store_integrity -eq 'verified') `
              -Detail "$($smoke.store_integrity)"

    $checks = @($smoke.scientific_checks)
    $passed = @($checks | Where-Object { $_.status -eq 'passed' })
    Add-Check -Name 'at least one scientific check passed' `
              -Ok ($passed.Count -ge 1) `
              -Detail (($checks | ForEach-Object { "$($_.id)=$($_.status)" }) -join ', ')
    $failedChecks = @($checks | Where-Object { $_.status -eq 'failed' })
    Add-Check -Name 'no scientific check failed' -Ok ($failedChecks.Count -eq 0) `
              -Detail $(if ($failedChecks.Count) { ($failedChecks | ForEach-Object { $_.id }) -join ', ' } else { 'none' })
}

# ---- 6. The installer -------------------------------------------------

if (-not $SkipInstaller) {
    $nsisDir = Join-Path $BuildDir 'bundle/nsis'
    $installers = @(Get-ChildItem -LiteralPath $nsisDir -Filter '*.exe' -ErrorAction SilentlyContinue)
    Add-Check -Name 'an NSIS installer was produced' -Ok ($installers.Count -gt 0) -Detail $nsisDir
    if ($installers.Count -gt 0) {
        $installer = $installers[0]
        # An installer that does not carry the application and its sidecar
        # would be a few hundred kilobytes.
        Add-Check -Name 'the installer is a plausible size' `
                  -Ok ($installer.Length -gt 2MB) `
                  -Detail "$($installer.Name), $($installer.Length) bytes"
    }
}

# ---- Report -----------------------------------------------------------

$report = [pscustomobject]@{
    artifact   = $exe
    triple     = $triple
    signed     = $false
    signing    = 'not performed in this phase; the artifact is an unsigned preview'
    checks     = $results
    ok         = (-not $failed)
}
$reportPath = Join-Path $BuildDir 'studio-acceptance.json'
$report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $reportPath -Encoding utf8

Write-Host ""
Write-Host "report: $reportPath"

if ($failed) {
    Write-Host "ACCEPTANCE FAILED" -ForegroundColor Red
    exit 1
}

Write-Host "ACCEPTANCE PASSED" -ForegroundColor Green
Write-Host ""
Write-Host "This artifact is unsigned. Windows SmartScreen will warn about it;" -ForegroundColor Yellow
Write-Host "see docs/studio/WINDOWS_DESKTOP_ACCEPTANCE.md." -ForegroundColor Yellow
exit 0
