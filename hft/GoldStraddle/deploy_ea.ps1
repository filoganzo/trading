# ---------------------------------------------------------------------------
#  GoldStraddle - deploy + compile the EA into the IC Markets EU terminal
# ---------------------------------------------------------------------------
#  MT5 compiles whatever sits in the terminal's own MQL5 sandbox, NOT the copy
#  in this repo. Editing the repo file alone leaves a stale .ex5 running - so
#  this script always copies first, then compiles, then reports.
#
#  Usage:  powershell -ExecutionPolicy Bypass -File deploy_ea.ps1
# ---------------------------------------------------------------------------

$ErrorActionPreference = 'Stop'

$repoDir     = Split-Path -Parent $MyInvocation.MyCommand.Path
$srcFile     = Join-Path $repoDir 'mt5\GoldStraddle.mq5'

$terminalId  = '<TERMINAL_ID>'   # IC Markets EU
$dataDir     = Join-Path $env:APPDATA "MetaQuotes\Terminal\$terminalId"
$expertsDir  = Join-Path $dataDir 'MQL5\Experts'
$dstFile     = Join-Path $expertsDir 'GoldStraddle.mq5'
$metaEditor  = 'C:\Program Files\MetaTrader 5 IC Markets EU\metaeditor64.exe'
$logFile     = Join-Path $env:TEMP 'GoldStraddle_compile.log'

Write-Host ''
Write-Host '=== GoldStraddle deploy ===' -ForegroundColor Cyan

if (-not (Test-Path $srcFile))    { throw "Source not found: $srcFile" }
if (-not (Test-Path $expertsDir)) { throw "Experts folder not found: $expertsDir" }

# --- 1. copy into the terminal sandbox -------------------------------------
Copy-Item -Path $srcFile -Destination $dstFile -Force
Write-Host "  copied  -> $dstFile" -ForegroundColor Green

# --- 2. compile -------------------------------------------------------------
if (-not (Test-Path $metaEditor)) {
    Write-Host "  MetaEditor not found, skipping compile." -ForegroundColor Yellow
    Write-Host "  Compile manually: open the file in MetaEditor and press F7."
    exit 0
}

if (Test-Path $logFile) { Remove-Item $logFile -Force }

Write-Host '  compiling...' -NoNewline
$proc = Start-Process -FilePath $metaEditor `
                      -ArgumentList @("/compile:`"$dstFile`"", "/log:`"$logFile`"") `
                      -Wait -PassThru -WindowStyle Hidden
Write-Host ''

# --- 3. report --------------------------------------------------------------
# MetaEditor writes its log as UTF-16LE and returns the error count as exit code.
if (Test-Path $logFile) {
    $log = Get-Content $logFile -Encoding Unicode | Where-Object { $_ -match '\S' }
    Write-Host ''
    Write-Host '--- compiler log ---' -ForegroundColor Cyan
    foreach ($line in $log) {
        if     ($line -match ' error ')   { Write-Host $line -ForegroundColor Red }
        elseif ($line -match ' warning ') { Write-Host $line -ForegroundColor Yellow }
        else                              { Write-Host $line -ForegroundColor DarkGray }
    }
    Write-Host '--------------------' -ForegroundColor Cyan
}

$ex5 = Join-Path $expertsDir 'GoldStraddle.ex5'
# MetaEditor can return a non-zero process exit code even when its compiler
# log reports a successful build (notably when launched from PowerShell).
# The compiler result line is authoritative; do not report a valid .ex5 as a
# failed deployment merely because the wrapper process returned 1.
$compileOk = $false
if ($log) {
    $compileOk = [bool]($log -match 'Result:\s+0 errors')
}
Write-Host ''
if ($compileOk -and (Test-Path $ex5)) {
    $stamp = (Get-Item $ex5).LastWriteTime
    Write-Host "  COMPILED OK  -> GoldStraddle.ex5  ($stamp)" -ForegroundColor Green
    Write-Host ''
    Write-Host '  Next: in MT5 press Ctrl+N (Navigator), right-click Expert Advisors ->'
    Write-Host '  Refresh, then drag GoldStraddle onto an XAUUSD chart.'
} else {
    Write-Host "  COMPILE FAILED (exit code $($proc.ExitCode)) - see log above." -ForegroundColor Red
    exit 1
}
