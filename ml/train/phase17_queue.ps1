# Phase 17 Track B queue -- ONE GPU cell at a time (concurrency level 1 for every cell; 4 GB card).
#
#   powershell -ExecutionPolicy Bypass -File ml\train\phase17_queue.ps1 -Stage B1 -Deadline "2026-10-02T16:33:00" `
#       -Cells "ml\train\phase17_b1.py neural --seed 7;ml\train\phase17_b1.py neural --seed 17;..."
#
# Stage cap: a cell is NOT started if (now + the slowest finished cell's wall time in this stage, or 35 min before
# any has finished) would pass -Deadline. The remaining cells are written to <stage>.STOPPED.json with the reason.
# Resumable: a finished cell leaves <log>.done and is skipped on the next run.
param([Parameter(Mandatory = $true)][string]$Stage, [Parameter(Mandatory = $true)][string]$Deadline,
      [Parameter(Mandatory = $true)][string[]]$Cells)
$ErrorActionPreference = "Continue"
# `powershell -File` hands a quoted list over as ONE string: accept ';'-separated cells as well
$Cells = @($Cells | ForEach-Object { $_ -split ';' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
Set-Location (Join-Path $PSScriptRoot "..\..")
$LOG = "ml\artifacts\phase17\logs"
New-Item -ItemType Directory -Force $LOG | Out-Null
$PY = "venv\Scripts\python.exe"
$env:HADES_DEVICE = "cuda"; $env:HADES_PANEL_HOST = "1"; $env:HADES_TCN_CHUNK = "4096"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$dl = [datetime]::Parse($Deadline)
$est = [timespan]::FromMinutes(35); $done = @(); $stopped = @()
foreach ($c in $Cells) {
    $name = ($c -replace '^ml\\train\\', '' -replace '\.py', '' -replace '[^A-Za-z0-9_.-]+', '_').Trim('_')
    $f = Join-Path $LOG "$Stage`_$name.log"
    if (Test-Path "$f.done") { Write-Host "[skip] $c"; continue }
    if ((Get-Date) + $est -gt $dl) { $stopped += $c; Write-Host "[STOP] $c -- would pass the $Stage cap ($Deadline)"; continue }
    Write-Host "=== START $Stage  $c  $(Get-Date -Format HH:mm:ss)"
    $t = Get-Date
    $p = Start-Process -FilePath $PY -ArgumentList ("-u " + $c) -RedirectStandardOutput $f -RedirectStandardError "$f.err" -NoNewWindow -PassThru -Wait
    $wall = (Get-Date) - $t
    if ($p.ExitCode -eq 0) { New-Item -ItemType File -Force "$f.done" | Out-Null; if ($wall -gt $est -or $done.Count -eq 0) { $est = $wall }; $done += $c }
    $last = (Get-Content $f -ErrorAction SilentlyContinue | Where-Object { $_ -match '\S' } | Select-Object -Last 1)
    Write-Host "=== END   $Stage  $c  $(Get-Date -Format HH:mm:ss)  exit $($p.ExitCode)  wall $([int]$wall.TotalMinutes) min  $last"
}
if ($stopped.Count) {
    @{ stage = $Stage; deadline = $Deadline; stopped_at = (Get-Date -Format s); not_run = $stopped; finished = $done;
       reason = "stage cap: the next cell's estimated wall time would pass the deadline" } |
        ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $LOG "$Stage.STOPPED.json")
}
Write-Host "=== STAGE $Stage COMPLETE $(Get-Date -Format HH:mm:ss)  finished $($done.Count)  stopped $($stopped.Count)"
