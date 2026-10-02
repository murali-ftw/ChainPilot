# Phase 17 Track B chain: wait for the B1 queue, then B2 -> B3 -> B4, one GPU cell at a time throughout.
# Each stage: a 1-epoch smoke run into a scratch root (no real bundle path is touched); if it fails, the stage is
# skipped with a STOPPED marker. Otherwise phase17_queue.ps1 runs the stage's cells under its cap, which starts when
# the stage starts.
#   powershell -ExecutionPolicy Bypass -File ml\train\phase17_chain.ps1 -WaitPid <B1 queue pid>
param([int]$WaitPid = 0)
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..\..")
$LOG = "ml\artifacts\phase17\logs"; New-Item -ItemType Directory -Force $LOG | Out-Null
$SMOKE = "ml\artifacts\phase17\smoke"
$PY = "venv\Scripts\python.exe"
$env:HADES_DEVICE = "cuda"; $env:HADES_PANEL_HOST = "1"; $env:HADES_TCN_CHUNK = "4096"; $env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
if ($WaitPid) { while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep 30 } }
Write-Host "=== CHAIN start $(Get-Date -Format HH:mm:ss)"

$stages = @(
  @{ name = "B2"; cap = 240; smoke = "ml\train\phase17_b2.py train --arm b2c --seed 7 --max-epochs 1 --bundle-root $SMOKE";
     cells = @(foreach ($s in 7, 17, 27, 37, 47) { "ml\train\phase17_b2.py train --arm b2b --seed $s"; "ml\train\phase17_b2.py train --arm b2c --seed $s" }) },
  @{ name = "B3"; cap = 180; smoke = "ml\train\loop.py train --config ml\configs\shipped.json --task fill_rate --world v8 --arch none --depth 0 --lr 1.25e-4 --fill-head band5 --seed 7 --max-epochs 1 --bundle-root $SMOKE";
     cells = @(foreach ($s in 7, 17, 27, 37, 47) { "ml\train\loop.py train --config ml\configs\shipped.json --task fill_rate --world v8 --arch none --depth 0 --lr 1.25e-4 --fill-head band5 --seed $s" }) },
  @{ name = "B4"; cap = 180; smoke = "ml\train\phase17_b4.py train --seed 7 --max-epochs 1 --bundle-root $SMOKE";
     cells = @(foreach ($s in 7, 17, 27, 37, 47) { "ml\train\phase17_b4.py train --seed $s" }) }
)
foreach ($st in $stages) {
    $start = Get-Date; $deadline = $start.AddMinutes($st.cap).ToString("s")
    $sf = Join-Path $LOG "$($st.name)_smoke.log"
    Write-Host "=== $($st.name) smoke  $(Get-Date -Format HH:mm:ss)"
    $p = Start-Process -FilePath $PY -ArgumentList ("-u " + $st.smoke) -RedirectStandardOutput $sf -RedirectStandardError "$sf.err" -NoNewWindow -PassThru -Wait
    if ($p.ExitCode -ne 0) {
        Write-Host "=== $($st.name) SMOKE FAILED (exit $($p.ExitCode)) -- stage skipped, see $sf.err"
        @{ stage = $st.name; stopped_at = (Get-Date -Format s); reason = "1-epoch smoke run failed (exit $($p.ExitCode))"; not_run = $st.cells } |
            ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $LOG "$($st.name).STOPPED.json")
        continue
    }
    Write-Host "=== $($st.name) smoke ok; cells under a $($st.cap)-min cap, deadline $deadline"
    & powershell -NoProfile -ExecutionPolicy Bypass -File ml\train\phase17_queue.ps1 -Stage $st.name -Deadline $deadline -Cells ($st.cells -join ';')
}
Write-Host "=== CHAIN COMPLETE $(Get-Date -Format HH:mm:ss)"
