# Phase 1 on v8 -- Windows/CUDA port of phase1_v8_queue.sh. Same cells, same order, same arguments.
#
# shipped.json is NOT modified: the world is passed on the command line and every other
# hyper-parameter comes from the config as it stands. No learning rate is retuned.
# fill_rate trains its NEURAL h0 (deviation 46: b5flat22 has no loader) -- see the .sh header.
#
# Memory mode for a 4 GB CUDA card (commit db27671): the panel is held in host RAM and the TCN is
# checkpointed in 4096-channel chunks, so nothing spills into shared memory. ONE cell at a time:
# two concurrent cells refill VRAM and run ~10x slower each (measured). TF32 stays off.
#
#   powershell -ExecutionPolicy Bypass -File ml\train\phase1_v8_queue.ps1
# Resumable: a finished cell leaves <log>.done and is skipped on the next run.
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..\..")
$LOG = if ($env:LOG) { $env:LOG } else { "ml\artifacts\phase1_v8_logs" }
New-Item -ItemType Directory -Force $LOG | Out-Null
$PY = "venv\Scripts\python.exe"
$CFG = "ml\configs\shipped.json"
$env:HADES_DEVICE = if ($env:HADES_DEVICE) { $env:HADES_DEVICE } else { "cuda" }
$env:HADES_PANEL_HOST = "1"
$env:HADES_TCN_CHUNK = "4096"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

function Run-Cells([string]$Task, [string]$Tag, [string[]]$CellArgs) {
    foreach ($s in 7, 17, 27) {
        $f = Join-Path $LOG "${Task}_${Tag}_s$s.log"
        if (Test-Path "$f.done") { Write-Host "[skip] $Task $Tag s$s"; continue }
        Write-Host "=== START $Task $Tag seed $s  $(Get-Date -Format HH:mm:ss)"
        $argv = @("-u", "ml\train\loop.py", "train", "--config", $CFG, "--task", $Task, "--world", "v8", "--seed", "$s") + $CellArgs
        $p = Start-Process -FilePath $PY -ArgumentList $argv -RedirectStandardOutput $f -RedirectStandardError "$f.err" -NoNewWindow -PassThru -Wait
        if ($p.ExitCode -eq 0) { New-Item -ItemType File -Force "$f.done" | Out-Null }
        $last = (Get-Content $f -ErrorAction SilentlyContinue | Select-Object -Last 2 | Select-Object -First 1)
        Write-Host "=== END   $Task $Tag seed $s  $(Get-Date -Format HH:mm:ss)  exit $($p.ExitCode)  $last"
    }
}

# 1. arrival: shipped h4 SHARE-lite, then the h0 reference that is actually SERVED
Run-Cells arrival_week    h4 @("--arch", "lite", "--depth", "4", "--lr", "2.5e-4")
Run-Cells arrival_week    h0 @("--arch", "none", "--depth", "0", "--lr", "2.5e-4")
# 2. capacity: shipped h4 message-passing, then h0
Run-Cells capacity_strain h4 @("--arch", "mp",   "--depth", "4", "--lr", "2.5e-4")
Run-Cells capacity_strain h0 @("--arch", "none", "--depth", "0", "--lr", "2.5e-4")
# 3. fill: neural h0 (deviation 46 -- see header)
Run-Cells fill_rate       h0 @("--arch", "none", "--depth", "0", "--lr", "1.25e-4")
# 4. shortage: shipped h1, diagnostic only
Run-Cells shortage_qty    h1 @("--arch", "mp",   "--depth", "1", "--lr", "1.25e-4")
Run-Cells shortage_qty    h0 @("--arch", "none", "--depth", "0", "--lr", "1.25e-4")
Write-Host "=== QUEUE COMPLETE $(Get-Date -Format HH:mm:ss)"
