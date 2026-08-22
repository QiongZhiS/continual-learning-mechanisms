# B1 probe batch runner: 5 arms x 3 seeds sequentially + judge.
# Usage: powershell -ExecutionPolicy Bypass -File run_b1_batch.ps1
# ASCII-only comments (PS 5.1 ANSI parse workaround). Per-run auto-resume via result.json.
# Env: run from repo dir; trainer = ..\.venv\Scripts\python.exe (default python has no torch).
$ErrorActionPreference = "Continue"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"
$env:DISABLE_COMPILE = "1"
$py = "..\.venv\Scripts\python.exe"
$arms = @("a0", "a1", "a2", "a3", "a4")
$seeds = @(0, 1, 2)
$logDir = "outputs/2026-08-20/m1_b1_probe/logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
foreach ($arm in $arms) {
  foreach ($seed in $seeds) {
    $sum = "outputs/2026-08-20/m1_b1_probe/$arm/s$seed/summary.json"
    if (Test-Path $sum) {
      Write-Host "SKIP $arm s$seed (summary exists)"
      continue
    }
    $log = "$logDir/${arm}_s${seed}.txt"
    Write-Host "=== RUN $arm s$seed -> $log ==="
    & $py m1_b1_probe.py --arm $arm --seed $seed *> $log
    if ($LASTEXITCODE -ne 0) {
      Write-Host "FAILED $arm s$seed exit=$LASTEXITCODE (see $log)"
    } else {
      Write-Host "OK $arm s$seed"
    }
  }
}
Write-Host "=== JUDGE ==="
& $py m1_b1_probe.py --judge-only *> "$logDir/judge.txt"
Write-Host "done. judge.txt under $logDir"
