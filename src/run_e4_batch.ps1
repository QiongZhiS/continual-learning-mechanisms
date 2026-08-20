# E4 六运行批处理（单轨/双轨 × seed 0/1/2，各 6000 步）
# 任一失败：记录到 _batch_status.log 并继续下一个；全部结束后汇总
$ErrorActionPreference = "Continue"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"
$env:DISABLE_COMPILE = "1"
$log = "outputs/2026-08-18/m4_e4/_batch_status.log"
$runs = @(
  @{arm="single"; seed=0},
  @{arm="dual";   seed=0},
  @{arm="single"; seed=1},
  @{arm="dual";   seed=1},
  @{arm="single"; seed=2},
  @{arm="dual";   seed=2}
)
foreach ($r in $runs) {
  $out = "outputs/2026-08-18/m4_e4/$($r.arm)/seed$($r.seed)"
  "=== $($r.arm) seed $($r.seed) start $(Get-Date -Format 'yyyy-MM-dd HH:mm') ===" | Out-File -Append -Encoding utf8 $log
  & ..\.venv\Scripts\python.exe m4_e4.py --arm $r.arm --seed $r.seed --steps 6000 --out $out 2>&1 | Tee-Object -FilePath "outputs/2026-08-18/m4_e4/$($r.arm)_seed$($r.seed).log" -Append
  if ($LASTEXITCODE -eq 0) {
    "=== $($r.arm) seed $($r.seed) OK $(Get-Date -Format 'HH:mm') ===" | Out-File -Append -Encoding utf8 $log
  } else {
    "=== $($r.arm) seed $($r.seed) FAIL exit=$LASTEXITCODE $(Get-Date -Format 'HH:mm') ===" | Out-File -Append -Encoding utf8 $log
  }
}
"=== ALL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm') ===" | Out-File -Append -Encoding utf8 $log
