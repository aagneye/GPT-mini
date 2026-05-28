# PowerShell script for continuous backup
# Run this in a separate PowerShell window while training is running

# Configuration - UPDATE THESE!
$VM_IP = "YOUR_VM_IP"
$VM_USER = "YOUR_USERNAME"
$LOCAL_BACKUP_DIR = ".\live_backup"
$SYNC_INTERVAL = 600  # Sync every 10 minutes (600 seconds)

New-Item -ItemType Directory -Force -Path $LOCAL_BACKUP_DIR | Out-Null
New-Item -ItemType Directory -Force -Path "$LOCAL_BACKUP_DIR\checkpoints" | Out-Null

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Live Backup Monitor" -ForegroundColor Cyan
Write-Host "Syncing every $SYNC_INTERVAL seconds" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

while ($true) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] Syncing from VM..." -ForegroundColor Yellow
    
    # Sync checkpoints
    try {
        scp "${VM_USER}@${VM_IP}:~/gpt-mini-training/checkpoints/step_*.pth" "$LOCAL_BACKUP_DIR\checkpoints\" 2>$null
        Write-Host "  Checkpoints synced" -ForegroundColor Green
    } catch {
        Write-Host "  Warning: Checkpoint sync failed" -ForegroundColor Red
    }
    
    # Sync progress report
    try {
        scp "${VM_USER}@${VM_IP}:~/gpt-mini-training/training_progress.json" "$LOCAL_BACKUP_DIR\" 2>$null
        Write-Host "  Progress report synced" -ForegroundColor Green
    } catch {
        Write-Host "  Warning: Progress report sync failed" -ForegroundColor Red
    }
    
    # Show current progress
    $progressFile = Join-Path $LOCAL_BACKUP_DIR "training_progress.json"
    if (Test-Path $progressFile) {
        $progress = Get-Content $progressFile | ConvertFrom-Json
        Write-Host ""
        Write-Host "  Current Progress:" -ForegroundColor Cyan
        Write-Host "    Step: $($progress.last_step)/$($progress.max_iters)" -ForegroundColor White
        Write-Host "    Progress: $([math]::Round($progress.progress_percentage, 2))%" -ForegroundColor White
        Write-Host "    Cost: `$$([math]::Round($progress.cost_estimate_usd, 2))" -ForegroundColor White
        Write-Host "    Elapsed: $([math]::Round($progress.elapsed_hours, 2))h" -ForegroundColor White
    }
    
    Write-Host ""
    Write-Host "[$timestamp] Sync complete. Next sync in $SYNC_INTERVAL seconds..." -ForegroundColor Green
    Write-Host "----------------------------------------"
    
    Start-Sleep -Seconds $SYNC_INTERVAL
}
