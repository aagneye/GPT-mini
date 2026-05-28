# PowerShell script to download model from AMD GPU Droplet
# Run this on Windows in PowerShell

# Configuration - UPDATE THESE!
$VM_IP = "YOUR_VM_IP"
$VM_USER = "YOUR_USERNAME"
$LOCAL_DIR = ".\downloaded_models"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Model Download Script (PowerShell)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "VM: $VM_USER@$VM_IP"
Write-Host "Local directory: $LOCAL_DIR"
Write-Host "==========================================" -ForegroundColor Cyan

# Create local directory
New-Item -ItemType Directory -Force -Path $LOCAL_DIR | Out-Null

# Download final model
Write-Host "[1/4] Downloading final model..." -ForegroundColor Yellow
scp "${VM_USER}@${VM_IP}:~/gpt-mini-training/model.pth" "$LOCAL_DIR/"

# Download latest checkpoint
Write-Host "[2/4] Downloading latest checkpoint..." -ForegroundColor Yellow
scp "${VM_USER}@${VM_IP}:~/gpt-mini-training/checkpoints/step_*.pth" "$LOCAL_DIR/"

# Download tokenizer
Write-Host "[3/4] Downloading tokenizer..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "$LOCAL_DIR\tokenizer" | Out-Null
scp "${VM_USER}@${VM_IP}:~/gpt-mini-training/tokenizer/spm.model" "$LOCAL_DIR\tokenizer\"

# Download training progress report
Write-Host "[4/4] Downloading progress report..." -ForegroundColor Yellow
scp "${VM_USER}@${VM_IP}:~/gpt-mini-training/training_progress.json" "$LOCAL_DIR/"

Write-Host "==========================================" -ForegroundColor Green
Write-Host "Download complete!" -ForegroundColor Green
Write-Host "Files saved to: $LOCAL_DIR" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# List downloaded files
Get-ChildItem -Path $LOCAL_DIR -Recurse | Format-Table Name, Length, LastWriteTime
