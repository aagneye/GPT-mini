# Downloads Alpaca + Dolly into data/dataset.txt (same as Kaggle prepare step).
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (Get-Command python -ErrorAction SilentlyContinue) {
    python prepare_dataset.py
    exit $LASTEXITCODE
}
if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3 prepare_dataset.py
    exit $LASTEXITCODE
}

Write-Error @"
Python not found in PATH.
Install Python 3, then from this repo root run:
  pip install -r requirements.txt
  python prepare_dataset.py
"@
exit 1
