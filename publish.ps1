param([string]$Message = "chore: update open-ai-hci")
if (Test-Path ".venv\Scripts\Activate.ps1") { . .venv\Scripts\Activate.ps1 }
ruff check open_ai_hci tests examples tools
if ($LASTEXITCODE -ne 0) { Write-Host "Ruff failed! Push aborted." -ForegroundColor Red; exit 1 }
pytest
if ($LASTEXITCODE -ne 0) { Write-Host "Tests failed! Push aborted." -ForegroundColor Red; exit 1 }
git add .
git commit -m $Message
git push
