# NEXUS-SLM Local Demo Launch Script for Windows PowerShell
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   NEXUS-SLM — Neural EXecution Support Intelligence     " -ForegroundColor Cyan
Write-Host "   Local Demonstration Launcher (Windows PowerShell)     " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$WorkspaceRoot = Split-Path -Parent $PSScriptRoot
Set-Location $WorkspaceRoot

# Check Python virtual environment
$PythonExe = Join-Path $WorkspaceRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Error "Virtual environment not found at $PythonExe. Please run python -m venv .venv and install requirements."
    exit 1
}

# Verify model weights
$ModelDir = Join-Path $WorkspaceRoot "models\nexus_qlora_v1"
if (-not (Test-Path (Join-Path $ModelDir "adapter_model.safetensors"))) {
    Write-Warning "Primary QLoRA weights not found at $ModelDir. API will initialize with deterministic heuristic fallback."
} else {
    Write-Host "Found primary candidate weights at $ModelDir." -ForegroundColor Green
}

Write-Host "`n[1/3] Starting FastAPI Inference Service on port 8000..." -ForegroundColor Yellow
$BackendProcess = Start-Process -FilePath $PythonExe -ArgumentList "-m uvicorn api.main:app --host 127.0.0.1 --port 8000" -PassThru -WindowStyle Minimized

Write-Host "Waiting for FastAPI service to initialize..."
$MaxRetries = 15
$Retry = 0
$BackendReady = $false

while ($Retry -lt $MaxRetries) {
    Start-Sleep -Seconds 1
    try {
        $Health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get -TimeoutSec 2
        if ($Health.status -eq "healthy") {
            $BackendReady = $true
            Write-Host "FastAPI Service is ONLINE [Mode: $($Health.inference_mode), Version: $($Health.model_version)]" -ForegroundColor Green
            break
        }
    } catch {
        $Retry++
    }
}

if (-not $BackendReady) {
    Write-Error "FastAPI backend failed to respond within $MaxRetries seconds."
    Stop-Process -Id $BackendProcess.Id -Force -ErrorAction SilentlyContinue
    exit 1
}

Write-Host "`n[2/3] Starting Next.js Dashboard on port 3000..." -ForegroundColor Yellow
$DashboardDir = Join-Path $WorkspaceRoot "dashboard"
$FrontendProcess = Start-Process -FilePath "cmd.exe" -ArgumentList "/c cd /d `"$DashboardDir`" && npm run dev" -PassThru -WindowStyle Minimized

Write-Host "Waiting for Next.js dashboard to initialize..."
Start-Sleep -Seconds 4

Write-Host "`n[3/3] DEMONSTRATION READY!" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Interactive UI: http://localhost:3000" -ForegroundColor White
Write-Host "  Swagger API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host "  Health Endpoint: http://localhost:8000/health" -ForegroundColor White
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "To terminate demo, press Enter or Ctrl+C..."

Read-Host "Press Enter to stop services"

Write-Host "Shutting down background demo processes..." -ForegroundColor Yellow
Stop-Process -Id $BackendProcess.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $FrontendProcess.Id -Force -ErrorAction SilentlyContinue
Write-Host "Demo shut down cleanly." -ForegroundColor Green
