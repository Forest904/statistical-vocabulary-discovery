param(
    [string]$Config = "configs/core.yaml",
    [int]$Port = 8000,
    [string]$HostName = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "Creating Python virtual environment..."
    py -3.12 -m venv .venv
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"
Write-Host "Checking backend Python dependencies..."
& $Python -c "import fastapi, uvicorn, statvocab" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing backend dependencies..."
    & $Python -m pip install --upgrade pip
    & $Python -m pip install -e ".[api,ml]"
}

$env:STATVOCAB_CONFIG = $Config

$PortInUse = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($PortInUse) {
    Write-Host "Port $Port is already in use. Stop the existing backend or choose another port with -Port."
    exit 1
}

Write-Host "Starting StatVocab backend at http://$HostName`:$Port"
Write-Host "Using config: $Config"
& $Python -m uvicorn api.app.main:app --reload --host $HostName --port $Port
