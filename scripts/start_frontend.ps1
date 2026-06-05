param(
    [int]$Port = 5173,
    [string]$HostName = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$WebRoot = Join-Path $Root "web"
Set-Location $WebRoot

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "npm was not found on PATH. Install Node.js, then rerun this task."
    exit 1
}

if (-not (Test-Path "node_modules")) {
    Write-Host "Installing frontend dependencies..."
    npm install
}

$BackendPortInUse = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if (-not $BackendPortInUse) {
    Write-Host "Backend does not appear to be listening on port 8000."
    Write-Host "Start the 'Start Backend' task first for API-backed pages."
}

Write-Host "Starting StatVocab frontend at http://$HostName`:$Port"
npm run dev -- --host $HostName --port $Port
