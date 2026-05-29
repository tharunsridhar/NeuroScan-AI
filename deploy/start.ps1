param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Missing .venv. Create it with Python 3.11 and install requirements first."
}

& ".venv\Scripts\python.exe" -m uvicorn app.main:app --host $HostName --port $Port
