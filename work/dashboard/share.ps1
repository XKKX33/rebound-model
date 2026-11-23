param([int]$GatewayPort = 8880)
$ErrorActionPreference = 'Stop'
$DashboardPort = 8875
$projectDirectory = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$env:DASHBOARD_SHARE_UPSTREAM = "http://127.0.0.1:$DashboardPort"
$env:DASHBOARD_SHARE_PORT = "$GatewayPort"
$env:PYTHONDONTWRITEBYTECODE = '1'
$pythonRuntime = $env:DASHBOARD_PYTHON
if (-not $pythonRuntime) {
    $localRuntime = Join-Path $projectDirectory 'work\.venv\Scripts\python.exe'
    $anacondaRuntime = Join-Path $env:USERPROFILE 'anaconda3\python.exe'
    if (Test-Path -LiteralPath $localRuntime) { $pythonRuntime = $localRuntime }
    elseif (Test-Path -LiteralPath $anacondaRuntime) { $pythonRuntime = $anacondaRuntime }
    else { $pythonRuntime = (Get-Command python -ErrorAction Stop).Source }
}
& $pythonRuntime -B -X utf8 (Join-Path $PSScriptRoot 'manage_sharing.py') --port $GatewayPort
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
