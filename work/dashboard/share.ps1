param([int]$GatewayPort = 8880)
$ErrorActionPreference = 'Stop'
$DashboardPort = 8875
$projectDirectory = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$sharingDirectory = Join-Path $projectDirectory 'work\sharing'
New-Item -Path $sharingDirectory -ItemType Directory -Force | Out-Null
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$DashboardPort/api/health" -TimeoutSec 3
    $ready = $health.status -eq 'ready'
} catch { $ready = $false }
if (-not $ready) { throw '请先启动 8875 看板。' }
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
$gatewayReady = $false
$accessFile = Join-Path $PSScriptRoot 'share_access.json'
if (Test-Path -LiteralPath $accessFile) {
    $access = Get-Content -LiteralPath $accessFile -Raw | ConvertFrom-Json
    $auth = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("$($access.username):$($access.password)"))
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$GatewayPort/api/health" -Headers @{Authorization="Basic $auth"} -TimeoutSec 3
        $gatewayReady = $health.status -eq 'ready'
    } catch {}
}
if (-not $gatewayReady) {
    if (Get-NetTCPConnection -LocalPort $GatewayPort -State Listen -ErrorAction SilentlyContinue) { throw "端口 $GatewayPort 已占用，且未通过本项目共享检查。" }
    $gatewayScript = Join-Path $PSScriptRoot 'share.py'
    Start-Process -FilePath $pythonRuntime -ArgumentList @('-X','utf8',"`"$gatewayScript`"") `
        -WorkingDirectory $projectDirectory -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $sharingDirectory 'gateway.log') `
        -RedirectStandardError (Join-Path $sharingDirectory 'gateway-error.log') | Out-Null
    for ($attempt=0; $attempt -lt 15; $attempt++) {
        Start-Sleep -Seconds 1
        if (Test-Path -LiteralPath $accessFile) {
            $access = Get-Content -LiteralPath $accessFile -Raw | ConvertFrom-Json
            $auth = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("$($access.username):$($access.password)"))
            try {
                $health = Invoke-RestMethod -Uri "http://127.0.0.1:$GatewayPort/api/health" -Headers @{Authorization="Basic $auth"} -TimeoutSec 2
                if ($health.status -eq 'ready') { $gatewayReady=$true; break }
            } catch {}
        }
    }
}
if (-not $gatewayReady) { throw '共享网关未就绪，请查看 work\sharing\gateway-error.log。' }
Write-Host "本机看板：http://127.0.0.1:$DashboardPort/"
Write-Host "共享用户名为 team；密码保存在 $accessFile。"
Write-Host '健康检查通过后会显示可分享地址；保持此窗口和电脑运行，按 Ctrl+C 结束共享。'
& $pythonRuntime -X utf8 (Join-Path $PSScriptRoot 'tunnel.py') --port $GatewayPort
