param([switch]$CleanLegacyOnly)
$ErrorActionPreference = 'Stop'
$dashboardDirectory = [System.IO.Path]::GetFullPath($PSScriptRoot)
$projectDirectory = Split-Path -Parent (Split-Path -Parent $dashboardDirectory)
$standardPort = 8875
$env:PYTHONDONTWRITEBYTECODE = '1'
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $OutputEncoding

$pythonRuntime = $env:DASHBOARD_PYTHON
if (-not $pythonRuntime) {
    $localRuntime = Join-Path $projectDirectory 'work\.venv\Scripts\python.exe'
    $anacondaRuntime = Join-Path $env:USERPROFILE 'anaconda3\python.exe'
    if (Test-Path -LiteralPath $localRuntime) { $pythonRuntime = $localRuntime }
    elseif (Test-Path -LiteralPath $anacondaRuntime) { $pythonRuntime = $anacondaRuntime }
    else { $pythonRuntime = (Get-Command python -ErrorAction Stop).Source }
}
if (-not (Test-Path -LiteralPath $pythonRuntime)) { throw "Python 路径不存在：$pythonRuntime" }

# Keep two restart invocations from stopping or starting the service simultaneously.
$lockPath = Join-Path $dashboardDirectory '.restart.lock'
try {
    $restartLock = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
} catch { throw "无法取得重启锁，请检查是否已有重启窗口或文件权限：$($_.Exception.Message)" }

try {
    $controlScript = Join-Path $dashboardDirectory 'service_control.py'
    $controlArguments = @('-X','utf8','-B',$controlScript)
    if ($CleanLegacyOnly) { $controlArguments += '--legacy-only' }
    $controlOutput = & $pythonRuntime @controlArguments
    $controlExitCode = $LASTEXITCODE
    try { $controlState = ($controlOutput -join "`n") | ConvertFrom-Json }
    catch { throw "无法读取服务管理结果：$($controlOutput -join ' ')" }
    if ($controlExitCode -ne 0) { throw $controlState.error }
    foreach ($service in $controlState.stopped) { Write-Host "已关闭端口 $($service.port)。" }

    $activeDirectory = $null
    $stateFile = Join-Path $dashboardDirectory 'service_state.json'
    if ($CleanLegacyOnly -and @($controlState.current).Count) {
        if (-not (Test-Path -LiteralPath $stateFile)) { throw '当前服务未登记生成目录，未执行清理。' }
        $serviceState = Get-Content -LiteralPath $stateFile -Raw -Encoding UTF8 | ConvertFrom-Json
        if (-not $serviceState.DebateDirectory -or $serviceState.Port -ne $standardPort -or
            $serviceState.PID -notin @($controlState.current.pid)) { throw '当前服务状态不符，未执行清理。' }
        $activeDirectory = [System.IO.Path]::GetFullPath($serviceState.DebateDirectory)
        if (-not $activeDirectory.StartsWith($dashboardDirectory + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw '当前辩论目录不在本项目内，未执行清理。'
        }
    }

    $removedBytes = 0L
    $removedCount = 0
    $pendingCleanup = @()
    foreach ($directory in @(Get-ChildItem -LiteralPath $dashboardDirectory -Directory -Filter 'rebound-debate-*')) {
        $resolved = [System.IO.Path]::GetFullPath($directory.FullName)
        if (-not $resolved.StartsWith($dashboardDirectory + '\', [StringComparison]::OrdinalIgnoreCase)) { throw '清理路径超出本项目。' }
        if ($resolved -eq $activeDirectory) { continue }
        if ($directory.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { $pendingCleanup += $directory.Name; continue }
        try {
            $children = @(Get-ChildItem -LiteralPath $resolved -Recurse -Force -ErrorAction Stop)
            if ($children | Where-Object { $_.Attributes -band [System.IO.FileAttributes]::ReparsePoint }) { $pendingCleanup += $directory.Name; continue }
            $generatedFiles = @($children | Where-Object { -not $_.PSIsContainer })
            if ($generatedFiles | Where-Object { $_.Name -notin @('app.js','charts.js','index.html','styles.css') }) { $pendingCleanup += $directory.Name; continue }
            $bytes = ($generatedFiles | Measure-Object Length -Sum).Sum
            Remove-Item -LiteralPath $resolved -Recurse -Force -ErrorAction Stop
            $removedBytes += $bytes
            $removedCount++
        } catch { $pendingCleanup += $directory.Name }
    }
    Write-Host ("清理 {0} 个旧网页目录，共 {1:N2} MB。" -f $removedCount, ($removedBytes / 1MB))
    if ($pendingCleanup.Count) { Write-Warning "仍有 $($pendingCleanup.Count) 个目录无法清理；清理权限不影响看板启动。" }

    if (-not $CleanLegacyOnly) {
        $env:DASHBOARD_PORT = "$standardPort"
        $serviceScript = Join-Path $dashboardDirectory 'server.py'
        $started = Start-Process -FilePath $pythonRuntime -ArgumentList @('-X','utf8','-B',"`"$serviceScript`"") `
            -WorkingDirectory $projectDirectory -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $dashboardDirectory 'dashboard-service.log') `
            -RedirectStandardError (Join-Path $dashboardDirectory 'dashboard-service-error.log')
        $deadline = [DateTime]::UtcNow.AddSeconds(45)
        $ready = $false
        do {
            $started.Refresh()
            if ($started.HasExited) { throw '看板进程已退出，请查看 dashboard-service-error.log。' }
            try {
                $health = Invoke-RestMethod -Uri "http://127.0.0.1:$standardPort/api/health" -TimeoutSec 2
                $registered = Get-Content -LiteralPath $stateFile -Raw -Encoding UTF8 | ConvertFrom-Json
                if ($health.status -eq 'ready' -and $registered.PID -eq $started.Id) { $ready = $true; break }
            } catch {}
            Start-Sleep -Milliseconds 500
        } while ([DateTime]::UtcNow -lt $deadline)
        if (-not $ready) { throw '45 秒内未完成启动；保留进程与日志，请检查后再重试。' }
    }
    Write-Host "看板入口：http://127.0.0.1:$standardPort/"
} finally { $restartLock.Dispose() }
