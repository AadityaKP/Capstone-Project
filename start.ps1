<#
.SYNOPSIS
    Bring up the founder product and leave it running.

.DESCRIPTION
    The frontend is not useful on its own - every analysis goes through the API,
    and the API needs Ollama. So this starts the whole stack, checks each piece is
    actually answering before it says so, and shuts all of it down together on
    Ctrl+C.

    Two modes:

      dev   (default)  Vite on 5173 with hot reload, proxying /api to 8000.
                       What you want while testing or changing the UI.
      -Prod            Rebuilds frontend/dist and serves everything from 8000.
                       What you want for a demo: one port, one URL, and no
                       chance of a stale bundle (see -Prod notes below).

.PARAMETER Prod
    Build the frontend and serve it from the API port instead of running Vite.
    Always rebuilds, because a stale frontend/dist is invisible: the page loads
    and looks correct with the previous build's JavaScript.

.PARAMETER Reload
    Restart the API when backend .py files change. Off by default - the reloader
    runs a child process that complicates clean shutdown, and you do not want it
    mid-demo.

.PARAMETER NoOllama
    Skip the Ollama check and start. The stack still runs; analyses come back
    with llm_ok=false and the UI shows its rules-only state, which is a
    legitimate thing to test (runbook 4.2).

.PARAMETER NoBrowser
    Do not open a browser window.

.EXAMPLE
    .\start.ps1
    Dev mode. Open http://localhost:5173

.EXAMPLE
    .\start.ps1 -Prod
    Rebuild and serve everything from http://localhost:8000

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\start.ps1
    If script execution is blocked on this machine.
#>

[CmdletBinding()]
param(
    [switch]$Prod,
    [switch]$Reload,
    [switch]$NoOllama,
    [switch]$NoBrowser,
    [int]$ApiPort = 8000,
    [int]$UiPort  = 5173
)

$ErrorActionPreference = 'Stop'
$Root      = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python    = Join-Path $Root 'venv\Scripts\python.exe'
$FrontEnd  = Join-Path $Root 'frontend'
$LogDir    = Join-Path $Root '.logs'
$OllamaModel = 'llama3.1:8b'

$script:Started = @()   # processes we own and must clean up

# ---------------------------------------------------------------- helpers ---

function Write-Step($msg)  { Write-Host "  $msg" -ForegroundColor Cyan }
function Write-Ok($msg)    { Write-Host "  [ok]   $msg" -ForegroundColor Green }
function Write-Warn2($msg) { Write-Host "  [warn] $msg" -ForegroundColor Yellow }
function Write-Bad($msg)   { Write-Host "  [fail] $msg" -ForegroundColor Red }

function Test-Port {
    param([int]$Port)
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne(400)) { return $false }
        $client.EndConnect($async)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

# Poll until a URL answers. Ports open before a server is ready to serve, so
# waiting on the port alone hands you a browser tab that fails on first request.
function Wait-ForHttp {
    param([string]$Url, [int]$TimeoutSec = 90, [string]$Label = 'service')
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
            if ($response.StatusCode -eq 200) { return $true }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    Write-Bad "$Label did not answer at $Url within ${TimeoutSec}s"
    return $false
}

# Under PowerShell, `Get-Command npm` resolves to npm.ps1 - an ExternalScript,
# not an executable. Start-Process cannot launch it and fails with the unhelpful
# "%1 is not a valid Win32 application", which reads like a corrupt install
# rather than a wrong file extension. The npm.cmd shim beside it is a real
# executable and behaves identically.
#
# Only Start-Process needs this. Calling `npm install` inline elsewhere in this
# script works fine, because PowerShell can run the .ps1 itself.
function Resolve-NpmExecutable {
    $npm = Get-Command npm -ErrorAction SilentlyContinue
    if ($null -eq $npm) { return $null }
    $shim = Join-Path (Split-Path $npm.Source) 'npm.cmd'
    if (Test-Path $shim) { return $shim }
    return $npm.Source
}

function Start-Tracked {
    param([string]$File, [string]$Arguments, [string]$WorkDir, [string]$LogName)
    $out = Join-Path $LogDir "$LogName.out.log"
    $err = Join-Path $LogDir "$LogName.err.log"
    $process = Start-Process -FilePath $File -ArgumentList $Arguments `
        -WorkingDirectory $WorkDir -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $script:Started += $process
    return $process
}

function Stop-All {
    Write-Host ""
    Write-Host "Shutting down..." -ForegroundColor Yellow
    foreach ($process in $script:Started) {
        if ($null -eq $process) { continue }
        try {
            if (-not $process.HasExited) {
                # taskkill /T so uvicorn's reloader child and npm's node child go
                # too; Stop-Process alone orphans them and the port stays held.
                taskkill /PID $process.Id /T /F | Out-Null
            }
        } catch {
            Write-Warn2 "could not stop PID $($process.Id)"
        }
    }
    Write-Host "Stopped." -ForegroundColor Yellow
}

# ------------------------------------------------------------ preflight -----

Write-Host ""
Write-Host "Startup Society of Minds - founder product" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor DarkGray

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host ""
Write-Host "Checking prerequisites" -ForegroundColor White

if (-not (Test-Path $Python)) {
    Write-Bad "No virtualenv at venv\. Create it, then: venv\Scripts\pip.exe install -r requirements.txt"
    exit 1
}
Write-Ok "python  $((& $Python --version) -replace 'Python ','')"

if (-not (Test-Path (Join-Path $FrontEnd 'node_modules'))) {
    Write-Warn2 "frontend\node_modules missing - running npm install (one time, a minute or two)"
    Push-Location $FrontEnd
    try {
        npm install
        if ($LASTEXITCODE -ne 0) { Write-Bad "npm install failed"; exit 1 }
    } finally {
        Pop-Location
    }
}
Write-Ok "node modules present"

# Ollama. Not fatal: the product has a designed rules-only path when the model is
# unreachable, and testing that path is legitimate. But it must be said clearly,
# because llm_ok=false looks like a bug if you were not expecting it.
if (-not $NoOllama) {
    if (Test-Port -Port 11434) {
        Write-Ok "ollama already running"
    } else {
        $ollama = Get-Command ollama -ErrorAction SilentlyContinue
        if ($null -eq $ollama) {
            Write-Warn2 "ollama not on PATH - analyses will return llm_ok=false (rules-only)"
        } else {
            Write-Step "starting ollama..."
            Start-Tracked -File $ollama.Source -Arguments 'serve' -WorkDir $Root -LogName 'ollama' | Out-Null
            if (Wait-ForHttp -Url 'http://127.0.0.1:11434/api/tags' -TimeoutSec 30 -Label 'ollama') {
                Write-Ok "ollama started"
            }
        }
    }

    try {
        $tags = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 5
        if ($tags.models.name -contains $OllamaModel) {
            Write-Ok "model $OllamaModel present"
        } else {
            Write-Warn2 "model $OllamaModel not pulled. Run: ollama pull $OllamaModel"
        }
    } catch {
        Write-Warn2 "could not query ollama - analyses will be rules-only"
    }
} else {
    Write-Warn2 "-NoOllama: skipping. Analyses will return llm_ok=false"
}

# Neo4j is optional and the backend degrades on its own, so this is information
# only. Without it you lose the causal evidence line, nothing else.
if (Test-Port -Port 7687) {
    Write-Ok "neo4j reachable (causal evidence enabled)"
} else {
    Write-Warn2 "neo4j not reachable - no causal evidence line. Set FOUNDER_ORACLE_MODE=oracle_v4 to silence"
}

foreach ($port in @($ApiPort, $UiPort)) {
    if ($port -eq $UiPort -and $Prod) { continue }
    if (Test-Port -Port $port) {
        Write-Bad "port $port is already in use. Stop whatever holds it, or pass -ApiPort / -UiPort"
        exit 1
    }
}

# -------------------------------------------------------------- frontend ----

if ($Prod) {
    Write-Host ""
    Write-Host "Building frontend" -ForegroundColor White
    # Always rebuild. A stale dist is the worst failure mode here because it is
    # silent: the page loads, looks right, and runs the previous build.
    Push-Location $FrontEnd
    try {
        npm run build
        if ($LASTEXITCODE -ne 0) { Write-Bad "build failed"; exit 1 }
    } finally {
        Pop-Location
    }
    $bundle = Get-ChildItem (Join-Path $FrontEnd 'dist\assets') -Filter '*.js' |
              Select-Object -First 1
    Write-Ok "built $($bundle.Name)"
}

# ------------------------------------------------------------------ start ---

try {
    Write-Host ""
    Write-Host "Starting services" -ForegroundColor White

    $uvicornArgs = "-m uvicorn backend.main:app --host 127.0.0.1 --port $ApiPort"
    if ($Reload) { $uvicornArgs += ' --reload' }
    Write-Step "api on $ApiPort..."
    Start-Tracked -File $Python -Arguments $uvicornArgs -WorkDir $Root -LogName 'api' | Out-Null

    if (-not (Wait-ForHttp -Url "http://127.0.0.1:$ApiPort/api/health" -TimeoutSec 60 -Label 'api')) {
        Write-Bad "see .logs\api.err.log"
        Stop-All
        exit 1
    }
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/api/health" -TimeoutSec 5
    Write-Ok "api ready - advisor mode: $($health.advisor_mode)"

    if ($Prod) {
        $appUrl = "http://localhost:$ApiPort"
    } else {
        Write-Step "vite on $UiPort..."
        $npm = Resolve-NpmExecutable
        if ($null -eq $npm) {
            Write-Bad "npm not on PATH"
            Stop-All
            exit 1
        }
        # -UiPort used to change only what this script waited on, not what Vite
        # bound to: Vite took its default 5173 and the wait timed out against an
        # empty port while a working dev server sat next door. `--` forwards the
        # rest to Vite; --strictPort makes a clash fail loudly instead of
        # silently landing on the next free port.
        #
        # VITE_API_PORT is read by frontend/vite.config.js for the /api proxy, so
        # -ApiPort moves the backend and the proxy that talks to it together.
        $env:VITE_API_PORT = "$ApiPort"
        Start-Tracked -File $npm -Arguments "run dev -- --port $UiPort --strictPort" `
            -WorkDir $FrontEnd -LogName 'ui' | Out-Null
        if (-not (Wait-ForHttp -Url "http://127.0.0.1:$UiPort" -TimeoutSec 60 -Label 'vite')) {
            Write-Bad "see .logs\ui.err.log"
            Stop-All
            exit 1
        }
        Write-Ok "vite ready"
        $appUrl = "http://localhost:$UiPort"
    }

    Write-Host ""
    Write-Host "  Open  $appUrl" -ForegroundColor White
    if ($Prod) {
        Write-Host "  If the page looks like an older build, hard-reload: Ctrl+Shift+R" -ForegroundColor DarkGray
    }
    Write-Host "  Logs  .logs\api.out.log  .logs\ui.out.log" -ForegroundColor DarkGray
    Write-Host "  Stop  Ctrl+C" -ForegroundColor DarkGray
    Write-Host ""

    if (-not $NoBrowser) { Start-Process $appUrl | Out-Null }

    # Hold the console so Ctrl+C reaches the finally block below. Also notices if
    # a child dies on its own rather than sitting on a stack that is half up.
    while ($true) {
        Start-Sleep -Seconds 2
        foreach ($process in $script:Started) {
            if ($process.HasExited) {
                Write-Bad "a service exited (PID $($process.Id), code $($process.ExitCode)) - check .logs\"
                return
            }
        }
    }
} finally {
    Stop-All
}
