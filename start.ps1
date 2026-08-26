<#
.SYNOPSIS
    Bring up the founder product and leave it running.

.DESCRIPTION
    The frontend is not useful on its own - every analysis goes through the API.
    So this starts the whole stack, checks each piece is actually answering before
    it says so, and shuts all of it down together on Ctrl+C.

    ON THIS BRANCH (demo-offline) the API does not need Ollama or Neo4j. Recorded
    answers cover the demo dataset and a model-free board covers everything else,
    so the whole product runs on a laptop with neither service:

        .\start.ps1 -Prod -NoOllama

    See docs/demo_walkthrough.md for the dataset and what each screen shows.
    Set FOUNDER_DEMO_FIXTURES=0 to force the live stack back on.

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
    Skip the Ollama check and start. On this branch that is the normal way to
    demonstrate: the demo dataset replays recorded answers with llm_ok=true, and
    any other numbers get a real model-free analysis with llm_ok=false and the
    UI's honest rules-only banner (runbook 4.2).

.PARAMETER NoBrowser
    Do not open a browser window.

.PARAMETER Force
    Kill whatever is holding the ports and start anyway. Handles the case the
    plain error cannot: when the PID in the connection table is already dead and
    a child process inherited the socket, so taskkill on the named PID reports
    "not found" while the port stays busy.

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
    [switch]$Force,
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

# Who actually holds a port, and is it one of ours.
#
# The connection table names the process that *created* the socket, which is not
# always the process still holding it. uvicorn --reload spawns a child; kill the
# parent and the child inherits the handle, but the table keeps naming the dead
# parent. taskkill on that PID reports "not found" while the port stays busy - a
# failure mode that reads like a broken machine instead of an orphaned process.
function Get-PortHolder {
    param([int]$Port)

    $info = [pscustomobject]@{
        Port         = $Port
        ProcessId    = $null
        Name         = $null
        Stale        = $false
        IsOurBackend = $false
        LiveChildren = @()
    }

    $conn = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    if ($conn) {
        $info.ProcessId = ($conn | Select-Object -First 1).OwningProcess
        $process = Get-Process -Id $info.ProcessId -ErrorAction SilentlyContinue
        if ($process) {
            $info.Name = $process.ProcessName
        } else {
            $info.Stale = $true
            # The live holder is usually a child of the dead PID.
            $info.LiveChildren = @(
                Get-CimInstance Win32_Process -Filter "ParentProcessId=$($info.ProcessId)" -ErrorAction SilentlyContinue
            )
        }
    }

    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/health" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $info.IsOurBackend = $true }
    } catch { }

    $info | Add-Member -MemberType ScriptMethod -Name Describe -Value {
        $who = if ($this.Name) { "$($this.Name) (pid $($this.ProcessId))" }
               elseif ($this.ProcessId) { "pid $($this.ProcessId), no longer running" }
               else { 'an unidentified process' }
        if ($this.LiveChildren.Count -gt 0) {
            $kids = ($this.LiveChildren | ForEach-Object { "$($_.Name) pid $($_.ProcessId)" }) -join ', '
            $who += " -> live child: $kids"
        }
        return $who
    }
    return $info
}

function Clear-Port {
    param([int]$Port)
    $holder = Get-PortHolder -Port $Port

    # Children first: killing the parent can reparent them and lose the trail.
    foreach ($child in $holder.LiveChildren) {
        taskkill /PID $child.ProcessId /T /F 2>&1 | Out-Null
    }
    if ($holder.ProcessId) {
        taskkill /PID $holder.ProcessId /T /F 2>&1 | Out-Null
    }

    foreach ($attempt in 1..10) {
        Start-Sleep -Milliseconds 400
        if (-not (Test-Port -Port $Port)) { return $true }
    }
    return -not (Test-Port -Port $Port)
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
                taskkill /PID $process.Id /T /F 2>&1 | Out-Null
            }
        } catch {
            Write-Warn2 "could not stop PID $($process.Id)"
        }
    }

    # Killing tracked PIDs is not enough on its own. npm.cmd is a shim: it spawns
    # node and exits immediately, so by shutdown time the tracked PID is long
    # dead, its child has been reparented, and `taskkill /T` on the dead PID
    # finds nothing to kill. The real server survives, still holding the port -
    # which is exactly how a later run ends up reporting "port already in use"
    # against a PID that no longer exists.
    #
    # So finish the job against the ports themselves. Anything still listening
    # here is ours: the preflight check refuses to start when these ports are
    # occupied, so nothing else can have claimed them since.
    $ports = @($ApiPort)
    if (-not $Prod) { $ports += $UiPort }
    foreach ($port in $ports) {
        if (Test-Port -Port $port) {
            if (Clear-Port -Port $port) {
                Write-Ok "released port $port"
            } else {
                Write-Warn2 "port $port is still held - run: .\start.ps1 -Force"
            }
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
    Write-Warn2 "-NoOllama: skipping. Recorded demo answers replay; anything else runs model-free (llm_ok=false)"
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
    if (-not (Test-Port -Port $port)) { continue }

    $holder = Get-PortHolder -Port $port

    if ($Force) {
        Write-Warn2 "port $port held by $($holder.Describe()) - reclaiming (-Force)"
        if (Clear-Port -Port $port) {
            Write-Ok "port $port freed"
            continue
        }
        Write-Bad "could not free port $port"
        exit 1
    }

    Write-Bad "port $port is already in use - held by $($holder.Describe())"
    if ($holder.IsOurBackend) {
        Write-Host "         It is answering /api/health, so this looks like a backend of ours" -ForegroundColor Yellow
        Write-Host "         left over from an earlier run." -ForegroundColor Yellow
    }
    if ($holder.Stale) {
        # The common and confusing case. uvicorn --reload spawns a child; kill the
        # parent and the child inherits the socket, but the connection table still
        # names the dead parent. taskkill on that PID then says "not found" while
        # the port stays busy, which looks like a broken machine rather than an
        # orphaned process.
        Write-Host "         The PID in the connection table no longer exists - a child process" -ForegroundColor Yellow
        Write-Host "         inherited the socket. taskkill on that PID will say 'not found'." -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "         Reclaim it:   .\start.ps1 -Force" -ForegroundColor Yellow
    Write-Host "         Or sidestep:  .\start.ps1 -ApiPort 8010 -UiPort 5183" -ForegroundColor Yellow
    exit 1
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
    # a service dies on its own rather than sitting on a stack that is half up.
    #
    # Watch the PORTS, not the tracked processes. npm.cmd is a launcher shim: it
    # spawns node and exits straight away, so a process-liveness check sees it
    # "exit" seconds after Vite comes up and tears down a perfectly healthy
    # stack - then leaves the real node and uvicorn orphaned, holding the ports,
    # which is how the next run inherits a phantom "port already in use".
    #
    # Whether the ports still answer is the thing actually worth knowing, and it
    # does not care how many shims sit between this script and the server.
    while ($true) {
        Start-Sleep -Seconds 3
        if (-not (Test-Port -Port $ApiPort)) {
            Write-Bad "api stopped answering on $ApiPort - check .logs\api.err.log"
            return
        }
        if (-not $Prod -and -not (Test-Port -Port $UiPort)) {
            Write-Bad "vite stopped answering on $UiPort - check .logs\ui.err.log"
            return
        }
    }
} finally {
    Stop-All
}
