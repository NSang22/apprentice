<#
Demo startup on the Windows laptop: SSH tunnel to the Jetson, colors.json, brain.server, ui/serve.py.

    powershell -ExecutionPolicy Bypass -File scripts\laptop_up.ps1 -Mode human
    powershell -ExecutionPolicy Bypass -File scripts\laptop_up.ps1 -Mode robot
    powershell -ExecutionPolicy Bypass -File scripts\laptop_up.ps1 -StartJetson -Lesson lessons\stack-test-v2.json
    powershell -ExecutionPolicy Bypass -File scripts\laptop_up.ps1 -Sim -Lesson lessons\stack-test-v2.json

-StartJetson: ssh in first, git pull on the Jetson, run scripts/jetson_up.sh there (--no-robot in
human mode). Set JETSON_DISPLAY=:0 in jetson.env to get the perception debug window on the Jetson's
own monitor. Use an SSH key (README section 8) or you type the password three times.
-Sim: no Jetson at all. The brain runs a simulated board and the fake arm (--grid sim --robot fake,
robot mode), plus the UI.
-Lesson: lesson JSON for the brain (default: the brain's demo lesson).

Run scripts/jetson_up.sh on the Jetson first. Reads JETSON_HOST, JETSON_USER (and optional
JETSON_REPO, default "apprentice") from scripts\jetson.env (copy scripts\jetson.env.example).
Python is $env:PYTHON when set, else python on PATH.

Steps: stop whatever listens on 8765, 8766, 8770, 8000 (old tunnels, brains, UI servers); open the
tunnel (local 8765 and 8770 to the Jetson) in its own window; scp colors.json; wait for a GridState
through the tunnel; robot mode: check GET :8770/status; start brain.server and ui/serve.py in their
own windows; open http://localhost:8000. Stops at the first step that fails.
#>
param(
    [ValidateSet("human", "robot")]
    [string]$Mode = "human",
    [string]$Lesson = "",
    [switch]$StartJetson,
    [switch]$Sim
)

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

function Ok([string]$m)   { Write-Host "OK    $m" -ForegroundColor Green }
function Warn([string]$m) { Write-Host "WARN  $m" -ForegroundColor Yellow }
function Fail([string]$m) { Write-Host "FAIL  $m" -ForegroundColor Red; exit 1 }

function Test-Port([int]$Port) {
    $c = New-Object System.Net.Sockets.TcpClient
    try { return $c.ConnectAsync("127.0.0.1", $Port).Wait(500) -and $c.Connected }
    catch { return $false }
    finally { $c.Close() }
}

function Wait-Port([int]$Port, [int]$TimeoutS) {
    $deadline = (Get-Date).AddSeconds($TimeoutS)
    while ((Get-Date) -lt $deadline) {
        if (Test-Port $Port) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Start-Window([string]$Title, [string]$Command) {
    $full = "`$host.UI.RawUI.WindowTitle = '$Title'; Set-Location '$RepoRoot'; $Command"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $full | Out-Null
}

# ---------------------------------------------------------------- config
$envFile = Join-Path $PSScriptRoot "jetson.env"
if ($Sim) { $envFile = $null }
if ($envFile -and -not (Test-Path $envFile)) { Fail "config: $envFile not found (copy scripts\jetson.env.example and fill it in)" }
$cfg = @{ JETSON_REPO = "apprentice"; JETSON_DISPLAY = "" }
if ($envFile) {
    foreach ($line in Get-Content $envFile) {
        if ($line -match '^\s*([A-Z_]+)\s*=\s*(.*?)\s*$') { $cfg[$Matches[1]] = $Matches[2].Trim('"') }
    }
    foreach ($k in "JETSON_HOST", "JETSON_USER") {
        if (-not $cfg[$k]) { Fail "config: $k missing in $envFile" }
    }
}
$Target = "$($cfg.JETSON_USER)@$($cfg.JETSON_HOST)"

if ($env:PYTHON) { $Python = $env:PYTHON } else {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $cmd) { Fail "python: not on PATH (or set `$env:PYTHON)" }
    $Python = $cmd.Source
}
foreach ($tool in "ssh", "scp") {
    if ($null -eq (Get-Command $tool -ErrorAction SilentlyContinue)) { Fail "$tool not found (Settings > Optional features > OpenSSH Client)" }
}
$lessonArg = ""
if ($Lesson) {
    if (-not (Test-Path $Lesson)) { Fail "lesson: $Lesson not found" }
    $lessonArg = "--lesson $Lesson"
}
if ($Sim) { $Mode = "robot"; Ok "config: simulation, fake arm, python $Python $lessonArg" }
else { Ok "config: $Target, mode $Mode, python $Python $lessonArg" }

# ---------------------------------------------------------------- stop old listeners
foreach ($port in 8765, 8766, 8770, 8000) {
    $conns = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
    $procIds = @($conns | ForEach-Object { $_.OwningProcess } | Where-Object { $_ -gt 4 } | Sort-Object -Unique)
    foreach ($procId in $procIds) {
        $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
        $name = if ($proc) { $proc.ProcessName } else { "?" }
        try {
            Stop-Process -Id $procId -Force -ErrorAction Stop
            Ok "port ${port}: stopped $name (pid $procId)"
        } catch {
            Fail "port ${port}: could not stop $name (pid $procId): $($_.Exception.Message)"
        }
    }
}
Start-Sleep -Milliseconds 500
foreach ($port in 8765, 8766, 8770, 8000) {
    if (Test-Port $port) { Fail "port ${port}: still in use after stopping its owner" }
}
Ok "ports 8765 8766 8770 8000 free"

function Start-BrainAndUi([string]$BrainArgs) {
    Start-Window "apprentice brain" "& '$Python' -m brain.server $BrainArgs"
    if (-not (Wait-Port 8766 20)) { Fail "brain.server: ws :8766 not listening after 20 s (see the 'apprentice brain' window)" }
    Ok "brain.server: ws://localhost:8766 ($BrainArgs)"
    Start-Window "apprentice ui" "& '$Python' ui/serve.py"
    if (-not (Wait-Port 8000 15)) { Fail "ui/serve.py: :8000 not listening after 15 s (see the 'apprentice ui' window)" }
    Ok "ui/serve.py: http://localhost:8000"
    Start-Process "http://localhost:8000"
    Ok "browser opened. Space starts, R resets."
}

# ---------------------------------------------------------------- simulation (no Jetson)
if ($Sim) {
    Start-BrainAndUi "--mode robot --robot fake --grid sim $lessonArg"
    exit 0
}

# ---------------------------------------------------------------- Jetson services
if ($StartJetson) {
    $noRobot = if ($Mode -eq "human") { " --no-robot" } else { "" }
    $display = if ($cfg.JETSON_DISPLAY) { "DISPLAY=$($cfg.JETSON_DISPLAY) " } else { "" }
    $remote = "cd ~/$($cfg.JETSON_REPO) && (git pull --ff-only || echo git pull skipped) && ${display}scripts/jetson_up.sh$noRobot"
    Write-Host "ssh $Target : $remote"
    & ssh -o ServerAliveInterval=5 $Target "bash -lc '$remote'"
    if ($LASTEXITCODE -ne 0) { Fail "jetson_up.sh failed (exit $LASTEXITCODE), see its output above" }
    Ok "Jetson services up"
}

# ---------------------------------------------------------------- tunnel
$sshCmd = "ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=3 " +
          "-L 8765:127.0.0.1:8765 -L 8770:127.0.0.1:8770 $Target; Write-Host 'tunnel closed' -ForegroundColor Red"
Start-Window "apprentice tunnel" $sshCmd
if (-not (Wait-Port 8765 60)) { Fail "tunnel: local port 8765 not listening after 60 s (check the 'apprentice tunnel' window: password prompt, host key, or 'Connection refused')" }
Ok "tunnel: localhost:8765 and :8770 -> $Target"

# ---------------------------------------------------------------- colors.json
$remote = "${Target}:$($cfg.JETSON_REPO)/perception/calib/colors.json"
$local = Join-Path $RepoRoot "perception\calib\colors.json"
$scpOut = (& scp -q $remote $local 2>&1 | ForEach-Object { "$_" }) -join " "
if ($LASTEXITCODE -eq 0) { Ok "colors.json copied from $remote" }
else { Warn "colors.json: scp failed (exit $LASTEXITCODE): $scpOut. The UI uses default colors." }

# ---------------------------------------------------------------- GridState through the tunnel
function Wait-GridState([int]$TimeoutS) {
    $deadline = (Get-Date).AddSeconds($TimeoutS)
    $script:GridError = "no reply"
    while ((Get-Date) -lt $deadline) {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $Python
        $psi.Arguments = "-m tests.listen"
        $psi.WorkingDirectory = $RepoRoot
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.CreateNoWindow = $true
        $p = [System.Diagnostics.Process]::Start($psi)
        $errTask = $p.StandardError.ReadToEndAsync()
        $lineTask = $p.StandardOutput.ReadLineAsync()
        $found = $null
        while ((Get-Date) -lt $deadline) {
            if (-not $lineTask.Wait(500)) { continue }
            $line = $lineTask.Result
            if ($null -eq $line) { break }
            if ($line -match '^#\d') { $found = $line; break }
            $lineTask = $p.StandardOutput.ReadLineAsync()
        }
        if (-not $p.HasExited) { $p.Kill() }
        $p.WaitForExit(2000) | Out-Null
        if ($found) { return $found }
        $errLines = @($errTask.Result -split "`r?`n" | Where-Object { $_.Trim() })
        if ($errLines.Count -gt 0) { $script:GridError = $errLines[-1] }
        Start-Sleep -Seconds 1
    }
    return $null
}
$first = Wait-GridState 30
if ($null -eq $first) { Fail "GridState: none through the tunnel in 30 s. Last error: $script:GridError (is perception.publish running on the Jetson? scripts/jetson_up.sh)" }
Ok "GridState: $first"

# ---------------------------------------------------------------- robot
$robotArg = ""
if ($Mode -eq "robot") {
    try {
        $st = Invoke-RestMethod -Uri "http://localhost:8770/status" -TimeoutSec 5 -ErrorAction Stop
        Ok "robot.api: busy=$($st.busy) at_home=$($st.at_home) last_error=$($st.last_error)"
    } catch {
        Fail "robot.api: GET http://localhost:8770/status failed: $($_.Exception.Message) (is robot.api running on the Jetson?)"
    }
    $robotArg = "--robot http://localhost:8770"
}

# ---------------------------------------------------------------- brain + UI
Start-BrainAndUi "--mode $Mode $robotArg $lessonArg"
Ok "final check: python scripts/check.py"
