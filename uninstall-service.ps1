# Uninstall the WhisperAPI Windows Service.
# Run from an elevated PowerShell prompt:
#   .\uninstall-service.ps1
#
# Removes the service registration from the SCM. Does NOT delete:
#   - The repo directory
#   - The venv
#   - Log files in .\logs\
#   - The Hugging Face model cache (~1.5 GB at %USERPROFILE%\.cache\huggingface)
#   - nssm.exe (left in place for re-install)
# Use -RemoveLocal to also delete logs and nssm.exe.

param(
    [switch] $RemoveLocal
)

$ErrorActionPreference = "Stop"

# --- elevate to admin if needed ---------------------------------------------
# Stop-Service / sc.exe delete both require administrator rights.
$identity  = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "This script needs admin rights. Triggering UAC..." -ForegroundColor Yellow
    $argList = @(
        "-NoExit",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$PSCommandPath`""
    )
    if ($RemoveLocal) { $argList += "-RemoveLocal" }
    Start-Process powershell -Verb RunAs -ArgumentList $argList
    exit
}

$ServiceName = "WhisperAPI"
$RepoDir     = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogsDir     = Join-Path $RepoDir "logs"
$LocalNssm   = Join-Path $RepoDir "nssm.exe"

# --- check service exists ---------------------------------------------------
$svc = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if (-not $svc) {
    Write-Host "Service '$ServiceName' is not installed - nothing to remove." -ForegroundColor Yellow
} else {
    # --- stop -----------------------------------------------------------
    if ($svc.Status -ne "Stopped") {
        Write-Host "Stopping $ServiceName..."
        try {
            Stop-Service -Name $ServiceName -Force -ErrorAction Stop
        } catch {
            # Stop-Service can throw if the service is in a transient state.
            # Fall through to the polling loop below.
            Write-Host "  (stop signal sent; waiting for service to settle)" -ForegroundColor DarkGray
        }
        # Poll until actually stopped, max 30s.
        $deadline = (Get-Date).AddSeconds(30)
        while ((Get-Date) -lt $deadline) {
            $cur = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
            if (-not $cur -or $cur.Status -eq "Stopped") { break }
            Start-Sleep -Milliseconds 500
        }
    }

    # --- delete ---------------------------------------------------------
    # Use sc.exe rather than NSSM so this script doesn't depend on nssm.exe
    # being present (the user might have already deleted it).
    Write-Host "Removing $ServiceName from the SCM..."
    & sc.exe delete $ServiceName | Out-Null

    # Poll until SCM forgets the service (it can linger briefly).
    $deadline = (Get-Date).AddSeconds(15)
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue)) { break }
        Start-Sleep -Milliseconds 500
    }

    if (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue) {
        Write-Host "WARNING: '$ServiceName' is still registered. A reboot may be required." -ForegroundColor Yellow
    } else {
        Write-Host "Service removed." -ForegroundColor Green
    }
}

# --- optional local cleanup -------------------------------------------------
if ($RemoveLocal) {
    if (Test-Path $LogsDir) {
        Write-Host "Removing logs directory: $LogsDir"
        Remove-Item -Recurse -Force $LogsDir
    }
    if (Test-Path $LocalNssm) {
        Write-Host "Removing local nssm.exe: $LocalNssm"
        Remove-Item -Force $LocalNssm
    }
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
if (-not $RemoveLocal) {
    Write-Host "Logs are preserved at: $LogsDir"
    Write-Host "Run with -RemoveLocal to also delete logs and nssm.exe."
}
Write-Host ""
Write-Host "To reinstall: .\install-service.ps1"
