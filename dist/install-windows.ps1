################################################################################
# DREAM ML - Windows Installation Script
################################################################################
# This script installs and runs DREAM ML on Windows 10/11
#
# Requirements:
#   - Windows 10/11 (64-bit)
#   - Docker Desktop for Windows
#   - WSL 2 (Windows Subsystem for Linux)
#
# Usage:
#   .\install-windows.ps1
#   Or right-click install-windows.bat and "Run as Administrator"
################################################################################

# Set strict mode
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

################################################################################
# Color functions for output
################################################################################
function Write-InfoLog {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-SuccessLog {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-WarningLog {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-ErrorLog {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-StepLog {
    param([string]$Message)
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Header {
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Magenta
    Write-Host "                                                                " -ForegroundColor Magenta
    Write-Host "                   DREAM ML - Windows Installer                 " -ForegroundColor Magenta
    Write-Host "                       Windows 10/11                            " -ForegroundColor Magenta
    Write-Host "                                                                " -ForegroundColor Magenta
    Write-Host "================================================================" -ForegroundColor Magenta
    Write-Host ""
}

################################################################################
# Error handler
################################################################################
function Exit-WithError {
    param([string]$Message)
    Write-ErrorLog $Message
    Write-ErrorLog "Installation failed. Please check the errors above."
    Write-InfoLog "For troubleshooting, see README.txt"
    Read-Host "Press Enter to exit"
    exit 1
}

################################################################################
# Cleanup function
################################################################################
function Invoke-Cleanup {
    Write-StepLog "Cleanup Mode"

    Write-WarningLog "This will stop and remove all DREAM ML containers and data."
    Write-WarningLog "All experiments, MLflow runs, and uploaded files will be deleted."
    $confirmation = Read-Host "Are you sure you want to continue? (yes/no)"

    if ($confirmation -ne "yes") {
        Write-InfoLog "Cleanup cancelled"
        exit 0
    }

    Write-InfoLog "Stopping containers..."
    try {
        docker compose down -v
        Write-SuccessLog "Containers stopped and volumes removed"
    }
    catch {
        Write-WarningLog "Failed to stop containers (they may not be running)"
    }

    Write-InfoLog "Removing images..."
    try {
        docker images | Select-String "tomasmanriquez480/dreaml-ml" | ForEach-Object {
            $line = $_ -split '\s+'
            $image = "$($line[0]):$($line[1])"
            docker rmi $image
        }
        Write-SuccessLog "Images removed"
    }
    catch {
        Write-WarningLog "Failed to remove some images"
    }

    Write-SuccessLog "Cleanup complete"
    Read-Host "Press Enter to exit"
    exit 0
}

################################################################################
# Main installation
################################################################################
function Main {
    param([string[]]$Args)

    Write-Header

    # Check for cleanup flag
    if ($Args -contains "--cleanup" -or $Args -contains "-c") {
        Invoke-Cleanup
    }

    ############################################################################
    # Step 1: System Check
    ############################################################################
    Write-StepLog "Step 1: System Verification"

    Write-InfoLog "Checking operating system..."
    $os = Get-CimInstance Win32_OperatingSystem
    if ($os.Caption -notmatch "Windows (10|11)") {
        Exit-WithError "This script requires Windows 10 or Windows 11"
    }
    Write-SuccessLog "Operating system: $($os.Caption)"

    Write-InfoLog "Checking architecture..."
    if ($env:PROCESSOR_ARCHITECTURE -ne "AMD64") {
        Write-WarningLog "Expected AMD64 architecture, found: $env:PROCESSOR_ARCHITECTURE"
    }
    else {
        Write-SuccessLog "Architecture: AMD64 64-bit"
    }

    Write-InfoLog "Checking WSL 2..."
    try {
        $wslVersion = wsl --status 2>&1
        if ($wslVersion -match "WSL 2") {
            Write-SuccessLog "WSL 2 is available"
        }
        else {
            Write-WarningLog "WSL 2 may not be configured. Docker Desktop requires WSL 2."
        }
    }
    catch {
        Write-WarningLog "Could not verify WSL status. Docker Desktop requires WSL 2."
        Write-InfoLog "To install WSL 2, run: wsl --install"
    }

    ############################################################################
    # Step 2: Check Prerequisites
    ############################################################################
    Write-StepLog "Step 2: Checking Prerequisites"

    Write-InfoLog "Checking if Docker is installed..."
    try {
        $dockerVersion = docker --version
        Write-SuccessLog "Docker is installed: $dockerVersion"
    }
    catch {
        Write-ErrorLog "Docker is not installed"
        Write-Host ""
        Write-Host "Please install Docker Desktop for Windows:"
        Write-Host "  1. Visit: https://www.docker.com/products/docker-desktop"
        Write-Host "  2. Download Docker Desktop for Windows"
        Write-Host "  3. Install and start Docker Desktop"
        Write-Host "  4. Ensure WSL 2 is enabled (wsl --install)"
        Write-Host "  5. Re-run this script"
        Write-Host ""
        Read-Host "Press Enter to exit"
        exit 1
    }

    Write-InfoLog "Checking if Docker daemon is running..."
    try {
        docker info | Out-Null
        Write-SuccessLog "Docker daemon is running"
    }
    catch {
        Exit-WithError "Docker daemon is not running. Please start Docker Desktop and try again."
    }

    Write-InfoLog "Checking Docker Compose..."
    try {
        $composeVersion = docker compose version
        Write-SuccessLog "Docker Compose is available: $composeVersion"
    }
    catch {
        Exit-WithError "Docker Compose is not available. Please update Docker Desktop."
    }

    ############################################################################
    # Step 3: Verify Required Files
    ############################################################################
    Write-StepLog "Step 3: Verifying Installation Files"

    $ScriptDir = $PSScriptRoot

    Write-InfoLog "Checking for docker-compose.yml..."
    if (-not (Test-Path "$ScriptDir\docker-compose.yml")) {
        Exit-WithError "docker-compose.yml not found in $ScriptDir"
    }
    Write-SuccessLog "Found docker-compose.yml"

    Write-InfoLog "Checking for .env file..."
    if (-not (Test-Path "$ScriptDir\.env")) {
        Exit-WithError ".env file not found in $ScriptDir"
    }
    Write-SuccessLog "Found .env file"

    ############################################################################
    # Step 3a: Setup Experimentos Directory
    ############################################################################
    Write-StepLog "Step 3a: Setting Up Experimentos Directory"

    $ExperimentosPath = Join-Path $ScriptDir "experimentos"

    Write-InfoLog "Creating experimentos directory for experiment data..."
    if (-not (Test-Path $ExperimentosPath)) {
        New-Item -ItemType Directory -Path $ExperimentosPath | Out-Null
        Write-SuccessLog "Created experimentos directory"
    }
    else {
        Write-SuccessLog "Experimentos directory already exists"
    }

    # Docker Desktop for Windows (with WSL2) handles permissions automatically
    # No manual permission setting needed
    Write-InfoLog "Experiment files will be accessible at: $ExperimentosPath"
    Write-InfoLog "You can access this folder from File Explorer"

    ############################################################################
    # Step 4: Pull Docker Images
    ############################################################################
    Write-StepLog "Step 4: Pulling Docker Images"

    Set-Location $ScriptDir

    Write-InfoLog "Pulling images from Docker Hub..."
    Write-InfoLog "This may take several minutes depending on your internet connection..."
    Write-Host ""

    try {
        docker compose pull
        Write-SuccessLog "Images pulled successfully"
    }
    catch {
        Exit-WithError "Failed to pull images. Check your internet connection and Docker Hub access."
    }

    ############################################################################
    # Step 5: Start Services
    ############################################################################
    Write-StepLog "Step 5: Starting Services"

    Write-InfoLog "Starting containers in detached mode..."
    try {
        docker compose up -d
        Write-SuccessLog "Containers started"
    }
    catch {
        Exit-WithError "Failed to start containers"
    }

    ############################################################################
    # Step 6: Wait for Services to be Healthy
    ############################################################################
    Write-StepLog "Step 6: Waiting for Services to be Ready"

    Write-InfoLog "Waiting for backend to be healthy (this may take 30-60 seconds)..."
    $timeout = 120
    $elapsed = 0
    $backendHealthy = $false

    while ($elapsed -lt $timeout) {
        $status = docker compose ps
        if ($status -match "backend.*healthy") {
            $backendHealthy = $true
            break
        }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 2
        $elapsed += 2
    }
    Write-Host ""

    if ($backendHealthy) {
        Write-SuccessLog "Backend is healthy"
    }
    else {
        Write-WarningLog "Backend health check timeout - checking status..."
        docker compose ps
        Write-WarningLog "Backend may still be starting. Check logs below."
    }

    Write-InfoLog "Waiting for frontend to be healthy..."
    $elapsed = 0
    $frontendHealthy = $false

    while ($elapsed -lt $timeout) {
        $status = docker compose ps
        if ($status -match "frontend.*healthy") {
            $frontendHealthy = $true
            break
        }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 2
        $elapsed += 2
    }
    Write-Host ""

    if ($frontendHealthy) {
        Write-SuccessLog "Frontend is healthy"
    }
    else {
        Write-WarningLog "Frontend health check timeout - checking status..."
        docker compose ps
        Write-WarningLog "Frontend may still be starting. Check logs below."
    }

    ############################################################################
    # Step 7: Display Container Status
    ############################################################################
    Write-StepLog "Step 7: Container Status"

    Write-InfoLog "Current container status:"
    docker compose ps

    ############################################################################
    # Step 8: Display Logs
    ############################################################################
    Write-StepLog "Step 8: Recent Container Logs"

    Write-InfoLog "Backend logs (last 20 lines):"
    Write-Host "----------------------------------------"
    docker compose logs --tail=20 backend
    Write-Host "----------------------------------------"
    Write-Host ""

    Write-InfoLog "Frontend logs (last 20 lines):"
    Write-Host "----------------------------------------"
    docker compose logs --tail=20 frontend
    Write-Host "----------------------------------------"
    Write-Host ""

    ############################################################################
    # Step 9: Check for Errors
    ############################################################################
    Write-StepLog "Step 9: Error Detection"

    Write-InfoLog "Checking for errors in logs..."
    $errorCount = 0

    $backendLogs = docker compose logs backend | Select-String -Pattern "error|exception|failed" | Where-Object { $_ -notmatch "Failed to find" } | Select-Object -Last 5
    if ($backendLogs) {
        $backendLogs
        $errorCount++
        Write-WarningLog "Found errors in backend logs (see above)"
    }
    else {
        Write-SuccessLog "No critical errors in backend logs"
    }

    $frontendLogs = docker compose logs frontend | Select-String -Pattern "error|exception|failed" | Select-Object -Last 5
    if ($frontendLogs) {
        $frontendLogs
        $errorCount++
        Write-WarningLog "Found errors in frontend logs (see above)"
    }
    else {
        Write-SuccessLog "No critical errors in frontend logs"
    }

    if ($errorCount -gt 0) {
        Write-WarningLog "Some errors were detected. The application may still work."
        Write-InfoLog "Check the full logs with: docker compose logs"
    }

    ############################################################################
    # Step 10: Success Message & Access Instructions
    ############################################################################
    Write-StepLog "Installation Complete!"

    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "                                                                " -ForegroundColor Green
    Write-Host "                 INSTALLATION SUCCESSFUL                        " -ForegroundColor Green
    Write-Host "                                                                " -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host ""

    Write-SuccessLog "DREAM ML is now running!"
    Write-Host ""
    Write-Host "Access the application at:"
    Write-Host ""
    Write-Host "  Frontend (Web UI):     http://localhost:5173" -ForegroundColor Cyan
    Write-Host "  Backend API:           http://localhost:8000" -ForegroundColor Cyan
    Write-Host "  MLflow UI:             http://localhost:5000" -ForegroundColor Cyan
    Write-Host "  API Documentation:     http://localhost:8000/api/docs/" -ForegroundColor Cyan
    Write-Host ""

    Write-InfoLog "Useful commands:"
    Write-Host ""
    Write-Host "  View logs:           docker compose logs -f"
    Write-Host "  Stop services:       docker compose stop"
    Write-Host "  Start services:      docker compose start"
    Write-Host "  Restart services:    docker compose restart"
    Write-Host "  Remove everything:   .\install-windows.ps1 -cleanup"
    Write-Host ""
    Write-Host "  Access experiments:  explorer $ScriptDir\experimentos"
    Write-Host "  Backup experiments:  Copy-Item -Recurse experimentos\ experimentos-backup\"
    Write-Host ""

    Write-InfoLog "For troubleshooting, see README.txt"
    Write-Host ""

    # Open browser (optional)
    $openBrowser = Read-Host "Would you like to open the application in your browser? (y/N)"
    if ($openBrowser -eq "y" -or $openBrowser -eq "Y") {
        Write-InfoLog "Opening browser..."
        Start-Process "http://localhost:5173"
    }

    Write-SuccessLog "Installation script completed successfully!"
    Write-Host ""
    Read-Host "Press Enter to exit"
}

################################################################################
# Entry point
################################################################################
try {
    Main -Args $args
}
catch {
    Write-ErrorLog "Unexpected error: $_"
    Write-ErrorLog $_.ScriptStackTrace
    Read-Host "Press Enter to exit"
    exit 1
}
