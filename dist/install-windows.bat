@echo off
REM ############################################################################
REM DREAM ML - Windows Installation Script Wrapper
REM ############################################################################
REM This batch file provides an easy way to run the PowerShell installation
REM script. Simply double-click this file or run it from Command Prompt.
REM ############################################################################

setlocal

REM Check if running as Administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo.
    echo ========================================
    echo WARNING: Not running as Administrator
    echo ========================================
    echo.
    echo For best results, right-click this file and select "Run as Administrator"
    echo.
    echo Press any key to continue anyway, or close this window to exit...
    pause >nul
)

echo.
echo ============================================================
echo           DREAM ML - Windows Installation
echo ============================================================
echo.
echo Starting PowerShell installation script...
echo.

REM Check if PowerShell is available
where powershell >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: PowerShell is not available on this system.
    echo Please run install-windows.ps1 manually.
    pause
    exit /b 1
)

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Run PowerShell script with bypass execution policy
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install-windows.ps1" %*

REM Check if PowerShell script succeeded
if %errorLevel% neq 0 (
    echo.
    echo ========================================
    echo Installation encountered an error
    echo ========================================
    echo.
    echo If you see an execution policy error, try running this command:
    echo   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    echo.
    echo Then run this batch file again.
    echo.
    pause
    exit /b 1
)

echo.
echo Installation wrapper completed.
echo.

endlocal
