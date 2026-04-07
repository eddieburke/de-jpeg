@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   JPEG Restorer - Auto Update Script
echo ========================================
echo.
echo This will fetch the latest version from GitHub and update your local files.
echo.

set /p CONFIRM="Are you sure you want to update? (y/n): "
if /i not "%CONFIRM%"=="y" (
    echo Update cancelled.
    exit /b 0
)

echo.
echo Fetching latest changes...

cd /d "%~dp0"

git fetch origin

echo.
echo Checking for updates...

git log --oneline HEAD..origin/main > nul 2>&1
if errorlevel 1 (
    echo No updates available. You are already on the latest version.
    echo.
    pause
    exit /b 0
)

echo Updates found! Pulling changes...

git pull origin main

if errorlevel 1 (
    echo.
    echo ERROR: Failed to pull updates. There may be merge conflicts.
    echo Please resolve them manually and try again.
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Update complete!
echo ========================================
echo.
echo Latest changes have been applied.
echo You can now run the updated application.
echo.
pause