@echo off
setlocal
chcp 65001 >nul
title HalgOM Screener - setup
rem One-file installer: the only file the user downloads. It downloads the latest screener (main) from
rem GitHub into %LOCALAPPDATA%\HalgOM and starts it with run_screener.bat, which installs uv, Python
rem and the packages, turns the folder into a checkout of main (screener\update.py) and puts the
rem "HalgOM Screener" shortcut on the desktop. Running it again only starts the installed screener.
rem No admin rights and no git needed. The download is anonymous HTTPS: it needs the repo to be public.
rem HALGOM_DIR (an environment variable) installs into another folder; CI uses it.

rem --- 1. already installed? start it ---
rem The desktop shortcut points at the launcher of an existing install, also one unpacked from a ZIP
rem into any folder: start that one, so a second copy without the user's portfolios is not created
rem (no parenthesized blocks here: the paths may contain parentheses)
if defined HALGOM_DIR goto :folder
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$l = Join-Path ([Environment]::GetFolderPath('Desktop')) 'HalgOM Screener.lnk'; if (Test-Path -LiteralPath $l) { $t = (New-Object -ComObject WScript.Shell).CreateShortcut($l).TargetPath; if ($t -and (Test-Path -LiteralPath $t)) { $t } }"`) do set "EXISTING=%%i"
set "HALGOM_DIR=%LOCALAPPDATA%\HalgOM"
if not defined EXISTING goto :folder
echo  HalgOM Screener is already installed - starting it.
call "%EXISTING%"
exit /b

:folder
if exist "%HALGOM_DIR%\run_screener.bat" goto :start

rem --- 2. download and unpack main; run_screener.bat is copied last, so its presence means installed ---
echo.
echo  Installing HalgOM Screener into your user folder - AppData\Local\HalgOM.
echo  [setup] Downloading the screener...
set "HALGOM_TMP=%TEMP%\halgom_setup_%RANDOM%"
set "PS=$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue'; try {"
set "PS=%PS% [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;"
set "PS=%PS% $t = $env:HALGOM_TMP; $d = $env:HALGOM_DIR; $z = Join-Path $t 'main.zip'; $x = Join-Path $t 'x';"
set "PS=%PS% New-Item -ItemType Directory -Force -Path $t, $d | Out-Null;"
set "PS=%PS% Invoke-WebRequest -UseBasicParsing -TimeoutSec 120 -Uri 'https://github.com/orimaimon/HalgOM1/archive/refs/heads/main.zip' -OutFile $z;"
set "PS=%PS% Expand-Archive -LiteralPath $z -DestinationPath $x -Force;"
set "PS=%PS% $src = @(Get-ChildItem -LiteralPath $x -Directory)[0].FullName;"
set "PS=%PS% Get-ChildItem -LiteralPath $src -Force | Where-Object { $_.Name -ne 'run_screener.bat' } | Copy-Item -Destination $d -Recurse -Force;"
set "PS=%PS% Copy-Item -LiteralPath (Join-Path $src 'run_screener.bat') -Destination $d -Force;"
set "PS=%PS% Remove-Item -LiteralPath $t -Recurse -Force -ErrorAction SilentlyContinue"
set "PS=%PS% } catch { Write-Host ('  ' + $_.Exception.Message); exit 1 }"
powershell -NoProfile -ExecutionPolicy Bypass -Command "%PS%"
if errorlevel 1 goto :fail
if not exist "%HALGOM_DIR%\run_screener.bat" goto :fail
echo  [setup] Downloaded. Preparing the first run...

:start
rem --- 3. start: run_screener.bat goes to its own folder and does the rest ---
call "%HALGOM_DIR%\run_screener.bat"
exit /b

:fail
echo.
echo  The download did not work. Check the internet connection and run this file again.
echo  If it keeps failing, send a screenshot of this window to Ori.
pause
exit /b 1
