@echo off
setlocal
chcp 65001 >nul
title HalgOM Screener
cd /d "%~dp0"
call :running
if not errorlevel 1 goto :open

rem --- 1. uv: installs Python + packages ---
where uv >nul 2>nul
if errorlevel 1 (
    echo [1/3] Installing uv...
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

rem --- old git pull below: off, see NOTE ---
set "SKIP_UPDATE=%SCREENER_NO_UPDATE%"
set "SCREENER_NO_UPDATE=1"
rem -----
if not defined SCREENER_NO_UPDATE if exist ".git" (
    where git >nul 2>nul
    if not errorlevel 1 (
        echo [2/3] Checking for updates...
        git pull --ff-only --quiet || echo        update skipped - running the current version
    )
)

rem --- 2. restore SCREENER_NO_UPDATE (a user setting: skip updates); find uv ---
set "SCREENER_NO_UPDATE=%SKIP_UPDATE%"
set "SKIP_UPDATE="
set "UV="
where uv >nul 2>nul
if not errorlevel 1 set "UV=1"

rem --- 3. Python environment (first run takes 1-2 minutes) ---
set "PY=screener\.venv\Scripts\python.exe"
rem A venv copied from another PC (Drive, ZIP) points at a Python that is not here: rebuild it
if exist "%PY%" (
    "%PY%" -c "import sys" >nul 2>nul
    if errorlevel 1 rmdir /s /q "screener\.venv"
)
if not exist "%PY%" (
    echo [2/3] First run - preparing the Python environment, this takes a minute or two...
    if defined UV (uv venv --python 3.12 screener\.venv --quiet) else (call :syspython)
)
if not exist "%PY%" goto :fail
if defined UV (uv pip install --python screener\.venv -r screener\requirements.lock --quiet) else ("%PY%" -m pip install --quiet --disable-pip-version-check -r screener\requirements.lock)
if errorlevel 1 goto :fail

rem --- 4. updates - ONE line, no parentheses (a folder name may contain them); see NOTE ---
if not defined SCREENER_NO_UPDATE if not defined SCREENER_UPDATED echo [3/3] Checking for updates... & "%PY%" -m screener.update & if errorlevel 3 set "SCREENER_UPDATED=1" & cmd /c ""%~f0"" & exit /b

rem Skip Streamlit's one-time "enter your email" prompt
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    mkdir "%USERPROFILE%\.streamlit" 2>nul
    > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
    >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

rem Desktop shortcut, created once - delete screener\_data\shortcut.done to create it again
if not exist "screener\_data\shortcut.done" (
    powershell -NoProfile -Command "$l = Join-Path ([Environment]::GetFolderPath('Desktop')) 'HalgOM Screener.lnk'; $s = (New-Object -ComObject WScript.Shell).CreateShortcut($l); $s.TargetPath = '%~f0'; $s.WorkingDirectory = '%~dp0'; $s.Save()" >nul 2>nul
    if not errorlevel 1 (
        mkdir "screener\_data" 2>nul
        > "screener\_data\shortcut.done" echo created
        echo  Added a "HalgOM Screener" shortcut to the desktop.
    )
)

echo.
echo  HalgOM Screener is starting - the browser will open in a few seconds.
echo  To close it: click Exit at the top of the screener, or close this window.
echo  It also closes by itself 15 minutes after the last browser tab is closed.
echo.
"%PY%" -m streamlit run screener\app.py --server.port 8501 --browser.gatherUsageStats false --theme.base light --client.toolbarMode minimal --server.runOnSave false
if errorlevel 1 goto :fail
goto :eof

:running
rem errorlevel 0 when a screener already answers on port 8501 - then just open the browser
powershell -NoProfile -Command "try { if ((Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://localhost:8501/_stcore/health).Content -eq 'ok') { exit 0 } } catch {}; exit 1" >nul 2>nul
exit /b %errorlevel%

:open
echo  HalgOM Screener is already running - opening it in the browser.
if not defined SCREENER_NO_BROWSER start "" http://localhost:8501
exit /b 0

:syspython
rem uv could not be installed: build the environment with the Python installed on this PC
echo        uv is not available - using the Python installed on this PC
py -3 -m venv screener\.venv >nul 2>nul
if not exist "%PY%" python -m venv screener\.venv >nul 2>nul
exit /b 0

:fail
echo.
echo  Something went wrong. Send the screenshot of this window, and the file
echo  screener\_data\logs\screener.log if it exists, to Ori.
pause
exit /b 1

rem NOTE - self-update. Launchers before 2026-10-19 ran git pull in step 2, which can rewrite
rem THIS file while cmd runs it; cmd then continues at the same byte offset in the new file.
rem So that old update block - "if not defined SCREENER_NO_UPDATE if exist" - stays
rem byte-identical and ends at byte 785, and the text above it keeps its total length
rem (adjust the "rem ---" padding line). The block is now switched off by
rem SCREENER_NO_UPDATE=1 just before it; step 4 runs screener\update.py instead, which needs
rem no git. update.py may rewrite this file too, so step 4 is a single line: cmd has read the
rem whole line before it runs, and the line itself restarts the launcher (cmd /c) when
rem update.py exits with 3. tests/test_screener.py checks all of this.
