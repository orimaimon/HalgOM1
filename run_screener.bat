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
where uv >nul 2>nul
if errorlevel 1 goto :fail

rem --- 2. updates (see NOTE at the end) ---
rem -------------------------
if not defined SCREENER_NO_UPDATE if exist ".git" (
    where git >nul 2>nul
    if not errorlevel 1 (
        echo [2/3] Checking for updates...
        git pull --ff-only --quiet || echo        update skipped - running the current version
    )
)

rem --- 3. Python environment (first run takes 1-2 minutes) ---
if not exist "screener\.venv\Scripts\python.exe" (
    echo [3/3] First run - preparing the Python environment, this takes a minute or two...
    uv venv --python 3.12 screener\.venv --quiet
    if errorlevel 1 goto :fail
)
uv pip install --python screener\.venv -r screener\requirements.lock --quiet
if errorlevel 1 goto :fail

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
screener\.venv\Scripts\python.exe -m streamlit run screener\app.py --server.port 8501 --browser.gatherUsageStats false --theme.base light --client.toolbarMode minimal --server.runOnSave false
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

:fail
echo.
echo  Something went wrong. Send the screenshot of this window, and the file
echo  screener\_data\logs\screener.log if it exists, to Ori.
pause
exit /b 1

rem NOTE - self-update: step 2 runs git pull, which can rewrite THIS file while cmd is
rem running it, and cmd then continues reading at the same byte offset in the new file.
rem So the update block - the "if not defined SCREENER_NO_UPDATE" block - must stay
rem byte-identical and must end at byte 785, where it ended in the version users first
rem installed. Edit freely below it; above it keep the total length unchanged by adjusting
rem the "rem---" padding line. tests/test_screener.py checks both.
