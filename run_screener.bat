@echo off
setlocal
chcp 65001 >nul
title HalgOM Screener
cd /d "%~dp0"

rem --- 1. uv: installs Python + packages, no admin rights needed ---
where uv >nul 2>nul
if errorlevel 1 (
    echo [1/3] Installing uv...
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
where uv >nul 2>nul
if errorlevel 1 goto :fail

rem --- 2. updates (only when this folder is a git checkout; set SCREENER_NO_UPDATE=1 to skip) ---
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

echo.
echo  HalgOM Screener is starting - the browser will open in a few seconds.
echo  Keep this window open while you work. Close it to stop the screener.
echo.
screener\.venv\Scripts\python.exe -m streamlit run screener\app.py --browser.gatherUsageStats false --theme.base light --client.toolbarMode minimal --server.runOnSave false
goto :eof

:fail
echo.
echo  Something went wrong. Send the screenshot of this window, and the file
echo  screener\_data\logs\screener.log if it exists, to Ori.
pause
exit /b 1
