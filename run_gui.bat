@echo off
setlocal
cd /d "%~dp0"
echo ======================================================================
echo Launching GHTSP Java Swing GUI Application...
echo ======================================================================
javac -encoding UTF-8 -d . gui\GHTSPApp.java
if %errorlevel% equ 0 (
    echo Starting GHTSP Java Swing Desktop App...
    start javaw -cp . gui.GHTSPApp
) else (
    echo [ERROR] Java compilation failed.
    pause
)
