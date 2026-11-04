@echo off
echo ======================================================================
echo Launching GHTSP Java Swing GUI Application...
echo ======================================================================
cd /d "%~dp0\.."
javac -encoding UTF-8 -d . gui\GHTSPApp.java
if %errorlevel% equ 0 (
    echo Starting GHTSP GUI...
    start javaw -cp . gui.GHTSPApp
) else (
    echo Compilation failed.
    pause
)
