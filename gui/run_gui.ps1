# PowerShell Launcher for GHTSP Java Swing GUI
$ProjectDir = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectDir
Write-Host "Compiling GHTSP Java Swing GUI..." -ForegroundColor Cyan
javac -d . gui/GHTSPApp.java
if ($LASTEXITCODE -eq 0) {
    Write-Host "Launching GHTSP GUI Window..." -ForegroundColor Green
    Start-Process javaw -ArgumentList "-cp . gui.GHTSPApp"
} else {
    Write-Host "Compilation failed with error code $LASTEXITCODE" -ForegroundColor Red
}
