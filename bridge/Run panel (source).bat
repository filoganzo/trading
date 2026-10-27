@echo off
REM Run the panel from source - fallback if GoldBridge.exe has trouble.
cd /d "%~dp0"
python panel.py
if errorlevel 1 pause
