@echo off
REM FiloHFT native Tk dashboard launcher.
REM Double-click or run from cmd.exe. Window stays up until you close it.

cd /d "%~dp0python"
set PYTHONIOENCODING=utf-8
start "FiloHFT Dashboard" /B pythonw -m filo_hft.gui.dashboard_tk
