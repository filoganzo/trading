@echo off
REM Watch-only. Never arms, never trades.
cd /d "%~dp0"
python run_headless.py --minutes 60
pause
