@echo off
setlocal EnableExtensions
title GoldStraddle Panel

rem ---------------------------------------------------------------------
rem  GoldStraddle panel - one-double-click launcher.
rem
rem  Resolves its own folder from %~dp0 (which on this machine contains
rem  both spaces and an "&", so every expansion below is quoted), finds a
rem  Python 3 interpreter, and hands over to panel\launch.py.
rem
rem  On failure it prints the reason and pauses, so a double-clicked
rem  window never vanishes before the user can read the error.
rem ---------------------------------------------------------------------

cd /d "%~dp0"
if errorlevel 1 (
    echo ERROR: could not switch to "%~dp0"
    goto :failed
)

set "LAUNCHER=%~dp0panel\launch.py"
if not exist "%LAUNCHER%" (
    echo ERROR: cannot find "%LAUNCHER%"
    echo The panel folder looks incomplete - panel\launch.py is missing.
    goto :failed
)

rem Find a usable interpreter: bare "python" first, then the py launcher.
rem Each candidate is actually executed, so the Microsoft Store stub named
rem python.exe (which exits non-zero) is skipped rather than used.
set "PY="
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
if not errorlevel 1 set "PY=python"
if not defined PY (
    py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY=py -3"
)
if not defined PY (
    echo ERROR: no working Python 3.9+ interpreter found.
    echo Tried "python" and "py -3".
    echo.
    echo Install Python 3 from https://www.python.org/downloads/ and tick
    echo "Add python.exe to PATH" during setup, then run this file again.
    goto :failed
)

echo Starting the GoldStraddle panel using: %PY%
echo Folder: "%~dp0"
echo.
%PY% "%LAUNCHER%" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo ERROR: the panel exited with code %RC%.
    goto :failed
)
endlocal
exit /b 0

:failed
echo.
echo ---------------------------------------------------------------------
echo The panel did not start. The message above explains why.
echo ---------------------------------------------------------------------
pause
endlocal
exit /b 1
