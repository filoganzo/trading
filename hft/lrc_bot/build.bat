@echo off
cd /d "%~dp0"
echo ================================================================
echo  LRC Bot  --  PyInstaller build
echo ================================================================
echo.

if not exist "%~dp0lrc_bot.py" (
    echo ================================================================
    echo  ERROR: lrc_bot.py not found next to this bat file.
    echo.
    echo  You are probably running this from INSIDE the zip archive.
    echo  Please EXTRACT ALL FILES first, then run build.bat again.
    echo ================================================================
    pause
    exit /b 1
)

echo [1/3] Installing dependencies...
pip install pyinstaller pyqt6 MetaTrader5 numpy pandas keyring
if %ERRORLEVEL% neq 0 (echo ERROR: pip install failed & pause & exit /b 1)

echo.
echo [2/3] Building executable...
python -m PyInstaller ^
  --onefile ^
  --console ^
  --name "LRC Bot" ^
  --add-data "%~dp0tickers.json;." ^
  --hidden-import keyring.backends.Windows ^
  --hidden-import win32timezone ^
  "%~dp0lrc_bot.py"
if %ERRORLEVEL% neq 0 (echo ERROR: PyInstaller build failed & pause & exit /b 1)

echo.
echo [3/3] Done!
echo.
echo  LRC Bot.exe is in the  dist\  folder.
echo  Share that single file with anyone running IC Markets MT5.
echo  They need no Python, no dependencies -- just the .exe.
echo.
echo  Config is saved to:  %%USERPROFILE%%\lrc_bot_config.json
echo  Trade log saved to:  %%USERPROFILE%%\lrc_bot_trades.csv
echo  Full log saved to:   %%USERPROFILE%%\lrc_bot.log
echo.
pause
