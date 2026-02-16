@echo off
chcp 65001 > nul
cls

echo ========================================================================
echo Sentirax Scalping Bot v1.0
echo ========================================================================
echo.
echo Current Time: %date% %time%
echo.
echo WARNING: Paper Trading Mode
echo WARNING: Scalping bot for surging stocks (11 models)
echo WARNING: TP/SL auto-managed per model
echo.
echo Press any key to continue...
pause > nul

cd /d "%~dp0"

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run scalping bot
python scripts\scalping_bot.py

echo.
echo.
echo Execution complete! You can close this window.
pause
