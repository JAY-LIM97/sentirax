@echo off
chcp 65001 > nul
cls

echo ========================================================================
echo Sentirax Trading System - ALL BOTS
echo ========================================================================
echo.
echo Current Time: %date% %time%
echo.
echo [1] Swing Trading  - TOP 20 stocks (14 models, 500-day backtested)
echo [2] Scalping Bot    - Surging stocks (11 models, 1-min intraday)
echo.
echo Both will run simultaneously in separate windows.
echo.
echo Press any key to launch...
pause > nul

cd /d "%~dp0"

REM Launch Swing Trading in new window
start "Sentirax - Swing Trading" cmd /k "chcp 65001 > nul && cd /d "%~dp0" && call venv\Scripts\activate.bat && python scripts\auto_trading_bot.py && pause"

REM Wait 3 seconds to avoid API token conflict
timeout /t 3 /nobreak > nul

REM Launch Scalping Bot in new window
start "Sentirax - Scalping Bot" cmd /k "chcp 65001 > nul && cd /d "%~dp0" && call venv\Scripts\activate.bat && python scripts\scalping_bot.py && pause"

echo.
echo Both bots launched in separate windows!
echo You can close this window.
echo.
pause
