@echo off
chcp 65001 > nul
cls

echo ========================================================================
echo Sentirax Auto Trading Bot
echo ========================================================================
echo.
echo Current Time: %date% %time%
echo.
echo WARNING: Paper Trading Mode
echo WARNING: Real orders will be executed for TOP 20 stocks (14 qualified)
echo.
echo Press any key to continue...
pause > nul

cd /d "%~dp0"

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run trading bot
python scripts\auto_trading_bot.py

echo.
echo.
echo Execution complete! You can close this window.
pause
