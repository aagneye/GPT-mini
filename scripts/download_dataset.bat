@echo off
REM Downloads Alpaca + Dolly into ..\data\dataset.txt (requires Python + deps).
cd /d "%~dp0.."

where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
  python prepare_dataset.py
  goto :done
)

where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
  py -3 prepare_dataset.py
  goto :done
)

echo Python not found. Install Python 3 and run:
echo   pip install -r requirements.txt
echo   python prepare_dataset.py
exit /b 1

:done
exit /b %ERRORLEVEL%
