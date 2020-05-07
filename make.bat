@echo off

if "%1"=="venv" (
    echo Building venv...
    python -m pip install --user virtualenv
    python -m venv venv
    cmd /k ".\\venv\\Scripts\\activate & python -m pip install -r requirements.txt"
    echo venv built!
) else if "%1"=="clean" (
    echo Quitting venv...
    deactivate
    echo Cleaning venv...
    rmdir /S /Q ".\venv"
    echo Done!
) else if "%1"=="help" (
    echo usage: make.bat [venv] [clean] [help]
) else (
    echo Command not found
    echo usage: make.bat [venv] [clean] [help]
)