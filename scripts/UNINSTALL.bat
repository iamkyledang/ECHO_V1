@echo off

REM Delete the Handbrake folder in drive C
rmdir /S /Q "C:\Handbrake"

REM Delete the desktop icon named Echo
del "%UserProfile%\OneDrive\Desktop\Echo.lnk"

REM Delete the myenv environment folder in the user's miniforge3 directory
rmdir /S /Q "%UserProfile%\miniforge3\envs\myenv"

REM Delete the Echo folder in C:\Program Files
rmdir /S /Q "C:\Program Files\Echo"

echo Uninstallation complete.