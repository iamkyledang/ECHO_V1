@echo off

REM Copy the Handbrake folder to C:\
xcopy /E /I /Y "%~dp0Handbrake" "C:\Handbrake"

REM Copy the main.exe file to the Desktop from one level higher than the batch file's directory
set "parentDir=%~dp0.."
copy "%parentDir%\main.exe" "%UserProfile%\Desktop\Echo.exe"

REM Retrieve the current SYSTEM PATH
for /f "tokens=2* delims= " %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path') do set currentPath=%%b

REM Construct the paths to miniforge3\Library\bin, miniforge3\Scripts, and Handbrake in the user's directory
set "userProfile=%UserProfile%"
set "miniforgeLibraryPath=%userProfile%\miniforge3\Library\bin"
set "miniforgeScriptsPath=%userProfile%\miniforge3\Scripts"
set "handbrakePath=C:\Handbrake"

REM Add the new paths to the SYSTEM PATH if they are not already present
call :AddPathIfNotExists "%miniforgeLibraryPath%"
call :AddPathIfNotExists "%miniforgeScriptsPath%"
call :AddPathIfNotExists "%handbrakePath%"

REM Set the new PATH
setx /M PATH "%currentPath%"

echo PATH updated successfully.

goto :eof

:AddPathIfNotExists
setlocal
set "pathToAdd=%~1"
echo %currentPath% | find /i "%pathToAdd%" >nul
if errorlevel 1 (
    set "currentPath=%currentPath%;%pathToAdd%"
)
endlocal & set "currentPath=%currentPath%"
goto :eof