@echo off
rem Create a shortcut for Echo.exe on the OneDrive Desktop

rem Create a temporary VBScript file
echo Set oWS = WScript.CreateObject("WScript.Shell") > "%temp%\CreateShortcut.vbs"
echo sLinkFile = oWS.ExpandEnvironmentStrings("%%UserProfile%%\OneDrive\Desktop\Echo.lnk") >> "%temp%\CreateShortcut.vbs"
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%temp%\CreateShortcut.vbs"
echo oLink.TargetPath = "C:\Program Files\Echo\Echo.exe" >> "%temp%\CreateShortcut.vbs"
echo oLink.WorkingDirectory = "C:\Program Files\Echo" >> "%temp%\CreateShortcut.vbs"
echo oLink.IconLocation = "%~dp0ECHO.ico" >> "%temp%\CreateShortcut.vbs"
echo oLink.Save >> "%temp%\CreateShortcut.vbs"

rem Run the VBScript to create the shortcut
cscript //nologo "%temp%\CreateShortcut.vbs"

rem Clean up the temporary VBScript file
del "%temp%\CreateShortcut.vbs"

call "%~dp0Setup\create_env.bat"
call "%~dp0Setup\packages.bat"
call "%~dp0Setup\path.bat"