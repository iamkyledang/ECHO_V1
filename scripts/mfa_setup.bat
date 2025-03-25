@echo off

REM Define the path to the global_config.yaml file
set "userProfile=%UserProfile%"
set "configPath=%userProfile%\Documents\MFA\global_config.yaml"

REM Extract the existing temporary_directory path
set "tempDirLine="
for /f "tokens=*" %%i in ('type "%configPath%" ^| findstr /b "temporary_directory:"') do (
    set "tempDirLine=%%i"
)


REM Define the content to write to the global_config.yaml file
(
    echo auto_server: true
    echo blas_num_threads: 1
    echo bytes_limit: 400000000.0
    echo clean: true
    echo cleanup_textgrids: true
    echo database_limited_mode: false
    echo debug: false
    echo github_token: null
    echo num_jobs: 20
    echo overwrite: false
    echo profiles: {}
    echo quiet: false
    echo seed: 0
    echo single_speaker: false
    REM Skip 6 lines for temporary_directory
    echo use_mp: true
    echo use_postgres: false
    echo use_threading: true
    echo verbose: false
    echo temporary_directory: !!python/object/apply:pathlib.WindowsPath
    echo - C:\
    echo - Users
    echo - %USERNAME%
    echo - Documents
    echo - MFA
) > "%configPath%"

echo global_config.yaml updated successfully.
echo Installation complete.