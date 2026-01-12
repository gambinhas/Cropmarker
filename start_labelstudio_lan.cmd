@echo off
setlocal

REM Start Label Studio (no Docker) accessible to other PCs on the same network (LAN).
REM Uses SUBST to avoid spaces/non-latin chars in DOCUMENT_ROOT.

cd /d "%~dp0"

set "LS_DRIVE=L:"

if exist "%LS_DRIVE%\" (
  echo Using existing %LS_DRIVE% mapping.
) else (
  echo Mapping %LS_DRIVE% to "%CD%" ...
  subst %LS_DRIVE% "%CD%"
  if errorlevel 1 (
    echo.
    echo Failed to map %LS_DRIVE%.
    echo Edit LS_DRIVE in start_labelstudio_lan.cmd or run as Administrator.
    exit /b 1
  )
)

set "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true"
set "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=%LS_DRIVE%\\"

set "LS_EXE=%CD%\.venv-labelstudio\Scripts\label-studio.exe"
if not exist "%LS_EXE%" (
  echo.
  echo Could not find: %LS_EXE%
  echo Create the venv in .venv-labelstudio and install label-studio first.
  exit /b 1
)

REM Get first non-APIPA, non-loopback IPv4 without using scriptblocks (more robust in .cmd parsing)
for /f "usebackq delims=" %%I in (`powershell.exe -NoProfile -Command "Get-NetIPAddress -AddressFamily IPv4 | Where-Object IPAddress -notlike '169.254*' | Where-Object IPAddress -ne '127.0.0.1' | Select-Object -First 1 -ExpandProperty IPAddress"`) do set "LAN_IP=%%I"

echo.
if defined LAN_IP (
  echo Label Studio [LAN] URL: http://%LAN_IP%:8080
) else (
  echo Label Studio [LAN] URL: http://<YOUR_PC_IP>:8080
)
echo Local-files root: %LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT%
echo.
echo NOTE: You may need to allow inbound TCP 8080 in Windows Firewall.
echo.

"%LS_EXE%" start --internal-host 0.0.0.0 --port 8080

echo.
echo Label Studio stopped (exit code: %ERRORLEVEL%).
pause
