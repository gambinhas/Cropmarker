@echo off
setlocal

REM Start Label Studio (no Docker) with Local Files serving enabled.
REM This script uses SUBST to avoid spaces/non-latin chars in DOCUMENT_ROOT.

cd /d "%~dp0"

set "LS_DRIVE=L:"

REM If drive is already mapped, leave it as-is; otherwise map it to this folder.
if exist "%LS_DRIVE%\" (
  echo Using existing %LS_DRIVE% mapping.
) else (
  echo Mapping %LS_DRIVE% to "%CD%" ...
  subst %LS_DRIVE% "%CD%"
  if errorlevel 1 (
    echo.
    echo Failed to map %LS_DRIVE%.
    echo Edit LS_DRIVE in start_labelstudio.cmd or run as Administrator.
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

echo.
echo Label Studio will be available at: http://localhost:8080
echo Local-files root: %LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT%
echo.

"%LS_EXE%" start --port 8080
