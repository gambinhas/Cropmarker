# Start Label Studio (no Docker) for the Cropmarker dataset
# Run from PowerShell:  .\start_labelstudio.ps1

Set-Location "$PSScriptRoot"

$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = (Get-Location).Path

& .\.venv-labelstudio\Scripts\label-studio.exe start --port 8080 --no-browser
