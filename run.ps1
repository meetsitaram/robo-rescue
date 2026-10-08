# Runs Python 3.12 from the project root (python is not on PATH on this machine).
$py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
if (-not (Test-Path $py)) { $py = "py" }
Push-Location $PSScriptRoot
try { & $py @args } finally { Pop-Location }
