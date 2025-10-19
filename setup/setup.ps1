<#
Simple Setup: create a virtual environment and install a Jupyter kernel.
Usage example:
  PowerShell -ExecutionPolicy Bypass -File .\Setup-Project.ps1 -PythonVersion "3.10" -EnvName "har_env"
#>

param(
    [string]$PythonVersion = "3.10",
    [string]$EnvName = "har_env"
)

function WriteOk { param($m) Write-Host $m -ForegroundColor Green }
function WriteErr { param($m) Write-Host $m -ForegroundColor Red }

# 1) Locate Python (prefer py launcher)
$pythonPath = $null
try {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $out = & py -$PythonVersion -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { $pythonPath = $out.Trim() }
    }
} catch {}

if (-not $pythonPath) {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $pythonPath = (Get-Command python).Source
    } else {
        WriteErr "Python not found. Install Python $PythonVersion or ensure 'python' or 'py' is in PATH."
        exit 1
    }
}

WriteOk "Using Python: $pythonPath"

# 2) Create virtual environment
if (-not (Test-Path $EnvName)) {
    & "$pythonPath" -m venv $EnvName
    if ($LASTEXITCODE -ne 0) { WriteErr "Failed to create venv"; exit 1 }
    WriteOk "venv created: $EnvName"
} else {
    WriteOk "venv folder already exists: $EnvName (using existing)"
}

# 3) Path to venv python
$envPython = Join-Path $EnvName "Scripts\python.exe"
if (-not (Test-Path $envPython)) {
    WriteErr "venv python.exe not found: $envPython"
    exit 1
}

# 4) Install ipykernel (without activating the env)
& "$envPython" -m pip install --upgrade pip ipykernel
if ($LASTEXITCODE -ne 0) { WriteErr "Failed to install ipykernel"; exit 1 }

# 5) Create Jupyter kernel for this venv
$kernelName = $EnvName
$displayName = "Python ($EnvName)"
& "$envPython" -m ipykernel install --user --name $kernelName --display-name $displayName
if ($LASTEXITCODE -ne 0) { WriteErr "Failed to create Jupyter kernel"; exit 1 }

WriteOk "✅ Jupyter kernel installed: $displayName (internal name: $kernelName)"
Write-Host "To start Jupyter, run: jupyter notebook (or jupyter lab) and select '$displayName'."
