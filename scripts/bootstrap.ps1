$ErrorActionPreference = "Stop"

# Course-friendly bootstrap for Windows PowerShell
# - creates .env from .env.example
# - creates venv (./.venv)
# - installs project in editable mode with recommended extras

$Root = (Resolve-Path (Join-Path $PSScriptRoot ".."))
Set-Location $Root

if (!(Test-Path ".env")) {
  if (Test-Path ".env.example") {
    Copy-Item ".env.example" ".env"
    Write-Host "[bootstrap] Created .env from .env.example"
  } else {
    throw "[bootstrap] ERROR: .env.example not found"
  }
}

$Py = if ($env:PYTHON) { $env:PYTHON } else { "python" }

# Default extras for the course.
# - agents: optional smolagents backend
# - g4f: open/free LLM access (can be unstable, but useful for the course)
$Extras = "dev,agents,g4f"

foreach ($a in $args) {
  switch ($a) {
    "--gnn" { $Extras = "$Extras,gnn" }
    "--gnn-ext" { $Extras = "$Extras,gnn,gnn_ext" }
    "--gnn_ext" { $Extras = "$Extras,gnn,gnn_ext" }
    "--mm" { $Extras = "$Extras,mm" }
    "--agents-hf" { $Extras = "$Extras,agents_hf" }
    "--agents_hf" { $Extras = "$Extras,agents_hf" }
    "--hf" { $Extras = "$Extras,agents_hf" }
  }
}

if (!(Test-Path ".venv")) {
  & $Py -m venv .venv
  Write-Host "[bootstrap] Created venv at .venv"
}

& .\.venv\Scripts\Activate.ps1

python -m pip install -U pip wheel setuptools
python -m pip install -e ".[${Extras}]"
python -c "import scireason; print('[bootstrap] Import OK')"

Write-Host "`n[bootstrap] Done. Next: top-papers-graph demo-run`n"
Write-Host "Optional extras: --gnn / --gnn-ext / --mm / --agents-hf`n"
