$ErrorActionPreference = "Stop"

# Install PyG extension wheels (pyg-lib, torch-scatter, torch-sparse, ...)
# using the official wheel index at https://data.pyg.org
#
# This script is optional. The repo works without it.

$Py = if ($env:PYTHON) { $env:PYTHON } else { "python" }

$TorchVer = & $Py -c "import torch; print(torch.__version__.split('+')[0])"
$CudaVer = & $Py -c "import torch; print(torch.version.cuda or 'cpu')"

if ($CudaVer -eq "cpu" -or $CudaVer -eq "None") {
  $CudaTag = "cpu"
} else {
  $CudaTag = "cu" + ($CudaVer -replace "\.", "")
}

$parts = $TorchVer.Split('.')
$BaseVer = "$($parts[0]).$($parts[1]).0"

$Url1 = "https://data.pyg.org/whl/torch-$TorchVer+$CudaTag.html"
$Url2 = "https://data.pyg.org/whl/torch-$BaseVer+$CudaTag.html"

& $Py -m pip install -U pip

$Ok = $false
foreach ($url in @($Url1, $Url2)) {
  Write-Host "[install_pyg_extensions] Trying: $url"
  try {
    & $Py -m pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f $url
    $Ok = $true
    break
  } catch {
    Write-Host "[install_pyg_extensions] Failed for $url, trying fallback..."
  }
}

if (-not $Ok) {
  throw "[install_pyg_extensions] ERROR: Could not install PyG extension wheels. See docs/gnn.md"
}

& $Py -m pip install -U torch_geometric
Write-Host "[install_pyg_extensions] Done."
