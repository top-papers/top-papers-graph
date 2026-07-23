# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

[CmdletBinding()]
param(
    [string]$Action = "Help",
    [ValidateSet("fp16-primary", "nf4-sensitivity")][string]$Mode = "fp16-primary",
    [string]$Owner = "",
    [string]$Config = "",
    [string]$PreparedRun = "",
    [string]$InputSlug = "",
    [string]$BaseStateSlug = "",
    [string]$FinalStateSlug = "",
    [string]$BaseKernelSlug = "",
    [string]$TunedKernelSlug = "",
    [ValidateSet("base", "tuned")][string]$Arm = "base",
    [string]$StateSlug = "",
    [string]$StagingDir = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

if (-not $InputSlug) { $InputSlug = "vlm-ab-$Mode-input" }
if (-not $BaseStateSlug) { $BaseStateSlug = "vlm-ab-$Mode-base-state" }
if (-not $FinalStateSlug) { $FinalStateSlug = "vlm-ab-$Mode-final-state" }
if (-not $BaseKernelSlug) { $BaseKernelSlug = "vlm-ab-$Mode-base" }
if (-not $TunedKernelSlug) { $TunedKernelSlug = "vlm-ab-$Mode-tuned" }

$KaggleRoot = $PSScriptRoot
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $KaggleRoot "..\..\..")).Path
$BuildRoot = Join-Path $KaggleRoot "build"
$StagingRoot = Join-Path $KaggleRoot "staging"
$DownloadsRoot = Join-Path $KaggleRoot "downloads"
$WorkflowBuildRoot = Join-Path $BuildRoot $Mode
$WorkflowStagingRoot = Join-Path $StagingRoot $Mode
$WorkflowDownloadsRoot = Join-Path $DownloadsRoot $Mode
$CredentialPath = Join-Path $HOME ".kaggle\kaggle.json"
$KernelVersionIo = Join-Path $KaggleRoot "kernel_version_io.py"
$StateValidator = Join-Path $KaggleRoot "validate_state.py"
$script:AuthenticatedKaggleOwner = ""
$script:ExpectedVersionIoSha256 = ""
$script:ExpectedStateValidatorSha256 = ""

function Invoke-NativeCapture {
    param([string]$File, [string[]]$Arguments)
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $lines = @(& $File @Arguments 2>&1 | ForEach-Object { [string]$_ })
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    $text = $lines -join "`n"
    if ($text) { Write-Host $text }
    $failurePattern = "(?im)^\s*(error|fatal|failed|failure|traceback|apiexception)\s*[:\-]|" +
        "\bstatus\s*:\s*(error|failed|failure|cancelled|canceled)\b|" +
        "\b(dataset|kernel)(?:\s+version)?\s+(creation|upload|push)\s+(failed|failure|error)\b|" +
        "\bnot valid (dataset|competition|kernel|model) sources\b"
    if ($exitCode -ne 0 -or $text -match $failurePattern) {
        throw "Command reported failure ($exitCode): $File $($Arguments -join ' ')"
    }
    return $text
}

function Invoke-Native {
    param([string]$File, [string[]]$Arguments)
    Invoke-NativeCapture $File $Arguments | Out-Null
}

function Invoke-KaggleCapture {
    param([string[]]$Arguments)
    return Invoke-NativeCapture "python" ([string[]]@("-m", "kaggle") + $Arguments)
}

function Invoke-Kaggle {
    param([string[]]$Arguments)
    Invoke-KaggleCapture $Arguments | Out-Null
}

function Ensure-Directory {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Write-AtomicNewUtf8File {
    param([string]$Path, [string]$Text)
    $parent = Split-Path -Parent $Path
    $temporary = Join-Path $parent ("." + (Split-Path -Leaf $Path) + "." + [guid]::NewGuid().ToString("N") + ".tmp")
    try {
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes($Text)
        $stream = [IO.File]::Open($temporary, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        try {
            $stream.Write($bytes, 0, $bytes.Length)
            $stream.Flush($true)
        } finally {
            $stream.Dispose()
        }
        [IO.File]::Move($temporary, $Path)
    } finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
    }
}

function Get-KaggleOwner {
    if (-not $script:AuthenticatedKaggleOwner) {
        throw "Kaggle client identity has not been authenticated."
    }
    if ($Owner -and $Owner -ne $script:AuthenticatedKaggleOwner) {
        throw "-Owner '$Owner' differs from authenticated Kaggle owner '$script:AuthenticatedKaggleOwner'."
    }
    return $script:AuthenticatedKaggleOwner
}

function Assert-KaggleClient {
    if (-not (Test-Path -LiteralPath $CredentialPath -PathType Leaf)) {
        throw "Save the API credential at $CredentialPath (never in the repository)."
    }
    $version = & python -c "import importlib.metadata; print(importlib.metadata.version('kaggle'))"
    if ($LASTEXITCODE -ne 0 -or ($version.Trim() -ne "2.2.3")) {
        throw "Kaggle client 2.2.3 is required; found '$version'."
    }
    if (-not (Test-Path -LiteralPath $KernelVersionIo -PathType Leaf)) {
        throw "Missing exact-version Kaggle SDK bridge: $KernelVersionIo"
    }
    $authText = Invoke-NativeCapture "python" @($KernelVersionIo, "auth")
    try {
        $auth = $authText | ConvertFrom-Json
    } catch {
        throw "Kaggle SDK bridge returned malformed authentication JSON: $authText"
    }
    if ($auth.authenticated -ne $true -or
        [string]$auth.username -notmatch "^[A-Za-z0-9][A-Za-z0-9_-]*$") {
        throw "Kaggle API authentication did not return a valid owner."
    }
    $script:AuthenticatedKaggleOwner = [string]$auth.username
}

function Assert-Slug {
    param([string]$Value, [string]$Label)
    if ($Value -notmatch "^[a-z0-9][a-z0-9-]{1,49}$") {
        throw "$Label must match ^[a-z0-9][a-z0-9-]{1,49}$"
    }
}

function Assert-ModeSlug {
    param([string]$Value, [string]$Label)
    $otherMode = if ($Mode -eq "fp16-primary") { "nf4-sensitivity" } else { "fp16-primary" }
    if (-not $Value.Contains($Mode) -or $Value.Contains($otherMode)) {
        throw "$Label must contain only the selected mode token '$Mode'."
    }
}

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-FileSha256 {
    param([string]$Path, [string]$ExpectedSha256)
    if ($ExpectedSha256 -and (
        -not (Test-Path -LiteralPath $Path -PathType Leaf) -or
        (Get-Sha256 $Path) -ne $ExpectedSha256
    )) {
        throw "At-use helper hash differs from the immutable payload: $Path"
    }
}

function Get-TextSha256 {
    param([string]$Text)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes($Text)
        return (($algorithm.ComputeHash($bytes) | ForEach-Object { $_.ToString("x2") }) -join "")
    } finally {
        $algorithm.Dispose()
    }
}

function Read-JsonFile {
    param([string]$Path)
    return (Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json)
}

function Test-JsonInteger {
    param([AllowNull()][object]$Value, [int64]$Minimum = 0)
    if ($null -eq $Value -or $Value -is [bool]) { return $false }
    if ($Value -isnot [int32] -and $Value -isnot [int64]) { return $false }
    return [int64]$Value -ge $Minimum
}

function Assert-StateInventory {
    param([string]$ExportRoot, [object]$State)
    $resolvedRoot = (Resolve-Path -LiteralPath $ExportRoot).Path.TrimEnd("\")
    $rootPrefix = $resolvedRoot + "\"
    $declared = @{}
    $entries = @($State.files)
    if ($entries.Count -eq 0) { throw "State manifest inventory is empty." }
    foreach ($entry in $entries) {
        $relative = [string]$entry.path
        $parts = @($relative -split "/")
        if (-not $relative -or $relative.Contains("\") -or $relative.Contains(":") -or
            $relative.StartsWith("/") -or $parts.Count -eq 0 -or
            @($parts | Where-Object { -not $_ -or $_ -eq "." -or $_ -eq ".." }).Count -gt 0) {
            throw "State manifest contains an unsafe path: $relative"
        }
        if ($declared.ContainsKey($relative)) { throw "Duplicate state manifest path: $relative" }
        $expectedHash = [string]$entry.sha256
        if ($expectedHash -notmatch "^[0-9a-f]{64}$" -or
            -not (Test-JsonInteger $entry.size 0)) {
            throw "State manifest contains invalid metadata: $relative"
        }
        $path = Join-Path $resolvedRoot ($relative.Replace("/", "\"))
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "State manifest file is missing: $relative"
        }
        $item = Get-Item -LiteralPath $path -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
            -not $item.FullName.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "State manifest path is unsafe: $relative"
        }
        if ([int64]$item.Length -ne [int64]$entry.size -or
            (Get-Sha256 $path) -ne $expectedHash) {
            throw "State manifest file size/hash mismatch: $relative"
        }
        $declared[$relative] = $true
    }

    $actual = @{}
    foreach ($item in @(Get-ChildItem -LiteralPath $resolvedRoot -Recurse -Force)) {
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "State directory contains a reparse point: $($item.FullName)"
        }
        if ($item.PSIsContainer) { continue }
        $relative = $item.FullName.Substring($rootPrefix.Length).Replace("\", "/")
        if ($relative -eq "state_manifest.json") { continue }
        $actual[$relative] = $true
    }
    if ($actual.Count -ne $declared.Count -or
        @($actual.Keys | Where-Object { -not $declared.ContainsKey($_) }).Count -gt 0) {
        throw "State directory differs from the complete manifest inventory."
    }
}

function Get-DatasetStatusInfo {
    param([string]$DatasetId)
    Assert-FileSha256 $KernelVersionIo $script:ExpectedVersionIoSha256
    $text = Invoke-NativeCapture "python" @(
        $KernelVersionIo, "dataset-status", "--dataset", $DatasetId
    )
    try {
        $info = $text | ConvertFrom-Json
    } catch {
        throw "Kaggle returned malformed dataset status JSON for ${DatasetId}: $text"
    }
    if ($info.dataset -ne $DatasetId -or -not ([string]$info.status).Trim() -or
        -not (Test-JsonInteger $info.current_version_number 1)) {
        throw "Kaggle returned incomplete dataset status for ${DatasetId}: $text"
    }
    return $info
}

function Assert-PrivateDataset {
    param([string]$DatasetId, [int64]$ExpectedVersion = 0)
    Assert-FileSha256 $KernelVersionIo $script:ExpectedVersionIoSha256
    $text = Invoke-NativeCapture "python" @(
        $KernelVersionIo, "dataset-info", "--dataset", $DatasetId
    )
    try {
        $info = $text | ConvertFrom-Json
    } catch {
        throw "Kaggle returned malformed dataset metadata JSON for ${DatasetId}: $text"
    }
    if ($info.dataset -ne $DatasetId -or $info.ref -ne $DatasetId -or
        $info.is_private -ne $true -or
        -not (Test-JsonInteger $info.current_version_number 1) -or
        ($ExpectedVersion -gt 0 -and [int64]$info.current_version_number -ne $ExpectedVersion)) {
        throw "Dataset is public, has the wrong identity, or differs from expected version: $DatasetId"
    }
    return $info
}

function Assert-PrivateDatasetSources {
    param([string[]]$Sources)
    foreach ($source in $Sources) {
        if ($source -notmatch "^(?<dataset>[A-Za-z0-9_-]+/[a-z0-9][a-z0-9-]{1,49})/(?<version>[1-9][0-9]*)$") {
            throw "Kernel metadata contains an invalid version-pinned dataset source: $source"
        }
        $datasetId = [string]$Matches.dataset
        Assert-PrivateDataset $datasetId | Out-Null
    }
}

function Wait-DatasetReady {
    param([string]$DatasetId, [int64]$ExpectedVersion, [int]$TimeoutSeconds = 900)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $info = Get-DatasetStatusInfo $DatasetId
        } catch {
            if ([string]$_.Exception.Message -match "^At-use helper hash") { throw }
            if ((Get-Date) -ge $deadline) { throw }
            Start-Sleep -Seconds 10
            continue
        }
        $status = [string]$info.status
        $version = [int64]$info.current_version_number
        if ($status -eq "ready") {
            if ($version -ne $ExpectedVersion) {
                if ($version -lt $ExpectedVersion) {
                    Start-Sleep -Seconds 10
                    continue
                }
                throw "Dataset $DatasetId is ready at version $version, expected $ExpectedVersion."
            }
            try {
                Assert-PrivateDataset $DatasetId $ExpectedVersion | Out-Null
            } catch {
                $message = [string]$_.Exception.Message
                if ($message -match "^(Dataset is public|At-use helper hash)") { throw }
                if ((Get-Date) -ge $deadline) { throw }
                Start-Sleep -Seconds 10
                continue
            }
            return $info
        }
        if ($version -gt $ExpectedVersion) {
            throw "Dataset $DatasetId advanced to unexpected version $version."
        }
        if ($status -match "^(error|failed|failure|cancelled|canceled)$") {
            throw "Dataset entered a terminal failure state: $DatasetId"
        }
        Start-Sleep -Seconds 10
    }
    throw "Timed out waiting for dataset ready: $DatasetId"
}

function Get-KernelStatus {
    param([string]$KernelId)
    Assert-FileSha256 $KernelVersionIo $script:ExpectedVersionIoSha256
    $text = Invoke-NativeCapture "python" @(
        $KernelVersionIo, "status", "--kernel", $KernelId
    )
    try {
        $result = $text | ConvertFrom-Json
    } catch {
        throw "Kaggle returned malformed exact-version kernel status JSON: $text"
    }
    $status = ([string]$result.status).Trim().ToLowerInvariant()
    if ($result.kernel -ne $KernelId -or
        @(
            "error", "failed", "failure", "cancelled", "canceled",
            "cancel_requested", "cancel_acknowledged"
        ) -contains $status -or
        $null -ne $result.failure_message) {
        throw "Kernel entered a terminal failure state: $KernelId"
    }
    if ($status -notmatch "^[a-z][a-z0-9_]*$") {
        throw "Kaggle returned an invalid kernel status: $KernelId"
    }
    return $status
}

function Assert-KernelAccepted {
    param([string]$KernelId)
    $status = Get-KernelStatus $KernelId
    if (@("new", "new_script", "queued", "running", "complete") -notcontains $status) {
        throw "Kernel status is not recognized as new/queued/running/complete: $KernelId"
    }
}

function Assert-KernelComplete {
    param([string]$KernelId)
    $status = Get-KernelStatus $KernelId
    if ($status -ne "complete") {
        throw "Kernel output is unavailable because status is not complete: $KernelId"
    }
}

function Wait-KernelComplete {
    param([string]$KernelId, [int]$TimeoutSeconds = 43200)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $status = Get-KernelStatus $KernelId
        } catch {
            $message = [string]$_.Exception.Message
            if ($message -match "^(Kernel entered a terminal|At-use helper hash)") { throw }
            if ((Get-Date) -ge $deadline) { throw }
            Start-Sleep -Seconds 30
            continue
        }
        if ($status -eq "complete") { return }
        if (@("new", "new_script", "queued", "running") -notcontains $status) {
            throw "Kernel entered an unrecognized non-complete state '$status': $KernelId"
        }
        Start-Sleep -Seconds 30
    }
    throw "Timed out waiting for kernel completion: $KernelId"
}

function Get-RequiredStatePaths {
    param([string]$SelectedArm)
    $paths = @("inference_summary.json")
    $arms = if ($SelectedArm -eq "base") { @("base") } else { @("base", "tuned") }
    foreach ($selected in $arms) {
        $paths += @(
            "predictions/$selected.jsonl",
            "predictions/$selected.jsonl.manifest.json",
            "predictions/$selected.summary.json"
        )
    }
    $paths += @(
        "runtime/environment.json",
        "runtime/pip-freeze.txt",
        "runtime/install.log",
        "runtime/editable-install.log",
        "runtime/pip-check.log",
        "runtime/infer-$SelectedArm.log"
    )
    return @($paths | Sort-Object)
}

function Assert-ExactStatePaths {
    param([object]$State, [string]$SelectedArm)
    $expected = @(Get-RequiredStatePaths $SelectedArm)
    $actual = @($State.files | ForEach-Object { [string]$_.path } | Sort-Object)
    if ($actual.Count -ne $expected.Count -or
        @(Compare-Object -ReferenceObject $expected -DifferenceObject $actual -SyncWindow 0).Count -ne 0) {
        throw "Complete $SelectedArm state does not declare the exact required inventory."
    }
}

function Assert-PredictionSidecar {
    param(
        [string]$ExportRoot,
        [string]$SelectedArm,
        [object]$State,
        [string]$ExpectedScope,
        [int64]$ExpectedRows,
        [string[]]$ExpectedConditions
    )
    $sidecarPath = Join-Path $ExportRoot "predictions\$SelectedArm.jsonl.manifest.json"
    $predictionPath = Join-Path $ExportRoot "predictions\$SelectedArm.jsonl"
    foreach ($path in @($sidecarPath, $predictionPath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Complete state is missing required $SelectedArm prediction file: $path"
        }
    }
    $sidecar = Read-JsonFile $sidecarPath
    if ($sidecar.status -ne "complete" -or $sidecar.arm -ne $SelectedArm -or
        $sidecar.backend -ne "transformers" -or $sidecar.result_scope -ne $ExpectedScope -or
        $null -ne $sidecar.limit -or -not (Test-JsonInteger $sidecar.expected_rows 1) -or
        -not (Test-JsonInteger $sidecar.completed_rows 1) -or
        -not (Test-JsonInteger $sidecar.successful_rows 1) -or
        -not (Test-JsonInteger $sidecar.error_rows 0) -or
        [int64]$sidecar.expected_rows -ne $ExpectedRows -or
        [int64]$sidecar.completed_rows -ne [int64]$sidecar.expected_rows -or
        [int64]$sidecar.successful_rows -ne [int64]$sidecar.expected_rows -or
        [int64]$sidecar.error_rows -ne 0 -or
        [string]$sidecar.experiment_fingerprint -ne [string]$State.run_fingerprint -or
        [string]$sidecar.protocol_fingerprint -ne [string]$State.protocol_fingerprint -or
        [string]$sidecar.input_fingerprint -ne [string]$State.input_fingerprint -or
        [string]$sidecar.config_fingerprint -notmatch "^[0-9a-f]{64}$" -or
        (Get-Sha256 $predictionPath) -ne ([string]$sidecar.output_sha256).ToLowerInvariant()) {
        throw "$SelectedArm prediction sidecar is not complete or semantically bound to the state."
    }
    $conditions = @($sidecar.conditions)
    if ($conditions.Count -eq 0 -or
        @($conditions | Where-Object { -not ($_ -is [string]) -or -not $_ }).Count -gt 0 -or
        @($conditions | Select-Object -Unique).Count -ne $conditions.Count -or
        ([int64]$sidecar.expected_rows % $conditions.Count) -ne 0) {
        throw "$SelectedArm prediction sidecar has invalid conditions."
    }
    if ($conditions.Count -ne $ExpectedConditions.Count -or
        @(Compare-Object -ReferenceObject $ExpectedConditions -DifferenceObject $conditions -SyncWindow 0).Count -ne 0) {
        throw "$SelectedArm prediction conditions differ from the immutable payload protocol."
    }
    $conditionSet = @{}
    foreach ($condition in $conditions) { $conditionSet[[string]$condition] = $true }
    $keys = @{}
    $sampleConditions = @{}
    $rowCount = 0
    foreach ($line in @(Get-Content -LiteralPath $predictionPath -Encoding UTF8)) {
        $rowCount += 1
        if (-not $line.Trim()) { throw "$SelectedArm prediction JSONL contains a blank row." }
        try {
            $row = $line | ConvertFrom-Json
        } catch {
            throw "$SelectedArm prediction JSONL contains invalid JSON at row $rowCount."
        }
        $sampleId = [string]$row.sample_id
        $condition = [string]$row.condition
        $key = "$sampleId`0$condition"
        if (-not $sampleId -or -not $condition -or -not $conditionSet.ContainsKey($condition) -or
            $keys.ContainsKey($key) -or $row.arm -ne $SelectedArm -or
            $row.backend -ne $sidecar.backend -or $row.status -ne "success" -or
            $null -ne $row.error -or
            [string]$row.protocol_fingerprint -ne [string]$sidecar.protocol_fingerprint -or
            [string]$row.config_fingerprint -ne [string]$sidecar.config_fingerprint) {
            throw "$SelectedArm prediction row $rowCount differs from its sidecar or duplicates a key."
        }
        $keys[$key] = $true
        if (-not $sampleConditions.ContainsKey($sampleId)) { $sampleConditions[$sampleId] = @{} }
        $sampleConditions[$sampleId][$condition] = $true
    }
    $expectedSamples = [int64]$sidecar.expected_rows / $conditions.Count
    if ($rowCount -ne [int64]$sidecar.expected_rows -or
        $sampleConditions.Count -ne $expectedSamples -or
        @($sampleConditions.Values | Where-Object { $_.Count -ne $conditions.Count }).Count -gt 0) {
        throw "$SelectedArm prediction JSONL does not cover every expected sample/condition pair."
    }
    return $sidecar
}

function Assert-PredictionSummary {
    param(
        [string]$ExportRoot,
        [string]$SelectedArm,
        [object]$Sidecar,
        [object]$EmbeddedSummary
    )
    $summaryPath = Join-Path $ExportRoot "predictions\$SelectedArm.summary.json"
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        throw "Complete state is missing the $SelectedArm arm summary."
    }
    $summary = Read-JsonFile $summaryPath
    $summaryJson = $summary | ConvertTo-Json -Compress -Depth 30
    $embeddedJson = $EmbeddedSummary | ConvertTo-Json -Compress -Depth 30
    if ($summaryJson -ne $embeddedJson -or
        $summary.config_fingerprint -ne $Sidecar.config_fingerprint -or
        $summary.protocol_fingerprint -ne $Sidecar.protocol_fingerprint -or
        -not (Test-JsonInteger $summary.expected_rows 1) -or
        -not (Test-JsonInteger $summary.existing_rows 0) -or
        -not (Test-JsonInteger $summary.written_rows 0) -or
        -not (Test-JsonInteger $summary.successful_rows 1) -or
        -not (Test-JsonInteger $summary.error_rows 0) -or
        [int64]$summary.expected_rows -ne [int64]$Sidecar.expected_rows -or
        [int64]$summary.successful_rows -ne [int64]$Sidecar.expected_rows -or
        [int64]$summary.error_rows -ne 0 -or
        ([int64]$summary.existing_rows + [int64]$summary.written_rows) -ne [int64]$Sidecar.expected_rows -or
        $summary.output_sha256 -ne $Sidecar.output_sha256) {
        throw "$SelectedArm arm summary differs from its complete prediction sidecar."
    }
}

function Assert-PythonState {
    param(
        [string]$ExportRoot,
        [string]$SelectedArm,
        [string]$ExpectedDatasetId,
        [string]$ExpectedParentHash,
        [string]$ExpectedPayloadHash,
        [string]$ExpectedConfigFingerprint,
        [string]$ExpectedExecutionLock,
        [int64]$ExpectedRows
    )
    $runnerPath = Join-Path (Join-Path $WorkflowBuildRoot $SelectedArm) "runner.py"
    if (-not (Test-Path -LiteralPath $StateValidator -PathType Leaf) -or
        -not (Test-Path -LiteralPath $runnerPath -PathType Leaf)) {
        throw "Independent state validator or exact rendered runner is missing."
    }
    $rendered = Assert-RenderedKernel $script:AuthenticatedKaggleOwner $SelectedArm
    Assert-FileSha256 $StateValidator $script:ExpectedStateValidatorSha256
    if ($rendered.Lock.output_dataset_id -ne $ExpectedDatasetId -or
        $rendered.Lock.payload_manifest_sha256 -ne $ExpectedPayloadHash -or
        $rendered.Lock.config_fingerprint -ne $ExpectedConfigFingerprint -or
        $rendered.Lock.execution_lock_sha256 -ne $ExpectedExecutionLock -or
        [int64]$rendered.Lock.expected_rows -ne $ExpectedRows) {
        throw "At-use rendered state-validation locks differ from the expected workflow."
    }
    $arguments = @(
        $StateValidator,
        "--runner", $runnerPath,
        "--payload-manifest", (Get-PayloadManifestPath),
        "--export-root", $ExportRoot,
        "--arm", $SelectedArm,
        "--expected-dataset", $ExpectedDatasetId,
        "--expected-payload-sha256", $ExpectedPayloadHash,
        "--expected-config-fingerprint", $ExpectedConfigFingerprint,
        "--expected-execution-lock", $ExpectedExecutionLock,
        "--expected-rows", ([string]$ExpectedRows)
    )
    if ($ExpectedParentHash) {
        $arguments += @("--expected-parent-sha256", $ExpectedParentHash)
        $parentRoot = Join-Path $WorkflowDownloadsRoot "base\kaggle_export"
        $arguments += @("--parent-state-root", $parentRoot)
    }
    Invoke-Native "python" ([string[]]$arguments)
}

function Assert-CompleteBaseState {
    param(
        [string]$ExportRoot,
        [string]$ExpectedDatasetId,
        [string]$ExpectedMode,
        [string]$ExpectedPayloadHash,
        [string]$ExpectedConfigFingerprint,
        [string]$ExpectedExecutionLock,
        [int64]$ExpectedRows,
        [string[]]$ExpectedConditions
    )
    $manifestPath = Join-Path $ExportRoot "state_manifest.json"
    $summaryPath = Join-Path $ExportRoot "inference_summary.json"
    foreach ($path in @($manifestPath, $summaryPath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Complete base state is missing required file: $path"
        }
    }
    $state = Read-JsonFile $manifestPath
    $expectedScope = if ($ExpectedMode -eq "fp16-primary") { "publication_candidate" } else { "automatic_sensitivity_only" }
    if ($state.artifact_version -ne 4 -or $state.complete -ne $true -or
        $state.kernel_succeeded -ne $true -or $state.base_state_complete -ne $true -or
        $state.arm -ne "base" -or
        $state.workflow_mode -ne $ExpectedMode -or
        $state.dataset_slug -ne $ExpectedDatasetId -or
        $state.result_scope -ne $expectedScope -or
        $null -ne $state.parent_state_manifest_sha256 -or
        $state.config_fingerprint -ne $ExpectedConfigFingerprint -or
        $state.payload_manifest_sha256 -ne $ExpectedPayloadHash -or
        $state.execution_lock_sha256 -ne $ExpectedExecutionLock -or
        [string]$state.run_fingerprint -notmatch "^[0-9a-f]{64}$" -or
        [string]$state.protocol_fingerprint -notmatch "^[0-9a-f]{64}$" -or
        [string]$state.input_fingerprint -notmatch "^[0-9a-f]{64}$") {
        throw "Base state manifest is incomplete or has the wrong arm, mode, or dataset slug."
    }
    Assert-ExactStatePaths $state "base"
    $sidecar = Assert-PredictionSidecar `
        $ExportRoot "base" $state $expectedScope $ExpectedRows $ExpectedConditions
    $summary = Read-JsonFile $summaryPath
    $summaryArms = @($summary.arms.PSObject.Properties.Name | Sort-Object)
    if ($summary.experiment_config_fingerprint -ne $state.config_fingerprint -or
        $summary.run_fingerprint -ne $state.run_fingerprint -or
        $summary.protocol_fingerprint -ne $state.protocol_fingerprint -or
        $summary.precision_mode -ne $ExpectedMode -or
        $summary.result_scope -ne $expectedScope -or $summary.exploratory -ne $false -or
        $summaryArms.Count -ne 1 -or $summaryArms[0] -ne "base") {
        throw "Base inference summary is not bound to the complete state."
    }
    Assert-PredictionSummary $ExportRoot "base" $sidecar $summary.arms.base
    Assert-StateInventory $ExportRoot $state
    Assert-PythonState `
        $ExportRoot "base" $ExpectedDatasetId "" $ExpectedPayloadHash `
        $ExpectedConfigFingerprint $ExpectedExecutionLock $ExpectedRows
}

function Assert-CompleteFinalState {
    param(
        [string]$ExportRoot,
        [string]$ExpectedDatasetId,
        [string]$ExpectedMode,
        [string]$ExpectedParentHash,
        [string]$ExpectedPayloadHash,
        [string]$ExpectedConfigFingerprint,
        [string]$ExpectedExecutionLock,
        [int64]$ExpectedRows,
        [string[]]$ExpectedConditions
    )
    $manifestPath = Join-Path $ExportRoot "state_manifest.json"
    $summaryPath = Join-Path $ExportRoot "inference_summary.json"
    foreach ($path in @($manifestPath, $summaryPath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Complete final state is missing required file: $path"
        }
    }
    $state = Read-JsonFile $manifestPath
    $expectedScope = if ($ExpectedMode -eq "fp16-primary") { "publication_candidate" } else { "automatic_sensitivity_only" }
    if ($state.artifact_version -ne 4 -or $state.complete -ne $true -or
        $state.kernel_succeeded -ne $true -or $state.base_state_complete -ne $true -or
        $state.arm -ne "tuned" -or $state.workflow_mode -ne $ExpectedMode -or
        $state.dataset_slug -ne $ExpectedDatasetId -or $state.result_scope -ne $expectedScope -or
        $state.parent_state_manifest_sha256 -ne $ExpectedParentHash -or
        $state.payload_manifest_sha256 -ne $ExpectedPayloadHash -or
        $state.config_fingerprint -ne $ExpectedConfigFingerprint -or
        $state.execution_lock_sha256 -ne $ExpectedExecutionLock -or
        [string]$state.run_fingerprint -notmatch "^[0-9a-f]{64}$" -or
        [string]$state.protocol_fingerprint -notmatch "^[0-9a-f]{64}$" -or
        [string]$state.input_fingerprint -notmatch "^[0-9a-f]{64}$") {
        throw "Final state manifest is incomplete or differs from the rendered workflow locks."
    }
    Assert-ExactStatePaths $state "tuned"
    $baseSidecar = Assert-PredictionSidecar `
        $ExportRoot "base" $state $expectedScope $ExpectedRows $ExpectedConditions
    $tunedSidecar = Assert-PredictionSidecar `
        $ExportRoot "tuned" $state $expectedScope $ExpectedRows $ExpectedConditions
    if ([int64]$baseSidecar.expected_rows -ne [int64]$tunedSidecar.expected_rows) {
        throw "Base and tuned final sidecars have different expected row counts."
    }
    $summary = Read-JsonFile $summaryPath
    $summaryArms = @($summary.arms.PSObject.Properties.Name | Sort-Object)
    if ($summary.experiment_config_fingerprint -ne $state.config_fingerprint -or
        $summary.run_fingerprint -ne $state.run_fingerprint -or
        $summary.protocol_fingerprint -ne $state.protocol_fingerprint -or
        $summary.precision_mode -ne $ExpectedMode -or
        $summary.result_scope -ne $expectedScope -or $summary.exploratory -ne $false -or
        $summaryArms.Count -ne 2 -or $summaryArms[0] -ne "base" -or
        $summaryArms[1] -ne "tuned" -or
        ($ExpectedMode -eq "nf4-sensitivity" -and $null -eq $summary.automatic_metrics)) {
        throw "Final inference summary is not bound to both completed arms."
    }
    Assert-PredictionSummary $ExportRoot "base" $baseSidecar $summary.arms.base
    Assert-PredictionSummary $ExportRoot "tuned" $tunedSidecar $summary.arms.tuned
    Assert-StateInventory $ExportRoot $state
    Assert-PythonState `
        $ExportRoot "tuned" $ExpectedDatasetId $ExpectedParentHash $ExpectedPayloadHash `
        $ExpectedConfigFingerprint $ExpectedExecutionLock $ExpectedRows
}

function Assert-Inputs {
    if (-not $Config -or -not (Test-Path -LiteralPath $Config -PathType Leaf)) {
        throw "-Config must name the generated config for -Mode $Mode."
    }
    if (-not $PreparedRun -or -not (Test-Path -LiteralPath $PreparedRun -PathType Container)) {
        throw "-PreparedRun must name its fresh strict prepared run."
    }
}

function New-InputStaging {
    param([string]$KaggleOwner)
    Assert-Inputs
    Ensure-Directory $StagingRoot
    Ensure-Directory $WorkflowStagingRoot
    $target = $StagingDir
    if (-not $target) { $target = Join-Path $WorkflowStagingRoot $InputSlug }
    if (Test-Path -LiteralPath $target) { throw "Refusing to overwrite staging: $target" }
    Invoke-Native "python" @(
        (Join-Path $KaggleRoot "build_payload.py"),
        "--repo-root", $RepoRoot,
        "--config", $Config,
        "--prepared-run", $PreparedRun,
        "--staging-dir", $target,
        "--owner", $KaggleOwner,
        "--slug", $InputSlug,
        "--mode", $Mode
    )
    return $target
}

function Convert-ToJsonLiteral {
    param([AllowNull()][object]$Value)
    return ($Value | ConvertTo-Json -Compress)
}

function Get-PayloadManifestPath {
    $payloadRoot = $StagingDir
    if (-not $payloadRoot) { $payloadRoot = Join-Path $WorkflowStagingRoot $InputSlug }
    $manifestPath = Join-Path $payloadRoot "payload_manifest.json"
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Create the exact local payload before rendering; missing $manifestPath"
    }
    return (Resolve-Path -LiteralPath $manifestPath).Path
}

function Get-VerifiedPayloadBinding {
    param(
        [string]$ManifestPath,
        [string]$KaggleOwner,
        [string]$ExpectedConfigPath = ""
    )
    $manifest = Read-JsonFile $ManifestPath
    $relativeConfig = [string]$manifest.config_relative
    $parts = @($relativeConfig -split "/")
    $conditions = @($manifest.conditions)
    if ($manifest.artifact_version -ne 3 -or
        $manifest.dataset_slug -ne "$KaggleOwner/$InputSlug" -or
        $manifest.workflow_mode -ne $Mode -or
        [string]$manifest.config_fingerprint -notmatch "^[0-9a-f]{64}$" -or
        $manifest.n_items -ne 150 -or
        -not (Test-JsonInteger $manifest.expected_rows 1) -or
        $conditions.Count -eq 0 -or
        @($conditions | Where-Object { -not ($_ -is [string]) -or -not $_ }).Count -gt 0 -or
        @($conditions | Select-Object -Unique).Count -ne $conditions.Count -or
        [int64]$manifest.expected_rows -ne 150 * $conditions.Count -or
        -not $relativeConfig -or $relativeConfig.Contains("\") -or
        $relativeConfig.Contains(":") -or $relativeConfig.StartsWith("/") -or
        @($parts | Where-Object { -not $_ -or $_ -eq "." -or $_ -eq ".." }).Count -gt 0) {
        throw "Payload manifest is not bound to the selected mode, slug, and safe config path."
    }
    $rootPrefix = $RepoRoot.TrimEnd("\") + "\"
    $configPath = (Resolve-Path -LiteralPath (Join-Path $RepoRoot $relativeConfig.Replace("/", "\"))).Path
    if (-not $configPath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Payload config path escapes the repository."
    }
    if ($ExpectedConfigPath) {
        $resolvedExpected = (Resolve-Path -LiteralPath $ExpectedConfigPath).Path
        if ($resolvedExpected -ne $configPath) {
            throw "Selected config differs from payload_manifest.json."
        }
    }
    $configInventoryPath = "top-papers-graph/$relativeConfig"
    $templateRelative = "experiments/vlm_ab_evaluation/kaggle/kernel_runner.py.template"
    $templateInventoryPath = "top-papers-graph/$templateRelative"
    $versionIoRelative = "experiments/vlm_ab_evaluation/kaggle/kernel_version_io.py"
    $versionIoInventoryPath = "top-papers-graph/$versionIoRelative"
    $stateValidatorRelative = "experiments/vlm_ab_evaluation/kaggle/validate_state.py"
    $stateValidatorInventoryPath = "top-papers-graph/$stateValidatorRelative"
    $configEntries = @($manifest.files | Where-Object { $_.path -eq $configInventoryPath })
    $templateEntries = @($manifest.files | Where-Object { $_.path -eq $templateInventoryPath })
    $versionIoEntries = @($manifest.files | Where-Object { $_.path -eq $versionIoInventoryPath })
    $stateValidatorEntries = @($manifest.files | Where-Object { $_.path -eq $stateValidatorInventoryPath })
    if ($configEntries.Count -ne 1 -or $templateEntries.Count -ne 1 -or
        $versionIoEntries.Count -ne 1 -or $stateValidatorEntries.Count -ne 1) {
        throw "Payload inventory lacks an exact config, runner template, or local validation helper."
    }
    $configHash = Get-Sha256 $configPath
    $templatePath = Join-Path $RepoRoot $templateRelative.Replace("/", "\")
    $templateHash = Get-Sha256 $templatePath
    $versionIoPath = Join-Path $RepoRoot $versionIoRelative.Replace("/", "\")
    $versionIoHash = Get-Sha256 $versionIoPath
    $stateValidatorPath = Join-Path $RepoRoot $stateValidatorRelative.Replace("/", "\")
    $stateValidatorHash = Get-Sha256 $stateValidatorPath
    if ([string]$configEntries[0].sha256 -ne $configHash -or
        [string]$templateEntries[0].sha256 -ne $templateHash -or
        [string]$versionIoEntries[0].sha256 -ne $versionIoHash -or
        [string]$stateValidatorEntries[0].sha256 -ne $stateValidatorHash) {
        throw "Local config, runner, or validation bytes differ from the immutable payload inventory."
    }
    $script:ExpectedVersionIoSha256 = $versionIoHash
    $script:ExpectedStateValidatorSha256 = $stateValidatorHash
    return [PSCustomObject]@{
        Manifest = $manifest
        ManifestPath = (Resolve-Path -LiteralPath $ManifestPath).Path
        ManifestSha256 = Get-Sha256 $ManifestPath
        ConfigRelative = $relativeConfig
        ConfigPath = $configPath
        ConfigSha256 = $configHash
        ConfigFingerprint = [string]$manifest.config_fingerprint
        ExpectedRows = [int64]$manifest.expected_rows
        Conditions = [string[]]$conditions
        TemplatePath = $templatePath
        TemplateSha256 = $templateHash
        VersionIoSha256 = $versionIoHash
        StateValidatorSha256 = $stateValidatorHash
    }
}

function Get-ExecutionLockSha256 {
    param(
        [string]$KaggleOwner,
        [string]$SelectedArm,
        [string]$KernelId,
        [string]$OutputDatasetId,
        [string]$ConfigRelative,
        [string]$ConfigSha256,
        [string]$ConfigFingerprint,
        [string]$TemplateSha256,
        [string]$PayloadManifestSha256,
        [string]$StateManifestSha256,
        [int64]$PayloadDatasetVersion,
        [AllowNull()][object]$StateDatasetVersion,
        [string[]]$DatasetSources
    )
    $binding = [ordered]@{
        artifact_version = 1
        workflow_mode = $Mode
        arm = $SelectedArm
        owner = $KaggleOwner
        kernel_id = $KernelId
        output_dataset_id = $OutputDatasetId
        config_relative = $ConfigRelative
        config_sha256 = $ConfigSha256
        config_fingerprint = $ConfigFingerprint
        template_sha256 = $TemplateSha256
        payload_manifest_sha256 = $PayloadManifestSha256
        state_manifest_sha256 = $StateManifestSha256
        payload_dataset_version = $PayloadDatasetVersion
        state_dataset_version = $StateDatasetVersion
        dataset_sources = $DatasetSources
    } | ConvertTo-Json -Compress -Depth 6
    return Get-TextSha256 $binding
}

function New-RunnerText {
    param(
        [string]$SelectedArm,
        [string]$KaggleOwner,
        [string]$SelectedStateSlug,
        [string]$PayloadManifestSha256,
        [string]$StateManifestSha256,
        [string]$ExecutionLockSha256,
        [string]$ConfigRelative,
        [string]$OutputStateSlug
    )
    $template = Get-Content -LiteralPath (Join-Path $KaggleRoot "kernel_runner.py.template") -Raw -Encoding UTF8
    $runner = $template.Replace("__ARM_JSON__", (Convert-ToJsonLiteral $SelectedArm))
    $runner = $runner.Replace("__WORKFLOW_MODE_JSON__", (Convert-ToJsonLiteral $Mode))
    $runner = $runner.Replace("__PAYLOAD_DATASET_SLUG_JSON__", (Convert-ToJsonLiteral "$KaggleOwner/$InputSlug"))
    $stateId = if ($SelectedStateSlug) { "$KaggleOwner/$SelectedStateSlug" } else { "" }
    $runner = $runner.Replace("__STATE_DATASET_SLUG_JSON__", (Convert-ToJsonLiteral $stateId))
    $runner = $runner.Replace("__PAYLOAD_MANIFEST_SHA256_JSON__", (Convert-ToJsonLiteral $PayloadManifestSha256))
    $runner = $runner.Replace("__STATE_MANIFEST_SHA256_JSON__", (Convert-ToJsonLiteral $StateManifestSha256))
    $runner = $runner.Replace("__EXECUTION_LOCK_SHA256_JSON__", (Convert-ToJsonLiteral $ExecutionLockSha256))
    $runner = $runner.Replace("__CONFIG_RELATIVE_JSON__", (Convert-ToJsonLiteral $ConfigRelative))
    return $runner.Replace("__OUTPUT_DATASET_SLUG__", "$KaggleOwner/$OutputStateSlug")
}

function New-KernelMetadataJson {
    param(
        [string]$KaggleOwner,
        [string]$KernelSlug,
        [string[]]$DatasetSources
    )
    $metadata = [ordered]@{
        id = "$KaggleOwner/$KernelSlug"
        title = $KernelSlug
        code_file = "runner.py"
        language = "python"
        kernel_type = "script"
        is_private = $true
        enable_gpu = $true
        enable_internet = $true
        machine_shape = "NvidiaTeslaT4"
        docker_image_pinning_type = "original"
        dataset_sources = $DatasetSources
        competition_sources = @()
        kernel_sources = @()
        model_sources = @()
    }
    return (($metadata | ConvertTo-Json -Depth 6) + "`n")
}

function Render-Kernel {
    param([string]$KaggleOwner, [string]$SelectedArm, [string]$SelectedStateSlug)
    Ensure-Directory $BuildRoot
    Ensure-Directory $WorkflowBuildRoot
    if ($SelectedArm -eq "base" -and $SelectedStateSlug) {
        throw "Automated partial base-state resume is not supported."
    }
    if ($SelectedArm -eq "tuned" -and $SelectedStateSlug -ne $BaseStateSlug) {
        throw "Tuned kernel must use exactly -StateSlug $BaseStateSlug."
    }
    if (-not $Config -or -not (Test-Path -LiteralPath $Config -PathType Leaf)) {
        throw "RenderKernel requires the exact local -Config used to build the payload."
    }
    $payloadManifestPath = Get-PayloadManifestPath
    $payload = Get-VerifiedPayloadBinding $payloadManifestPath $KaggleOwner $Config
    $payloadStatus = Get-DatasetStatusInfo "$KaggleOwner/$InputSlug"
    if ($payloadStatus.status -ne "ready") { throw "Input dataset is not ready." }
    $payloadDatasetVersion = [int64]$payloadStatus.current_version_number
    Assert-PrivateDataset "$KaggleOwner/$InputSlug" $payloadDatasetVersion | Out-Null
    $stateManifestPath = ""
    $stateManifestHash = ""
    $stateDatasetVersion = $null
    if ($SelectedArm -eq "tuned") {
        $stateRoot = Join-Path $WorkflowDownloadsRoot "base\kaggle_export"
        $stateManifestPath = Join-Path $stateRoot "state_manifest.json"
        $baseRendered = Assert-RenderedKernel $KaggleOwner "base"
        Assert-CompleteBaseState `
            $stateRoot "$KaggleOwner/$BaseStateSlug" $Mode `
            ([string]$baseRendered.Lock.payload_manifest_sha256) `
            ([string]$baseRendered.Lock.config_fingerprint) `
            ([string]$baseRendered.Lock.execution_lock_sha256) `
            ([int64]$baseRendered.Lock.expected_rows) `
            ([string[]]@($baseRendered.Lock.conditions))
        $stateManifestHash = Get-Sha256 $stateManifestPath
        $stateStatus = Get-DatasetStatusInfo "$KaggleOwner/$BaseStateSlug"
        if ($stateStatus.status -ne "ready") { throw "Base-state dataset is not ready." }
        $stateDatasetVersion = [int64]$stateStatus.current_version_number
        Assert-PrivateDataset "$KaggleOwner/$BaseStateSlug" $stateDatasetVersion | Out-Null
    }
    $kernelSlug = if ($SelectedArm -eq "base") { $BaseKernelSlug } else { $TunedKernelSlug }
    $outputStateSlug = if ($SelectedArm -eq "base") { $BaseStateSlug } else { $FinalStateSlug }
    $target = Join-Path $WorkflowBuildRoot $SelectedArm
    if (Test-Path -LiteralPath $target) { throw "Refusing to overwrite rendered kernel: $target" }
    New-Item -ItemType Directory -Path $target | Out-Null
    $sources = @("$KaggleOwner/$InputSlug/$payloadDatasetVersion")
    if ($SelectedStateSlug) { $sources += "$KaggleOwner/$SelectedStateSlug/$stateDatasetVersion" }
    $executionLockHash = Get-ExecutionLockSha256 `
        $KaggleOwner $SelectedArm "$KaggleOwner/$kernelSlug" "$KaggleOwner/$outputStateSlug" `
        $payload.ConfigRelative $payload.ConfigSha256 $payload.ConfigFingerprint `
        $payload.TemplateSha256 $payload.ManifestSha256 $stateManifestHash `
        $payloadDatasetVersion $stateDatasetVersion $sources
    $runner = New-RunnerText `
        $SelectedArm $KaggleOwner $SelectedStateSlug $payload.ManifestSha256 `
        $stateManifestHash $executionLockHash $payload.ConfigRelative $outputStateSlug
    [IO.File]::WriteAllText((Join-Path $target "runner.py"), $runner, [Text.UTF8Encoding]::new($false))

    $metadataPath = Join-Path $target "kernel-metadata.json"
    $runnerPath = Join-Path $target "runner.py"
    $metadataJson = New-KernelMetadataJson $KaggleOwner $kernelSlug $sources
    [IO.File]::WriteAllText($metadataPath, $metadataJson, [Text.UTF8Encoding]::new($false))
    $renderLock = [ordered]@{
        artifact_version = 3
        workflow_mode = $Mode
        arm = $SelectedArm
        owner = $KaggleOwner
        kernel_id = "$KaggleOwner/$kernelSlug"
        output_dataset_id = "$KaggleOwner/$outputStateSlug"
        config_relative = $payload.ConfigRelative
        config_sha256 = $payload.ConfigSha256
        config_fingerprint = $payload.ConfigFingerprint
        expected_rows = $payload.ExpectedRows
        conditions = $payload.Conditions
        template_sha256 = $payload.TemplateSha256
        kernel_version_io_sha256 = $payload.VersionIoSha256
        state_validator_sha256 = $payload.StateValidatorSha256
        payload_manifest_path = $payload.ManifestPath
        payload_manifest_sha256 = $payload.ManifestSha256
        state_manifest_path = $stateManifestPath
        state_manifest_sha256 = $stateManifestHash
        payload_dataset_version = $payloadDatasetVersion
        state_dataset_version = $stateDatasetVersion
        execution_lock_sha256 = $executionLockHash
        runner_sha256 = Get-Sha256 $runnerPath
        runner_source_sha256 = Get-TextSha256 ($runner.Replace("`r`n", "`n").Replace("`r", "`n"))
        metadata_sha256 = Get-Sha256 $metadataPath
        dataset_sources = $sources
    } | ConvertTo-Json -Depth 6
    [IO.File]::WriteAllText((Join-Path $target "render-lock.json"), $renderLock + "`n", [Text.UTF8Encoding]::new($false))
    return $target
}

function Assert-RenderedKernel {
    param([string]$KaggleOwner, [string]$SelectedArm)
    $target = Join-Path $WorkflowBuildRoot $SelectedArm
    $runnerPath = Join-Path $target "runner.py"
    $metadataPath = Join-Path $target "kernel-metadata.json"
    $lockPath = Join-Path $target "render-lock.json"
    foreach ($path in @($runnerPath, $metadataPath, $lockPath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Rendered kernel is missing a locked file: $path"
        }
    }
    $lock = Read-JsonFile $lockPath
    $metadata = Read-JsonFile $metadataPath
    $kernelSlug = if ($SelectedArm -eq "base") { $BaseKernelSlug } else { $TunedKernelSlug }
    $outputSlug = if ($SelectedArm -eq "base") { $BaseStateSlug } else { $FinalStateSlug }
    if (-not (Test-JsonInteger $lock.payload_dataset_version 1) -or
        ($SelectedArm -eq "base" -and $null -ne $lock.state_dataset_version) -or
        ($SelectedArm -eq "tuned" -and -not (Test-JsonInteger $lock.state_dataset_version 1))) {
        throw "Rendered kernel has invalid dataset version locks."
    }
    $expectedSources = @("$KaggleOwner/$InputSlug/$($lock.payload_dataset_version)")
    if ($SelectedArm -eq "tuned") {
        $expectedSources += "$KaggleOwner/$BaseStateSlug/$($lock.state_dataset_version)"
    }
    $actualSources = @($metadata.dataset_sources)
    if ($lock.artifact_version -ne 3 -or $lock.workflow_mode -ne $Mode -or
        $lock.arm -ne $SelectedArm -or $lock.owner -ne $KaggleOwner -or
        $lock.kernel_id -ne "$KaggleOwner/$kernelSlug" -or
        $lock.output_dataset_id -ne "$KaggleOwner/$outputSlug" -or
        $metadata.id -ne "$KaggleOwner/$kernelSlug" -or $metadata.title -ne $kernelSlug -or
        $metadata.code_file -ne "runner.py" -or $metadata.language -ne "python" -or
        $metadata.kernel_type -ne "script" -or $metadata.is_private -ne $true -or
        $metadata.enable_gpu -ne $true -or $metadata.enable_internet -ne $true -or
        $metadata.machine_shape -ne "NvidiaTeslaT4" -or
        $metadata.docker_image_pinning_type -ne "original" -or
        @($metadata.PSObject.Properties | Where-Object { $_.Name -eq "docker_image" }).Count -ne 0 -or
        @($metadata.competition_sources).Count -ne 0 -or
        @($metadata.kernel_sources).Count -ne 0 -or @($metadata.model_sources).Count -ne 0 -or
        $actualSources.Count -ne $expectedSources.Count -or
        @(Compare-Object -ReferenceObject $expectedSources -DifferenceObject $actualSources -SyncWindow 0).Count -ne 0) {
        throw "Rendered kernel metadata/lock differs from the selected mode, arm, or slugs."
    }
    $payloadManifestPath = Get-PayloadManifestPath
    $payload = Get-VerifiedPayloadBinding $payloadManifestPath $KaggleOwner
    $selectedStateSlug = ""
    $stateManifestPath = ""
    $stateManifestHash = ""
    if ($SelectedArm -eq "base") {
        if ($lock.state_manifest_path -or $lock.state_manifest_sha256) {
            throw "Rendered base kernel unexpectedly contains a state lock."
        }
    } else {
        $selectedStateSlug = $BaseStateSlug
        $stateManifestPath = Join-Path $WorkflowDownloadsRoot "base\kaggle_export\state_manifest.json"
        if (-not (Test-Path -LiteralPath $stateManifestPath -PathType Leaf) -or
            (Get-Sha256 $stateManifestPath) -ne [string]$lock.state_manifest_sha256) {
            throw "Rendered tuned kernel parent-state manifest lock is unavailable or changed."
        }
        $stateManifestHash = Get-Sha256 $stateManifestPath
    }
    $expectedExecutionLock = Get-ExecutionLockSha256 `
        $KaggleOwner $SelectedArm "$KaggleOwner/$kernelSlug" "$KaggleOwner/$outputSlug" `
        $payload.ConfigRelative $payload.ConfigSha256 $payload.ConfigFingerprint `
        $payload.TemplateSha256 $payload.ManifestSha256 $stateManifestHash `
        ([int64]$lock.payload_dataset_version) $lock.state_dataset_version $expectedSources
    $expectedRunner = New-RunnerText `
        $SelectedArm $KaggleOwner $selectedStateSlug $payload.ManifestSha256 `
        $stateManifestHash $expectedExecutionLock $payload.ConfigRelative $outputSlug
    $expectedMetadata = New-KernelMetadataJson $KaggleOwner $kernelSlug $expectedSources
    $expectedSourceHash = Get-TextSha256 (
        $expectedRunner.Replace("`r`n", "`n").Replace("`r", "`n")
    )
    if ($lock.config_relative -ne $payload.ConfigRelative -or
        $lock.config_sha256 -ne $payload.ConfigSha256 -or
        $lock.config_fingerprint -ne $payload.ConfigFingerprint -or
        [int64]$lock.expected_rows -ne $payload.ExpectedRows -or
        @($lock.conditions).Count -ne $payload.Conditions.Count -or
        @(Compare-Object -ReferenceObject $payload.Conditions -DifferenceObject @($lock.conditions) -SyncWindow 0).Count -ne 0 -or
        $lock.template_sha256 -ne $payload.TemplateSha256 -or
        $lock.kernel_version_io_sha256 -ne $payload.VersionIoSha256 -or
        $lock.state_validator_sha256 -ne $payload.StateValidatorSha256 -or
        $lock.payload_manifest_path -ne $payload.ManifestPath -or
        $lock.payload_manifest_sha256 -ne $payload.ManifestSha256 -or
        $lock.state_manifest_path -ne $stateManifestPath -or
        $lock.state_manifest_sha256 -ne $stateManifestHash -or
        $lock.execution_lock_sha256 -ne $expectedExecutionLock -or
        (Get-Sha256 $runnerPath) -ne (Get-TextSha256 $expectedRunner) -or
        [string]$lock.runner_sha256 -ne (Get-TextSha256 $expectedRunner) -or
        [string]$lock.runner_source_sha256 -ne $expectedSourceHash -or
        (Get-Sha256 $metadataPath) -ne (Get-TextSha256 $expectedMetadata) -or
        [string]$lock.metadata_sha256 -ne (Get-TextSha256 $expectedMetadata)) {
        throw "Rendered kernel cannot be reconstructed from the immutable payload and workflow inputs."
    }
    return [PSCustomObject]@{ Metadata = $metadata; Lock = $lock }
}

function Push-Kernel {
    param([string]$KaggleOwner, [string]$SelectedArm)
    $target = Join-Path $WorkflowBuildRoot $SelectedArm
    $rendered = Assert-RenderedKernel $KaggleOwner $SelectedArm
    Assert-PrivateDatasetSources ([string[]]@($rendered.Metadata.dataset_sources))
    $receiptPath = Join-Path $target "push-receipt.json"
    $attemptPath = Join-Path $WorkflowBuildRoot "$SelectedArm-push-attempt.json"
    if (Test-Path -LiteralPath $receiptPath) {
        Wait-PushReceiptReady $KaggleOwner $SelectedArm | Out-Null
        return
    }
    if (Test-Path -LiteralPath $attemptPath) {
        throw "A prior push attempt has no receipt; resolve its remote version before retrying: $attemptPath"
    }
    $attempt = [ordered]@{
        artifact_version = 1
        kernel_id = [string]$rendered.Metadata.id
        execution_lock_sha256 = [string]$rendered.Lock.execution_lock_sha256
        runner_sha256 = [string]$rendered.Lock.runner_sha256
        started_at = [DateTimeOffset]::UtcNow.ToString("o")
    } | ConvertTo-Json -Depth 4
    Write-AtomicNewUtf8File $attemptPath ($attempt + "`n")
    $attemptHash = Get-Sha256 $attemptPath
    $pushOutput = Invoke-KaggleCapture @(
        "kernels", "push", "-p", $target,
        "--accelerator", "NvidiaTeslaT4", "--timeout", "43200"
    )
    $matches = [regex]::Matches(
        $pushOutput,
        "(?im)\bKernel version (?<version>[1-9][0-9]*) successfully pushed\b"
    )
    if ($matches.Count -ne 1) {
        throw "Kaggle did not report exactly one pushed kernel version."
    }
    $version = [int64]$matches[0].Groups["version"].Value
    $receipt = [ordered]@{
        artifact_version = 2
        kernel_id = [string]$rendered.Metadata.id
        kernel_version = $version
        push_attempt_sha256 = $attemptHash
        execution_lock_sha256 = [string]$rendered.Lock.execution_lock_sha256
        runner_sha256 = [string]$rendered.Lock.runner_sha256
        runner_source_sha256 = [string]$rendered.Lock.runner_source_sha256
        pushed_at = [DateTimeOffset]::UtcNow.ToString("o")
        push_output_sha256 = Get-TextSha256 $pushOutput
    } | ConvertTo-Json -Depth 4
    Write-AtomicNewUtf8File $receiptPath ($receipt + "`n")
    Wait-PushReceiptReady $KaggleOwner $SelectedArm | Out-Null
}

function Assert-PushReceipt {
    param([string]$KaggleOwner, [string]$SelectedArm)
    $rendered = Assert-RenderedKernel $KaggleOwner $SelectedArm
    Assert-PrivateDatasetSources ([string[]]@($rendered.Metadata.dataset_sources))
    $receiptPath = Join-Path (Join-Path $WorkflowBuildRoot $SelectedArm) "push-receipt.json"
    $attemptPath = Join-Path $WorkflowBuildRoot "$SelectedArm-push-attempt.json"
    if (-not (Test-Path -LiteralPath $receiptPath -PathType Leaf)) {
        throw "Push the exact rendered $SelectedArm kernel first; receipt is missing."
    }
    if (-not (Test-Path -LiteralPath $attemptPath -PathType Leaf)) {
        throw "Kernel push receipt has no atomic push-attempt lock."
    }
    $receipt = Read-JsonFile $receiptPath
    $attempt = Read-JsonFile $attemptPath
    if ($attempt.artifact_version -ne 1 -or
        $attempt.kernel_id -ne $rendered.Metadata.id -or
        $attempt.execution_lock_sha256 -ne $rendered.Lock.execution_lock_sha256 -or
        $attempt.runner_sha256 -ne $rendered.Lock.runner_sha256 -or
        $receipt.artifact_version -ne 2 -or
        $receipt.kernel_id -ne $rendered.Metadata.id -or
        -not (Test-JsonInteger $receipt.kernel_version 1) -or
        $receipt.push_attempt_sha256 -ne (Get-Sha256 $attemptPath) -or
        $receipt.execution_lock_sha256 -ne $rendered.Lock.execution_lock_sha256 -or
        $receipt.runner_sha256 -ne $rendered.Lock.runner_sha256 -or
        $receipt.runner_source_sha256 -ne $rendered.Lock.runner_source_sha256 -or
        [string]$receipt.push_output_sha256 -notmatch "^[0-9a-f]{64}$") {
        throw "Kernel push receipt differs from the exact rendered execution."
    }
    $kernelReference = "$($receipt.kernel_id)/$($receipt.kernel_version)"
    Assert-FileSha256 $KernelVersionIo ([string]$rendered.Lock.kernel_version_io_sha256)
    $descriptionText = Invoke-NativeCapture "python" @(
        $KernelVersionIo, "describe", "--kernel", $kernelReference
    )
    try {
        $description = $descriptionText | ConvertFrom-Json
    } catch {
        throw "Kaggle returned malformed exact-version kernel metadata JSON: $descriptionText"
    }
    $remoteSources = @($description.dataset_sources)
    $expectedSources = @($rendered.Metadata.dataset_sources)
    if ($description.kernel -ne $kernelReference -or
        $description.ref -ne $receipt.kernel_id -or
        $description.source_sha256 -ne $receipt.runner_source_sha256 -or
        $description.is_private -ne $true -or $description.enable_gpu -ne $true -or
        $description.enable_internet -ne $true -or
        $description.machine_shape -ne "NvidiaTeslaT4" -or
        $remoteSources.Count -ne $expectedSources.Count -or
        @(Compare-Object -ReferenceObject $expectedSources -DifferenceObject $remoteSources -SyncWindow 0).Count -ne 0) {
        throw "Remote exact kernel version differs from the receipted source or metadata."
    }
    return [PSCustomObject]@{
        Rendered = $rendered
        Receipt = $receipt
        KernelReference = $kernelReference
    }
}

function Wait-PushReceiptReady {
    param([string]$KaggleOwner, [string]$SelectedArm, [int]$TimeoutSeconds = 600)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $pushed = Assert-PushReceipt $KaggleOwner $SelectedArm
            Assert-KernelAccepted $pushed.KernelReference
            return $pushed
        } catch {
            $message = [string]$_.Exception.Message
            if ($message -match "^(Rendered kernel|Kernel push receipt|Remote exact kernel|Dataset is public|At-use helper hash)") {
                throw
            }
            if ((Get-Date) -ge $deadline) { throw }
            Start-Sleep -Seconds 10
        }
    }
    throw "Timed out waiting for the receipted kernel version to become queryable."
}

function Download-KernelOutput {
    param(
        [string]$KernelReference,
        [string]$Target,
        [bool]$RequireComplete = $true,
        [scriptblock]$Validator = $null
    )
    if (Test-Path -LiteralPath $Target) { throw "Refusing to overwrite download: $Target" }
    if ($RequireComplete) { Wait-KernelComplete $KernelReference }
    $parent = Split-Path -Parent $Target
    Ensure-Directory $parent
    $temporary = Join-Path $parent ("." + (Split-Path -Leaf $Target) + "." + [guid]::NewGuid().ToString("N") + ".tmp")
    New-Item -ItemType Directory -Path $temporary | Out-Null
    try {
        Assert-FileSha256 $KernelVersionIo $script:ExpectedVersionIoSha256
        Invoke-Native "python" @(
            $KernelVersionIo, "output", "--kernel", $KernelReference, "--target", $temporary
        )
        if ($null -ne $Validator) { & $Validator $temporary }
        Move-Item -LiteralPath $temporary -Destination $Target
    } catch {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Recurse -Force
        }
        throw
    }
    return (Resolve-Path -LiteralPath $Target).Path
}

function Create-StateDataset {
    param([string]$KaggleOwner)
    $basePush = Assert-PushReceipt $KaggleOwner "base"
    $source = Join-Path $WorkflowDownloadsRoot "base\kaggle_export"
    if (-not (Test-Path -LiteralPath (Join-Path $source "state_manifest.json") -PathType Leaf)) {
        throw "Download base output first; state_manifest.json is missing."
    }
    Assert-CompleteBaseState `
        $source "$KaggleOwner/$BaseStateSlug" $Mode `
        ([string]$basePush.Rendered.Lock.payload_manifest_sha256) `
        ([string]$basePush.Rendered.Lock.config_fingerprint) `
        ([string]$basePush.Rendered.Lock.execution_lock_sha256) `
        ([int64]$basePush.Rendered.Lock.expected_rows) `
        ([string[]]@($basePush.Rendered.Lock.conditions))
    Ensure-Directory $WorkflowStagingRoot
    $target = Join-Path $WorkflowStagingRoot "$BaseStateSlug-state-dataset"
    if (Test-Path -LiteralPath $target) {
        throw "Refusing to overwrite state dataset staging: $target"
    }
    New-Item -ItemType Directory -Path $target | Out-Null
    foreach ($item in @(Get-ChildItem -LiteralPath $source -Force)) {
        Copy-Item -LiteralPath $item.FullName -Destination $target -Recurse -Force
    }
    $metadata = [ordered]@{
        title = $BaseStateSlug
        id = "$KaggleOwner/$BaseStateSlug"
        licenses = @(@{ name = "GPL-3.0" })
        isPrivate = $true
    } | ConvertTo-Json -Depth 5
    [IO.File]::WriteAllText((Join-Path $target "dataset-metadata.json"), $metadata + "`n", [Text.UTF8Encoding]::new($false))
    Invoke-Kaggle @("datasets", "create", "-p", $target, "--dir-mode", "zip")
    Wait-DatasetReady "$KaggleOwner/$BaseStateSlug" 1 | Out-Null
}

function Show-Help {
    @"
Kaggle T4x2 FP16-primary / NF4-sensitivity API workflow (PowerShell 5.1):
  CheckAuth                Validate standard credential and kaggle==2.2.3.
  CreateInputDataset       Verify/build fresh payload, then create private dataset.
  VersionInputDataset      Verify/build fresh payload, then upload a private version.
  RenderKernel             Render a byte-locked base/tuned runner and metadata.
  PushBase                 Push the already-rendered base kernel.
  Status                   Show status for -Arm base|tuned.
  Logs                     Download receipted selected kernel output/log files safely.
  DownloadBase             Wait for and download receipted base output to kaggle/downloads/<mode>/base.
  CreateBaseStateDataset   Publish downloaded complete base state privately.
  PushTuned                Download receipted base output again, then push rendered tuned kernel.
  DownloadFinal            Wait for and download receipted tuned output to kaggle/downloads/<mode>/final.

Select -Mode fp16-primary|nf4-sensitivity. Required for payload actions: -Config, -PreparedRun.
RenderKernel uses -Arm; tuned requires
-StateSlug equal to BaseStateSlug, while base state input is rejected. Override -Owner only when it
differs from kaggle.json.
Reuse the same -StagingDir on every action for one payload. Payload staging is write-once; choose a
fresh -StagingDir for a new dataset version. Rendered build directories are also write-once; use a
fresh worktree after a partial render rather than editing generated locks.
"@
}

if ($Action -eq "Help") { Show-Help; exit 0 }

Assert-KaggleClient
$KaggleOwner = Get-KaggleOwner
Assert-Slug $InputSlug "InputSlug"
Assert-Slug $BaseStateSlug "BaseStateSlug"
Assert-Slug $FinalStateSlug "FinalStateSlug"
Assert-Slug $BaseKernelSlug "BaseKernelSlug"
Assert-Slug $TunedKernelSlug "TunedKernelSlug"
Assert-ModeSlug $InputSlug "InputSlug"
Assert-ModeSlug $BaseStateSlug "BaseStateSlug"
Assert-ModeSlug $FinalStateSlug "FinalStateSlug"
Assert-ModeSlug $BaseKernelSlug "BaseKernelSlug"
Assert-ModeSlug $TunedKernelSlug "TunedKernelSlug"
$datasetSlugs = @($InputSlug, $BaseStateSlug, $FinalStateSlug)
if (@($datasetSlugs | Select-Object -Unique).Count -ne $datasetSlugs.Count) {
    throw "InputSlug, BaseStateSlug, and FinalStateSlug must be distinct."
}
if ($BaseKernelSlug -eq $TunedKernelSlug) {
    throw "BaseKernelSlug and TunedKernelSlug must be distinct."
}
switch ($Action) {
    "CheckAuth" { Write-Output "Authenticated Kaggle API owner: $KaggleOwner" }
    "CreateInputDataset" {
        $target = New-InputStaging $KaggleOwner
        Get-VerifiedPayloadBinding (Join-Path $target "payload_manifest.json") $KaggleOwner $Config | Out-Null
        Invoke-Kaggle @("datasets", "create", "-p", $target, "--dir-mode", "zip")
        Wait-DatasetReady "$KaggleOwner/$InputSlug" 1 | Out-Null
    }
    "VersionInputDataset" {
        $target = New-InputStaging $KaggleOwner
        Get-VerifiedPayloadBinding (Join-Path $target "payload_manifest.json") $KaggleOwner $Config | Out-Null
        $before = Get-DatasetStatusInfo "$KaggleOwner/$InputSlug"
        Assert-PrivateDataset `
            "$KaggleOwner/$InputSlug" ([int64]$before.current_version_number) | Out-Null
        Invoke-Kaggle @("datasets", "version", "-p", $target, "-m", "verified $Mode payload", "--dir-mode", "zip")
        Wait-DatasetReady "$KaggleOwner/$InputSlug" ([int64]$before.current_version_number + 1) | Out-Null
    }
    "RenderKernel" { Render-Kernel $KaggleOwner $Arm $StateSlug | Write-Output }
    "PushBase" { Push-Kernel $KaggleOwner "base" }
    "Status" {
        $pushed = Assert-PushReceipt $KaggleOwner $Arm
        $status = Get-KernelStatus $pushed.KernelReference
        Write-Output "$($pushed.KernelReference) has status `"$status`""
    }
    "Logs" {
        $pushed = Assert-PushReceipt $KaggleOwner $Arm
        $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        Ensure-Directory $WorkflowDownloadsRoot
        Download-KernelOutput $pushed.KernelReference (Join-Path $WorkflowDownloadsRoot "logs-$Arm-$stamp") $false | Write-Output
    }
    "DownloadBase" {
        $pushed = Assert-PushReceipt $KaggleOwner "base"
        Ensure-Directory $WorkflowDownloadsRoot
        $validator = {
            param($root)
            Assert-CompleteBaseState `
                (Join-Path $root "kaggle_export") "$KaggleOwner/$BaseStateSlug" $Mode `
                ([string]$pushed.Rendered.Lock.payload_manifest_sha256) `
                ([string]$pushed.Rendered.Lock.config_fingerprint) `
                ([string]$pushed.Rendered.Lock.execution_lock_sha256) `
                ([int64]$pushed.Rendered.Lock.expected_rows) `
                ([string[]]@($pushed.Rendered.Lock.conditions))
        }
        $download = Download-KernelOutput `
            $pushed.KernelReference (Join-Path $WorkflowDownloadsRoot "base") $true $validator
        $download | Write-Output
    }
    "CreateBaseStateDataset" { Create-StateDataset $KaggleOwner }
    "PushTuned" {
        $rendered = Assert-RenderedKernel $KaggleOwner "tuned"
        Wait-DatasetReady `
            "$KaggleOwner/$BaseStateSlug" ([int64]$rendered.Lock.state_dataset_version) | Out-Null
        $basePush = Assert-PushReceipt $KaggleOwner "base"
        $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        Ensure-Directory $WorkflowDownloadsRoot
        $baseValidator = {
            param($root)
            Assert-CompleteBaseState `
                (Join-Path $root "kaggle_export") "$KaggleOwner/$BaseStateSlug" $Mode `
                ([string]$basePush.Rendered.Lock.payload_manifest_sha256) `
                ([string]$basePush.Rendered.Lock.config_fingerprint) `
                ([string]$basePush.Rendered.Lock.execution_lock_sha256) `
                ([int64]$basePush.Rendered.Lock.expected_rows) `
                ([string[]]@($basePush.Rendered.Lock.conditions))
        }
        $latest = Download-KernelOutput `
            $basePush.KernelReference `
            (Join-Path $WorkflowDownloadsRoot "base-before-tuned-$stamp") $true $baseValidator
        $latestManifest = Join-Path $latest "kaggle_export\state_manifest.json"
        $publishedManifest = Join-Path $WorkflowDownloadsRoot "base\kaggle_export\state_manifest.json"
        if (-not (Test-Path -LiteralPath $latestManifest -PathType Leaf) -or
            -not (Test-Path -LiteralPath $publishedManifest -PathType Leaf)) {
            throw "Cannot prove that the latest base output matches the published state dataset."
        }
        if ((Get-Sha256 $latestManifest) -ne (Get-Sha256 $publishedManifest)) {
            throw "Latest base state differs from the state used to create the tuned input dataset."
        }
        Push-Kernel $KaggleOwner "tuned"
    }
    "DownloadFinal" {
        $pushed = Assert-PushReceipt $KaggleOwner "tuned"
        Ensure-Directory $WorkflowDownloadsRoot
        $validator = {
            param($root)
            Assert-CompleteFinalState `
                (Join-Path $root "kaggle_export") `
                "$KaggleOwner/$FinalStateSlug" `
                $Mode `
                ([string]$pushed.Rendered.Lock.state_manifest_sha256) `
                ([string]$pushed.Rendered.Lock.payload_manifest_sha256) `
                ([string]$pushed.Rendered.Lock.config_fingerprint) `
                ([string]$pushed.Rendered.Lock.execution_lock_sha256) `
                ([int64]$pushed.Rendered.Lock.expected_rows) `
                ([string[]]@($pushed.Rendered.Lock.conditions))
        }
        $download = Download-KernelOutput `
            $pushed.KernelReference (Join-Path $WorkflowDownloadsRoot "final") $true $validator
        $download | Write-Output
    }
    default { throw "Unknown action '$Action'. Run -Action Help." }
}
