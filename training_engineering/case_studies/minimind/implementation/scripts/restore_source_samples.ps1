param(
    [ValidateSet("pretrain", "sft")]
    [string]$DatasetType = "pretrain"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$sourceDir = Join-Path $root "data/source_samples"
$manifest = Get-Content -LiteralPath (Join-Path $sourceDir "manifest.json") -Raw -Encoding UTF8 | ConvertFrom-Json
$sampleName = "${DatasetType}_sample.jsonl"
$entry = $manifest.samples.PSObject.Properties[$sampleName].Value
if ($null -eq $entry) {
    throw "No manifest entry for $sampleName"
}
$samplePath = Join-Path $sourceDir $sampleName

if (Test-Path -LiteralPath $samplePath) {
    $existingHash = (Get-FileHash -LiteralPath $samplePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($existingHash -ne $entry.sha256) {
        throw "Existing sample differs from manifest; refusing to overwrite $samplePath"
    }
    Write-Output "Verified existing sample: $samplePath"
    exit 0
}

$chunkDir = Join-Path $root "data/download_chunks"
New-Item -ItemType Directory -Path $chunkDir -Force | Out-Null
$chunkPath = Join-Path $chunkDir "${DatasetType}.part"
$url = "https://huggingface.co/datasets/$($manifest.source_repo)/resolve/$($manifest.source_revision)/$($entry.upstream_file)"

& curl.exe --fail --location --range $manifest.extraction.byte_range --max-filesize 2097152 --connect-timeout 20 --max-time 120 --retry 2 $url --output $chunkPath
if ($LASTEXITCODE -ne 0) {
    throw "Sample download failed with exit code $LASTEXITCODE"
}

# Preserve the original bytes and only decode complete lines, not the partial tail.
$bytes = [System.IO.File]::ReadAllBytes($chunkPath)
$lineCount = 0
$lastLineEnd = -1
for ($i = 0; $i -lt $bytes.Length; $i++) {
    if ($bytes[$i] -eq 10) {
        $lineCount++
        if ($lineCount -eq $entry.record_count) {
            $lastLineEnd = $i
            break
        }
    }
}
if ($lastLineEnd -lt 0) {
    throw "Downloaded chunk has fewer than $($entry.record_count) complete lines"
}
[byte[]]$sampleBytes = $bytes[0..$lastLineEnd]
$utf8 = [System.Text.UTF8Encoding]::new($false, $true)
$text = $utf8.GetString($sampleBytes)
foreach ($line in ($text -split "`n" | Select-Object -First $entry.record_count)) {
    $null = $line | ConvertFrom-Json
}

$sha = [System.Security.Cryptography.SHA256]::Create()
try {
    $actualHash = ([System.BitConverter]::ToString($sha.ComputeHash($sampleBytes))).Replace("-", "").ToLowerInvariant()
} finally {
    $sha.Dispose()
}
if ($actualHash -ne $entry.sha256) {
    throw "Sample SHA256 differs from manifest: $actualHash"
}

[System.IO.File]::WriteAllBytes($samplePath, $sampleBytes)
Write-Output "Restored $($entry.record_count) records: $samplePath"
Write-Output "SHA256: $actualHash"
