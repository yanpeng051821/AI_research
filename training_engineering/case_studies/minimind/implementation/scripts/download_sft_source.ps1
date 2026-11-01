param(
    [ValidateRange(1, 16)][int]$Connections = 8,
    [ValidateRange(1, 64)][int]$ChunkMiB = 16
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = Split-Path -Parent $PSScriptRoot
$revision = '312afb4f76391145c6902f765bb51691c09a12f5'
$size = 1739201170L
$expectedHash = 'abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518'
$url = "https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/$revision/sft_t2t_mini.jsonl"
$target = Join-Path $root 'data/raw/minimind/sft_t2t_mini.jsonl'
$prefix = "$target.download"
$chunkDir = Join-Path $root 'data/download_chunks/sft_fixed_312afb4_v1'
$metadataPath = Join-Path $chunkDir 'download_identity.json'

if (Test-Path -LiteralPath $target) {
    if ((Get-Item -LiteralPath $target).Length -ne $size -or
        (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHash) {
        throw 'Existing source differs; refusing to overwrite.'
    }
    Write-Output 'Fixed SFT source already exists and its identity is verified.'
    exit 0
}
$prefixSize = if (Test-Path -LiteralPath $prefix) { (Get-Item -LiteralPath $prefix).Length } else { 0L }
if ($prefixSize -gt $size) { throw 'Partial source exceeds the expected size.' }
$chunkBytes = $ChunkMiB * 1024L * 1024L
if (Test-Path -LiteralPath $metadataPath) {
    $saved = Get-Content -LiteralPath $metadataPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($saved.revision -ne $revision -or $saved.expected_sha256 -ne $expectedHash -or
        $saved.prefix_size -ne $prefixSize -or $saved.chunk_bytes -ne $chunkBytes) {
        throw 'Download inputs changed; do not mix incompatible chunks.'
    }
} else {
    if (Test-Path -LiteralPath $chunkDir) { throw 'Unidentified chunk directory already exists.' }
    New-Item -ItemType Directory -Path $chunkDir | Out-Null
    @{ revision = $revision; expected_size = $size; expected_sha256 = $expectedHash;
       prefix_size = $prefixSize; chunk_bytes = $chunkBytes } |
        ConvertTo-Json | Set-Content -LiteralPath $metadataPath -Encoding utf8
}

$chunks = @()
$index = 0
for ($start = $prefixSize; $start -lt $size; $start += $chunkBytes) {
    $end = [Math]::Min($start + $chunkBytes, $size) - 1
    $chunks += [pscustomobject]@{ Index = $index; Start = $start; End = $end;
                                Length = $end - $start + 1;
                                Path = Join-Path $chunkDir ('{0:D4}.part' -f $index) }
    $index++
}
Write-Output "Fixed source: $size bytes; prefix: $prefixSize; ranges: $($chunks.Count); connections: $Connections."
$results = @($chunks | ForEach-Object -ThrottleLimit $Connections -Parallel {
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'
    $chunk = $_
    if ((Test-Path -LiteralPath $chunk.Path) -and
        (Get-Item -LiteralPath $chunk.Path).Length -eq $chunk.Length) {
        [pscustomobject]@{ Index = $chunk.Index; Status = 'cached' }
    } else {
        $complete = $false
        for ($attempt = 1; $attempt -le 3; $attempt++) {
            try {
                $response = Invoke-WebRequest -Uri $using:url -Headers @{
                    Range = "bytes=$($chunk.Start)-$($chunk.End)"
                } -OutFile $chunk.Path -PassThru -ConnectionTimeoutSeconds 20 -OperationTimeoutSeconds 120
                $range = $response.Headers['Content-Range'] -join ''
                if ([int]$response.StatusCode -ne 206 -or
                    $range -ne "bytes $($chunk.Start)-$($chunk.End)/$using:size" -or
                    (Get-Item -LiteralPath $chunk.Path).Length -ne $chunk.Length) {
                    throw 'Unexpected range response or segment size.'
                }
                $complete = $true
                break
            } catch {
                if ($attempt -eq 3) { throw "Segment $($chunk.Index) failed: $($_.Exception.Message)" }
            }
        }
        if ($complete) {
            Write-Host "Segment $($chunk.Index) verified ($($chunk.Length) bytes)."
            [pscustomobject]@{ Index = $chunk.Index; Status = 'downloaded' }
        }
    }
})
if ($results.Count -ne $chunks.Count) { throw 'Not all segments completed; retained chunks for retry.' }

$assembly = "$target.assembling"
if (Test-Path -LiteralPath $assembly) { throw 'An assembly file already exists; inspect it before retrying.' }
New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
$output = [IO.File]::Open($assembly, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
try {
    $inputs = @()
    if ($prefixSize -gt 0) { $inputs += $prefix }
    $inputs += @($chunks | Sort-Object Index | ForEach-Object { $_.Path })
    foreach ($path in $inputs) {
        $input = [IO.File]::OpenRead($path)
        try { $input.CopyTo($output) } finally { $input.Dispose() }
    }
} finally { $output.Dispose() }
if ((Get-Item -LiteralPath $assembly).Length -ne $size) { throw 'Assembled size mismatch.' }
$actualHash = (Get-FileHash -LiteralPath $assembly -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $expectedHash) { throw 'Assembled SHA256 mismatch; no source file was published.' }
Move-Item -LiteralPath $assembly -Destination $target
Write-Output "Verified full source: $target"
Write-Output "SHA256: $actualHash"
