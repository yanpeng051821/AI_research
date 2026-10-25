param(
    [Parameter(Mandatory = $true)]
    [string]$ProcessDirectory
)

$ErrorActionPreference = 'Stop'
$implementationRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$processRoot = [IO.Path]::GetFullPath($ProcessDirectory)
if (-not $processRoot.StartsWith($implementationRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Process directory must be inside the implementation workspace.'
}
if (-not (Test-Path -LiteralPath $processRoot -PathType Container)) {
    throw 'Create a new process directory before starting the supervisor.'
}

$statePath = Join-Path $processRoot 'process.json'
$state = [ordered]@{
    status = 'starting'
    supervisor_pid = $PID
    python_pid = $null
    started_at_utc = [DateTime]::UtcNow.ToString('o')
    working_directory = $implementationRoot
    stdout = (Join-Path $processRoot 'stdout.log')
    stderr = (Join-Path $processRoot 'stderr.log')
    exit_code = $null
}

function Save-ProcessState {
    $temporary = $statePath + '.pending'
    $json = $state | ConvertTo-Json -Depth 5
    [IO.File]::WriteAllText($temporary, $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $statePath -Force
}

try {
    Add-Type @'
using System.Runtime.InteropServices;
public static class PretrainPowerState {
    [DllImport("kernel32.dll")]
    private static extern uint SetThreadExecutionState(uint flags);
    public static bool Hold() { return SetThreadExecutionState(0x80000001u) != 0; }
    public static void Release() { SetThreadExecutionState(0x80000000u); }
}
'@
    $state['prevent_idle_sleep'] = [PretrainPowerState]::Hold()
    if (-not $state['prevent_idle_sleep']) {
        throw 'Unable to prevent idle system sleep for the background run.'
    }
    $python = Join-Path $implementationRoot '.venv\Scripts\python.exe'
    $arguments = @(
        '-u', 'scripts/train_frozen_pretrain.py',
        '--freeze', '../experiments/pretrain_mini_one_epoch_v1/training_recipe/training_freeze.json',
        '--run', '--report', 'runs/pretrain_mini_one_epoch_v1.frozen-launch.json'
    )
    $state['command'] = @($python) + $arguments
    Save-ProcessState
    $process = Start-Process -FilePath $python -ArgumentList $arguments `
        -WorkingDirectory $implementationRoot -WindowStyle Hidden `
        -RedirectStandardOutput $state['stdout'] -RedirectStandardError $state['stderr'] -PassThru
    $null = $process.Handle
    $state['python_pid'] = $process.Id
    $state['status'] = 'running'
    Save-ProcessState
    $process.WaitForExit()
    $state['exit_code'] = $process.ExitCode
    $state['status'] = if ($process.ExitCode -eq 0) { 'exited_successfully' } else { 'failed' }
}
catch {
    $state['status'] = 'supervisor_failed'
    $state['error'] = $_.Exception.Message
}
finally {
    if ('PretrainPowerState' -as [type]) { [PretrainPowerState]::Release() }
    $state['finished_at_utc'] = [DateTime]::UtcNow.ToString('o')
    Save-ProcessState
}
