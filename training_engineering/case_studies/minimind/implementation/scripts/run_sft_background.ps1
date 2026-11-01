param([Parameter(Mandatory = $true)][string]$ProcessDirectory)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$directory = [IO.Path]::GetFullPath($ProcessDirectory)
if (-not $directory.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Process directory must stay within the implementation workspace.'
}
if (-not (Test-Path -LiteralPath $directory -PathType Container)) { throw 'Process directory does not exist.' }
$path = Join-Path $directory 'process.json'
if (Test-Path -LiteralPath $path) { throw 'Refusing to overwrite supervisor state.' }
$state = [ordered]@{
    status = 'starting'; supervisor_pid = $PID; python_pid = $null
    started_at_utc = [DateTime]::UtcNow.ToString('o'); working_directory = $root
    stdout = (Join-Path $directory 'stdout.log'); stderr = (Join-Path $directory 'stderr.log')
    exit_code = $null
}
function Save-State {
    [IO.File]::WriteAllText($path + '.pending', (($state | ConvertTo-Json -Depth 5) + [Environment]::NewLine), [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath ($path + '.pending') -Destination $path -Force
}
try {
    Add-Type @'
using System.Runtime.InteropServices;
public static class SftPowerState {
    [DllImport("kernel32.dll")] private static extern uint SetThreadExecutionState(uint flags);
    public static bool Hold() { return SetThreadExecutionState(0x80000001u) != 0; }
    public static void Release() { SetThreadExecutionState(0x80000000u); }
}
'@
    $state['prevent_idle_sleep'] = [SftPowerState]::Hold()
    if (-not $state['prevent_idle_sleep']) { throw 'Unable to prevent idle system sleep.' }
    $python = Join-Path $root '.venv\Scripts\python.exe'
    $arguments = @('-X', 'utf8', '-u', 'scripts/train_frozen_sft.py',
        '--freeze', '../experiments/sft_mini_v1/training_recipe/training_freeze.json',
        '--run', '--output-dir', 'runs/sft_mini_one_epoch_v1')
    $state['command'] = @($python) + $arguments
    Save-State
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $root `
        -WindowStyle Hidden -RedirectStandardOutput $state['stdout'] -RedirectStandardError $state['stderr'] -PassThru
    $null = $process.Handle
    $state['python_pid'] = $process.Id
    $state['status'] = 'running'
    Save-State
    $process.WaitForExit()
    $state['exit_code'] = $process.ExitCode
    $state['status'] = if ($process.ExitCode -eq 0) { 'exited_successfully' } else { 'failed' }
}
catch { $state['status'] = 'supervisor_failed'; $state['error'] = $_.Exception.Message }
finally {
    if ('SftPowerState' -as [type]) { [SftPowerState]::Release() }
    $state['finished_at_utc'] = [DateTime]::UtcNow.ToString('o')
    Save-State
}
