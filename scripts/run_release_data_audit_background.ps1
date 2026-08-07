$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\11231\miniconda3\envs\Altolia_v1\python.exe"
$logDir = Join-Path $repoRoot "outputs\release_audit"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir "data_audit.stdout.log"
$stderr = Join-Path $logDir "data_audit.stderr.log"
$exitFile = Join-Path $logDir "data_audit.exitcode"

try {
    & $python (Join-Path $PSScriptRoot "audit_release_data.py") 1> $stdout 2> $stderr
    $code = $LASTEXITCODE
} catch {
    $_ | Out-String | Set-Content -Encoding utf8 $stderr
    $code = 99
}
Set-Content -Encoding ascii -Path $exitFile -Value $code
exit $code
