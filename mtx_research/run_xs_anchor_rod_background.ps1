$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$OutDir = Join-Path $ProjectRoot "report_outputs\xs_anchor_rod_18816"
$LogPath = Join-Path $OutDir "run_last.log"
$HtmlPath = Join-Path $OutDir "xs_anchor_rod_report.html"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Write-RunLog {
    param([string]$Message)
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8
}

Set-Content -LiteralPath $LogPath -Value "" -Encoding UTF8
Write-RunLog "Start MTX XS Anchor ROD 18816 backtest"
Write-RunLog "Project=$ProjectRoot"
Write-RunLog "Instrument=mtx"
Write-RunLog "Output=$OutDir"

try {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        $cmd = "py"
        $args = @(
            "-3",
            "mtx_research\run_xs_anchor_rod.py",
            "--instrument", "mtx",
            "--outdir", "report_outputs\xs_anchor_rod_18816",
            "--progress-every", "100000"
        )
    } else {
        $cmd = "python"
        $args = @(
            "mtx_research\run_xs_anchor_rod.py",
            "--instrument", "mtx",
            "--outdir", "report_outputs\xs_anchor_rod_18816",
            "--progress-every", "100000"
        )
    }

    Write-RunLog "Command=$cmd $($args -join ' ')"
    & $cmd @args 2>&1 | ForEach-Object {
        Add-Content -LiteralPath $LogPath -Value $_.ToString() -Encoding UTF8
    }
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        throw "Backtest failed with exit code $exitCode"
    }

    if (-not (Test-Path -LiteralPath $HtmlPath)) {
        throw "HTML report not found: $HtmlPath"
    }

    Write-RunLog "Finished successfully"
    Write-RunLog "Open HTML=$HtmlPath"
    Start-Process -FilePath $HtmlPath
} catch {
    Write-RunLog "ERROR: $($_.Exception.Message)"
    Start-Process -FilePath "notepad.exe" -ArgumentList "`"$LogPath`""
}
