# Lightning Studios Result Retrieval Script (Windows Edition)
# Usage: ./scripts/fetch_results.ps1

$StudioName = "scratch-studio-devbox"
$RemotePath = "/home/teamspace/studios/this_studio/benchmarks"
$LocalPath = "./benchmarks"

Write-Host "--- Initializing Benchmark Retrieval from Lightning Cloud ---" -ForegroundColor Cyan

# Ensure the Lightning CLI is available
if (!(Get-Command lightning -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Lightning CLI not found locally. Please run: pip install lightning" -ForegroundColor Red
    exit 1
}

Write-Host "Syncing results from Studio: $StudioName..." -ForegroundColor Yellow

# Download the benchmarks folder
# -s specifies the Studio name
# --delete can be added to ensure a exact mirror, but we'll stick to additive for safety
lightning download $RemotePath "." --studio $StudioName

Write-Host "--- Sync Complete ---" -ForegroundColor Green
Write-Host "Local ledger (run_ledger.csv) and XAI results updated."
Write-Host "You can now run: python tools/aggregate_benchmarks.py"
