# Senior Benchmark Suite: Orchestrated Training Suite
# Usage: ./scripts/train_all.ps1

$ErrorActionPreference = "Stop"
$LogFile = "train_all.log"

function Log($msg) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $msg"
    Write-Host $line -ForegroundColor Cyan
    $line | Out-File -FilePath $LogFile -Append
}

Log("--- Initializing FULL THESIS BENCHMARK SUITE ---")

$Models = @(
    "evidential",
    "convnext_tiny",
    "convnext_tiny_evidential"
)

# Step 1: Train 5 Independent Base ResNet-18 blocks for Deep Ensembles Comparison
Log("Starting Run: Deep Ensemble Base (N=5 ResNet-18)")
for ($i=1; $i -le 5; $i++) {
    Log("Training Ensemble Node $i/5")
    try {
        $seed = 40 + $i
        # Dynamically seed each run to enforce distinct variance across the nodes
        python train.py model=resnet18 trainer.max_epochs=50 +seed=$seed | Tee-Object -FilePath $LogFile -Append
        Log("Successfully Completed Ensemble Node $i/5")
    } catch {
        Log("CRITICAL FAILURE during Ensemble Node $i. Check logs.")
        exit 1
    }
}

# Step 2: Single-pass / Evidential Models
foreach ($model in $Models) {
    Log("Starting Run: $model")
    try {
        # Note: Datamodule num_workers relies on cfg defaults (e.g. 2 instances to prevent OS memory blocks)
        python train.py model=$model trainer.max_epochs=50 | Tee-Object -FilePath $LogFile -Append
        Log("Successfully Completed: $model")
    } catch {
        Log("CRITICAL FAILURE during $model. Check logs.")
        exit 1
    }
}

Log("--- BENCHMARK SUITE COMPLETE ---")
Log("Results stored in checkpoints. Run python eval.py directly or looping over modes.")
