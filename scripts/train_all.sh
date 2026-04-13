#!/bin/bash

# Senior Benchmark Suite: Orchestrated Training Suite (Cloud Edition)
# Features: Smart-Skip logic to avoid redundant training of secured baselines.

# Exit on error
set -e

LOG_FILE="train_all.log"

# Stability configurations for Cloud T4 GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_SILENT=true

log() {
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo -e "\033[0;36m[$timestamp] $1\033[0m"
    echo "[$timestamp] $1" >> "$LOG_FILE"
}

echo "[$(date +'%Y-%m-%d %H:%M:%S')] --- Initializing FULL THESIS BENCHMARK SUITE (Lightning Cloud) ---"

# Array of models for the primary Trust Portfolio
MODELS=(
    "evidential"
    "mc_dropout"
    "convnext_tiny"
    "convnext_tiny_evidential"
)

# Step 1: Deep Ensemble Baseline (RQ5: Cost vs Gain)
log "🏁 Training/Verifying Deep Ensemble Nodes (Nodes 2-5)"
for i in {2..5}; do
    WANDB_NAME="resnet18_ensemble_node_$i"
    # Smart Skip Check: If ledger exists in the unique benchmark folder, skip training
    if [ -f "benchmarks/${WANDB_NAME}/run_ledger.csv" ]; then
        log "✅ Skipping $WANDB_NAME: Results already secured in local ledger."
        continue
    fi

    SEED=$((40 + i))
    log "🚀 Launching Ensemble Node $i (Seed: $SEED)"
    python train.py model=resnet18 +seed=$SEED logger.wandb.name="$WANDB_NAME" trainer.max_epochs=50 2>&1 | tee -a "$LOG_FILE"
done

# Step 2: Ultimate Trust Portfolio (RQ1-RQ6)
log "🏁 Training/Verifying Core Portfolio"
for model in "${MODELS[@]}"; do
    WANDB_NAME="${model}_research"
    # Smart Skip Check
    if [ -f "benchmarks/${WANDB_NAME}/run_ledger.csv" ]; then
        log "✅ Skipping $WANDB_NAME: Results already secured in local ledger."
        continue
    fi

    log "🚀 Launching Core Portfolio: $model"
    python train.py model="$model" +seed=42 logger.wandb.name="$WANDB_NAME" trainer.max_epochs=50 2>&1 | tee -a "$LOG_FILE"
done

log "--- BENCHMARK SUITE COMPLETE ---"
log "Ph.D. Evidence secured in benchmarks/ folder."
