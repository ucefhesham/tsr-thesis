#!/bin/bash

# Senior Benchmark Suite: Orchestrated Training Suite (Cloud Edition)
# Usage: ./scripts/train_all.sh

# Exit on error
set -e

LOG_FILE="train_all.log"

log() {
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo -e "\033[0;36m[$timestamp] $1\033[0m"
    echo "[$timestamp] $1" >> "$LOG_FILE"
}

log "--- Initializing FULL THESIS BENCHMARK SUITE (Lightning Cloud) ---"

# Array of models to train (Single-pass)
MODELS=(
    "evidential"
    "convnext_tiny"
    "convnext_tiny_evidential"
)

# Step 1: Train 5 Independent Base ResNet-18 blocks for Deep Ensembles Comparison
log "Starting Run: Deep Ensemble Base (N=5 ResNet-18)"
for i in {1..5}; do
    log "Training Ensemble Node $i/5"
    seed=$((40 + i))
    # Dynamically seed each run to enforce distinct variance across the nodes
    python train.py model=resnet18 trainer.max_epochs=50 +seed=$seed trainer.devices=1 trainer.accelerator="gpu" | tee -a "$LOG_FILE"
    log "Successfully Completed Ensemble Node $i/5"
done

# Step 2: Single-pass / Evidential Models
for model in "${MODELS[@]}"; do
    log "Starting Run: $model"
    python train.py model="$model" trainer.max_epochs=50 trainer.devices=1 trainer.accelerator="gpu" | tee -a "$LOG_FILE"
    log "Successfully Completed: $model"
done

log "--- BENCHMARK SUITE COMPLETE ---"
log "Results stored in benchmarks/ folder. Use scripts/fetch_results.sh to sync back home."
