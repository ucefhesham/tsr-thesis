#!/bin/bash
# scripts/train_ensemble.sh
# Automates the training of 5 independent ResNet-18 models for Deep Ensemble baseline.

SEEDS=(42 43 44 45 46)
MODEL="resnet18"
EPOCHS=30

echo "Starting Deep Ensemble Training ($#SEEDS models)..."

for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    NAME="resnet18_ensemble_m$i"
    
    echo "------------------------------------------------"
    echo "Training Member $i (Seed: $SEED, Name: $NAME)"
    echo "------------------------------------------------"
    
    python train.py \
        model=$MODEL \
        trainer.max_epochs=$EPOCHS \
        seed=$SEED \
        model.name=$NAME \
        logger.name=$NAME \
        datamodule.num_workers=4
        
    echo "Member $i complete."
done

echo "Double Ensemble Training Complete. Check checkpoints/ for results."
