import hydra
from omegaconf import DictConfig
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import List
import torch
import wandb
import os

@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def train(cfg: DictConfig):
    """
    Main training script for Trust Analysis of Traffic Sign Classifiers.
    
    Orchestrates the lifecycle from data loading to model training and 
    final baseline evaluation using Hydra-composed configurations.
    """
    
    # CRITICAL: Strict Reproducibility
    # Ensures deterministic behavior across different runs
    L.seed_everything(42, workers=True)

    # Instantiate Loggers from configuration
    logger: List[L.loggers.Logger] = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            print(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    # Instantiate Callbacks (ModelCheckpoint, EarlyStopping, etc.)
    callbacks: List[L.Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            print(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    # Senior ML Engineer: Explicit Trust-Specific Checkpointing
    # Bypassing cfg to ensure strict adherence to optimal weight storage
    checkpoint_callback = ModelCheckpoint(
        monitor='val/ece', 
        mode='min', 
        save_top_k=1, 
        dirpath='checkpoints/', 
        filename='best-trust-baseline'
    )
    callbacks.append(checkpoint_callback)

    # Instantiate DataModule (GTSRB) from configuration
    print(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    # Instantiate Model (ResNetBaselineModule) from configuration
    print(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    # Initialize PyTorch Lightning Trainer
    print(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger, 
        callbacks=callbacks
    )

    # Optional: Log the model architecture and gradients if using Wandb
    if logger:
        for lg in logger:
            if isinstance(lg, L.loggers.WandbLogger):
                lg.watch(model, log="all")

    print("\n--- Phase 1: Training Lifecycle ---")
    # Fit the model using the training and validation sets
    trainer.fit(model, datamodule=datamodule)

    print("\n--- Phase 2: Evaluation Lifecycle ---")
    # Immediately evaluate accuracy and reliability metrics on the official GTSRB test split
    # Explicitly loads the 'best' checkpoint (optimal validation ECE) instead of final weights
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

if __name__ == "__main__":
    train()
