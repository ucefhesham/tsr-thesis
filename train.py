import hydra
from omegaconf import DictConfig
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb

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

    # Initialize Weights & Biases Logger for experiment tracking
    wandb_logger = WandbLogger(project='trust-tsr-thesis')

    # Instantiate DataModule (GTSRB) from configuration
    print(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    # Instantiate Model (ResNetBaselineModule) from configuration
    print(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    # Initialize PyTorch Lightning Trainer
    # Hardware acceleration is set to 'auto' to support both GPU and CPU environments
    trainer = L.Trainer(
        max_epochs=cfg.model.max_epochs,
        accelerator='auto',
        devices=1,
        logger=wandb_logger,
        deterministic=True  # Force deterministic algorithms where possible
    )

    # Optional: Log the model architecture and gradients
    wandb_logger.watch(model, log="all")

    print("\n--- Phase 1: Training Lifecycle ---")
    # Fit the model using the training and validation sets
    trainer.fit(model, datamodule=datamodule)

    print("\n--- Phase 2: Evaluation Lifecycle ---")
    # Immediately evaluate accuracy and reliability metrics on the official GTSRB test split
    # Uses the 'best' checkpoint found during the validation phase
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

if __name__ == "__main__":
    train()
