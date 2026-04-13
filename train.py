import sys
import os

# --- CRITICAL: Windows Environment Isolation ---
os.environ["PYTHONNOUSERSITE"] = "1"
venv_site = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "Lib", "site-packages")
if os.path.exists(venv_site):
    if venv_site not in sys.path:
        sys.path.insert(0, venv_site)
    elif sys.path[0] != venv_site:
        sys.path.remove(venv_site)
        sys.path.insert(0, venv_site)

import hydra
from omegaconf import DictConfig
import pytorch_lightning as L
from typing import List
import torch
import cv2
import csv
from datetime import datetime
import shutil
from src.metrics.efficiency import compute_model_flops

# Disable OpenCV multithreading to prevent hangs in multi-worker dataloaders
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import wandb

@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def train(cfg: DictConfig):
    """
    Main training script for Trust Analysis of Traffic Sign Classifiers.
    """
    # Set seed for reproducibility
    seed = cfg.get("seed", 42)
    L.seed_everything(seed)

    # 1. Init DataModule
    print(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # 2. Init Model
    print(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    # 3. Init Callbacks & Loggers
    callbacks: List[L.Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                print(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    logger = []
    wandb_name = "default_run"
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                print(f"Instantiating logger <{lg_conf._target_}>")
                # Capture wandb name for folder organization
                if "wandb" in lg_conf._target_.lower():
                    wandb_name = lg_conf.get("name", "unnamed_wandb_run")
                logger.append(hydra.utils.instantiate(lg_conf))

    # 4. Init Trainer
    print(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger
    )

    # --- Phase 1: Standard Training ---
    trainer.fit(model, datamodule=datamodule)

    # --- Phase 2: Evaluation & Best Model Recovery ---
    best_ckpt_path = ""
    for cb in callbacks:
        if isinstance(cb, L.callbacks.ModelCheckpoint):
            best_ckpt_path = cb.best_model_path
            break
            
    # Loading 'best' weights for final report
    results = trainer.test(model, datamodule=datamodule, ckpt_path=best_ckpt_path if best_ckpt_path else "best")

    # --- Phase 3: Run Ledger & Permanent Tracking ---
    if results:
        res = results[0]
        # Use wandb_name if model_name is generic (like resnet18_baseline)
        model_name = cfg.model.name
        run_name = wandb_name if wandb_name != "unnamed_wandb_run" else model_name
        
        benchmark_dir = os.path.join(cfg.paths.root_dir, "benchmarks", run_name)
        ledger_path = os.path.join(benchmark_dir, "run_ledger.csv")
        
        # 1. Compute Efficiency Metrics (Corrected Input Argument)
        gflops = compute_model_flops(model, input_res=cfg.datamodule.input_size)
        
        # 2. Append to Ledger
        headers = ["Timestamp", "Best_Checkpoint", "Top1_Acc", "SWE", "ESP", "ECE", "Brier", "GFLOPs"]
        os.makedirs(benchmark_dir, exist_ok=True)
        file_exists = os.path.isfile(ledger_path)
        
        with open(ledger_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                os.path.basename(best_ckpt_path) if best_ckpt_path else "final",
                res.get("test/acc", 0.0),
                res.get("test/swe", 0.0), # Selection-With-Evidence / Risk
                res.get("test/esp", 0.0), # Expected Selection Performance
                res.get("test/ece", 0.0),
                res.get("test/brier", 0.0),
                gflops
            ])

    # Shutdown WandB cleanly
    if logger:
        wandb.finish()

if __name__ == "__main__":
    train()
