import sys
import os

# --- CRITICAL: Windows Environment Isolation ---
# Prevent system-wide NumPy (often 2.x) from leaking into dataloader workers.
# This must happen BEFORE any other imports to ensure strict isolation.
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
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import List
import torch
import cv2
import os
import csv
from datetime import datetime
import shutil
from src.metrics.efficiency import compute_model_flops

# Disable OpenCV multithreading to prevent hangs in multi-worker dataloaders
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
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
    L.seed_everything(42, workers=True)

    # SECURE: Windows-specific memory commitment check
    # Helps diagnose "paging file too small" errors during worker spawning
    if os.name == 'nt':
        try:
            import subprocess
            cmd = "Get-CimInstance Win32_OperatingSystem | Select-Object FreeVirtualMemory, TotalVirtualMemorySize"
            output = subprocess.check_output(["powershell", "-Command", cmd], text=True)
            print(f"\n--- System Environment Check ---")
            print(output.strip())
            print("Note: If 'FreeVirtualMemory' is low, consider reducing num_workers or increasing Paging File.\n")
        except Exception:
            pass

    # SECURE: Allowlist Hydra/OmegaConf for older PyTorch versions where needed
    # (Note: add_safe_globals is only available in torch 2.4+, skipping for 2.0.1)
    pass

    # Instantiate Loggers from configuration
    logger: List[L.loggers.Logger] = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            print(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    # Instantiate Callbacks (ModelCheckpoint, EarlyStopping, etc.)
    callbacks: List[L.Callback] = []
    if "callbacks" in cfg:
        for name, cb_conf in cfg.callbacks.items():
            print(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))


    # Instantiate DataModule (GTSRB) from configuration
    print(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    # Instantiate Model from configuration
    print(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

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
    # Supports automatic resumption if ckpt_path is provided via Hydra config
    ckpt_path = cfg.get("ckpt_path")
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    print("\n--- Phase 2: Evaluation Lifecycle ---")
    # Finding the best checkpoint found by the callback
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
        model_name = cfg.model.name
        benchmark_dir = os.path.join(cfg.paths.root_dir, "benchmarks", model_name)
        ledger_path = os.path.join(benchmark_dir, "run_ledger.csv")
        
        # 1. Compute Efficiency Metrics
        gflops = compute_model_flops(model, input_size=(1, 3, cfg.datamodule.input_size, cfg.datamodule.input_size))
        
        # 2. Append to Ledger
        headers = ["Timestamp", "Best_Checkpoint", "Top1_Acc", "SWE", "ESP", "ECE", "Brier", "GFLOPs"]
        file_exists = os.path.isfile(ledger_path)
        
        with open(ledger_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                os.path.basename(best_ckpt_path),
                res.get("test/acc", 0.0),
                res.get("test/swe", 0.0),
                res.get("test/esp", 0.0),
                res.get("test/ece", 0.0),
                res.get("test/brier", 0.0),
                gflops
            ])
            
        print(f"Run data permanently logged to: {ledger_path}")
        
        # 3. Convenience Copy: latest_completed.ckpt
        if best_ckpt_path:
            latest_path = os.path.join(benchmark_dir, "latest_completed.ckpt")
            shutil.copy(best_ckpt_path, latest_path)
            print(f"Best weight file promoted to parent directory: {latest_path}")

if __name__ == "__main__":
    train()
