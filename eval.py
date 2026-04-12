import hydra
from omegaconf import DictConfig
import pytorch_lightning as L
from typing import List
import torch
import cv2
import os

# Disable OpenCV multithreading to prevent hangs in multi-worker dataloaders
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import csv
import os
import gc
from src.transforms.corruptions import TrustStressTester
from src.models.calibration import ModelWithTemperature
from src.metrics.efficiency import compute_model_flops
from pytorch_lightning.callbacks import RichProgressBar

@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def evaluate(cfg: DictConfig):
    """
    Stress Evaluation Suite for Traffic Sign Classifiers.
    
    Orchestrates a comprehensive sweep across 4 corruption categories and 
    5 severity levels to evaluate model performance and calibration under uncertainty.
    Results are exported to CSV for thesis plotting.
    """
    # SECURE: Allowlist Hydra/OmegaConf for PyTorch 2.6+ loading
    from omegaconf.listconfig import ListConfig
    from omegaconf.base import ContainerMetadata
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata])
    
    # Ensure logs directory exists for result extraction
    os.makedirs("logs", exist_ok=True)
    
    # Senior ML Engineer: Use dynamic naming based on the model module path 
    # to avoid overwriting ResNet vs Evidential results.
    model_name = "evidential" if "evidential" in cfg.model._target_.lower() else "resnet"
    results_path = f"logs/{model_name}_stress_test_results.csv"
    
    # Initialize CSV with Header
    # Added GFLOPs to help plot the Trust-Per-Efficiency (TPE) metric
    with open(results_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Corruption", "Severity", "Calibration_Method", "Accuracy", "ECE", "GFLOPs"])

    # Instantiate DataModule
    print(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    
    # Ensure data is downloaded (especially on Colab)
    datamodule.prepare_data()
    datamodule.setup(stage="test")

    # Load Model from checkpoint
    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        print("\n[ERROR] Please provide a valid checkpoint path using 'ckpt_path=/path/to/checkpoint.ckpt'")
        return

    print(f"Loading model weights from: {ckpt_path}")
    model = hydra.utils.instantiate(cfg.model)
    # Load state dict manually to prevent mismatches when backbone is swapped later
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    
    # --- PHASE -1: Efficiency Profiling ---
    # Compute GFLOPs once for the backbone
    gflops = compute_model_flops(model.backbone, input_res=cfg.datamodule.input_size)
    print(f"Model Complexity: {gflops:.3f} GFLOPs")
    
    # Instantiate Trainer for evaluation
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        logger=False, 
        callbacks=[], # Disabled RichProgressBar to fix IndexError on older Lightning versions
    )

    print("\n--- Phase 0: Post-Hoc Calibration ---")
    # Use 0 workers for calibration set to avoid MemoryError during process spawning
    datamodule.hparams.num_workers = 0
    cal_loader = datamodule.calibration_dataloader()
    
    # Instantiate the wrapper with the existing backbone
    # We use model.backbone (ResNet-18) to ensure we're scaling the raw features
    original_backbone = model.backbone
    cal_model = ModelWithTemperature(original_backbone)
    
    # Move model to device manually for calibration since we aren't using Trainer.test yet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    cal_model.to(device)
    
    # Calculate optimal temperature
    cal_model.set_temperature(cal_loader)

    print("\n--- Phase 1: Standard Evaluation (Clean Baseline) ---")
    evaluation_modes = [
        ("None", original_backbone),
        ("Temperature_Scaling", cal_model)
    ]

    for mode_name, backbone in evaluation_modes:
        print(f"\nEvaluating Clean Baseline [Mode: {mode_name}]...")
        model.backbone = backbone
        # USE ckpt_path=None because weights are already loaded
        results = trainer.test(model=model, datamodule=datamodule, ckpt_path=None)
        
        if results:
            with open(results_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["clean", 0, mode_name, results[0]["test/acc"], results[0]["test/ece"], gflops])

    print("\n--- Phase 2: Uncertainty Stress Sweep ---")
    # Stress sweep respects the datamodule configuration, but we disable 
    # persistent workers to prevent memory leaks during the loop.
    datamodule.hparams.persistent_workers = False
    
    categories = ["noise", "blur", "weather", "compression", "jitter"]
    severities = [1, 2, 3, 4, 5]

    for corruption in categories:
        for severity in severities:
            print(f"\n>>> Stress Test: [{corruption.upper()}] | Severity: {severity} <<<", flush=True)
            
            # 1. Create the specific Stress Transform
            stress_transform = TrustStressTester(
                corruption_type=corruption, 
                severity=severity, 
                input_size=cfg.datamodule.input_size
            )
            
            # 2. DEEP HARDENING: Overwrite the transform template and force a setup
            datamodule.eval_transforms = stress_transform
            datamodule.setup(stage="test")
            
            for mode_name, backbone in evaluation_modes:
                print(f"--- Running Eval [Mode: {mode_name}] ---")
                model.backbone = backbone
                
                # 3. CRITICAL: Re-instantiate Trainer to clear all data caches for EVERY pass
                fresh_trainer = L.Trainer(
                    accelerator="auto",
                    devices="auto",
                    precision="16-mixed",
                    logger=False,
                    callbacks=[], # Disabled RichProgressBar to fix IndexError on older Lightning versions
                )
                
                # 4. Run evaluation with ALREADY LOADED weights
                results = fresh_trainer.test(model=model, datamodule=datamodule, ckpt_path=None)
                
                # 5. Extract metrics and append to CSV
                if results:
                    acc = results[0].get("test/acc", 0.0)
                    ece = results[0].get("test/ece", 0.0)
                    with open(results_path, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([corruption, severity, mode_name, acc, ece, gflops])
                
                # Cleanup to keep memory stable
                del fresh_trainer
                gc.collect()
                torch.cuda.empty_cache()

    print(f"\n--- Stress Evaluation Complete ---")
    print(f"Results successfully exported to: {results_path}")

if __name__ == "__main__":
    evaluate()
