import hydra
from omegaconf import DictConfig
import pytorch_lightning as L
from typing import List
import torch
import csv
import os
from src.transforms.corruptions import TrustStressTester

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
    torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata])
    
    # Ensure logs directory exists for result extraction
    os.makedirs("logs", exist_ok=True)
    results_path = "logs/stress_test_results.csv"
    
    # Initialize CSV with Header
    with open(results_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Corruption", "Severity", "Accuracy", "ECE"])

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

    print(f"Loading model from: {ckpt_path}")
    model = hydra.utils.instantiate(cfg.model)
    
    # Instantiate Trainer for evaluation (Quiet mode with Auto-Hardware detection)
    trainer = L.Trainer(
        accelerator="auto", # Automatically detects GPU/TPU/CPU
        devices="auto",     # Automatically selects available units
        logger=False, 
        enable_progress_bar=False,
    )

    print("\n--- Phase 1: Standard Evaluation (Clean Baseline) ---")
    results = trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)
    
    # Optional: Log clean baseline to CSV as Severity 0
    if results:
        with open(results_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["clean", 0, results[0]["test/acc"], results[0]["test/ece"]])

    print("\n--- Phase 2: Uncertainty Stress Sweep ---")
    categories = ["noise", "blur", "weather", "compression"]
    severities = [1, 2, 3, 4, 5]

    for corruption in categories:
        for severity in severities:
            print(f">>> Stress Test: [{corruption.upper()}] | Severity: {severity}")
            
            # 1. Create the specific Stress Transform
            stress_transform = TrustStressTester(
                corruption_type=corruption, 
                severity=severity, 
                input_size=cfg.datamodule.input_size
            )
            
            # 2. Inject the transform into the test dataset
            datamodule.data_test.transform = stress_transform
            
            # 3. Run evaluation and capture results
            results = trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)
            
            # 4. Extract metrics and append to CSV
            if results:
                acc = results[0].get("test/acc", 0.0)
                ece = results[0].get("test/ece", 0.0)
                with open(results_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([corruption, severity, acc, ece])

    print(f"\n--- Stress Evaluation Complete ---")
    print(f"Results successfully exported to: {results_path}")

if __name__ == "__main__":
    evaluate()
