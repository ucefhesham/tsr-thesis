import hydra
from omegaconf import DictConfig
import pytorch_lightning as L
from typing import List
import torch
import cv2
import os
import random
import csv
import gc
import time
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Disable OpenCV multithreading to prevent hangs in multi-worker dataloaders
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from src.transforms.corruptions import TrustStressTester
from src.models.calibration import ModelWithTemperature, IsotonicCalibrator, IsotonicInferenceWrapper, SafePolicyWrapper
from src.metrics.efficiency import compute_model_flops
from src.models.trust_selector import ConformalRiskSuite
from src.utils.xai_utils import TrustInterpreter
from src.datamodules.stress_suite import StressSuiteDataModule
from tools.trust_dashboard import create_trust_dashboard
from src.datamodules.btsc_module import BTSCDataModule
from src.datamodules.robustness_datamodule import StickerOcclusionDataModule, JunkDataModule, DetectionJitterTransform

def run_domain_shift_suite(cfg: DictConfig, model: L.LightningModule, trainer: L.Trainer, evaluation_modes: List):
    """RQ Shift: Measure German-to-Belgium (BTSC) domain shift spikes."""
    print("\n=== Phase 5: Epistemic Domain Shift (BTSC) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_name = "evidential" if "evidential" in cfg.model._target_.lower() else "resnet"
    results_path = f"logs/{model_name}_domain_shift_results.csv"
    with open(results_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Calibration_Method", "Accuracy", "ECE", "Vacuity_Mean", "Entropy_Mean"])

    btsc_dm = BTSCDataModule(data_dir=cfg.datamodule.data_dir, batch_size=cfg.datamodule.batch_size)
    btsc_dm.prepare_data()
    btsc_dm.setup(stage="test")
    
    if btsc_dm.dataset is None:
        print("[SKIP] BTSC Dataset failed to download/load.")
        return

    test_loader = btsc_dm.test_dataloader()
    if test_loader is None:
        print("[SKIP] BTSC Dataloader is empty or failed to load.")
        return

    backbone_attr = "model" if hasattr(model, "model") else "backbone"
    for mode_name, backbone in evaluation_modes:
        print(f"[BTSC] Mode: {mode_name}")
        # Synchronize and register the submodule explicitly
        backbone.to(device)
        model.add_module(backbone_attr, backbone) 
        model.to(device)
        
        res = trainer.test(model=model, dataloaders=test_loader)[0]
        
        with open(results_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["BTSC", mode_name, res["test/acc"], res.get("test/ece", 0.0), res.get("test/vacuity", 0.0), res.get("test/entropy", 0.0)])

def run_physical_robustness_suite(cfg: DictConfig, model: L.LightningModule, trainer: L.Trainer, evaluation_modes: List):
    """RQ Robustness: Measure 'Sticker' flagging performance."""
    print("\n=== Phase 6: Physical Robustness (Stickers) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_name = "evidential" if "evidential" in cfg.model._target_.lower() else "resnet"
    results_path = f"logs/{model_name}_physical_robustness_results.csv"
    with open(results_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Sticker_Prob", "Calibration_Method", "Accuracy", "Vacuity", "Detection_Recall"])

    backbone_attr = "model" if hasattr(model, "model") else "backbone"
    for prob in [0.5, 1.0]:
        sticker_dm = StickerOcclusionDataModule(sticker_prob=prob, data_dir=cfg.datamodule.data_dir, num_workers=cfg.datamodule.num_workers)
        sticker_dm.setup(stage="test")
        
        for mode_name, backbone in evaluation_modes:
            backbone.to(device)
            model.add_module(backbone_attr, backbone)
            model.to(device)
            res = trainer.test(model=model, dataloaders=sticker_dm.test_dataloader())[0]
            with open(results_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([prob, mode_name, res["test/acc"], res.get("test/vacuity", 0.0), 1.0 if res.get("test/vacuity", 0.0) > 0.5 else 0.0])

def plot_risk_coverage(cfg: DictConfig, model: L.LightningModule, test_loader: torch.utils.data.DataLoader, suffix=""):
    """RQ3/RQ4: Plot Risk-Coverage curve."""
    print("\n--- Generating Risk-Coverage Plots ---")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_scores = {"entropy": [], "energy": [], "vacuity": [], "msp": []}
    all_is_correct = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if isinstance(out, dict):
                probs = out["prob"]
                all_scores["vacuity"].append(out.get("vacuity", torch.zeros_like(y).float()))
                all_scores["entropy"].append(out.get("entropy", -torch.sum(probs * torch.log(probs + 1e-10), dim=1)))
                all_scores["energy"].append(out.get("energy", torch.logsumexp(out.get("logits", out["prob"]), dim=1)))
                all_scores["msp"].append(probs.max(1)[0])
            else:
                probs = torch.softmax(out, dim=1)
                all_scores["msp"].append(probs.max(1)[0])
                all_scores["entropy"].append(-torch.sum(probs * torch.log(probs + 1e-10), dim=1))
                all_scores["energy"].append(torch.logsumexp(out, dim=1))
                all_scores["vacuity"].append(torch.zeros_like(y).float())
            
            all_is_correct.append((probs.max(1)[1] == y).float())

    for k in all_scores: all_scores[k] = torch.cat(all_scores[k]).cpu().numpy()
    all_is_correct = torch.cat(all_is_correct).cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    for name, scores in all_scores.items():
        if np.all(scores == 0): continue
        scores_to_plot = 1.0 - scores if name == "msp" else (scores if name != "energy" else -scores)
        idx = np.argsort(scores_to_plot)
        sorted_correct = all_is_correct[idx]
        coverage = np.linspace(0, 1, len(sorted_correct))
        rolling_acc = np.cumsum(sorted_correct) / (np.arange(len(sorted_correct)) + 1)
        risk = 1.0 - rolling_acc
        plt.plot(coverage, risk, label=f"Proxy: {name.upper()}")
        
    plt.xlabel("Coverage")
    plt.ylabel("Risk")
    plt.title(f"Risk-Coverage Curve {suffix}")
    plt.legend()
    plt.grid(True)
    out_path = f"logs/risk_coverage_{suffix.lower().replace(' ', '_')}.png"
    plt.savefig(out_path)
    print(f"✅ Risk-Coverage saved to {out_path}")

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, -2.5, 2.5)
    return perturbed_image

@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def evaluate_suites(cfg: DictConfig):
    os.makedirs("logs", exist_ok=True)
    model_name = "evidential" if "evidential" in cfg.model._target_.lower() else "resnet"
    
    # --- Phase 0: Logger Integration ---
    from pytorch_lightning.loggers import WandbLogger
    logger = False
    if "logger" in cfg and "wandb" in cfg.logger:
        run_name = f"eval_suites_{model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"Instantiating WandB Logger: {run_name}")
        logger = hydra.utils.instantiate(cfg.logger.wandb, name=run_name)

    # 1. Setup Data
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup(stage="test")
    
    # 2. Load Model
    ckpt_path = cfg.get("ckpt_path")
    print(f"Loading model from: {ckpt_path}")
    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Detect underlying backbone attribute (Evidential uses .net, others use .backbone or .model)
    if hasattr(model, "net"):
        backbone_attr = "net"
    elif hasattr(model, "model"):
        backbone_attr = "model"
    else:
        backbone_attr = "backbone"
        
    original_backbone = getattr(model, backbone_attr)
    print(f"✅ Detected backbone attribute: {backbone_attr}")

    # 3. Setup Calibration (Fast Pass)
    cal_loader = datamodule.calibration_dataloader()
    cal_model = ModelWithTemperature(original_backbone)
    cal_model.to(device)
    cal_model.set_temperature(cal_loader)
    
    isotonic_cal = IsotonicCalibrator(num_classes=cfg.model.num_classes)
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for x, y in cal_loader:
            x, y = x.to(device), y.to(device)
            # Direct backbone inference for probability collection
            out = original_backbone(x)
            if isinstance(out, dict):
                p = out["prob"]
            else:
                p = torch.softmax(out, dim=1)
            all_probs.append(p)
            all_targets.append(y)
    isotonic_cal.fit(torch.cat(all_probs), torch.cat(all_targets))
    isotonic_inference = IsotonicInferenceWrapper(original_backbone, isotonic_cal).to(device)

    evaluation_modes = [
        ("None", original_backbone.to(device)), 
        ("Temperature_Scaling", cal_model.to(device)),
        ("Isotonic_Calibration", isotonic_inference.to(device)),
        ("Safe_Policy_Fallback", SafePolicyWrapper(original_backbone).to(device))
    ]
    
    trainer = L.Trainer(accelerator="auto", devices="auto", precision="16-mixed", logger=logger)
    results_path = f"logs/{model_name}_suites_results.csv"
    with open(results_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Phase", "Metric", "Mode", "Value"])

    # --- Phase 5: Adversarial ---
    print("\n--- Phase 5: Adversarial Stress (FGSM) ---")
    epsilons = [0.01, 0.05, 0.1, 0.2]
    model.to(device)
    model.eval()
    
    for eps in epsilons:
        test_loader = datamodule.test_dataloader()
        limit = 50
        for i, (images, labels) in enumerate(test_loader):
            if i >= limit: break
            images, labels = images.to(device), labels.to(device)
            images.requires_grad = True
            
            # Direct backbone inference for attack generation
            outputs = original_backbone(images)
            logits = outputs.get("logits", outputs.get("prob", outputs)) if isinstance(outputs, dict) else outputs
            loss = F.cross_entropy(logits, labels)
            model.zero_grad()
            loss.backward()
            perturbed_data = fgsm_attack(images, eps, images.grad.data)
            
            with torch.no_grad():
                for mode_name, backbone in evaluation_modes:
                    # DIRECT INFERENCE: No attribute swapping on the LightningModule
                    out = backbone(perturbed_data)
                    probs = out["prob"] if isinstance(out, dict) else torch.softmax(out, dim=1)
                    acc = (probs.max(1)[1] == labels).float().mean().item()
                    with open(results_path, mode="a", newline="") as f:
                        csv.writer(f).writerow(["Adversarial", f"FGSM_eps_{eps}", mode_name, acc])

    # --- Ph.D. Suites ---
    run_domain_shift_suite(cfg, model, trainer, evaluation_modes)
    run_physical_robustness_suite(cfg, model, trainer, evaluation_modes)
    plot_risk_coverage(cfg, model, datamodule.test_dataloader(), suffix=model_name)

if __name__ == "__main__":
    evaluate_suites()
