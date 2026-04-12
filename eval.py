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
from src.models.calibration import ModelWithTemperature, IsotonicCalibrator, IsotonicInferenceWrapper, SafePolicyWrapper
from src.metrics.efficiency import compute_model_flops
from src.models.trust_selector import ConformalRiskSuite
from src.utils.xai_utils import TrustInterpreter
from src.datamodules.stress_suite import StressSuiteDataModule
from tools.trust_dashboard import create_trust_dashboard
from pytorch_lightning.callbacks import RichProgressBar
import time
import numpy as np
import torch.nn.functional as F
from scipy.stats import spearmanr
from src.models.detector import TrafficSignDetector

def run_full_pipeline_test(cfg: DictConfig):
    """
    Two-Stage Evaluation Loop (YOLOv11 + Evidential Classifier)
    """
    print("\n=== Executing End-to-End Perception Pipeline ===")
    results_path = "logs/perception_pipeline_results.csv"
    with open(results_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Stress_Severity", "mIoU", "Miss_Rate", "Combined_Accuracy", "FP_Vacuity", "Uncertainty_Overlap_Corr", "Latency_ms"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Classifier
    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        print("[ERROR] Classifier ckpt_path required for pipeline test.")
        return
        
    print(f"Loading classifier from: {ckpt_path}")
    classifier = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.to(device)
    classifier.eval()

    # Load Detector
    print("Loading YOLOv11 Detector...")
    detector = TrafficSignDetector(model_path="yolo11n.pt").to(device)
    
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    for severity in range(0, 6):
        print(f"\n[Pipeline Test] Stress Severity: {severity}")
        if severity > 0:
            dm = StressSuiteDataModule(
                corruption_type="blur", # Representative stress
                severity=severity,
                data_dir=cfg.datamodule.data_dir,
                input_size=cfg.datamodule.input_size,
                batch_size=1, # Process 1-by-1 for precise detection testing
                num_workers=cfg.datamodule.num_workers
            )
            dm.setup(stage="test")
            loader = dm.test_dataloader()
        else:
            loader = test_loader
            
        all_ious = []
        all_vacuities = []
        fp_vacuities = []
        correct_combined = 0
        total_gt = 0
        missed = 0
        total_latency = 0.0
        
        for i, batch in enumerate(loader):
            if i >= 100: break # Limit instances for testing duration constraints
            images, labels = batch
            for idx in range(images.size(0)):
                img = images[idx].to(device)
                lbl = labels[idx].item()
                total_gt += 1
                
                # Assume GT box covers the resized image for dataset fallback
                w, h = img.shape[2], img.shape[1]
                gt_box = [0, 0, w, h] 
                
                start_time = time.time()
                crops, metadata = detector.detect_crops(img)
                
                if len(crops) == 0:
                    missed += 1
                    total_latency += (time.time() - start_time)
                    continue
                
                best_iou = 0
                best_crop = None
                
                for crop, meta in zip(crops, metadata):
                    det_box = meta["box"]
                    ix1 = max(gt_box[0], det_box[0])
                    iy1 = max(gt_box[1], det_box[1])
                    ix2 = min(gt_box[2], det_box[2])
                    iy2 = min(gt_box[3], det_box[3])
                    
                    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    det_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
                    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                    iou = inter_area / float(det_area + gt_area - inter_area + 1e-6)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_crop = crop
                        
                    # False Positive Tracking (0 IoU)
                    if iou < 0.1:
                        with torch.no_grad():
                            resized_crop = F.interpolate(crop.unsqueeze(0), size=(cfg.datamodule.input_size, cfg.datamodule.input_size), mode='bilinear')
                            out = classifier(resized_crop)
                            if isinstance(out, dict) and "vacuity" in out:
                                fp_vacuities.append(out["vacuity"].mean().item())
                                
                if best_iou > 0.3 and best_crop is not None:
                    all_ious.append(best_iou)
                    with torch.no_grad():
                        resized_crop = F.interpolate(best_crop.unsqueeze(0), size=(cfg.datamodule.input_size, cfg.datamodule.input_size), mode='bilinear')
                        out = classifier(resized_crop)
                        if isinstance(out, dict):
                            vacuity = out.get("vacuity", torch.zeros(1)).mean().item()
                            all_vacuities.append(vacuity)
                            pred = out["prob"].max(1)[1].item()
                        else:
                            pred = torch.softmax(out, dim=1).max(1)[1].item()
                            all_vacuities.append(0.0)
                            
                        if pred == lbl:
                            correct_combined += 1
                else:
                    missed += 1
                total_latency += (time.time() - start_time)
                
        # Aggregate Metrics
        mIoU = np.mean(all_ious) if all_ious else 0.0
        miss_rate = missed / max(total_gt, 1)
        combined_acc = correct_combined / max(total_gt, 1)
        mean_fp_vacuity = np.mean(fp_vacuities) if fp_vacuities else 0.0
        lat_ms = (total_latency / max(total_gt, 1)) * 1000
        
        # Uncertainty Overlap Coefficient (Correlation)
        if len(all_ious) > 1 and len(all_vacuities) > 1:
            loc_errors = [1.0 - iou for iou in all_ious]
            corr, _ = spearmanr(loc_errors, all_vacuities)
            if np.isnan(corr): corr = 0.0
        else:
            corr = 0.0
            
        with open(results_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([severity, mIoU, miss_rate, combined_acc, mean_fp_vacuity, corr, lat_ms])
            
    print(f"\nPipeline evaluation complete. Dashboard metrics exported to: {results_path}")

def fgsm_attack(image, epsilon, data_grad):
    """Simple FGSM perturbation."""
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    # Re-normalize to ensure the image is still in range (roughly)
    # We clip here after clamping based on norm if needed, but 0-1 is standard
    perturbed_image = torch.clamp(perturbed_image, -2.5, 2.5) # Based on standard normalization
    return perturbed_image

@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def evaluate(cfg: DictConfig):
    """
    Stress Evaluation Suite for Traffic Sign Classifiers.
    """
    if cfg.get("mode") == "full_pipeline":
        return run_full_pipeline_test(cfg)
        
    from omegaconf.listconfig import ListConfig
    from omegaconf.base import ContainerMetadata
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata])
    
    os.makedirs("logs", exist_ok=True)
    
    model_name = "evidential" if "evidential" in cfg.model._target_.lower() else "resnet"
    results_path = f"logs/{model_name}_stress_test_results.csv"
    
    with open(results_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Corruption", "Severity", "Calibration_Method", "Accuracy", "ECE", "AECE", "STOP_ECE", "SWE", "ESP", "GFLOPs", "Latency_ms", "TPE"])

    # Instantiate DataModule
    print(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup(stage="test")

    # Load Model with Automated Checkpoint Resolution
    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        print("[INFO] No ckpt_path provided. Searching for the best local checkpoint...")
        # Search in standard logs directory based on model target
        search_dir = "logs/checkpoints" # Assumed generic ModelCheckpoint path, or traverse hydra dir
        
        # We can scan the active directory or the hydra output tree for recent checkpoints
        import glob
        possible_ckpts = []
        # Look for checkpoints matching the current model configuration
        model_type = "evidential" if "evidential" in cfg.model._target_.lower() else ("convnext" if "convnext" in cfg.model._target_.lower() else "resnet18")
        
        # Generic wildcard search through logs
        wildcard_path = os.path.join(os.getcwd(), "**", f"*{model_type}*.ckpt")
        possible_ckpts = glob.glob(wildcard_path, recursive=True)
        # Fallback to general epochs
        if not possible_ckpts:
            possible_ckpts = glob.glob(os.path.join(os.getcwd(), "logs", "**", "*.ckpt"), recursive=True)
            
        if possible_ckpts:
            # Pick the most recently modified checkpoint
            ckpt_path = max(possible_ckpts, key=os.path.getmtime)
            print(f"[AUTO-RESOLVED] Found best checkpoint: {ckpt_path}")
        else:
            print("\n[ERROR] No checkpoint found automatically. Provide ckpt_path=/path/to/checkpoint.ckpt")
            return

    print(f"Loading model weights from: {ckpt_path}")
    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    
    # Efficiency Profiling
    gflops = compute_model_flops(model.backbone, input_res=cfg.datamodule.input_size)
    print(f"Model Complexity: {gflops:.3f} GFLOPs")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Detect underlying backbone attribute (Evidential uses .model, others use .backbone)
    backbone_attr = "model" if hasattr(model, "model") else "backbone"
    original_backbone = getattr(model, backbone_attr)

    # --- Phase 0: Post-Hoc Calibration ---
    # Fix: Use dynamic num_workers to prevent Windows memory commitment crashes instead of hardcoding '6'
    datamodule.hparams.num_workers = cfg.datamodule.get("num_workers", 2)
    cal_loader = datamodule.calibration_dataloader()
    
    cal_model = ModelWithTemperature(original_backbone)
    cal_model.to(device)
    cal_model.set_temperature(cal_loader)

    # 2. Isotonic Calibration
    isotonic_cal = IsotonicCalibrator(num_classes=cfg.model.num_classes)
    
    # Collect probabilities for Isotonic fit
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for x, y in cal_loader:
            x, y = x.to(device), y.to(device)
            # Use forward to get prob for evidential or softmax for others
            if hasattr(model, "model"): # Evidential
                out = model.model(x)
                probs = out["prob"]
            else:
                probs = torch.softmax(model.backbone(x), dim=1)
            all_probs.append(probs)
            all_targets.append(y)
    
    isotonic_cal.fit(torch.cat(all_probs), torch.cat(all_targets))
    isotonic_inference = IsotonicInferenceWrapper(original_backbone, isotonic_cal).to(device)

    # --- Phase 0.1: Conformal Selection ---
    print("\n--- Phase 0.1: Conformal Risk Calibration ---")
    conformal_suite = ConformalRiskSuite(alphas=[0.01, 0.05])
    
    # Get all calibration scores
    model.eval()
    all_cal_probs = []
    all_cal_targets = []
    with torch.no_grad():
        for x, y in cal_loader:
            x, y = x.to(device), y.to(device)
            if hasattr(model, "forward"): # Evidential handles it via forward
                outputs = model(x)
                probs = outputs["prob"]
            else:
                probs = torch.softmax(model.backbone(x), dim=1)
            all_cal_probs.append(probs)
            all_cal_targets.append(y)
    
    conformal_suite.calibrate_all(torch.cat(all_cal_probs), torch.cat(all_cal_targets))

    # --- Setup Evaluations ---
    trainer = L.Trainer(accelerator="auto", devices="auto", precision="16-mixed", logger=False)
    evaluation_modes = [
        ("None", original_backbone), 
        ("Temperature_Scaling", cal_model),
        ("Isotonic_Calibration", isotonic_inference),
        ("Safe_Policy_Fallback", SafePolicyWrapper(model))
    ]
    
    # Store Base Metrics for TPE calculation
    base_metrics = {"ece": 1.0, "gflops": gflops} 

    # --- Phase 1: Standard Evaluation ---
    print("\n--- Phase 1: Standard Evaluation & Conformal Coverage ---")
    for mode_name, backbone in evaluation_modes:
        print(f"\nEvaluating Baseline [Mode: {mode_name}]...")
        setattr(model, backbone_attr, backbone)
        
        start_time = time.time()
        results = trainer.test(model=model, datamodule=datamodule, ckpt_path=None)
        latency = (time.time() - start_time) / (len(datamodule.data_test) / cfg.datamodule.batch_size)
        
        res = results[0]
        curr_ece = res.get("test/ece", 0.0)
        curr_aece = res.get("test/aece", 0.0)
        curr_stop_ece = res.get("test/ece_class_14", 0.0)
            
        # Calculate TPE relative to "None" mode (if it's the first run, store as base)
        if mode_name == "None":
            base_metrics["ece"] = curr_ece
            tpe = 0.0
        else:
            ece_gain = base_metrics["ece"] - curr_ece
            tpe = ece_gain / max(gflops - base_metrics["gflops"], 1e-6) if gflops > base_metrics["gflops"] else ece_gain
                
        with open(results_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["clean", 0, mode_name, res["test/acc"], curr_ece, curr_aece, curr_stop_ece, res["test/swe"], res["test/esp"], gflops, latency * 1000, tpe])
        print(f"\n>>> Running Robustness Test: {test_name} <<<")
        test_dm.prepare_data()
        test_dm.setup(stage="test")
        
        for mode_name, backbone in evaluation_modes:
            setattr(model, backbone_attr, backbone)
            results = trainer.test(model=model, datamodule=test_dm, ckpt_path=None)
            if results:
                res = results[0]
                curr_ece = res.get("test/ece", 0.0)
                curr_aece = res.get("test/aece", 0.0)
                curr_stop_ece = res.get("test/ece_class_14", 0.0)
                ece_gain = base_metrics["ece"] - curr_ece
                tpe = ece_gain / max(gflops - base_metrics["gflops"], 1e-6) if mode_name != "None" else 0.0
                
                with open(results_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([test_name, 5, mode_name, res.get("test/acc", 0.0), curr_ece, curr_aece, curr_stop_ece, res.get("test/swe", 0.0), res.get("test/esp", 0.0), gflops, 0.0, tpe])

    # --- Phase 3: Stress Sweep ---
    print("\n--- Phase 3: Systematic 5-Level Stress Sweep ---")
    categories = ["noise", "blur", "fog"]
    for corruption in categories:
        for severity in range(1, 6): # Full 1-5 severity sweep as requested
            print(f"\n[Stress] Corruption: {corruption} | Severity: {severity}")
            stress_dm = StressSuiteDataModule(
                corruption_type=corruption, 
                severity=severity,
                data_dir=cfg.datamodule.data_dir,
                input_size=cfg.datamodule.input_size,
                batch_size=cfg.datamodule.batch_size,
                num_workers=cfg.datamodule.num_workers
            )
            stress_dm.setup(stage="test")
            
            for mode_name, backbone in evaluation_modes:
                setattr(model, backbone_attr, backbone)
                results = trainer.test(model=model, datamodule=stress_dm, ckpt_path=None)
                if results:
                    res = results[0]
                    curr_ece = res.get("test/ece", 0.0)
                    curr_aece = res.get("test/aece", 0.0)
                    curr_stop_ece = res.get("test/ece_class_14", 0.0)
                    ece_gain = base_metrics["ece"] - curr_ece
                    tpe = ece_gain / max(gflops - base_metrics["gflops"], 1e-6) if mode_name != "None" else 0.0
                    
                    with open(results_path, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([corruption, severity, mode_name, res["test/acc"], curr_ece, curr_aece, curr_stop_ece, res["test/swe"], res["test/esp"], gflops, 0.0, tpe])

    # --- Phase 5: Adversarial Safety Stress (FGSM) ---
    print("\n--- Phase 5: Adversarial Safety Stress (FGSM) ---")
    epsilons = [0.01, 0.05, 0.1, 0.2]
    model.eval()
    
    for eps in epsilons:
        print(f"\n[Adversarial] FGSM Attack | Epsilon: {eps}")
        
        # We'll run a manual eval loop for FGSM to handle the gradient step
        correct = 0
        total = 0
        all_vacuity = []
        all_entropy = []
        
        # Use a subset of the test set for speed in the adversarial sweep
        test_loader = datamodule.test_dataloader()
        limit = 50 # batches (approx 1600 images)
        
        for i, (images, labels) in enumerate(test_loader):
            if i >= limit: break
            
            images, labels = images.to(device), labels.to(device)
            images.requires_grad = True
            
            # Forward pass to get logits for grad
            # Ensure we are using the base model for grad calculation
            setattr(model, backbone_attr, original_backbone)
            outputs = model(images)
            if isinstance(outputs, dict):
                logits = outputs.get("logits", outputs.get("prob", outputs))
            else:
                logits = outputs
                
            loss = F.cross_entropy(logits, labels)
            model.zero_grad()
            loss.backward()
            
            # Attack and re-eval
            data_grad = images.grad.data
            perturbed_data = fgsm_attack(images, eps, data_grad)
            
            with torch.no_grad():
                # Re-predict using the currently active evaluation mode
                for mode_name, backbone in evaluation_modes:
                    setattr(model, backbone_attr, backbone)
                    out = model(perturbed_data)
                    
                    if isinstance(out, dict):
                        probs = out["prob"]
                        vacuity = out.get("vacuity", torch.zeros(1)).mean().item()
                        entropy = out.get("entropy", torch.zeros(1)).mean().item()
                    else:
                        probs = torch.softmax(out, dim=1)
                        # Manual entropy for non-dict outputs
                        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean().item()
                        vacuity = 0.0
                    
                    preds = probs.max(1)[1]
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    
                    # Store results for the specific model/eps
                    # Columns: Corruption, Severity, Calibration_Method, Accuracy, ECE, AECE, STOP_ECE, SWE, ESP, GFLOPs, Latency_ms, TPE
                    with open(results_path, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([f"FGSM_eps_{eps}", 5, mode_name, (preds == labels).float().mean().item(), 0.0, 0.0, 0.0, 0.0, 0.0, gflops, 0.0, 0.0])
        
        print(f"FGSM Epsilon {eps} complete.")

    # --- Phase 4: Qualitative Trust Analysis (XAI) ---
    print("\n--- Phase 4: Qualitative Trust Analysis (Grad-CAM) ---")
    xai_dir = "logs/xai_insights"
    os.makedirs(xai_dir, exist_ok=True)
    
    # Identify Target Layers for Grad-CAM
    if model_name == "evidential":
        target_layers = [model.model.feature_extractor[-1]]
    else:
        target_layers = [model.backbone.features[-1]]

    interpreter = TrustInterpreter(model, target_layers=target_layers)
    
    # 1. Select a few interesting samples from Sticker Occlusion AND Jitter
    from src.datamodules.robustness_datamodule import DetectionJitterTransform
    jitter_fn = DetectionJitterTransform(translation_limit=0.12, scale_limit=0.12)
    
    # We'll use the clean test set and apply transforms manually for more control
    datamodule.setup(stage="test")
    raw_test_set = datamodule.data_test
    
    print(f"Generating Trust Dashboards for challenging samples in {xai_dir}...")
    model.eval()
    
    # Scenarios to visualize
    scenarios = [
        ("sticker", True, False),
        ("jitter", False, True)
    ]
    
    samples_saved = 0
    with torch.no_grad():
        for name, play_sticker, play_jitter in scenarios:
            for i in range(3): # 3 samples per scenario
                idx = random.randint(0, len(raw_test_set)-1)
                img_pil, y_true = raw_test_set[idx]
                w, h = img_pil.size
                bbox = (0, 0, w, h)
                
                # Apply simulated errors
                if play_jitter:
                    tx, ty, s = jitter_fn.get_params(w, h)
                    img_pil = jitter_fn(img_pil)
                    bbox = jitter_fn.apply_to_bbox(bbox, tx, ty, s, w, h)
                
                # Convert to Tensor for model
                # Use standard eval transforms (Resize + Norm)
                from torchvision import transforms
                prep = transforms.Compose([
                    transforms.Resize((cfg.datamodule.input_size, cfg.datamodule.input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                x_input = prep(img_pil).to(device).unsqueeze(0)
                
                # Predict for the best mode (Isotonic)
                setattr(model, backbone_attr, isotonic_inference)
                out = model(x_input)
                
                # Scale BBox to resolution
                rw, rh = cfg.datamodule.input_size, cfg.datamodule.input_size
                bbox_res = (
                    int(bbox[0] * rw / w), int(bbox[1] * rh / h),
                    int(bbox[2] * rw / w), int(bbox[3] * rh / h)
                )
                
                # Get image for visualization
                img_np = np.array(img_pil.resize((rw, rh)))
                
                # Switch back to torch.set_grad_enabled(True) for CAM calculation
                with torch.enable_grad():
                    maps = interpreter.generate_maps(x_input, img_np, bbox=bbox_res)
                
                # Prepare Metrics for Dashboard
                metrics = {
                    "gt_label": y_true,
                    "pred_class": maps["class_id"],
                    "vacuity": out["vacuity"][0].item() if isinstance(out, dict) else 0.0,
                    "entropy": out["entropy"][0].item() if isinstance(out, dict) else 0.0,
                    "energy": out["energy"][0].item() if isinstance(out, dict) else 0.0,
                    "sfe_score": maps["sfe"],
                    "probs": out["prob"][0].cpu().numpy() if isinstance(out, dict) else torch.softmax(out, dim=1)[0].cpu().numpy()
                }
                
                create_trust_dashboard(
                    img_np, 
                    maps["class_map"], 
                    maps["uncertainty_map"], 
                    metrics, 
                    f"{xai_dir}/{name}_{i}_trust.png"
                )
                samples_saved += 1

    print(f"\n--- Stress Evaluation Complete. Results: {results_path} ---")
    print(f"--- Qualitative Insights Saved to: {xai_dir} ---")

if __name__ == "__main__":
    evaluate()
