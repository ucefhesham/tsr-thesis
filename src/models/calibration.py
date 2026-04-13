import torch
import torch.nn as nn
from typing import Any
import torch.optim as optim
from torchmetrics.classification import CalibrationError
from typing import Optional
import numpy as np
from sklearn.isotonic import IsotonicRegression
from src.metrics.safety_config import CLASS_METADATA

def ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Bulletproof 2D reshape: Ensures (Batch, Classes) format."""
    if tensor is None:
        return torch.zeros((1, 1)) # Last resort empty
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    return tensor

class ModelWithTemperature(nn.Module):
    """
    Temperature Scaling wrapper for post-hoc calibration of multiclass classifiers.
    Based on "On Calibration of Modern Neural Networks" (Guo et al., 2017).
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # Learnable temperature parameter initialized to 1.5 as per requirements
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def _ensure_device(self, x: torch.Tensor):
        """Self-healing device sync: Ensures model matches input device."""
        # Check first parameter's device
        try:
            device = next(self.parameters()).device
            if device != x.device:
                self.to(x.device)
        except StopIteration:
            # No parameters, just move self
            self.to(x.device)

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass returns temperature-scaled output, preserving signature."""
        self._ensure_device(x)
        out = self.model(x)
        
        # Handle dict (evidential) or tensor (baseline)
        if isinstance(out, dict):
            # Explicit key lookup with safe fallbacks
            if "logits" in out:
                logits = out["logits"]
            elif "prob" in out:
                logits = torch.log(out["prob"] + 1e-9)
            elif "alpha" in out:
                # Infer probs from alpha if needed
                alpha = out["alpha"]
                probs = alpha / torch.sum(alpha, dim=-1, keepdim=True)
                logits = torch.log(probs + 1e-9)
            else:
                raise KeyError(f"Evidential output dict missing required keys. Found: {list(out.keys())}")
            
            logits = ensure_2d(logits)
            scaled_logits = self.temperature_scale(logits)
            
            # Reconstruct the dictionary with calibrated probabilities
            new_probs = torch.softmax(scaled_logits, dim=1)
            out["prob"] = new_probs
            # Essential: Reconstruct 'alpha' to satisfy EvidentialModule.forward
            out["alpha"] = new_probs * 100.0 
            return out
        else:
            logits = ensure_2d(out)
            return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by the temperature parameter."""
        # Expand temperature to match batch size
        temperature = self.temperature.expand(logits.size(0), 1)
        return logits / temperature

    def set_temperature(self, dataloader: torch.utils.data.DataLoader):
        """
        Extracts logits and labels from the dataloader and optimizes the 
        temperature parameter using LBFGS.
        """
        self.model.eval()
        n_classes = getattr(self.model, "num_classes", 43) # Default to 43 for GTSRB
        device = next(self.parameters()).device
        
        logits_list = []
        labels_list = []
        
        print(f"Extracting logits for calibration on device: {device}...")
        with torch.no_grad():
            for input, label in dataloader:
                input = input.to(device)
                out = self.model(input)
                # Support both raw tensors (baseline) and dictionaries (evidential)
                if isinstance(out, dict):
                    # Extract prob or logits if available, otherwise prob
                    # Note: temperature scaling is typically on logits, 
                    # but if only probs are available, we use log(probs)
                    val = out.get("logits", out.get("prob"))
                    logits_list.append(val.detach().cpu())
                    labels_list.append(label.cpu())
                else:
                    logits_list.append(out.detach().cpu())
                    labels_list.append(label.cpu())
            
            # Concatenate all collected data
            logits = torch.cat(logits_list).to(device)
            labels = torch.cat(labels_list).to(device)

        # Define metrics for Before/After comparison
        ece_metric = CalibrationError(task="multiclass", num_classes=n_classes, n_bins=15).to(device)

        # Calculate ECE BEFORE calibration
        before_ece = ece_metric(torch.softmax(logits, dim=1), labels)
        print(f"Before Temperature Scaling - ECE: {before_ece:.4f}")

        # Optimization Setup
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        criterion = nn.CrossEntropyLoss().to(device)

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        # Calculate ECE AFTER calibration
        after_ece = ece_metric(torch.softmax(self.temperature_scale(logits), dim=1), labels)
        print(f"After Temperature Scaling - ECE: {after_ece:.4f}")
        print(f"Optimal Temperature: {self.temperature.item():.4f}")

        return self

class IsotonicCalibrator:
    """
    Non-parametric Multiclass Isotonic Calibration (One-vs-Rest).
    Ref: "Predicting Good Probabilities with Supervised Learning" (Niculescu-Mizil & Caruana, 2005)
    """
    def __init__(self, num_classes: int = 43):
        self.num_classes = num_classes
        # We instantiate one isotonic regressor per class
        self.calibrators = [IsotonicRegression(out_of_bounds='clip') for _ in range(num_classes)]
        self.is_fitted = False

    def fit(self, probs: torch.Tensor, targets: torch.Tensor):
        """
        Fits the isotonic regressors on a validation set.
        Args:
            probs: Predicted probabilities [N, K]
            targets: Ground truth labels [N]
        """
        probs_np = probs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        print(f"Fitting Isotonic Calibrators for {self.num_classes} classes...")
        for i in range(self.num_classes):
            # Binary target for the current class
            target_bin = (targets_np == i).astype(float)
            # Probabilities predicted for the current class
            prob_i = probs_np[:, i]
            
            # Isotonic Regression requires non-decreasing mapping
            self.calibrators[i].fit(prob_i, target_bin)
            
        self.is_fitted = True
        return self

    def calibrate(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Applies the fitted calibration and normalizes result.
        """
        if not self.is_fitted:
            return probs
            
        device = probs.device
        probs_np = probs.detach().cpu().numpy()
        calibrated = np.zeros_like(probs_np)
        
        for i in range(self.num_classes):
            calibrated[:, i] = self.calibrators[i].transform(probs_np[:, i])
            
        # Post-hoc normalization (Multiclass requirement)
        row_sums = calibrated.sum(axis=1, keepdims=True)
        # Avoid division by zero in rare edge cases (e.g. all-zero pred)
        row_sums[row_sums == 0] = 1.0
        calibrated /= row_sums
        
        return torch.from_numpy(calibrated).to(probs.dtype).to(device)

class IsotonicInferenceWrapper(nn.Module):
    """
    Module wrapper that applies Isotonic Calibration to the output of a backbone.
    This behaves like an nn.Module so it can be swapped into LightningModules.
    """
    def __init__(self, model: nn.Module, calibrator: IsotonicCalibrator):
        super().__init__()
        self.model = model
        self.calibrator = calibrator

    def _ensure_device(self, x: torch.Tensor):
        """Self-healing device sync: Ensures model matches input device."""
        try:
            device = next(self.model.parameters()).device
            if device != x.device:
                self.model.to(x.device)
        except StopIteration:
            pass

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass returns calibrated output, preserving signature."""
        self._ensure_device(x)
        out = self.model(x)
        
        # Handle dict (evidential) or tensor (baseline)
        if isinstance(out, dict):
            if "prob" in out:
                probs = out["prob"]
            elif "logits" in out:
                probs = torch.softmax(out["logits"], dim=-1)
            elif "alpha" in out:
                alpha = out["alpha"]
                probs = alpha / torch.sum(alpha, dim=-1, keepdim=True)
            else:
                raise KeyError(f"Evidential output missing probability source. Found: {list(out.keys())}")
            
            probs = ensure_2d(probs)
            cal_probs = self.calibrator.calibrate(probs)
            out["prob"] = cal_probs
            # Reconstruct 'alpha' to satisfy EvidentialModule.forward
            out["alpha"] = cal_probs * 100.0
            return out
        else:
            logits = ensure_2d(out)
            probs = torch.softmax(logits, dim=1)
            cal_probs = self.calibrator.calibrate(probs)
            # We return log-probabilities for baselines to maintain softmax-transferability
            return torch.log(cal_probs + 1e-9)

class SafePolicyWrapper(nn.Module):
    """
    Senior-level Safety Policy Wrapper.
    Implements Conservative Fallback: When uncertainty is high, 
    override prediction with a 'Safe Default' (e.g., 30km/h).
    """
    def __init__(self, model: nn.Module, threshold: float = 0.6):
        super().__init__()
        self.model = model
        self.threshold = threshold
        
        # Identity 'Safe Defaults' from Metadata
        # Speed Limit Fallback: 30 km/h (Class 1)
        # Priority Fallback: Stop (Class 14)
        self.speed_indices = set(CLASS_METADATA["speed_limits"]["indices"])
        self.priority_indices = set(CLASS_METADATA["priority"]["indices"])
        self.safe_speed = 1
        self.safe_priority = 14

    def _ensure_device(self, x: torch.Tensor):
        """Self-healing device sync: Ensures model matches input device."""
        try:
            device = next(self.model.parameters()).device
            if device != x.device:
                self.model.to(x.device)
        except (StopIteration, AttributeError):
            pass

    def forward(self, x):
        self._ensure_device(x)
        outputs = self.model(x)
        
        # 1. Extract Probabilities and Uncertainty (Vacuity for EDL, Entropy for Baseline)
        if isinstance(outputs, dict):
            probs = outputs["prob"]
            # EDL uses Vacuity (Uncertainty mass)
            uncertainty = outputs.get("vacuity", -torch.sum(probs * torch.log(probs + 1e-10), dim=1))
        else:
            probs = torch.softmax(outputs, dim=1)
            # Baseline uses Normalized Entropy as a proxy for uncertainty
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            uncertainty = entropy / np.log(probs.shape[1])
        
        preds = probs.argmax(dim=1)
        
        # 2. Apply Fallback Policy: If uncertainty > threshold, override with a safer class
        for i in range(len(preds)):
            if uncertainty[i] > self.threshold:
                p = preds[i].item()
                if p in self.speed_indices:
                    preds[i] = self.safe_speed
                elif p in self.priority_indices:
                    preds[i] = self.safe_priority
        
        # 3. Reconstruct output to maintain compatibility with evaluation loops
        if isinstance(outputs, dict):
            # Create a 'pseudo-prob' that reflects the safe prediction (one-hot)
            # This ensures that standard accuracy metrics correctly count the safe fallback
            # but we preserve the dictionary structure for evidential processing
            new_probs = torch.zeros_like(probs)
            new_probs.scatter_(1, preds.unsqueeze(1), 1.0)
            outputs["prob"] = new_probs
            return outputs
        else:
            # For baselines, we return "one-hot" logits or similar
            # Since the eval loop does torch.softmax(out, dim=1), 
            # we return a tensor with a high value at the safe prediction index
            new_logits = torch.full_like(probs, -100.0)
            new_logits.scatter_(1, preds.unsqueeze(1), 100.0)
            return new_logits
