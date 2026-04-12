import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import CalibrationError
from typing import Optional
import numpy as np
from sklearn.isotonic import IsotonicRegression
from src.metrics.safety_config import CLASS_METADATA

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returns temperature-scaled logits."""
        logits = self.model(x)
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
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            
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
        probs_np = probs.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
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
        probs_np = probs.cpu().numpy()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns 'calibrated logits' (log-probabilities) so that 
        standard softmax in the main module still works or becomes identity-like.
        Actually, it's safer to return raw probabilities and let the caller know, 
        but to maintain compatibility with eval.py's backbone swapping, 
        we return the log(calibrated_probs) so that softmax(log(p)) = p.
        """
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        cal_probs = self.calibrator.calibrate(probs)
        
        # We return log-probabilities so that Softmax(log(p)) = p
        # Add epsilon to avoid log(0)
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

    def forward(self, x):
        outputs = self.model(x)
        
        # This wrapper expects evidentiary output or logits
        # For Evidential models, we use Vacuity
        # For non-evidential, we could use Entropy (if wrapped similarly)
        
        # Handle Evidential Dictionary Output
        if isinstance(outputs, dict):
            probs = outputs["prob"]
            vacuity = outputs["vacuity"]
            preds = probs.argmax(dim=1)
            
            # Apply Fallback Logic
            for i in range(len(preds)):
                if vacuity[i] > self.threshold:
                    p = preds[i].item()
                    if p in self.speed_indices:
                        preds[i] = self.safe_speed
                    elif p in self.priority_indices:
                        preds[i] = self.safe_priority
            
            # Reconstruct 'probs' as one-hot for the new predictions
            # to remain compatible with evaluation loops
            new_probs = torch.zeros_like(probs)
            new_probs.scatter_(1, preds.unsqueeze(1), 1.0)
            outputs["prob"] = new_probs
            
        return outputs
