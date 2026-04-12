import torch
import numpy as np
from typing import Dict, List, Optional
import torch.nn.functional as F

class SplitConformalSelector:
    """
    Advanced Selective Prediction using Split Conformal Classification.
    Provides a mathematically guaranteed coverage (1 - alpha).
    
    Ref: 'A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty'
    """
    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Miscoverage level (e.g., 0.05 for 95% coverage).
        """
        self.alpha = alpha
        self.quantile = None

    def calibrate(self, probs: torch.Tensor, targets: torch.Tensor):
        """
        Fit the conformal quantile on a held-out calibration set.
        
        Args:
            probs: Predicted probabilities [N, K]
            targets: Ground truth labels [N]
        """
        n = probs.shape[0]
        # Get probabilities of true classes
        true_probs = probs[torch.arange(n), targets]
        
        # LAC (Least Ambiguous Classification) scores: 1 - P(y_true)
        # Low score = high confidence in true class
        scores = 1 - true_probs
        
        # Calculate the (1-alpha) quantile
        # We use (n+1)(1-alpha) / n version for finite sample correction
        level = (1 - self.alpha) * (n + 1) / n
        if level > 1.0:
            level = 1.0
            
        self.quantile = np.quantile(scores.cpu().numpy(), level, method='higher')
        print(f"Calibration Complete: alpha={self.alpha}, n={n}, Quantile={self.quantile:.4f}")

    def predict_set(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Generate prediction sets for new inputs.
        
        Args:
            probs: Predicted probabilities [N, K]
        Returns:
            Binary mask of prediction sets [N, K]
        """
        if self.quantile is None:
            raise ValueError("Selector must be calibrated before prediction.")
            
        # Set is {k : 1 - probs[k] <= quantile} => {k : probs[k] >= 1 - quantile}
        threshold = 1 - self.quantile
        return (probs >= threshold).bool()

    def get_metrics(self, set_mask: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute standard Conformal Metrics.
        """
        batch_size = targets.size(0)
        
        # Coverage: Is true label in the set?
        covered = set_mask[torch.arange(batch_size), targets]
        coverage = covered.float().mean().item()
        
        # Set Size: How many labels in the set?
        set_size = set_mask.float().sum(dim=1).mean().item()
        
        # Empty Set rate (model is so uncertain it picks nothing)
        empty_sets = (set_mask.sum(dim=1) == 0).float().mean().item()
        
        return {
            "conformal/coverage": coverage,
            "conformal/avg_set_size": set_size,
            "conformal/empty_set_rate": empty_sets
        }

class ConformalRiskSuite:
    """
    Sweeps across multiple alpha levels to generate Risk-Coverage Curves.
    """
    def __init__(self, alphas: List[float] = [0.01, 0.05, 0.1, 0.2]):
        self.selectors = {alpha: SplitConformalSelector(alpha) for alpha in alphas}

    def calibrate_all(self, probs: torch.Tensor, targets: torch.Tensor):
        for selector in self.selectors.values():
            selector.calibrate(probs, targets)

    def evaluate_all(self, probs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        all_results = {}
        for alpha, selector in self.selectors.items():
            set_mask = selector.predict_set(probs)
            metrics = selector.get_metrics(set_mask, targets)
            for k, v in metrics.items():
                all_results[f"{k}_alpha_{alpha}"] = v
        return all_results
