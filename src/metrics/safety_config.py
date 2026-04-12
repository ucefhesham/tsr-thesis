import torch
import numpy as np

# Semantic Groups for GTSRB (43 Classes)
CLASS_METADATA = {
    "speed_limits": {
        "indices": [0, 1, 2, 3, 4, 5, 7, 8],
        "speeds": {0: 20, 1: 30, 2: 50, 3: 60, 4: 70, 5: 80, 7: 100, 8: 120}
    },
    "priority": {
        "indices": [11, 12, 13, 14], # Right-of-way, Priority Road, Yield, Stop
        "critical": [14] # Stop
    },
    "prohibitory": {
        "indices": [9, 10, 15, 16, 17], # No passing, No entry, etc.
        "critical": [17] # No Entry
    },
    "danger": {
        "indices": list(range(18, 32)) # General caution, curves, etc.
    },
    "mandatory": {
        "indices": list(range(33, 41)) # Directional arrows, roundabout
    }
}

class CostMatrixGenerator:
    """
    Senior-level Cost Matrix Generator that builds physically grounded 
    penalties for traffic sign misclassifications.
    """
    def __init__(self, num_classes: int = 43):
        self.num_classes = num_classes

    def get_asymmetric_speed_cost(self, gt_idx, pred_idx):
        """Asymmetric penalty: High-as-Low is dangerous, Low-as-High is inefficient."""
        speeds = CLASS_METADATA["speed_limits"]["speeds"]
        v_gt = speeds[gt_idx]
        v_pred = speeds[pred_idx]
        
        delta = v_pred - v_gt
        if delta > 0:
            # Dangerous over-prediction (e.g., predicted 120 in 30 zone)
            # Quadratic penalty to reflect exponential kinetic energy or risk
            return 2.0 + (delta / 20.0) ** 2
        else:
            # Inefficient under-prediction (e.g., predicted 30 in 120 zone)
            # Linear, lower penalty
            return 0.5 + abs(delta) / 50.0

    def generate(self) -> torch.Tensor:
        W = torch.ones((self.num_classes, self.num_classes))
        W.fill_diagonal_(0.0)
        
        # 1. Handle Speed Limit Costs
        speed_indices = CLASS_METADATA["speed_limits"]["indices"]
        for gt in speed_indices:
            for pred in range(self.num_classes):
                if gt == pred: continue
                
                if pred in speed_indices:
                    # Within speed group: Use asymmetric delta
                    W[gt, pred] = self.get_asymmetric_speed_cost(gt, pred)
                else:
                    # Missing a speed limit entirely: High risk
                    W[gt, pred] = 5.0

        # 2. Handle Priority Flipping (Yield/Stop -> Road Work/Priority)
        priority_indices = CLASS_METADATA["priority"]["indices"]
        critical_priority = CLASS_METADATA["priority"]["critical"] # STOP
        for gt in priority_indices:
            cost = 15.0 if gt in critical_priority else 10.0
            for pred in range(self.num_classes):
                if gt != pred and pred not in priority_indices:
                    W[gt, pred] = cost

        # 3. Handle Prohibitory Violations (No Entry -> Go)
        prohibitory_indices = CLASS_METADATA["prohibitory"]["indices"]
        for gt in prohibitory_indices:
            cost = 15.0 if gt == 17 else 8.0 # 17 is No Entry
            for pred in range(self.num_classes):
                if gt != pred and pred not in prohibitory_indices:
                    W[gt, pred] = cost

        # 4. Minority/Vulnerable Group Bias (Pedestrians/Children)
        vulnerable = [27, 28] # Pedestrians, Children
        for gt in vulnerable:
            for pred in range(self.num_classes):
                if gt != pred:
                    W[gt, pred] = 12.0

        # 5. Same group minor penalties
        for group in ["danger", "mandatory"]:
            indices = CLASS_METADATA[group]["indices"]
            for gt in indices:
                for pred in indices:
                    if gt != pred:
                        W[gt, pred] = 0.2

        return W

def get_risk_categories():
    """Returns classification of indices for sub-metric reporting."""
    return {
        "Speed_Risk": CLASS_METADATA["speed_limits"]["indices"],
        "Priority_Risk": CLASS_METADATA["priority"]["indices"],
        "Vulnerable_Risk": [27, 28],
        "General_Risk": list(range(43))
    }
