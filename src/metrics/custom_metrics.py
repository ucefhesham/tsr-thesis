import torch
from torchmetrics import Metric
import torch.nn.functional as F

class ExpectedCalibrationError(Metric):
    def __init__(self, n_bins: int = 15, **kwargs):
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.add_state("confidences", default=[], dist_reduce_fx="cat")
        self.add_state("accuracies", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds: Predicted probabilities (softmax output) [N, K]
            targets: Ground truth labels [N]
        """
        confidences, predictions = torch.max(preds, dim=1)
        accuracies = (predictions == targets).float()
        
        self.confidences.append(confidences)
        self.accuracies.append(accuracies)

    def compute(self):
        confidences = torch.cat(self.confidences)
        accuracies = torch.cat(self.accuracies)
        
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=confidences.device)
        ece = torch.zeros(1, device=confidences.device)
        
        for i in range(self.n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.any():
                bin_acc = accuracies[mask].mean()
                bin_conf = confidences[mask].mean()
                ece += mask.float().mean() * torch.abs(bin_acc - bin_conf)
                
        return ece

class MulticlassBrierScore(Metric):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds: Predicted probabilities (softmax) [N, K]
            targets: Ground truth labels [N]
        """
        num_classes = preds.shape[1]
        target_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        squared_error = torch.sum((preds - target_one_hot) ** 2, dim=1)
        
        self.sum_squared_error += squared_error.sum()
        self.total += targets.size(0)

    def compute(self):
        return self.sum_squared_error / self.total

class AdvancedSeverityRisk(Metric):
    def __init__(self, num_classes: int = 43, near_miss_threshold: float = 0.15, **kwargs):
        """
        Senior-level Advanced Severity Risk (ASR) suite.
        Calculates:
        - SWE: Top-1 based average severity cost.
        - ESP: Expected Severity Penalty (Probabilistic risk).
        - Category-specific risks (Speed, Priority, etc.).
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.near_miss_threshold = near_miss_threshold

        # Initialize tracking states
        self.add_state("sum_swe", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_esp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("near_misses", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
        # Breakdown states
        from src.metrics.safety_config import get_risk_categories, CostMatrixGenerator
        self.risk_categories = get_risk_categories()
        for cat in self.risk_categories.keys():
            self.add_state(f"sum_esp_{cat}", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"total_{cat}", default=torch.tensor(0), dist_reduce_fx="sum")

        # Precompute cost matrix using the senior generator
        generator = CostMatrixGenerator(num_classes)
        self.register_buffer("W", generator.generate())

    def update(self, preds: torch.Tensor, targets: torch.Tensor, vacuity: torch.Tensor = None):
        """
        Args:
            preds: Predicted probabilities [N, K]
            targets: Ground truth labels [N]
            vacuity: Optional evidential vacuity (epistemic uncertainty) [N]
        """
        batch_size = targets.size(0)
        
        # 1. Top-1 SWE (Argmax based)
        predictions = torch.argmax(preds, dim=1)
        swe_costs = self.W[targets, predictions]
        self.sum_swe += swe_costs.sum()
        
        # 2. ESP (Probabilistic Expected Severity Penalty)
        target_costs = self.W[targets] # Shape [N, K]
        esp_costs = torch.sum(preds * target_costs, dim=1)
        
        # 3. Evidential Tie-in (Senior Feature)
        # If vacuity is provided, the model's risk is a combination of 
        # probabilistic risk (ESP) and vacuity-driven "safe retreat".
        # We assume vacuity contributes to a high "Maximum Possible Cost" risk
        if vacuity is not None:
            # Risk = (1 - u) * ESP + u * MaxPotentialPenalty
            # Here we assume a high-vacuity prediction is a "potential disaster" 
            # unless the system can fail-safe.
            max_potential_costs, _ = torch.max(target_costs, dim=1)
            esp_costs = (1 - vacuity) * esp_costs + vacuity * max_potential_costs
            
        self.sum_esp += esp_costs.sum()
        
        # 4. Near Misses
        correct_mask = (predictions == targets)
        if correct_mask.any():
            fatal_mask = (target_costs > 5.0).float()
            fatal_probs = preds * fatal_mask
            max_fatal_prob, _ = torch.max(fatal_probs, dim=1)
            
            near_miss_mask = correct_mask & (max_fatal_prob > self.near_miss_threshold)
            self.near_misses += near_miss_mask.sum()

        # 5. Category Breakdown
        for cat, indices in self.risk_categories.items():
            cat_mask = torch.tensor([t.item() in indices for t in targets], device=targets.device)
            if cat_mask.any():
                getattr(self, f"sum_esp_{cat}").add_(esp_costs[cat_mask].sum())
                getattr(self, f"total_{cat}").add_(cat_mask.sum())

        self.total += batch_size

    def compute(self):
        results = {
            "asr/swe": self.sum_swe / self.total,
            "asr/esp": self.sum_esp / self.total,
            "asr/near_miss_rate": self.near_misses.float() / self.total
        }
        
        # Add category specific ESP
        for cat in self.risk_categories.keys():
            total = getattr(self, f"total_{cat}")
            if total > 0:
                results[f"asr/esp_{cat.lower()}"] = getattr(self, f"sum_esp_{cat}") / total
                
        return results

# Keep alias for backward compatibility if needed, but ASR is the new standard
SeverityWeightedError = AdvancedSeverityRisk
