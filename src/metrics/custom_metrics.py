import torch
from torchmetrics import Metric
import torch.nn.functional as F
from typing import List, Dict, Any

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

class AdaptiveECE(ExpectedCalibrationError):
    """
    Adaptive ECE uses equal-mass binning (each bin has N/n_bins samples).
    Provides a more robust estimate for sparse predictions.
    """
    def compute(self):
        confidences = torch.cat(self.confidences)
        accuracies = torch.cat(self.accuracies)
        n = len(confidences)
        
        # Sort and find equal-mass boundaries using quantiles
        sorted_conf, _ = torch.sort(confidences)
        # We need n_bins + 1 boundaries
        indices = torch.linspace(0, n-1, self.n_bins + 1, dtype=torch.long, device=confidences.device)
        bin_boundaries = sorted_conf[indices]
        
        ece = torch.zeros(1, device=confidences.device)
        for i in range(self.n_bins):
            # Use <= for the last bin to include the maximum confidence
            if i == self.n_bins - 1:
                mask = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            else:
                mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
                
            if mask.any():
                bin_acc = accuracies[mask].mean()
                bin_conf = confidences[mask].mean()
                ece += mask.float().mean() * torch.abs(bin_acc - bin_conf)
                
        return ece

class ClasswiseECE(Metric):
    """
    Calculates ECE specifically for safety-critical classes.
    """
    def __init__(self, target_classes: List[int], n_bins: int = 15, **kwargs):
        super().__init__(**kwargs)
        self.target_classes = target_classes
        self.n_bins = n_bins
        self.add_state("confidences", default=[], dist_reduce_fx="cat")
        self.add_state("accuracies", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        confidences, predictions = torch.max(preds, dim=1)
        accuracies = (predictions == targets).float()
        
        self.confidences.append(confidences)
        self.accuracies.append(accuracies)
        self.targets.append(targets)

    def compute(self):
        confidences = torch.cat(self.confidences)
        accuracies = torch.cat(self.accuracies)
        targets = torch.cat(self.targets)
        
        results = {}
        for cls_id in self.target_classes:
            mask = (targets == cls_id)
            if not mask.any():
                results[f"ece_class_{cls_id}"] = torch.tensor(0.0, device=confidences.device)
                continue
            
            c_cls = confidences[mask]
            a_cls = accuracies[mask]
            
            # Simple uniform ECE for the class subset
            bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=confidences.device)
            ece_val = torch.zeros(1, device=confidences.device)
            for i in range(self.n_bins):
                b_mask = (c_cls > bin_boundaries[i]) & (c_cls <= bin_boundaries[i + 1])
                if b_mask.any():
                    ece_val += b_mask.float().mean() * torch.abs(a_cls[b_mask].mean() - c_cls[b_mask].mean())
            results[f"ece_class_{cls_id}"] = ece_val
            
        return results

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

class NegativeLogLikelihood(Metric):
    """
    Standard Negative Log-Likelihood (NLL) for classification.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_nll", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds: Predicted probabilities (softmax) [N, K]
            targets: Ground truth labels [N]
        """
        # Cross entropy on probabilities is -log(p_target)
        nll = F.nll_loss(torch.log(preds + 1e-10), targets, reduction="sum")
        self.sum_nll += nll
        self.total += targets.size(0)

    def compute(self):
        return self.sum_nll / self.total

class ReliabilityDiagram:
    """
    Utility for generating Calibration Reliability Diagrams.
    """
    @staticmethod
    def calculate_bins(preds: torch.Tensor, targets: torch.Tensor, n_bins: int = 15):
        confidences, predictions = torch.max(preds, dim=1)
        accuracies = (predictions == targets).float()
        
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accs = []
        bin_confs = []
        bin_sizes = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            mask = (confidences > bin_lower) & (confidences <= bin_upper)
            if mask.any():
                bin_accs.append(accuracies[mask].mean().item())
                bin_confs.append(confidences[mask].mean().item())
                bin_sizes.append(mask.float().mean().item())
            else:
                bin_accs.append(0.0)
                bin_confs.append(0.0)
                bin_sizes.append(0.0)
                
        return {
            "bin_accs": bin_accs,
            "bin_confs": bin_confs,
            "bin_sizes": bin_sizes,
            "bin_boundaries": bin_boundaries.tolist()
        }

class NegativeLogLikelihood(Metric):
    """
    Standard Negative Log-Likelihood (NLL) for classification.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_nll", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        # Cross entropy on probabilities is -log(p_target)
        nll = F.nll_loss(torch.log(preds + 1e-10), targets, reduction="sum")
        self.sum_nll += nll
        self.total += targets.size(0)

    def compute(self):
        return self.sum_nll / self.total

class ReliabilityDiagram:
    """Utility for generating Calibration Reliability Diagrams."""
    @staticmethod
    def calculate_bins(preds: torch.Tensor, targets: torch.Tensor, n_bins: int = 15):
        confidences, predictions = torch.max(preds, dim=1)
        accuracies = (predictions == targets).float()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]
        
        bin_accs, bin_confs, bin_sizes = [], [], []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            mask = (confidences > bin_lower) & (confidences <= bin_upper)
            if mask.any():
                bin_accs.append(accuracies[mask].mean().item())
                bin_confs.append(confidences[mask].mean().item())
                bin_sizes.append(mask.float().mean().item())
            else:
                bin_accs.append(0.0); bin_confs.append(0.0); bin_sizes.append(0.0)
        return {"bin_accs": bin_accs, "bin_confs": bin_confs, "bin_sizes": bin_sizes}

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

class EntropyScore(Metric):
    """
    Measures the average entropy of the predicted probability distribution.
    Higher entropy indicates higher aleatoric (or sometimes epistemic) uncertainty.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total_entropy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor):
        """
        Args:
            preds: Predicted probabilities [N, K]
        """
        # H(p) = -sum(p * log(p))
        entropy = -torch.sum(preds * torch.log(preds + 1e-10), dim=1)
        self.total_entropy += entropy.sum()
        self.total += preds.size(0)

    def compute(self):
        return self.total_entropy / self.total

class EnergyScore(Metric):
    """
    Measures the average Energy-based uncertainty score.
    Higher energy (LogSumExp of logits) often indicates In-Distribution (ID).
    Lower energy often indicates Out-of-Distribution (OOD).
    """
    def __init__(self, tau: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.add_state("total_energy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor):
        """
        Args:
            logits: Pre-softmax network outputs [N, K]
        """
        energy = self.tau * torch.logsumexp(logits / self.tau, dim=1)
        self.total_energy += energy.sum()
        self.total += logits.size(0)

    def compute(self):
        return self.total_energy / self.total

# Keep alias for backward compatibility if needed, but ASR is the new standard
SeverityWeightedError = AdvancedSeverityRisk
