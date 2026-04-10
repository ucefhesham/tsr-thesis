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

# Placeholder for Severity-Weighted Error (SWE)
# Requires a cost matrix W as per the thesis proposal
def calculate_swe(preds, targets, cost_matrix):
    pass
