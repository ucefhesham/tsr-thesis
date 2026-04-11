import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import CalibrationError
from typing import Optional

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
