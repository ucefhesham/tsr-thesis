import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm

class TemperatureScaler(nn.Module):
    """
    A helper module to optimize the temperature T for model probabilities.
    Answers RQ 3: Post-hoc calibration gains.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, valid_loader, device="cuda"):
        """
        Tune the temperature of the model (using the validation set).
        Fits T by minimizing Cross-Entropy (NLL) on the validation set.
        """
        self.model.eval()
        self.to(device)
        nll_criterion = nn.CrossEntropyLoss().to(device)
        
        # 1. Collect all logits and labels from the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in tqdm(valid_loader, desc="Collecting val logits"):
                input = input.to(device)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).to(device)
            labels = torch.cat(labels_list).to(device)

        # 2. Optimize temperature T
        print(f"Before temperature scaling - NLL: {nll_criterion(logits, labels).item():.4f}")
        
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        print(f"Optimal temperature: {self.temperature.item():.4f}")
        print(f"After temperature scaling - NLL: {nll_criterion(self.temperature_scale(logits), labels).item():.4f}")
        
        return self

def fit_temperature(model, val_loader, device="cuda"):
    scaler = TemperatureScaler(model)
    scaler.set_temperature(val_loader, device)
    return scaler.temperature.item()
