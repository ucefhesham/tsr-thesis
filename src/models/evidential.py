import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class EvidentialNetwork(nn.Module):
    def __init__(self, num_classes: int = 43):
        """
        Evidential Deep Learning Network wrapper for ResNet-18.
        Outputs Dirichlet parameters (alpha) instead of logits.
        """
        super().__init__()
        self.num_classes = num_classes
        
        # 1. Initialize ResNet-18 with ImageNet weights
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # 2. Slice the backbone (remove original fc)
        # We take everything up to the global average pool
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        
        # 3. New Evidence Head
        self.evidence_head = nn.Linear(512, num_classes)
        
        # 4. Explicit Initialization for Stability
        # Kaiming normal is suitable for layers before Softplus/ReLU
        nn.init.kaiming_normal_(self.evidence_head.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.evidence_head.bias, 0)

    def forward(self, x: torch.Tensor):
        # Extract features (Batch, 512, 1, 1)
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1) # (Batch, 512)
        
        # Compute raw output
        out = self.evidence_head(features)
        
        # 4. Evidence e must be strictly positive and capped for stability
        evidence = F.softplus(out)
        evidence = torch.clamp(evidence, max=1e4)
        
        # 5. Dirichlet parameters alpha = evidence + 1
        alpha = evidence + 1
        
        # 6. Expected probabilities p = alpha / S
        S = torch.sum(alpha, dim=1, keepdim=True)
        prob = alpha / S
        
        # 7. Vacuity (Uncertainty mass) u = K / S
        vacuity = self.num_classes / S
        
        return {
            "evidence": evidence,
            "alpha": alpha,
            "prob": prob,
            "vacuity": vacuity
        }
