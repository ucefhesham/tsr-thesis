import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class EvidentialNetwork(nn.Module):
    def __init__(self, backbone: nn.Module = None, num_classes: int = 43):
        """
        Senior-Level Polymorphic Evidential Network.
        Outputs Dirichlet parameters (alpha) instead of logits.
        
        Args:
            backbone: An instantiated feature extractor (should output pooled features).
            num_classes: Number of target categories.
        """
        super().__init__()
        self.num_classes = num_classes
        
        # 1. Initialize Default Backbone (ResNet-18) if none provided
        if backbone is None:
            raw_resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.feature_extractor = nn.Sequential(*list(raw_resnet.children())[:-1])
        else:
            self.feature_extractor = backbone
            
        # 2. Automated Feature Dimensionality Detection
        # Pass a dummy tensor through the extractor to find out head size
        self.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_output = self.feature_extractor(dummy_input)
            in_features = torch.flatten(dummy_output, 1).shape[1]
        self.train()
        
        # 3. Dynamic Evidence Head
        self.evidence_head = nn.Linear(in_features, num_classes)
        
        # 4. Explicit Initialization for Stability
        nn.init.kaiming_normal_(self.evidence_head.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.evidence_head.bias, 0)

    def forward(self, x: torch.Tensor):
        # Extract features (Adaptive to backbone output shape)
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        
        # Compute raw output
        out = self.evidence_head(features)
        
        # Evidence e must be strictly positive (Softplus)
        evidence = F.softplus(out)
        evidence = torch.clamp(evidence, max=1e4) # Stability cap
        
        # Dirichlet parameters alpha = evidence + 1
        alpha = evidence + 1
        
        # Expected probabilities p = alpha / S
        S = torch.sum(alpha, dim=1, keepdim=True)
        prob = alpha / S
        
        # Vacuity (Uncertainty mass) u = K / S
        vacuity = self.num_classes / S
        
        return {
            "evidence": evidence,
            "alpha": alpha,
            "prob": prob,
            "vacuity": vacuity
        }
