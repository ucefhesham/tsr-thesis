import torch
import torch.nn as nn
import pytorch_lightning as L
from typing import List, Dict, Any, Optional
import os

class DeepEnsembleWrapper(nn.Module):
    """
    Wraps multiple independent models for ensemble inference.
    Answers RQ 5: Gold Standard Comparison.
    """
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Aggregate predictions from all ensemble members.
        Returns mean probability and uncertainty metrics.
        """
        all_logits = []
        for model in self.models:
            model.eval()
            # Ensure we get the raw logits (some models might return dicts)
            out = model(x)
            if isinstance(out, dict):
                logits = out.get("logits", out.get("prob", out)) # Fallback logic
            else:
                logits = out
            all_logits.append(logits)
            
        all_logits = torch.stack(all_logits) # [M, B, K]
        all_probs = torch.softmax(all_logits, dim=-1) # [M, B, K]
        
        # 1. Mean Predictive Probability
        mean_probs = torch.mean(all_probs, dim=0) # [B, K]
        
        # 2. Aleatoric Uncertainty (Average Entropy)
        # H_aleatoric = 1/M * sum(Entropy(p_m))
        entropies = -torch.sum(all_probs * torch.log(all_probs + 1e-10), dim=-1) # [M, B]
        avg_entropy = torch.mean(entropies, dim=0) # [B]
        
        # 3. Epistemic Uncertainty (Mutual Information / Predictive Variance)
        # H_total = Entropy(1/M * sum(p_m))
        total_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1) # [B]
        mutual_info = total_entropy - avg_entropy # [B]
        
        # Predictive Variance (alternate proxy)
        variance = torch.var(all_probs, dim=0).mean(dim=-1) # [B]
        
        return {
            "prob": mean_probs,
            "entropy": total_entropy, # Total uncertainty
            "aleatoric": avg_entropy,
            "epistemic": mutual_info,
            "variance": variance
        }

def load_ensemble(ckpt_paths: List[str], model_class: Any, device: str = "cuda") -> DeepEnsembleWrapper:
    """Helper to load multiple checkpoints into an ensemble."""
    models = []
    for path in ckpt_paths:
        if not os.path.exists(path):
            print(f"WARNING: Checkpoint {path} not found.")
            continue
        # We load with map_location to avoid CUDA OOM if many models are loaded
        model = model_class.load_from_checkpoint(path, map_location="cpu")
        model.to(device)
        model.eval()
        models.append(model)
    
    return DeepEnsembleWrapper(models)
