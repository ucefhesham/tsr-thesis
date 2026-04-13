import torch
import torch.nn as nn
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
from typing import List, Optional, Dict, Tuple

def calculate_focus_score(grayscale_cam: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    """
    Computes Saliency Focus Efficiency (SFE).
    The percentage of Grad-CAM energy contained within the bounding box.
    bbox: (x1, y1, x2, y2) in pixel coordinates.
    """
    x1, y1, x2, y2 = bbox
    total_energy = grayscale_cam.sum()
    if total_energy == 0: return 0.0
    
    # Ensure coordinates are within bounds
    h, w = grayscale_cam.shape
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    
    focus_energy = grayscale_cam[y1:y2, x1:x2].sum()
    return float(focus_energy / total_energy)
class UncertaintyTarget:
    """
    Custom Grad-CAM target for Uncertainty.
    Visualizes regions that contribute most to the uncertainty score (Vacuity or Entropy).
    """
    def __init__(self, mode: str = "vacuity"):
        self.mode = mode

    def __call__(self, model_output):
        # Handle dictionary vs raw tensor (Evidential Module forward returns raw alpha)
        if isinstance(model_output, dict):
            if self.mode == "vacuity" and "vacuity" in model_output:
                return model_output["vacuity"].squeeze()
            model_output = model_output.get("prob") or model_output.get("alpha")
            
        # Ensure 2D for consistent indexing [batch, classes]
        if model_output.ndim == 1:
            model_output = model_output.unsqueeze(0)
            
        if self.mode == "vacuity":
            # Dirichlet Uncertainty Saliency: Vacuity = K / S
            # Regions that increase total evidence (S) decrease vacuity.
            # We visualize the inverse to see what *reduces* uncertainty.
            K = model_output.shape[-1]
            S = torch.sum(model_output, dim=1)
            return K / (S + 1e-9)
        else:
            # Baseline Entropy Saliency
            probs = torch.softmax(model_output, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
            return entropy

class TrustInterpreter:
    """
    Senior-level XAI Engine for Traffic Sign Recognition.
    Handles both standard classification saliency and Trust-aware uncertainty saliency.
    """
    def __init__(self, model: nn.Module, target_layers: List[nn.Module]):
        self.model = model
        self.target_layers = target_layers
        self.cam = GradCAM(model=model, target_layers=target_layers)

    def generate_maps(self, 
                      input_tensor: torch.Tensor, 
                      original_image: np.ndarray, 
                      target_class: Optional[int] = None,
                      bbox: Optional[Tuple[int, int, int, int]] = None):
        """
        Generates a dual-saliency map profile.
        1. Class Saliency (Why this class?)
        2. Uncertainty Saliency (Why so confused?)
        """
        # Ensure input has batch dimension and gradient tracking
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        # MANDATORY for Grad-CAM backward pass
        input_tensor.requires_grad = True
            
        # 1. Class Saliency Map
        if target_class is None:
            # Re-run model to get predicted class if not provided
            self.model.eval()
            with torch.no_grad():
                out = self.model(input_tensor)
                if isinstance(out, dict):
                    target_class = out["prob"].argmax(dim=1).item()
                else:
                    target_class = out.argmax(dim=1).item()
        
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam_class = self.cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        # 2. Uncertainty Saliency Map
        # Determine mode based on architecture
        mode = "vacuity" if hasattr(self.model, "evidence_head") or "Evidential" in str(type(self.model)) else "entropy"
        uncertainty_targets = [UncertaintyTarget(mode=mode)]
        grayscale_cam_uncertainty = self.cam(input_tensor=input_tensor, targets=uncertainty_targets)[0, :]
        
        # 3. Calculate Focus Score (Saliency Focus Efficiency - SFE)
        if bbox is None:
            # Default to center 80% if no bbox provided
            h, w = grayscale_cam_class.shape
            bbox = (int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9))
            
        sfe_score = calculate_focus_score(grayscale_cam_class, bbox)
        
        # Overlay on original image
        # Normalize original image to [0, 1]
        img_norm = original_image.astype(np.float32) / 255.0
        
        visualization_class = show_cam_on_image(img_norm, grayscale_cam_class, use_rgb=True)
        visualization_uncertainty = show_cam_on_image(img_norm, grayscale_cam_uncertainty, use_rgb=True)
        
        return {
            "class_map": visualization_class,
            "uncertainty_map": visualization_uncertainty,
            "class_id": target_class,
            "sfe": sfe_score,
            "uncertainty_score": grayscale_cam_uncertainty.mean() # Quantitative proxy
        }
