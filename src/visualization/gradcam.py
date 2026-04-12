import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

class SaliencyAnalyzer:
    def __init__(self, model: nn.Module, target_layers: list = None):
        """
        Qualitative Trust Analyzer using Grad-CAM.
        Visualizes which parts of the traffic sign the model is 'trusting' for its decision.
        """
        self.model = model.eval()
        self.device = next(model.parameters()).device
        
        if target_layers is None:
            self.target_layers = self._auto_find_target_layers()
        else:
            self.target_layers = target_layers

        self.cam = GradCAM(
            model=self.model, 
            target_layers=self.target_layers,
            use_cuda=torch.cuda.is_available()
        )

    def _auto_find_target_layers(self):
        """Attempts to find the last convolutional layer automatically based on architecture."""
        # For ResNet architectures
        if hasattr(self.model, "layer4"):
            return [self.model.layer4[-1]]
        
        # For ConvNeXt architectures (via torchvision)
        if hasattr(self.model, "features"):
            return [self.model.features[-1]]
            
        # Fallback to the last submodule that is a Conv2d or LayerNorm (for modern blocks)
        conv_layers = []
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.LayerNorm)):
                conv_layers.append(module)
        
        if conv_layers:
            return [conv_layers[-1]]
        
        raise ValueError("Could not automatically find target layers for Grad-CAM.")

    def explain(self, input_tensor: torch.Tensor, original_img_path: str = None) -> np.ndarray:
        """
        Generates a Grad-CAM heatmap overlaid on the image.
        
        Args:
            input_tensor: Preprocessed tensor [1, 3, H, W]
            original_img_path: Optional path to raw image for better visualization.
            
        Returns:
            np.ndarray: RGB image with heatmap overlay.
        """
        # Ensure batch dimension
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
            
        input_tensor = input_tensor.to(self.device)
        
        # Generate heatmap
        # targets=None defaults to the highest scoring class
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=None)[0, :]
        
        # Prepare background image for overlay
        if original_img_path and os.path.exists(original_img_path):
            rgb_img = cv2.imread(original_img_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_img = cv2.resize(rgb_img, (input_tensor.shape[2], input_tensor.shape[3]))
        else:
            # Fallback: Denormalize the input tensor (rough approximation)
            # Standard ImageNet mean/std
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            rgb_img = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
            rgb_img = (rgb_img * std + mean).clip(0, 1)
            rgb_img = (rgb_img * 255).astype(np.uint8)

        # Overlay
        # Divide by 255 because show_cam_on_image expects 0-1 float
        visualization = show_cam_on_image(rgb_img.astype(np.float32) / 255.0, grayscale_cam, use_rgb=True)
        return visualization

def save_saliency_map(viz: np.ndarray, save_path: str):
    """Saves the RGB visualization to disk."""
    Image.fromarray(viz).save(save_path)
    print(f"Saliency map saved to {save_path}")
