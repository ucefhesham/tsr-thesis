import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
from PIL import Image

class TrustStressTester:
    """
    Scientific wrapper for Albumentations corruptions to ensure compatibility with 
    standard PyTorch Datasets. Handles PIL -> NumPy -> Tensor logic.
    """
    def __init__(self, corruption_type: str, severity: int = 1, input_size: int = 224):
        self.corruption_type = corruption_type
        self.severity = severity
        self.input_size = input_size
        
        # Build the pipeline with strict order:
        # Resize -> [Corruption] -> Normalize -> ToTensorV2
        ops = [A.Resize(input_size, input_size)]
        
        # Inject specific corruption logic
        ops.append(self._get_corruption(corruption_type, severity))
        
        # Finalize with ImageNet stats and PyTorch conversion
        ops.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        self.pipeline = A.Compose(ops)

    def _get_corruption(self, c_type: str, s: int):
        """Maps severity 1-5 to specific Albumentations parameters."""
        if c_type == "noise":
            # Albumentations 2.0+: use std_range instead of var_limit (scale 0 to 1)
            # Increasing scale: severity 5 will have up to ~27% noise
            return A.GaussNoise(
                std_range=(np.sqrt(50.0 * s) / 255.0, np.sqrt(400.0 * s) / 255.0), 
                p=1.0
            )
        
        elif c_type == "blur":
            # Scales motion blur kernel size
            return A.MotionBlur(blur_limit=(3 + 4 * s), p=1.0)
        
        elif c_type == "weather":
            # Albumentations 2.0+: use fog_coef_range
            # Tuning: original 0.1 was too thick. Using 0.03 steps for better curves.
            return A.RandomFog(
                fog_coef_range=(0.03 * s, 0.03 * s + 0.05), 
                alpha_coef=0.1, 
                p=1.0
            )
        
        elif c_type == "compression":
            # Albumentations 2.0+: use quality_range
            quality = max(5, 100 - (18 * s))
            return A.ImageCompression(quality_range=(quality, quality), p=1.0)
        
        else:
            # Identity op if type not found
            return A.NoOp(p=1.0)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """
        Accepts PIL Image, converts to NumPy, passes through Albumentations,
        and returns the extracted Tensor.
        """
        # Convert PIL to NumPy (Albumentations requirement)
        img_np = np.array(img)
        
        # Process through pipeline
        transformed = self.pipeline(image=img_np)
        
        # Explicitly return only the extracted tensor to satisfy torchvision/Lightning
        return transformed["image"]

def get_base_transforms(input_size=224):
    """Standard evaluation transforms."""
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
