import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
from PIL import Image
import cv2
import random

class FastFog:
    """
    Optimized Fog transform that pre-generates a pool of masks during 
    initialization to avoid heavy Perlin noise generation in the data loop.
    """
    def __init__(self, fog_coef_range: tuple, alpha_coef: float, size: int = 224, num_masks: int = 40):
        self.masks = []
        # Use standard Albumentations to generate high-quality base masks
        base_fog = A.RandomFog(fog_coef_range=fog_coef_range, alpha_coef=alpha_coef, p=1.0)
        dummy_img = np.zeros((size, size, 3), dtype=np.uint8)
        
        print(f"DEBUG: Pre-generating {num_masks} fog masks for speed optimization...")
        for _ in range(num_masks):
            # Apply RandomFog to black image to extract ONLY the fog layer
            fog_layer = base_fog(image=dummy_img)["image"]
            self.masks.append(fog_layer)

    def __call__(self, force_apply=False, **kwargs):
        # Compatible with Albumentations-style calling (image=...)
        image = kwargs["image"]
        
        # Pick a random mask from the pool
        mask = random.choice(self.masks)
        
        # Randomly flip/rotate the mask for added variety (zero cost)
        if random.random() > 0.5: mask = np.fliplr(mask)
        if random.random() > 0.5: mask = np.flipud(mask)
        
        # Super-fast blending: image + mask
        # Since the mask was generated on a black image, simple addition works
        # and is extremely fast in OpenCV.
        return {"image": cv2.add(image, mask)}

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
            return A.GaussNoise(
                std_range=(np.sqrt(50.0 * s) / 255.0, np.sqrt(400.0 * s) / 255.0), 
                p=1.0
            )
        
        elif c_type == "blur":
            return A.MotionBlur(blur_limit=(3 + 4 * s), p=1.0)
        
        elif c_type == "weather":
            # Optimization: Use Cached FastFog to avoid heavy Perlin noise generation overhead
            return FastFog(
                fog_coef_range=(0.03 * s, 0.03 * s + 0.05), 
                alpha_coef=0.1, 
                size=self.input_size
            )
        
        elif c_type == "compression":
            quality = max(5, 100 - (18 * s))
            return A.ImageCompression(quality_range=(quality, quality), p=1.0)
        
        else:
            return A.NoOp(p=1.0)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """
        Accepts PIL Image, converts to NumPy, passes through Albumentations,
        and returns the extracted Tensor.
        """
        img_np = np.array(img)
        transformed = self.pipeline(image=img_np)
        return transformed["image"]

def get_base_transforms(input_size=224):
    """Standard evaluation transforms."""
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
