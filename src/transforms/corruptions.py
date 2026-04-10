import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np

def get_base_transforms(input_size=224):
    """Basic transforms for training/validation."""
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_corruption_transform(corruption_type, severity=1, input_size=224):
    """
    Returns an Albumentations transform for a specific corruption.
    
    Args:
        corruption_type: String name of corruption (noise, blur, weather, compression)
        severity: Intensity level (1-5)
        input_size: Target image size
    """
    # Placeholder logic for the thesis's "stress suite"
    transforms = [A.Resize(input_size, input_size)]
    
    if corruption_type == "noise":
        transforms.append(A.GaussNoise(var_limit=(10.0 * severity, 50.0 * severity), p=1.0))
    elif corruption_type == "blur":
        transforms.append(A.MotionBlur(blur_limit=(3 + 2*severity), p=1.0))
    elif corruption_type == "weather":
        # Example: Simulating rain or fog could be complex, using placeholder
        transforms.append(A.RandomFog(fog_coef_lower=0.1*severity, fog_coef_upper=0.2*severity, p=1.0))
    elif corruption_type == "compression":
        transforms.append(A.ImageCompression(quality_lower=100 - 15*severity, quality_upper=100 - 10*severity, p=1.0))
    
    transforms.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms)
