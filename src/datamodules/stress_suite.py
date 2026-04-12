import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms, datasets
from src.datamodules.gtsrb_module import GTSRBDataModule
from typing import Optional, List, Tuple

class CorruptionEngine:
    """
    Implements a set of 5-level severities for Noise, Blur, and Fog.
    Based on Hendrycks & Dietterich (2019) benchmarks but optimized for OpenCV.
    """
    
    @staticmethod
    def gaussian_noise(img: np.ndarray, severity: int) -> np.ndarray:
        # Severity scales 1-5: sigma increases
        c = [0.04, 0.08, 0.12, 0.18, 0.26][severity - 1]
        noise = np.random.normal(0, c * 255.0, img.shape)
        out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return out

    @staticmethod
    def motion_blur(img: np.ndarray, severity: int) -> np.ndarray:
        # Severity scales 1-5: kernel size increases
        c = [5, 9, 15, 21, 27][severity - 1]
        kernel = np.zeros((c, c))
        kernel[int((c - 1) / 2), :] = np.ones(c)
        kernel = kernel / c
        return cv2.filter2D(img, -1, kernel)

    @staticmethod
    def synthetic_fog(img: np.ndarray, severity: int) -> np.ndarray:
        # Severity scales 1-5: fog density increases
        # We blend the image with a light gray [200, 200, 200]
        c = [0.15, 0.30, 0.45, 0.60, 0.75][severity - 1]
        fog_overlay = np.ones_like(img) * 200
        out = cv2.addWeighted(img, 1 - c, fog_overlay, c, 0)
        return out

class CorruptedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, corruption_type: str, severity: int, transform=None):
        self.base_dataset = base_dataset
        self.corruption_type = corruption_type
        self.severity = severity
        self.transform = transform
        self.engine = CorruptionEngine()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        # Convert PIL to Numpy
        img_np = np.array(img)
        
        # Apply corruption
        if self.corruption_type == "noise":
            img_np = self.engine.gaussian_noise(img_np, self.severity)
        elif self.corruption_type == "blur":
            img_np = self.engine.motion_blur(img_np, self.severity)
        elif self.corruption_type == "fog":
            img_np = self.engine.synthetic_fog(img_np, self.severity)
        
        # Convert back to PIL for torchvision transforms
        img_pil = Image.fromarray(img_np)
        
        if self.transform:
            img_pil = self.transform(img_pil)
            
        return img_pil, label

class StressSuiteDataModule(GTSRBDataModule):
    """
    DataModule that generates multi-severity test loaders on the fly.
    Answers RQ 1 & 2: How does calibration degrade across types and severities?
    """
    def __init__(self, corruption_type: str = "noise", severity: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        super().setup(stage)
        if stage == "test" or stage is None:
            # We initialize a vanilla GTSRB test set as the source
            # Transforms are NOT passed to datasets.GTSRB directly so we can apply 
            # the corruption in the intermediate Numpy stage.
            base_test = datasets.GTSRB(
                self.hparams.data_dir, 
                split="test", 
                transform=None, 
                download=False
            )
            
            self.data_test = CorruptedDataset(
                base_dataset=base_test,
                corruption_type=self.hparams.corruption_type,
                severity=self.hparams.severity,
                transform=self.eval_transforms
            )
