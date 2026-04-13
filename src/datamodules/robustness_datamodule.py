import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from src.datamodules.gtsrb_module import GTSRBDataModule
from typing import Optional, Tuple
import PIL.Image as Image
import random
import numpy as np

class StickerTransform:
    """
    Simulates a physical 'Sticker' occlusion by pasting a random patch 
    onto the traffic sign image.
    """
    def __init__(self, p: float = 0.5, size_range: Tuple[float, float] = (0.05, 0.2)):
        """
        Args:
            p: Probability of applying the sticker.
            size_range: Ratio of sticker size relative to image size.
        """
        self.p = p
        self.size_range = size_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        
        # Convert to numpy for easier patch pasting
        img_arr = np.array(img).copy()
        h, w, _ = img_arr.shape
        
        # Determine sticker size
        sticker_ratio = random.uniform(*self.size_range)
        sh, sw = int(h * sticker_ratio), int(w * sticker_ratio)
        
        # Determine sticker position
        y = random.randint(0, h - sh)
        x = random.randint(0, w - sw)
        
        # Create a random 'adversarial' sticker (noise or random color/logo)
        # For a thesis, random structured noise is a good proxy for stickers
        sticker_type = random.choice(["noise", "color", "logo"])
        
        if sticker_type == "noise":
            patch = np.random.randint(0, 256, (sh, sw, 3), dtype=np.uint8)
        elif sticker_type == "color":
            color = np.random.randint(0, 256, (3,), dtype=np.uint8)
            patch = np.tile(color, (sh, sw, 1))
        else:
            # Simple 'X' logo simulation
            patch = np.zeros((sh, sw, 3), dtype=np.uint8)
            patch[:, :] = [255, 255, 255] # White background
            for i in range(min(sh, sw)):
                patch[i, i] = [255, 0, 0] # Red diagonal
                patch[i, sw - 1 - i] = [255, 0, 0]
        
        # Paste with slight alpha blending (simulating lighting/reflection)
        alpha = random.uniform(0.7, 0.9)
        img_arr[y:y+sh, x:x+sw] = (alpha * patch + (1 - alpha) * img_arr[y:y+sh, x:x+sw]).astype(np.uint8)
        
        return Image.fromarray(img_arr)

class DetectionJitterTransform:
    """
    Simulates detection window jitter/misalignment.
    Used for 'Upstream Noise' stress cases.
    """
    def __init__(self, translation_limit: float = 0.15, scale_limit: float = 0.15):
        self.translation_limit = translation_limit
        self.scale_limit = scale_limit

    def get_params(self, w: int, h: int) -> Tuple[float, float, float]:
        tx = random.uniform(-self.translation_limit, self.translation_limit) * w
        ty = random.uniform(-self.translation_limit, self.translation_limit) * h
        s = random.uniform(1.0 - self.scale_limit, 1.0 + self.scale_limit)
        return tx, ty, s

    def apply_to_bbox(self, bbox: Tuple[int, int, int, int], tx: float, ty: float, s: float, w: int, h: int) -> Tuple[int, int, int, int]:
        cx, cy = w / 2, h / 2
        x1, y1, x2, y2 = bbox
        x1, x2 = (x1 - cx) * s + cx, (x2 - cx) * s + cx
        y1, y2 = (y1 - cy) * s + cy, (y2 - cy) * s + cy
        x1, x2 = x1 + tx, x2 + tx
        y1, y2 = y1 + ty, y2 + ty
        return (int(x1), int(y1), int(x2), int(y2))

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        tx, ty, s = self.get_params(w, h)
        cx, cy = w / 2, h / 2
        a, b, c = 1.0 / s, 0, cx - (cx + tx) / s
        d, e, f = 0, 1.0 / s, cy - (cy + ty) / s
        return img.transform((w, h), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)

class UpstreamNoiseDataModule(GTSRBDataModule):
    """
    DataModule that simulates 'Sloppy Detector' outputs using Jitter.
    """
    def __init__(self, jitter_level: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.jitter_level = jitter_level
        self.jitter_transforms = transforms.Compose([
            transforms.Resize((self.hparams.input_size, self.hparams.input_size)),
            DetectionJitterTransform(translation_limit=jitter_level, scale_limit=jitter_level),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage: Optional[str] = None):
        super().setup(stage)
        if stage == "test" or stage is None:
            self.data_test = datasets.GTSRB(
                self.hparams.data_dir, 
                split="test", 
                transform=self.jitter_transforms, 
                download=False
            )

class StickerOcclusionDataModule(GTSRBDataModule):
    """
    A GTSRB DataModule where the test set is occluded by synthetic stickers.
    """
    def __init__(self, sticker_prob: float = 1.0, sticker_size: Tuple[float, float] = (0.1, 0.3), **kwargs):
        super().__init__(**kwargs)
        self.sticker_prob = sticker_prob
        self.sticker_size = sticker_size
        
        # Add sticker transform to the eval chain for test set
        self.test_transforms = transforms.Compose([
            transforms.Resize((self.hparams.input_size, self.hparams.input_size)),
            StickerTransform(p=self.sticker_prob, size_range=self.sticker_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage: Optional[str] = None):
        super().setup(stage)
        if stage == "test" or stage is None:
            # Re-instantiate test set with the sticker transform
            self.data_test = datasets.GTSRB(
                self.hparams.data_dir, 
                split="test", 
                transform=self.test_transforms, 
                download=False
            )

class JunkDataModule(GTSRBDataModule):
    """
    DataModule for Open-Set Recognition (OSR). 
    Loads unrelated images (CIFAR-10) to test out-of-distribution (OOD) performance.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use same eval transforms for normalization consistency
        self.junk_transforms = self.eval_transforms

    def setup(self, stage: Optional[str] = None):
        if stage == "test" or stage is None:
            # We use CIFAR-10 as 'Junk' (Non-Traffic Signs)
            # All labels in CIFAR-10 are 'OOD' relative to GTSRB
            self.data_test = datasets.CIFAR10(
                self.hparams.data_dir,
                train=False,
                transform=self.junk_transforms,
                download=True
            )
            # We wrap the labels to indicate 'Unknown' (usually Class -1 or a special flag)
            # But for trust metrics, we just care about the model's confidence scores.

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
