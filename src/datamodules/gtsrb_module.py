import pytorch_lightning as L
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import GTSRB
from torchvision import transforms
import torch
from typing import Optional, List

class GTSRBDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        input_size: int = 224,
    ):
        """
        PyTorch Lightning DataModule for GTSRB.
        
        Args:
            data_dir: Directory to store/load the dataset.
            batch_size: Batch size for dataloaders.
            num_workers: Number of workers for dataloaders.
            pin_memory: Whether to pin memory in dataloaders.
            input_size: Target image size for resizing.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Preprocessing & Augmentation
        # Training transforms include basic augmentations for robustness
        self.train_transforms = transforms.Compose([
            transforms.Resize((self.hparams.input_size, self.hparams.input_size)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Evaluation transforms (Val/Test/Calibration) are strictly deterministic
        self.eval_transforms = transforms.Compose([
            transforms.Resize((self.hparams.input_size, self.hparams.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.data_train: Optional[Subset] = None
        self.data_val: Optional[Subset] = None
        self.data_test: Optional[GTSRB] = None
        self.data_cal: Optional[Subset] = None

    def prepare_data(self):
        """Download GTSRB data if not already present."""
        GTSRB(self.hparams.data_dir, split="train", download=True)
        GTSRB(self.hparams.data_dir, split="test", download=True)

    def setup(self, stage: Optional[str] = None):
        """
        Split data and set up datasets. 
        Uses a dual-transform strategy to apply augmentations only to the training subset.
        """
        # Instantiate full training dataset twice to handle different transforms
        full_train_aug = GTSRB(self.hparams.data_dir, split="train", transform=self.train_transforms)
        full_train_eval = GTSRB(self.hparams.data_dir, split="train", transform=self.eval_transforms)

        # Generate reproducible indices for splitting (70% Train, 20% Val, 10% Calibration)
        n_samples = len(full_train_aug)
        n_train = int(0.7 * n_samples)
        n_val = int(0.2 * n_samples)
        n_cal = n_samples - n_train - n_val

        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(n_samples, generator=generator).tolist()

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        cal_indices = indices[n_train + n_val:]

        # Create Subsets based on the stage
        if stage == "fit" or stage is None:
            self.data_train = Subset(full_train_aug, train_indices)
            self.data_val = Subset(full_train_eval, val_indices)

        if stage == "test" or stage is None:
            self.data_test = GTSRB(self.hparams.data_dir, split="test", transform=self.eval_transforms)

        # Always keep the calibration subset accessible
        self.data_cal = Subset(full_train_eval, cal_indices)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def calibration_dataloader(self):
        """Dedicated dataloader for post-hoc trust calibration."""
        return DataLoader(
            self.data_cal,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
