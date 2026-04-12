import pytorch_lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from typing import Optional, List
import os

class ExternalRobustnessDataModule(L.LightningDataModule):
    """
    Generic DataModule for Out-of-Distribution (OOD) and Robustness datasets
    like ESD (European Dataset) and Sticker Occlusion.
    """
    def __init__(
        self,
        dataset_name: str,
        data_dir: str = "data/external/",
        batch_size: int = 32,
        num_workers: int = 2,
        input_size: int = 224,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Deterministic evaluation transforms
        self.eval_transforms = transforms.Compose([
            transforms.Resize((self.hparams.input_size, self.hparams.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.dataset: Optional[datasets.ImageFolder] = None

    def prepare_data(self):
        """
        Note: ESD and Sticker datasets often require registration or manual download.
        We expect the data to be placed in data/external/{dataset_name} in ImageFolder format.
        """
        path = os.path.join(self.hparams.data_dir, self.hparams.dataset_name)
        if not os.path.exists(path):
            print(f"WARNING: External dataset '{self.hparams.dataset_name}' not found at {path}.")
            print("Please ensure the dataset is downloaded and structured as Country/Class/Images.")

    def setup(self, stage: Optional[str] = None):
        path = os.path.join(self.hparams.data_dir, self.hparams.dataset_name)
        if os.path.exists(path):
            self.dataset = datasets.ImageFolder(path, transform=self.eval_transforms)
        else:
            self.dataset = None

    def test_dataloader(self):
        if self.dataset is None:
            return None
            
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True
        )

# Specific wrappers for ease of config
class ESDDataModule(ExternalRobustnessDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_name="esd", **kwargs)

class StickerOcclusionDataModule(ExternalRobustnessDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_name="sticker_occlusion", **kwargs)
