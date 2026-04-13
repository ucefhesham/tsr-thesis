import pytorch_lightning as L
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from typing import Optional, Tuple
import os
import urllib.request
import zipfile

# BTSC (Belgium Traffic Sign Classification) to GTSRB (German Traffic Sign) Mapping
# This maps Belgium sign classes to their visually/semantically identical German counterparts.
# BTSC classes without a GTSRB equivalent are mapped to -1 (Ignored/OOD).
BTSC_TO_GTSRB_MAP = {
    # Speed limits
    17: 1,  # Speed limit (30km/h)
    18: 2,  # Speed limit (50km/h)
    19: 3,  # Speed limit (60km/h)
    20: 4,  # Speed limit (70km/h)
    21: 5,  # Speed limit (80km/h)
    22: 7,  # Speed limit (100km/h)
    23: 8,  # Speed limit (120km/h)
    
    # Prohibitory
    24: 9,  # No overtaking
    25: 10, # No overtaking (trucks)
    26: 15, # No vehicles
    27: 16, # No entry (trucks) - Approximate
    28: 17, # No entry
    
    # Danger
    33: 18, # General danger
    34: 19, # Dangerous curve left
    35: 20, # Dangerous curve right
    36: 21, # Double curve
    37: 22, # Bumpy road
    38: 23, # Slippery road
    39: 24, # Road narrows on the right
    40: 25, # Roadworks
    41: 26, # Traffic signals
    42: 27, # Pedestrians
    43: 28, # Children
    44: 29, # Bicycles
    45: 30, # Ice/snow
    46: 31, # Wild animals
    
    # Priority
    47: 11, # Right-of-way at intersection
    48: 12, # Priority road
    49: 13, # Yield
    50: 14, # Stop
    
    # Mandatory
    51: 33, # Turn right ahead
    52: 34, # Turn left ahead
    53: 35, # Ahead only
    54: 36, # Ahead or right
    55: 37, # Ahead or left
    56: 38, # Keep right
    57: 39, # Keep left
    58: 40, # Roundabout
    59: 41, # End of no passing
    60: 42  # End of no passing by trucks
}

class MappedImageFolder(Dataset):
    """Wraps ImageFolder to apply the subset mapping (BTSC to GTSRB)."""
    def __init__(self, root, transform=None):
        self.dataset = datasets.ImageFolder(root, transform=transform)
        # Filter indices to only those that exist in our mapping
        self.valid_indices = []
        for i, (path, label) in enumerate(self.dataset.samples):
            # The folder names in BTSC are usually format '00017' string -> int
            folder_class = int(os.path.basename(os.path.dirname(path)))
            if folder_class in BTSC_TO_GTSRB_MAP:
                self.valid_indices.append((i, BTSC_TO_GTSRB_MAP[folder_class]))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        original_idx, mapped_label = self.valid_indices[idx]
        img, _ = self.dataset[original_idx]
        return img, mapped_label

class BTSCDataModule(L.LightningDataModule):
    """
    BelgiumTS DataModule.
    Serves as an Epistemic Domain Shift alternative to ESD.
    """
    def __init__(
        self,
        data_dir: str = "data/external/btsc",
        batch_size: int = 32,
        num_workers: int = 2,
        input_size: int = 224,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.download_url = "https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip"

        self.eval_transforms = transforms.Compose([
            transforms.Resize((self.hparams.input_size, self.hparams.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.dataset: Optional[Dataset] = None

    def prepare_data(self):
        # Auto-download logic for BTSC Testing Set
        os.makedirs(self.hparams.data_dir, exist_ok=True)
        zip_path = os.path.join(self.hparams.data_dir, "BelgiumTSC_Testing.zip")
        extract_dir = os.path.join(self.hparams.data_dir, "Testing")
        
        if not os.path.exists(extract_dir):
            if not os.path.exists(zip_path):
                print(f"Downloading BelgiumTS (BTSC) to {zip_path}...")
                try:
                    urllib.request.urlretrieve(self.download_url, zip_path)
                    print("Download complete.")
                except Exception as e:
                    print(f"[WARNING] Could not automatically download BTSC: {e}")
                    return
            
            print(f"Extracting {zip_path}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.hparams.data_dir)
            except Exception as e:
                print(f"[WARNING] Extraction failed: {e}")

    def setup(self, stage: Optional[str] = None):
        extract_dir = os.path.join(self.hparams.data_dir, "Testing")
        if os.path.exists(extract_dir):
            try:
                self.dataset = MappedImageFolder(extract_dir, transform=self.eval_transforms)
                print(f"✅ BTSC Dataset loaded with {len(self.dataset)} mapped samples.")
            except Exception as e:
                print(f"[WARNING] BTSC Loading failed: {e}. Skipping domain shift evaluation.")
                self.dataset = None
        else:
            self.dataset = None

    def test_dataloader(self):
        if self.dataset is None or len(self.dataset) == 0:
            return None
            
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True
        )
