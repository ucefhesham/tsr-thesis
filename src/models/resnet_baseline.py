import torch
import torch.nn as nn
import pytorch_lightning as L
from torchvision.models import resnet18, ResNet18_Weights
from torchmetrics.classification import MulticlassAccuracy, CalibrationError
from src.metrics.custom_metrics import (
    MulticlassBrierScore, 
    SeverityWeightedError, 
    AdaptiveECE, 
    ClasswiseECE
)
from typing import Any, Dict, Optional

class ResNetBaselineModule(L.LightningModule):
    def __init__(
        self,
        num_classes: int = 43,
        lr: float = 0.001,
        max_epochs: int = 50,
        n_bins: int = 15,
        name: Optional[str] = None,
        **kwargs
    ):
        """
        ResNet-18 baseline classifier for Trust Analysis.
        """
        super().__init__()
        # Senior best practice: ignore catch-all kwargs in hparams
        self.save_hyperparameters(ignore=['kwargs'])

        # Architecture: ResNet-18 with ImageNet weights
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Modify final FC layer for GTSRB (43 classes)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Performance Metrics
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

        # Trust Metrics (Only for Val and Test)
        # ECE: Expected Calibration Error quantified in 15 bins
        self.val_ece = CalibrationError(task="multiclass", num_classes=num_classes, n_bins=n_bins)
        self.test_ece = CalibrationError(task="multiclass", num_classes=num_classes, n_bins=n_bins)

        # Multiclass Brier Score: Proper scoring rule that penalizes both calibration and refinement
        self.val_brier = MulticlassBrierScore(num_classes=num_classes)
        self.test_brier = MulticlassBrierScore(num_classes=num_classes)

        # SWE: Severity-Weighted Error (Trust Analysis)
        self.val_swe = SeverityWeightedError(num_classes=num_classes)
        self.test_swe = SeverityWeightedError(num_classes=num_classes)

        # Ph.D. Rigor: Advanced Calibration
        self.test_aece = AdaptiveECE(n_bins=n_bins)
        self.test_cece = ClasswiseECE(target_classes=[13, 14]) # Yield, Stop

    def forward(self, x: torch.Tensor):
        return self.backbone(x)

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        
        # Training accuracy (using predictions from logits)
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)

        # Logging training state
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def _eval_step(self, batch: Any, prefix: str):
        """Helper for shared validation and test logic."""
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        
        # CRITICAL: Transform logits to probabilities for reliability metrics
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        # Dynamic metric retrieval
        acc = getattr(self, f"{prefix}_acc")
        ece = getattr(self, f"{prefix}_ece")
        brier = getattr(self, f"{prefix}_brier")

        # Update metrics
        acc(preds, y)
        ece(probs, y)
        brier(probs, y)
        
        if hasattr(self, f"{prefix}_swe"):
            swe = getattr(self, f"{prefix}_swe")
            swe_results = swe(probs, y)
            
            # Log individual safety components
            self.log(f"{prefix}/swe", swe_results["asr/swe"], on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{prefix}/esp", swe_results["asr/esp"], on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{prefix}/near_miss_rate", swe_results["asr/near_miss_rate"], on_step=False, on_epoch=True)

            # Log Category Breakdown (Speed, Priority, etc.)
            for key, value in swe_results.items():
                if "esp_" in key:
                    self.log(f"{prefix}/{key.replace('asr/', '')}", value, on_step=False, on_epoch=True)

        if prefix == "test":
            self.test_aece(probs, y)
            cece_results = self.test_cece(probs, y)
            self.log("test/aece", self.test_aece, on_step=False, on_epoch=True)
            for k, v in cece_results.items():
                self.log(f"test/{k}", v, on_step=False, on_epoch=True)

        # Logging with prefix (val/ or test/)
        self.log(f"{prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/ece", ece, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/brier", brier, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        return self._eval_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int):
        return self._eval_step(batch, "test")

    def configure_optimizers(self):
        """Configure Adam optimizer with CosineAnnealingLR scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.hparams.max_epochs
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
