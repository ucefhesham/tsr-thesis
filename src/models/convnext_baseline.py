import torch
import torch.nn as nn
import pytorch_lightning as L
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchmetrics.classification import MulticlassAccuracy, CalibrationError
from src.metrics.custom_metrics import MulticlassBrierScore, SeverityWeightedError
from typing import Any, Dict, Optional

class ConvNextBaselineModule(L.LightningModule):
    def __init__(
        self,
        num_classes: int = 43,
        lr: float = 0.0005,  # ConvNeXt often prefers slightly lower LR than ResNet
        max_epochs: int = 50,
        cost_config: Optional[Dict] = None,
    ):
        """
        ConvNeXt-Tiny baseline classifier for Trust Analysis.
        ConvNeXt is a modern isotropic architecture that often offers better inherent 
        calibration than traditional pyramidal CNNs like ResNet.
        
        Args:
            num_classes: Number of target classes (GTSRB = 43).
            lr: Initial learning rate.
            max_epochs: Total number of epochs for the CosineAnnealingLR scheduler.
            cost_config: Configuration for Severity-Weighted Error.
        """
        super().__init__()
        self.save_hyperparameters()

        # Architecture: ConvNeXt-Tiny with ImageNet weights
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        
        # Modify the classifier head for GTSRB (43 classes)
        # ConvNeXt head is typically: (head): Sequential( (0): LayerNorm..., (1): Flatten, (2): Linear )
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Performance Metrics
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

        # Trust Metrics
        self.val_ece = CalibrationError(task="multiclass", num_classes=num_classes, n_bins=15)
        self.test_ece = CalibrationError(task="multiclass", num_classes=num_classes, n_bins=15)

        self.val_brier = MulticlassBrierScore(num_classes=num_classes)
        self.test_brier = MulticlassBrierScore(num_classes=num_classes)

        # SWE: Severity-Weighted Error
        if cost_config:
            self.val_swe = SeverityWeightedError(cost_config=cost_config, num_classes=num_classes)
            self.test_swe = SeverityWeightedError(cost_config=cost_config, num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        return self.backbone(x)

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def _eval_step(self, batch: Any, prefix: str):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        # Log metrics
        acc = getattr(self, f"{prefix}_acc")
        ece = getattr(self, f"{prefix}_ece")
        brier = getattr(self, f"{prefix}_brier")

        acc(preds, y)
        ece(probs, y)
        brier(probs, y)
        
        if hasattr(self, f"{prefix}_swe"):
            swe = getattr(self, f"{prefix}_swe")
            swe(preds, y)
            self.log(f"{prefix}/swe", swe, on_step=False, on_epoch=True, prog_bar=True)

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
