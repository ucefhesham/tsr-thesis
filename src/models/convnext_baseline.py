import torch
import torch.nn as nn
import pytorch_lightning as L
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchmetrics.classification import MulticlassAccuracy, CalibrationError
from src.metrics.custom_metrics import MulticlassBrierScore
from src.metrics.custom_metrics import AdvancedSeverityRisk, EntropyScore, EnergyScore
from typing import Any, Dict, Optional

class ConvNextBaselineModule(L.LightningModule):
    def __init__(
        self,
        num_classes: int = 43,
        lr: float = 0.0005,
        max_epochs: int = 50,
        n_bins: int = 15,
        name: Optional[str] = None,
        **kwargs,
    ):
        """
        ConvNeXt-Tiny baseline classifier for Trust Analysis.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['kwargs'])

        # Architecture: ConvNeXt-Tiny with ImageNet weights
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Performance Metrics
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

        # Trust Metrics
        try:
            self.val_ece = CalibrationError(task="multiclass", num_classes=num_classes, n_bins=15)
            self.test_ece = CalibrationError(task="multiclass", num_classes=num_classes, n_bins=15)
        except:
            # Older torchmetrics fallback (no task arg)
            self.val_ece = CalibrationError(num_classes=num_classes, n_bins=15)
            self.test_ece = CalibrationError(num_classes=num_classes, n_bins=15)

        try:
            self.val_brier = MulticlassBrierScore(num_classes=num_classes)
            self.test_brier = MulticlassBrierScore(num_classes=num_classes)
        except:
            self.val_brier = MulticlassBrierScore()
            self.test_brier = MulticlassBrierScore()

        # ASR: Advanced Severity Risk
        self.val_asr = AdvancedSeverityRisk(num_classes=num_classes)
        self.test_asr = AdvancedSeverityRisk(num_classes=num_classes)

        # Advanced Uncertainty Proxies
        self.val_entropy = EntropyScore()
        self.test_entropy = EntropyScore()
        self.val_energy = EnergyScore()
        self.test_energy = EnergyScore()

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
        
        entropy = getattr(self, f"{prefix}_entropy")
        energy = getattr(self, f"{prefix}_energy")
        entropy(probs)
        energy(logits)
        
        asr = getattr(self, f"{prefix}_asr")
        asr_results = asr(probs, y)
        
        # Log Top-level Safety Metrics
        self.log(f"{prefix}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/ece", ece, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/brier", brier, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/entropy", entropy, on_step=False, on_epoch=True)
        self.log(f"{prefix}/energy", energy, on_step=False, on_epoch=True)
        self.log(f"{prefix}/swe", asr_results["asr/swe"], on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/esp", asr_results["asr/esp"], on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/near_miss_rate", asr_results["asr/near_miss_rate"], on_step=False, on_epoch=True)

        # Log Category Breakdown
        for key, value in asr_results.items():
            if "esp_" in key:
                self.log(f"{prefix}/{key.replace('asr/', '')}", value, on_step=False, on_epoch=True)
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
