import torch
import torch.nn as nn
import pytorch_lightning as L
from torchmetrics.classification import MulticlassAccuracy, CalibrationError
from src.models.evidential import EvidentialNetwork
from src.losses.dirichlet_loss import EDLLoss
from src.metrics.custom_metrics import (
    AdvancedSeverityRisk, 
    EntropyScore, 
    AdaptiveECE, 
    ClasswiseECE
)
from typing import Any, Dict, Optional

class EvidentialModule(L.LightningModule):
    def __init__(
        self,
        backbone: nn.Module = None,
        num_classes: int = 43,
        lr: float = 0.001,
        max_epochs: int = 50,
        annealing_epochs: int = 10,
        kl_penalty_weight: float = 0.2,
        n_bins: int = 15,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        # Senior best practice: ignore catch-all kwargs in hparams
        self.save_hyperparameters(ignore=['kwargs', 'backbone'])
        
        # 1. Model & Loss
        self.model = EvidentialNetwork(backbone=backbone, num_classes=num_classes)
        self.criterion = EDLLoss(num_classes=num_classes, annealing_epochs=annealing_epochs, kl_penalty_weight=kl_penalty_weight)
        
        # 2. Performance Metrics
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        
        # ECE: To track if EDL improves calibration out-of-the-box
        self.val_ece = CalibrationError(task="multiclass", num_classes=num_classes, n_bins=n_bins)
        self.test_ece = CalibrationError(task="multiclass", num_classes=num_classes, n_bins=n_bins)

        # ASR: Advanced Severity Risk (Senior Level)
        self.val_asr = AdvancedSeverityRisk(num_classes=num_classes)
        self.test_asr = AdvancedSeverityRisk(num_classes=num_classes)

        # Advanced Uncertainty Proxies
        self.val_entropy = EntropyScore()
        self.test_entropy = EntropyScore()

        # Ph.D. Rigor: Advanced Calibration
        self.test_aece = AdaptiveECE(n_bins=n_bins)
        self.test_cece = ClasswiseECE(target_classes=[13, 14]) # Yield, Stop
        self.test_brier = MulticlassBrierScore(num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        outputs = self.forward(x)
        
        # Target must be one-hot for EDLLoss
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=self.hparams.num_classes).float()
        
        # Pass current epoch for KL annealing
        loss_dict = self.criterion(outputs["alpha"], y_one_hot, self.current_epoch)
        loss = loss_dict["loss"]
        
        # Log training state
        self.train_acc(outputs["prob"], y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mse_loss", loss_dict["mse"], on_step=False, on_epoch=True)
        self.log("train/kl_loss", loss_dict["kl"], on_step=False, on_epoch=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def _eval_step(self, batch: Any, prefix: str):
        x, y = batch
        outputs = self.forward(x)
        
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=self.hparams.num_classes).float()
        loss_dict = self.criterion(outputs["alpha"], y_one_hot, self.current_epoch)
        loss = loss_dict["loss"]
        
        # Extract predictions and probabilities
        probs = outputs["prob"]
        vacuity = outputs["vacuity"]
        
        # Update metrics
        acc = getattr(self, f"{prefix}_acc")
        ece = getattr(self, f"{prefix}_ece")
        
        acc(probs, y)
        ece(probs, y)
        
        entropy = getattr(self, f"{prefix}_entropy")
        entropy(probs)
        
        # Update ASR (Evidential version passes vacuity to the metric)
        asr = getattr(self, f"{prefix}_asr")
        asr_results = asr(probs, y, vacuity=vacuity)
        
        # Log accuracy, ECE, and critical Vacuity (Epistemic Uncertainty)
        self.log(f"{prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/ece", ece, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/entropy", entropy, on_step=False, on_epoch=True)
        self.log(f"{prefix}/vacuity", torch.mean(vacuity), on_step=False, on_epoch=True, prog_bar=True)
        
        # Log Safety Metrics
        self.log(f"{prefix}/swe", asr_results["asr/swe"], on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/esp", asr_results["asr/esp"], on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/near_miss_rate", asr_results["asr/near_miss_rate"], on_step=False, on_epoch=True)

        # Log Category Breakdown
        for key, value in asr_results.items():
            if "esp_" in key:
                self.log(f"{prefix}/{key.replace('asr/', '')}", value, on_step=False, on_epoch=True)

        if prefix == "test":
            self.test_aece(probs, y)
            self.test_brier(probs, y)
            cece_results = self.test_cece(probs, y)
            self.log("test/aece", self.test_aece, on_step=False, on_epoch=True)
            self.log("test/brier", self.test_brier, on_step=False, on_epoch=True)
            for k, v in cece_results.items():
                self.log(f"test/{k}", v, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        return self._eval_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int):
        return self._eval_step(batch, "test")

    def configure_optimizers(self):
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
