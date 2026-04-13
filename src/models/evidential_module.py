import torch
import torch.nn as nn
import pytorch_lightning as L
from torchmetrics.classification import MulticlassAccuracy
from src.metrics.custom_metrics import MulticlassBrierScore

from src.models.evidential import EvidentialNetwork
from src.losses.dirichlet_loss import EDLLoss
from src.metrics.custom_metrics import (
    AdvancedSeverityRisk, 
    EntropyScore, 
    AdaptiveECE, 
    NegativeLogLikelihood
)
import time
import numpy as np
from typing import List, Dict, Any, Optional

class EvidentialModule(L.LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        num_classes: int = 43,
        lr: float = 0.0005,
        max_epochs: int = 100,
        annealing_epochs: int = 30,
        kl_penalty_weight: float = 0.4,
        n_bins: int = 15,
        name: Optional[str] = None,
        **kwargs,
    ):
        """
        Evidential Deep Learning Module for Trust Analysis.
        Aligned with configs/model/evidential.yaml
        """
        super().__init__()
        self.save_hyperparameters(ignore=['backbone', 'kwargs'], logger=False)

        # Wrap the backbone in the EvidentialNetwork to handle the evidence_head and softplus
        self.net = EvidentialNetwork(backbone=backbone, num_classes=num_classes)
        self.criterion = EDLLoss(
            num_classes=num_classes,
            annealing_epochs=annealing_epochs,
            kl_penalty_weight=kl_penalty_weight
        )

        # Metrics
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        
        self.val_ece = AdaptiveECE(n_bins=n_bins)
        self.test_ece = AdaptiveECE(n_bins=n_bins)
        self.test_nll = NegativeLogLikelihood()
        
        self.val_brier = MulticlassBrierScore(num_classes=num_classes)
        self.test_brier = MulticlassBrierScore(num_classes=num_classes)
        
        self.val_severity_risk = AdvancedSeverityRisk(num_classes=num_classes)
        self.test_severity_risk = AdvancedSeverityRisk(num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        # EvidentialNetwork returns a dict; we primarily need 'alpha' for the loss and probs
        out = self.net(x)
        if isinstance(out, dict):
            return out["alpha"]
        # Fallback for polymorphic wrappers that might return raw tensors for baselines
        return out

    def on_train_start(self):
        self.val_acc.reset()
        self.val_ece.reset()

    def model_step(self, batch: Any):
        x, y = batch
        alpha = self.forward(x)
        
        # Ensure y is one-hot for EDLLoss
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=self.hparams.num_classes).float()
        
        # Call loss with only expected positional arguments: alpha, y_one_hot, epoch
        results = self.criterion(
            alpha, 
            y_one_hot, 
            self.current_epoch
        )
        
        # Handle the dictionary return from EDLLoss
        loss = results["loss"]
        loss_details = results
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / S
        
        return loss, alpha, probs, y, loss_details

    def training_step(self, batch: Any, batch_idx: int):
        loss, alpha, probs, y, details = self.model_step(batch)
        self.train_acc(probs, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss, alpha, probs, y, _ = self.model_step(batch)
        self.val_acc(probs, y)
        self.val_ece(probs, y)
        self.val_brier(probs, y)
        asr_results = self.val_severity_risk(probs, y)
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ece", self.val_ece, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/brier", self.val_brier, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/swe", asr_results["asr/swe"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/esp", asr_results["asr/esp"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/near_miss_rate", asr_results["asr/near_miss_rate"], on_step=False, on_epoch=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, alpha, probs, y, _ = self.model_step(batch)
        self.test_acc(probs, y)
        self.test_ece(probs, y)
        self.test_nll(probs, y)
        self.test_brier(probs, y)
        asr_results = self.test_severity_risk(probs, y)
        
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test/ece", self.test_ece, on_step=False, on_epoch=True)
        self.log("test/nll", self.test_nll, on_step=False, on_epoch=True)
        self.log("test/brier", self.test_brier, on_step=False, on_epoch=True)
        self.log("test/swe", asr_results["asr/swe"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/esp", asr_results["asr/esp"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/near_miss_rate", asr_results["asr/near_miss_rate"], on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return {"optimizer": optimizer}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y = batch
        alpha = self.forward(x)
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / S
        entropy = EntropyScore()(probs)
        vacuity = self.hparams.num_classes / S.squeeze()
        
        return {
            "logits": alpha,
            "probs": probs,
            "uncertainty": vacuity,
            "entropy": entropy,
            "targets": y
        }
