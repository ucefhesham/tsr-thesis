import torch
import torch.nn as nn
import pytorch_lightning as L
from torchmetrics.classification import MulticlassAccuracy, CalibrationError
from src.models.evidential import EvidentialNetwork
from src.losses.dirichlet_loss import EDLLoss
from typing import Any, Dict, Optional

class EvidentialResNetModule(L.LightningModule):
    def __init__(
        self,
        num_classes: int = 43,
        lr: float = 0.001,
        max_epochs: int = 50,
        annealing_epochs: int = 10,
        kl_penalty_weight: float = 0.2,
    ):
        """
        PyTorch Lightning Module for Evidential Deep Learning.
        """
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Model & Loss
        self.model = EvidentialNetwork(num_classes=num_classes)
        self.criterion = EDLLoss(num_classes=num_classes, annealing_epochs=annealing_epochs, kl_penalty_weight=kl_penalty_weight)
        
        # 2. Performance Metrics
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        
        # ECE: To track if EDL improves calibration out-of-the-box
        self.val_ece = CalibrationError(task="multiclass", num_classes=num_classes, n_bins=15)
        self.test_ece = CalibrationError(task="multiclass", num_classes=num_classes, n_bins=15)

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
        
        # Log accuracy, ECE, and critical Vacuity (Epistemic Uncertainty)
        self.log(f"{prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/ece", ece, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/vacuity", torch.mean(vacuity), on_step=False, on_epoch=True, prog_bar=True)
        
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
