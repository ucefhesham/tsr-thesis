import torch
import torch.nn as nn
import pytorch_lightning as L
from src.models.resnet_baseline import ResNetBaselineModule
from src.models.backbones import get_resnet18_deep_dropout
from typing import Any, Dict, Optional

class MCDropoutModule(ResNetBaselineModule):
    def __init__(
        self,
        num_classes: int = 43,
        lr: float = 0.001,
        max_epochs: int = 50,
        n_bins: int = 15,
        num_samples: int = 10,
        dropout_p: float = 0.15,
        **kwargs
    ):
        """
        Monte-Carlo Dropout wrapper for ResNet-18. 
        Answers RQ 5: Trust gains vs. Stochastic Overhead.
        """
        super().__init__(num_classes=num_classes, lr=lr, max_epochs=max_epochs, n_bins=n_bins, **kwargs)
        self.save_hyperparameters(ignore=['kwargs'])
        
        # Override the backbone with our Deep Dropout version
        self.backbone = get_resnet18_deep_dropout(p=dropout_p)
        self.model = nn.Sequential(
            self.backbone,
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward_stochastic(self, x: torch.Tensor, n: int):
        """Performs N stochastic forward passes and returns mean probs and variance."""
        self.train() # Force dropout to be active even during eval
        
        all_logits = []
        for _ in range(n):
            logits = self.model(x)
            all_logits.append(logits)
        
        all_logits = torch.stack(all_logits) # [N, B, K]
        all_probs = torch.softmax(all_logits, dim=-1) # [N, B, K]
        
        mean_probs = torch.mean(all_probs, dim=0) # [B, K]
        # Epistemic uncertainty can be measured as the variance of the predictions
        variance = torch.var(all_probs, dim=0) # [B, K]
        
        return mean_probs, variance

    def _eval_step(self, batch: Any, prefix: str):
        """Overridden eval step to use stochastic inference."""
        x, y = batch
        
        # We perform N passes to get the predictive distribution
        probs, variance = self.forward_stochastic(x, n=self.hparams.num_samples)
        
        # For loss, we still use the mean probability (or we could use the mean logit)
        # Here we use CrossEntropy on the mean log-probs roughly
        loss = torch.nn.functional.cross_entropy(torch.log(probs + 1e-10), y)
        
        preds = torch.argmax(probs, dim=1)

        # Dynamic metric retrieval
        acc = getattr(self, f"{prefix}_acc")
        ece = getattr(self, f"{prefix}_ece")
        brier = getattr(self, f"{prefix}_brier")
        swe = getattr(self, f"{prefix}_swe")

        # Update and log metrics
        acc(preds, y)
        ece(probs, y)
        brier(probs, y)
        
        swe_results = swe(probs, y)
        
        # Log results
        self.log(f"{prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/ece", ece, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/brier", brier, on_step=False, on_epoch=True)
        self.log(f"{prefix}/swe", swe_results["asr/swe"], on_step=False, on_epoch=True)
        self.log(f"{prefix}/esp", swe_results["asr/esp"], on_step=False, on_epoch=True)

        # Log average variance as an epistemic uncertainty proxy
        self.log(f"{prefix}/epistemic_variance", torch.mean(variance), on_step=False, on_epoch=True)
        
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        """Standard training step (one pass) for efficiency."""
        self.model.train()
        x, y = batch
        logits = self.model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
