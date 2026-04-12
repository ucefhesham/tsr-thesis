import torch
import torch.nn as nn
import torch.nn.functional as F

class EDLLoss(nn.Module):
    def __init__(self, num_classes: int = 43, annealing_epochs: int = 10, kl_penalty_weight: float = 0.2):
        """
        Evidential Deep Learning Loss (Sensoy et al., 2018).
        Calculates MSE Loss + KL Regularization for Dirichlet parameters.
        """
        super().__init__()
        self.num_classes = num_classes
        self.annealing_epochs = annealing_epochs
        self.kl_penalty_weight = kl_penalty_weight

    def kl_divergence(self, alpha, y):
        """
        KL Divergence between predicted Dirichlet and Flat Dirichlet.
        Forced to float32 to prevent overflow in lgamma/digamma.
        """
        # Ensure calculations are in float32
        alpha = alpha.float()
        y = y.float()
        
        # alpha_tilde = y + (1 - y) * alpha
        # Add epsilon to prevent issues with alpha=1 exactly in digamma
        alpha_tilde = y + (1 - y) * alpha + 1e-10
        
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
        K = self.num_classes
        device = alpha.device
        
        # Dirichlet KL formula in float32
        first_term = (
            torch.lgamma(S_tilde)
            - torch.lgamma(torch.tensor(float(K), device=device))
            - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)
        )
        
        second_term = torch.sum(
            (alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)),
            dim=1,
            keepdim=True
        )
        
        return first_term + second_term

    def forward(self, alpha, y_one_hot, epoch):
        """
        alpha: Expected (batch_size, num_classes), values > 1
        y_one_hot: Expected (batch_size, num_classes), values in {0, 1}
        epoch: Current training epoch for KL annealing
        """
        S = torch.sum(alpha, dim=1, keepdim=True)
        p = alpha / S
        
        # 1. Expected Mean Square Error (EDL-MSE)
        err = (y_one_hot - p) ** 2
        var = p * (1 - p) / (S + 1)
        mse_loss = torch.sum(err + var, dim=1, keepdim=True)
        mse_loss = torch.mean(mse_loss)
        
        # 2. KL Divergence Regularization
        kl_loss = self.kl_divergence(alpha, y_one_hot)
        kl_loss = torch.mean(kl_loss)
        
        # 3. Annealing factor lambda_t = min(1.0, epoch / annealing_epochs)
        annealing_factor = min(1.0, epoch / self.annealing_epochs)
        
        # Total loss: Scale KL by kl_penalty_weight to prioritize accuracy
        total_loss = mse_loss + self.kl_penalty_weight * annealing_factor * kl_loss
        
        return {
            "loss": total_loss,
            "mse": mse_loss,
            "kl": kl_loss,
            "annealing_factor": annealing_factor
        }
