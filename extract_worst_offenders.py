import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from src.models.resnet_baseline import ResNetBaselineModule
from src.datamodules.gtsrb_module import GTSRBDataModule

def denormalize(img):
    """Undo ImageNet normalization for visualization."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img.transpose((1, 2, 0)) # CHW -> HWC
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img

def main():
    # Setup paths
    ckpt_path = "checkpoints/epoch_017.ckpt" # Verified existing checkpoint
    save_path = "logs/worst_offenders.png"
    os.makedirs("logs", exist_ok=True)

    print(f"Loading model from {ckpt_path}...")
    model = ResNetBaselineModule.load_from_checkpoint(ckpt_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Initializing DataModule...")
    datamodule = GTSRBDataModule(batch_size=32)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    worst_offenders = []

    print("Running inference to find 'Worst Offenders' (Wrong but Confident)...")
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probs = F.softmax(logits, dim=1)
            confidences, preds = torch.max(probs, dim=1)

            # Filter for High-Confidence Failures: (pred != target) AND (conf > 0.90)
            mask = (preds != y) & (confidences > 0.90)
            
            if mask.any():
                idx_matches = torch.where(mask)[0]
                for idx in idx_matches:
                    offender = {
                        'image': x[idx].cpu().numpy(),
                        'true_label': y[idx].item(),
                        'pred_label': preds[idx].item(),
                        'confidence': confidences[idx].item()
                    }
                    worst_offenders.append(offender)
            
            # Optimization: Stop if we have enough candidates
            if len(worst_offenders) > 50:
                break

    if not worst_offenders:
        print("No high-confidence failures found with confidence > 0.90!")
        return

    # Sort by confidence descending and select top 5
    worst_offenders.sort(key=lambda x: x['confidence'], reverse=True)
    top_5 = worst_offenders[:5]

    print(f"Generating 1x5 plot with {len(top_5)} worst offenders...")
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    for i, offender in enumerate(top_5):
        img = denormalize(offender['image'])
        axes[i].imshow(img)
        axes[i].axis('off')
        
        title = (f"True: {offender['true_label']}\n"
                 f"Pred: {offender['pred_label']}\n"
                 f"Conf: {offender['confidence']*100:.1f}%")
        axes[i].set_title(title, fontsize=12, fontweight='bold', color='red')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Successfully saved worst offenders plot to: {save_path}")

if __name__ == "__main__":
    main()
