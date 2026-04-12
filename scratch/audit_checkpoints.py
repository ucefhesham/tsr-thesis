import torch
import os
from pathlib import Path

checkpoint_dir = Path("d:/thesis/checkpoints")
checkpoints = list(checkpoint_dir.glob("*.ckpt"))

print(f"{'Filename':<40} | {'Epoch':<6} | {'Val Acc':<10} | {'Step':<10}")
print("-" * 75)

for ckpt in checkpoints:
    try:
        # map_location='cpu' to avoid OOM or CUDA issues
        checkpoint = torch.load(ckpt, map_location="cpu")
        
        # PL stores the best model score in the ModelCheckpoint callback state
        callbacks = checkpoint.get("callbacks", {})
        best_score = "N/A"
        for key, val in callbacks.items():
            if "ModelCheckpoint" in key and "best_model_score" in val:
                best_score = f"{val['best_model_score']:.4f}"
                break
        
        epoch = checkpoint.get("epoch", "N/A")
        global_step = checkpoint.get("global_step", "N/A")
        
        print(f"{ckpt.name:<40} | {epoch:<6} | {best_score:<10} | {global_step:<10}")
    except Exception as e:
        print(f"{ckpt.name:<40} | Error: {e}")
