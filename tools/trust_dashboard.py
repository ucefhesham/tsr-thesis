import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from typing import Dict, Any

def create_trust_dashboard(
    image: np.ndarray,
    class_map: np.ndarray,
    uncertainty_map: np.ndarray,
    metrics: Dict[str, Any],
    save_path: str
):
    """
    Generates a premium, senior-level trust dashboard infographic.
    """
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 8))
    
    # 1. Main Image & Maps
    ax1 = plt.subplot2grid((2, 4), (0, 0))
    ax1.imshow(image)
    ax1.set_title("Input Sample", fontsize=12, pad=10)
    ax1.axis('off')
    
    ax2 = plt.subplot2grid((2, 4), (0, 1))
    ax2.imshow(class_map)
    ax2.set_title("Class Saliency (Logits)", fontsize=12, pad=10)
    ax2.axis('off')
    
    ax3 = plt.subplot2grid((2, 4), (0, 2))
    ax3.imshow(uncertainty_map)
    ax3.set_title("Uncertainty Saliency", fontsize=12, pad=10)
    ax3.axis('off')
    
    # 2. Metric Breakdown Panel
    ax_metrics = plt.subplot2grid((2, 4), (0, 3), rowspan=2)
    ax_metrics.axis('off')
    
    y_pos = 0.95
    ax_metrics.text(0, y_pos, "TRUST ANALYTICS", fontsize=18, fontweight='bold', color='#00d4ff')
    y_pos -= 0.1
    
    # Priority Metrics
    p_class = metrics.get("pred_class", "N/A")
    gt = metrics.get("gt_label", "N/A")
    res_text = "CORRECT" if str(p_class) == str(gt) else "ERROR"
    ax_metrics.text(0, y_pos, f"PRED: {p_class} ({res_text})", fontsize=12, fontweight='bold', color='#4bff4b' if res_text == "CORRECT" else '#ff4b4b')
    y_pos -= 0.08

    # Logic for displaying metrics
    display_keys = {
        "vacuity": "Epistemic Vacuity",
        "entropy": "Inherent Entropy",
        "energy": "Energy Score",
        "sfe_score": "Saliency Focus (SFE)"
    }
    
    for key, label in display_keys.items():
        if key not in metrics: continue
        val = metrics[key]
        color = 'white'
        if key == "vacuity" and val > 0.5: color = '#ff4b4b'
        if key == "sfe_score" and val < 0.6: color = '#ff9f4b' # Low focus
        
        ax_metrics.text(0, y_pos, f"{label}:", fontsize=11)
        ax_metrics.text(0.7, y_pos, f"{val:.4f}" if isinstance(val, float) else str(val), fontsize=11, color=color, fontweight='bold')
        y_pos -= 0.06

    # 3. Conformal Prediction Set (New)
    y_pos -= 0.02
    conf_set = metrics.get("conformal_set", "N/A")
    set_size = metrics.get("set_size", 0)
    
    ax_metrics.text(0, y_pos, "95% CONFIDENCE SET:", fontsize=11, fontweight='bold', color='#f1c40f')
    y_pos -= 0.05
    ax_metrics.text(0, y_pos, f"{conf_set}", fontsize=10, color='white', wrap=True)
    y_pos -= 0.08

    # 3. Model Recommendation & Policy
    y_pos -= 0.05
    vacuity = metrics.get("vacuity", 0)
    poly = "PROCEED"
    if vacuity > 0.6:
        poly = "SAFE FALLBACK (30km/h)"
    elif metrics.get("entropy", 0) > 1.8:
        poly = "ABSTAIN"
        
    ax_metrics.text(0, y_pos, "SAFETY POLICY:", fontsize=12, fontweight='bold')
    ax_metrics.text(0, y_pos - 0.08, poly, fontsize=16, fontweight='bold', color='#ff4b4b' if "SAFE" in poly or "ABSTAIN" in poly else '#4bff4b')

    # 4. Calibration Curve (Placeholder/Simple)
    ax4 = plt.subplot2grid((2, 4), (1, 0), colspan=3)
    ax4.text(0.5, 0.5, "Saliency Spectrum Analysis", ha='center', va='center', fontsize=14, alpha=0.3)
    # We could plot the class probability distribution here
    if "probs" in metrics:
        probs = metrics["probs"]
        top_k = torch.topk(torch.tensor(probs), k=min(10, len(probs)))
        ax4.bar(range(len(top_k.values)), top_k.values.numpy(), color='#00d4ff', alpha=0.7)
        ax4.set_xticks(range(len(top_k.indices)))
        ax4.set_xticklabels(top_k.indices.numpy(), rotation=45)
        ax4.set_title("Top-10 Predicted Class Probabilities")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Dashboard saved: {save_path}")

if __name__ == "__main__":
    # Test with dummy data
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_map = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_metrics = {
        "prediction": "Class 14 (Stop)",
        "vacuity": 0.78,
        "energy": -12.4,
        "entropy": 2.1,
        "conformal_set": "{14, 13}",
        "probs": np.random.dirichlet([1]*43)
    }
    create_trust_dashboard(dummy_img, dummy_map, dummy_map, dummy_metrics, "logs/test_dashboard.png")
