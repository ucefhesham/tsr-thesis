import torch
import sys
import os

# Add src to path for local execution
sys.path.append(os.getcwd())

from src.metrics.custom_metrics import AdvancedSeverityRisk

def test_asr_logic():
    print("--- Testing ASR Suite Logic ---")
    num_classes = 43
    asr = AdvancedSeverityRisk(num_classes=num_classes, near_miss_threshold=0.1)

    # 1. Test ASYMMETRY (30km/h vs 120km/h)
    # Class 1 is 30km/h, Class 8 is 120km/h
    # Case A: Predicted 120 when GT is 30 (DANGEROUS)
    # Case B: Predicted 30 when GT is 120 (INEFFICIENT)
    
    # Case A (Dangerous Over-speed)
    preds_a = torch.zeros((1, num_classes))
    preds_a[0, 8] = 1.0 # 100% confident in 120
    targets_a = torch.tensor([1]) # GT is 30
    
    # Case B (Inefficient Under-speed)
    preds_b = torch.zeros((1, num_classes))
    preds_b[0, 1] = 1.0 # 100% confident in 30
    targets_b = torch.tensor([8]) # GT is 120

    asr.reset()
    asr.update(preds_a, targets_a)
    cost_a = asr.compute()["asr/swe"].item()
    
    asr.reset()
    asr.update(preds_b, targets_b)
    cost_b = asr.compute()["asr/swe"].item()
    
    print(f"Cost 30->120 (Dangerous): {cost_a:.2f}")
    print(f"Cost 120->30 (Inefficient): {cost_b:.2f}")
    assert cost_a > cost_b, "Asymmetry failed: Dangerous over-speed should cost more than under-speed."

    # 2. Test SENSITIVITY (Top-1 correct, but mass on dangerous classes)
    # Target is 30km/h (Class 1). Pred is 30km/h (Top-1 correct).
    # Residual mass on 120km/h (Class 8).
    preds_c = torch.zeros((1, num_classes))
    preds_c[0, 1] = 0.8 # Correct
    preds_c[0, 8] = 0.2 # dangerous mass
    targets_c = torch.tensor([1])
    
    asr.reset()
    asr.update(preds_c, targets_c)
    results_c = asr.compute()
    swe_c = results_c["asr/swe"].item()
    esp_c = results_c["asr/esp"].item()
    near_miss_c = results_c["asr/near_miss_rate"].item()
    
    print(f"\nCorrect prediction with 20% mass on dangerous class:")
    print(f"SWE: {swe_c:.2f} (Should be 0)")
    print(f"ESP: {esp_c:.2f} (Should be > 0)")
    print(f"Near Miss Rate: {near_miss_c:.2f} (Should be 1.0)")
    
    assert swe_c == 0.0, "SWE should be zero for correct top-1."
    assert esp_c > 0.0, "ESP should reflect dangerous mass even if top-1 is correct."
    assert near_miss_c == 1.0, "Near miss should be detected."

    # 3. Test CONVERGENCE (ESP approaches SWE as confidence approaches 1.0)
    # Wrong prediction: GT is 30, Predicted is 120 with 0.999 confidence
    preds_d = torch.zeros((2, num_classes))
    preds_d[0, 8] = 0.9999
    preds_d[0, 1] = 0.0001
    targets_d = torch.tensor([1])
    
    asr.reset()
    asr.update(preds_d[0:1], targets_d)
    results_d = asr.compute()
    swe_d = results_d["asr/swe"].item()
    esp_d = results_d["asr/esp"].item()
    
    print(f"\nHighly confident wrong prediction (Confidence 0.9999):")
    print(f"SWE: {swe_d:.4f}")
    print(f"ESP: {esp_d:.4f}")
    assert abs(swe_d - esp_d) < 1e-2, "ESP should converge to SWE as confidence grows."

    print("\n--- All Safety Metric Tests Passed! ---")

if __name__ == "__main__":
    test_asr_logic()
