# Ph.D. Research Specification & Reproducibility Guide

This document provides the formal technical mapping and theoretical foundations of the Traffic Sign Reliability (TSR) Benchmarking Suite.

## 1. Theoretical Mapping (Research Questions)

| Research Question (RQ) | Code Implementation Module | Scientific Objective |
| :--- | :--- | :--- |
| **RQ 1: Calibration Decay** | `eval.py` (Phase 1-3) | Quantify ECE escalation under scaling severities (1-5). |
| **RQ 2: Severity-Utility** | `src/metrics/custom_metrics.py` | Implementation of SWE/ESP/ERS metrics. |
| **RQ 3: Selective Prediction** | `tools/analyze_selective_prediction.py` | Calculation of AURC and Coverage-Risk thresholds. |
| **RQ 4: Safe Policy** | `src/models/calibration.py` | Post-hoc SafePolicyWrapper fallback logic. |
| **RQ 5: Stochastic Epistemic** | `src/models/mc_dropout.py` & `ensemble.py` | Comparison of single-pass vs. sampling models. |
| **RQ 6: Visual Trust** | `tools/trust_dashboard.py` | Graduation of Grad-CAM and Conformal Set overlays. |

## 2. Advanced Methodology

### 2.1 Physics-Grounded Cost Matrix
The **AdvancedSeverityRisk (ASR)** metric utilizes a cost matrix $W \in \mathbb{R}^{K \times K}$ where penalties are grounded in physical consequences:
- **Speed-Sign Asymmetry**: Over-estimating speed is penalized quadratically (safety risk); under-estimating is penalized linearly (efficiency risk).
- **Priority-Flipping**: Misclassifying a `STOP` or `YIELD` sign into a passive group carries a maximum penalty (15.0).
- **Vulnerable Group Bias**: Misclassification of `Pedestrian` or `Children` signs is weighted at 12.0x standard error.

### 2.2 Adaptive Calibration
To avoid the bias of fixed-width binning in Expected Calibration Error (ECE), we utilize **Adaptive ECE (AECE)**. This method calculates boundaries such that each bin contains an equal number of samples ($N_{bin} = N/M$), providing a more robust estimate in high-confidence regions.

### 2.3 Statistical Significance
All results reported in the final thesis tables are aggregated over $N=5$ independent runs with unique random seeds. Significance is determined via the non-parametric **Mann-Whitney U test** ($p < 0.05$), ensuring that reported performance gains are mathematically defensible.

## 3. Reproducibility

To regenerate the entire thesis results ledger:
1. **Train Ensemble**: `bash scripts/train_ensemble.sh`
2. **Execute Systematic Stress**: `python eval.py ckpt_path=...` (Repeat for all members)
3. **Export Tables**: `python tools/latex_exporter.py`

---
*Author: Ph.D. Automated Finalization Engine*
*Date: 2026-04-12*
