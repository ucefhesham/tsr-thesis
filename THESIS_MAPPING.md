# Scientific Thesis Alignment Mapping

This document provides a formal cross-reference between the mathematical definitions in `Proposal After 3rd Edits.tex` and their programmatic implementation in the `tsr-thesis` repository.

## 1. Safety-Critical Metrics

| LaTeX Concept | Thesis Equation / Section | Code Implementation (Primary) |
| :--- | :---: | :--- |
| **SWE** (Severity Weighted Error) | Eq (3.1) / Line 232 | [`src/metrics/custom_metrics.py:L102-105`](file:///d:/thesis/src/metrics/custom_metrics.py#L102-105) |
| **ESP** (Expected Severity Penalty) | Section 3.4.2 / Line 238 | [`src/metrics/custom_metrics.py:L108-109`](file:///d:/thesis/src/metrics/custom_metrics.py#L108-109) |
| **Cost Matrix** ($W_{ij}$) | Eq (3.2) / Line 236 | [`src/metrics/safety_config.py:L50-99`](file:///d:/thesis/src/metrics/safety_config.py#L50-99) |
| **Near-Miss Rate** | Line 237 | [`src/metrics/custom_metrics.py:L125-132`](file:///d:/thesis/src/metrics/custom_metrics.py#L125-132) |

## 2. Calibration & Trust

| LaTeX Concept | Thesis Equation / Section | Code Implementation (Primary) |
| :--- | :---: | :--- |
| **ECE** (15-Bin) | Eq (2.4) / Line 222 | [`src/metrics/custom_metrics.py:L24-38`](file:///d:/thesis/src/metrics/custom_metrics.py#L24-38) |
| **Brier Score** | Eq (2.5) / Line 225 | [`src/metrics/custom_metrics.py:L60-61`](file:///d:/thesis/src/metrics/custom_metrics.py#L60-61) |
| **TPE** (Trust-Per-Efficiency) | Eq (3.3) / Line 240 | [`tools/calculate_tpe.py:L37-41`](file:///d:/thesis/tools/calculate_tpe.py#L37-41) |

## 3. Evidential Deep Learning (Subjective Logic)

| LaTeX Concept | Thesis Equation / Section | Code Implementation (Primary) |
| :--- | :---: | :--- |
| **Dirichlet Prior** | Line 183 | [`src/losses/dirichlet_loss.py:L16-46`](file:///d:/thesis/src/losses/dirichlet_loss.py#L16-46) |
| **Vacuity** (Uncertainty Mass) | Line 184 | [`src/models/evidential.py:L74-78`](file:///d:/thesis/src/models/evidential.py#L74-78) |
| **EDL-MSE Loss** | Section 3.2.1 | [`src/losses/dirichlet_loss.py:L57-61`](file:///d:/thesis/src/losses/dirichlet_loss.py#L57-61) |
| **KL Annealing** | Line 190 | [`src/losses/dirichlet_loss.py:L67-71`](file:///d:/thesis/src/losses/dirichlet_loss.py#L67-71) |

## 4. Hardware Efficiency

| LaTeX Concept | Thesis Equation / Section | Code Implementation (Primary) |
| :--- | :---: | :--- |
| **GFLOPs** (Ops Count) | Line 240 | [`src/metrics/efficiency.py:L31-33`](file:///d:/thesis/src/metrics/efficiency.py#L31-33) |

---
*Note: Logic implementations are strictly hardware-agnostic, using fvcore-based profiling to ensure reproducibility across local (Windows) and remote (Linux/T4) environments.*
