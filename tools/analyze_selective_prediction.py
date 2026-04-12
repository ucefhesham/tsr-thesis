import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
import os

class SelectivePredictionAnalyzer:
    """
    Calculates Risk-Coverage trade-offs and AURC for thesis results.
    Answers RQ 4: Selective Prediction efficacy.
    """
    def __init__(self, output_dir: str = "logs/selective_analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def calculate_risk_coverage(self, labels: np.ndarray, predictions: np.ndarray, uncertainty_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the Risk vs. Coverage curve.
        Risk is the error rate of the model on the classified (non-rejected) samples.
        Coverage is the percentage of samples that are not rejected.
        """
        # Sort by uncertainty (ascending: most certain first)
        indices = np.argsort(uncertainty_scores)
        sorted_labels = labels[indices]
        sorted_preds = predictions[indices]
        
        # Binary error array: 1 if incorrect, 0 if correct
        errors = (sorted_labels != sorted_preds).astype(float)
        
        # Cumulative average error (Risk) at each coverage level
        # cumsum / count
        n = len(errors)
        coverage = np.linspace(1/n, 1.0, n)
        risk = np.cumsum(errors) / np.arange(1, n + 1)
        
        return coverage, risk

    def calculate_aurc(self, coverage: np.ndarray, risk: np.ndarray) -> float:
        """
        Calculates the Area Under the Risk-Coverage Curve (AURC).
        Lower is better (means we identify errors earlier in the uncertainty rank).
        """
        return np.trapz(risk, coverage)

    def plot_rc_curves(self, results: Dict[str, Tuple[np.ndarray, np.ndarray]], title: str = "Risk-Coverage Curves"):
        """
        results: Dict mapping strategy name to (coverage, risk)
        """
        plt.figure(figsize=(10, 6))
        for name, (coverage, risk) in results.items():
            aurc = self.calculate_aurc(coverage, risk)
            plt.plot(coverage, risk, label=f"{name} (AURC: {aurc:.5f})")
        
        plt.xlabel("Coverage (Fraction of Dataset Accepted)")
        plt.ylabel("Risk (Error Rate on Accepted Samples)")
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig(f"{self.output_dir}/rc_curve_{title.lower().replace(' ', '_')}.png")
        plt.close()

    def generate_report(self, labels, predictions, uncertainty_dict: Dict[str, np.ndarray]):
        """
        Compares different uncertainty measures (MSP, Entropy, Energy, Vacuity).
        """
        rc_results = {}
        report_data = []

        for name, scores in uncertainty_dict.items():
            coverage, risk = self.calculate_risk_coverage(labels, predictions, scores)
            aurc = self.calculate_aurc(coverage, risk)
            rc_results[name] = (coverage, risk)
            report_data.append({"Method": name, "AURC": aurc})
            
            # Find Risk at 90% coverage
            idx_90 = np.argmin(np.abs(coverage - 0.9))
            report_data.append({"Method": name, "Risk@90%": risk[idx_90]})

        # Plot comparison
        self.plot_rc_curves(rc_results)
        
        # Save summary
        df = pd.DataFrame(report_data)
        df.to_csv(f"{self.output_dir}/selective_prediction_summary.csv", index=False)
        return df

if __name__ == "__main__":
    # Example usage / testing with dummy data
    analyzer = SelectivePredictionAnalyzer()
    n_samples = 1000
    y_true = np.random.randint(0, 10, n_samples)
    y_pred = np.random.randint(0, 10, n_samples)
    
    # Simulate a "good" uncertainty measure (correlated with errors)
    is_wrong = (y_true != y_pred).astype(float)
    uncertainty = is_wrong + np.random.normal(0, 0.5, n_samples)
    
    analyzer.generate_report(y_true, y_pred, {"Simulated_Uncertainty": uncertainty})
    print("Selective prediction analysis complete. Check logs/selective_analysis/")
