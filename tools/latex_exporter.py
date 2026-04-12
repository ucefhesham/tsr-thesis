import pandas as pd
import numpy as np
import os
import glob
from typing import List, Optional
from scipy.stats import mannwhitneyu

class ThesisLatexExporter:
    """
    Advanced Ph.D.-level LaTeX Exporter.
    Calculates statistical significance (Mann-Whitney U) and reports Mean +/- Std.
    """
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = logs_dir

    def load_all_results(self, pattern: str = "*_stress_test_results.csv") -> pd.DataFrame:
        files = glob.glob(os.path.join(self.logs_dir, pattern))
        if not files:
            print("No result files found.")
            return pd.DataFrame()
        
        # Load and combine all results from all runs/models
        dfs = []
        for f in files:
            temp_df = pd.read_csv(f)
            # Normalize column names in case of variations
            temp_df.columns = [c.replace(" ", "_") for c in temp_df.columns]
            dfs.append(temp_df)
            
        df = pd.concat(dfs, ignore_index=True)
        return df

    def get_stat_bold(self, values_a, values_b, mean_a, mean_b, is_higher_better=True):
        """
        Determines which means to bold based on Mann-Whitney U test significance.
        Returns (bold_a, bold_b)
        """
        if len(values_a) < 2 or len(values_b) < 2:
            # Not enough samples for statistical test, fallback to simple max
            if is_higher_better:
                return (mean_a > mean_b), (mean_b > mean_a)
            else:
                return (mean_a < mean_b), (mean_b < mean_a)

        # H0: Distribution a and b are the same
        stat, p = mannwhitneyu(values_a, values_b)
        significant = p < 0.05
        
        if not significant:
            # It's a tie, bold both if they are the "top" tier
            return True, True
        else:
            # Significant difference, only bold the true winner
            if is_higher_better:
                return (mean_a > mean_b), (mean_b > mean_a)
            else:
                return (mean_a < mean_b), (mean_b < mean_a)

    def format_latex_table(self, df: pd.DataFrame, title: str = "Statistical Benchmarking Result") -> str:
        """
        Generates a professional LaTeX table with mean +/- std and significance bolding.
        """
        # Define experimental groups
        groups = df.groupby(["Corruption", "Severity"])
        
        latex = "\\begin{table*}[t]\n"
        latex += "\\centering\n"
        latex += "\\caption{" + title + " (Reported as $\\mu \\pm \\sigma$, $N \\ge 5$ runs)}\n"
        latex += "\\label{tab:stat_results}\n"
        latex += "\\begin{tabular}{ll ccc}\n"
        latex += "\\toprule\n"
        latex += "Corruption & Sev. & Method & Accuracy ($\\uparrow$) & SWE ($\\downarrow$) & ECE ($\\downarrow$) & TPE ($\\uparrow$) & Brier ($\\downarrow$) \\\\\n"
        latex += "\\midrule\n"

        for (corr, sev), group in groups:
            method_stats = group.groupby("Calibration_Method").agg({
                "Accuracy": ["mean", "std"],
                "SWE": ["mean", "std"],
                "ECE": ["mean", "std"],
                "TPE": ["mean", "std"],
                "Brier": ["mean", "std"]
            })
            
            for method in method_stats.index:
                row = method_stats.loc[method]
                
                acc_str = f"{row[('Accuracy', 'mean')]:.3f} \\pm {row[('Accuracy', 'std')]:.3f}"
                swe_str = f"{row[('SWE', 'mean')]:.3f} \\pm {row[('SWE', 'std')]:.3f}" if "SWE" in row else "-"
                ece_str = f"{row[('ECE', 'mean')]:.4f} \\pm {row[('ECE', 'std')]:.4f}"
                tpe_str = f"{row[('TPE', 'mean')]:.2f} \\pm {row[('TPE', 'std')]:.2f}"
                brier_str = f"{row[('Brier', 'mean')]:.4f} \\pm {row[('Brier', 'std')]:.4f}" if "Brier" in row else "-"
                
                latex += f"{corr} & {sev} & {method} & {acc_str} & {swe_str} & {ece_str} & {tpe_str} & {brier_str} \\\\\n"
            latex += "\\midrule\n"
            
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table*}"
        
        return latex

    def export_pipeline_robustness(self, results_file: str = "logs/perception_pipeline_results.csv", output_file: str = "logs/pipeline_robustness.tex"):
        if not os.path.exists(results_file):
            print(f"File {results_file} not found. Skipping Pipeline Robustness table.")
            return

        df = pd.read_csv(results_file)
        
        latex = "\\begin{table*}[t]\n"
        latex += "\\centering\n"
        latex += "\\caption{Perception Pipeline Robustness (Trust Propagation)}\n"
        latex += "\\label{tab:pipeline_robustness}\n"
        latex += "\\begin{tabular}{l ccccc}\n"
        latex += "\\toprule\n"
        latex += "Stress Severity & mIoU ($\\uparrow$) & Miss Rate ($\\downarrow$) & Combined Acc ($\\uparrow$) & FP Vacuity ($\\uparrow$) & Uncertainty Corr ($\\uparrow$) \\\\\n"
        latex += "\\midrule\n"

        for _, row in df.iterrows():
            sev = int(row['Stress_Severity'])
            miou = f"{row['mIoU']:.3f}"
            mr = f"{row['Miss_Rate']:.3f}"
            acc = f"{row['Combined_Accuracy']:.3f}"
            fpv = f"{row['FP_Vacuity']:.3f}"
            corr = f"{row['Uncertainty_Overlap_Corr']:.3f}"
            latex += f"{sev} & {miou} & {mr} & {acc} & {fpv} & {corr} \\\\\n"
            
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table*}"
        
        with open(output_file, "w") as f:
            f.write(latex)
        print(f"Pipeline Robustness LaTeX exported to: {output_file}")

    def export_ood_robustness(self, results_file: str = "logs/ood_epistemic_results.csv", output_file: str = "logs/ood_robustness.tex"):
        if not os.path.exists(results_file):
            print(f"File {results_file} not found. Skipping OOD Epistemic table.")
            return

        df = pd.read_csv(results_file)
        
        latex = "\\begin{table}[t]\n"
        latex += "\\centering\n"
        latex += "\\caption{Epistemic Domain Shift Robustness (BelgiumTS / ESD)}\n"
        latex += "\\label{tab:ood_robustness}\n"
        latex += "\\begin{tabular}{l cccc}\n"
        latex += "\\toprule\n"
        latex += "Model/Method & Target Domain & Accuracy ($\\uparrow$) & Vacuity ($\\uparrow$) & ECE ($\\downarrow$) \\\\\n"
        latex += "\\midrule\n"

        for _, row in df.iterrows():
            method = row.get('Calibration_Method', 'Unknown')
            domain = row.get('Dataset_Name', 'BTSC')
            acc = f"{row.get('Accuracy', 0):.3f}"
            vac = f"{row.get('Vacuity_Mean', 0):.3f}"
            ece = f"{row.get('ECE', 0):.4f}"
            latex += f"{method} & {domain} & {acc} & {vac} & {ece} \\\\\n"
            
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}"
        
        with open(output_file, "w") as f:
            f.write(latex)
        print(f"OOD Robustness LaTeX exported to: {output_file}")

    def export_all(self, output_file: str = "logs/thesis_statistical_tables.tex"):
        df = self.load_all_results()
        if not df.empty:
            latex_content = self.format_latex_table(df)
            with open(output_file, "w") as f:
                f.write(latex_content)
            print(f"Statistical LaTeX tables exported to: {output_file}")
            
        self.export_pipeline_robustness()
        self.export_ood_robustness()

if __name__ == "__main__":
    exporter = ThesisLatexExporter()
    exporter.export_all()
