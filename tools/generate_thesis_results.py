import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from typing import List

def generate_latex_table(df: pd.DataFrame, model_name: str, output_path: str):
    """
    Converts a results dataframe into a LaTeX table format for thesis chapters.
    """
    # Filter for clean baseline and summary corruptions
    summary_df = df[df["Severity"] == 0].copy() # Clean
    
    # Calculate TPE if GFLOPs is present
    if "GFLOPs" in summary_df.columns:
        # TPE = (ECE_base - ECE_current) / GFLOPs
        # (Simplified implementation for the report table)
        pass

    latex_code = summary_df.to_latex(
        index=False, 
        caption=f"Calibration and Safety Metrics for {model_name} (Clean Baseline)",
        label=f"tab:{model_name.lower()}_baseline",
        column_format="llcccc",
        float_format="%.4f"
    )
    
    with open(output_path, "w") as f:
        f.write(latex_code)
    print(f"LaTeX table saved to {output_path}")

def plot_stress_resilience(df_list: List[pd.DataFrame], model_names: List[str]):
    """
    Plots Accuracy and ECE vs Severity for multiple models.
    """
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    for df, name in zip(df_list, model_names):
        # Average across all corruptions for a general trend
        avg_acc = df.groupby("Severity")["Accuracy"].mean()
        plt.plot(avg_acc.index, avg_acc.values, marker='o', label=name)
    
    plt.xlabel("Corruption Severity")
    plt.ylabel("Mean Accuracy")
    plt.title("Performance Resilience under Stress")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # ECE Plot
    plt.subplot(1, 2, 2)
    for df, name in zip(df_list, model_names):
        avg_ece = df.groupby("Severity")["ECE"].mean()
        plt.plot(avg_ece.index, avg_ece.values, marker='s', label=name)
    
    plt.xlabel("Corruption Severity")
    plt.ylabel("Expected Calibration Error (ECE)")
    plt.title("Confidence Reliability under Stress")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("logs/thesis_stress_resilience.png")
    print("Stress resilience plot saved to logs/thesis_stress_resilience.png")

def main():
    parser = argparse.ArgumentParser(description="Thesis Result Generator")
    parser.add_argument("--logs_dir", type=str, default="logs", help="Directory containing results.csv files")
    args = parser.parse_args()

    # Load available result files
    files = [f for f in os.listdir(args.logs_dir) if f.endswith("_stress_test_results.csv")]
    
    if not files:
        print(f"No result files found in {args.logs_dir}. Have you run eval.py yet?")
        return

    all_dfs = []
    model_names = []
    
    for f in files:
        path = os.path.join(args.logs_dir, f)
        df = pd.read_csv(path)
        all_dfs.append(df)
        model_name = f.replace("_stress_test_results.csv", "").capitalize()
        model_names.append(model_name)
        
        # Output individual LaTeX table
        tex_path = os.path.join(args.logs_dir, f.replace(".csv", ".tex"))
        generate_latex_table(df, model_name, tex_path)

    # Generate Comparison Visualization
    plot_stress_resilience(all_dfs, model_names)

if __name__ == "__main__":
    main()
