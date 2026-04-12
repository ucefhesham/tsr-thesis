import pandas as pd
import os
import glob
from pathlib import Path

def calculate_tpe_report(benchmarks_path="benchmarks"):
    """
    Consolidates all run ledgers and calculates Trust-Per-Efficiency (TPE).
    TPE = delta_ECE / delta_GFLOPs (relative to resnet18_baseline)
    """
    ledger_files = glob.glob(f"{benchmarks_path}/*/run_ledger.csv")
    
    if not ledger_files:
        print(f"No run ledgers found in {benchmarks_path}")
        return

    all_runs = []
    for f in ledger_files:
        df = pd.read_csv(f)
        # Get the latest run for each model
        latest_run = df.iloc[-1].to_dict()
        latest_run["model"] = Path(f).parent.name
        all_runs.append(latest_run)

    summary_df = pd.DataFrame(all_runs)
    
    # 1. Identify Baseline (ResNet18)
    baseline_match = summary_df[summary_df["model"].str.contains("resnet18_baseline")]
    if baseline_match.empty:
        print("Warning: resnet18_baseline not found. TPE will be calculated vs the first model.")
        baseline = summary_df.iloc[0]
    else:
        baseline = baseline_match.iloc[0]

    b_ece = baseline["ECE"]
    b_flops = baseline["GFLOPs"]

    # 2. Calculate TPE Gains
    def compute_tpe(row):
        delta_ece = b_ece - row["ECE"] # Positive if target is BETTER (lower ECE)
        delta_flops = row["GFLOPs"] - b_flops # Positive if target is HEAVIER
        
        # Avoid division by zero, use epsilon
        return delta_ece / (delta_flops + 1e-6)

    summary_df["TPE_Gain"] = summary_df.apply(compute_tpe, axis=1)
    
    # 3. Format Report
    report_cols = ["model", "Top1_Acc", "SWE", "ECE", "GFLOPs", "TPE_Gain"]
    final_report = summary_df[report_cols]
    
    # 4. Save to CSV
    output_path = os.path.join(benchmarks_path, "final_thesis_report.csv")
    final_report.to_csv(output_path, index=False)
    
    print("\n" + "="*50)
    print("      SCIENTIFIC BENCHMARK SUMMARY")
    print("="*50)
    print(final_report.to_string(index=False))
    print("="*50)
    print(f"Report saved to: {output_path}")

if __name__ == "__main__":
    calculate_tpe_report()
