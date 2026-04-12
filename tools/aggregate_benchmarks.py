import os
import csv
import json
import pandas as pd
from datetime import datetime

def aggregate_results(benchmarks_path="benchmarks"):
    """
    Scans the benchmarks directory and aggregates the latest run metrics 
    for every model variant into a Master Thesis Table.
    """
    all_data = []
    
    if not os.path.exists(benchmarks_path):
        print(f"Error: {benchmarks_path} directory not found.")
        return

    # Iterate through model directories
    for model_dir in os.listdir(benchmarks_path):
        dir_path = os.path.join(benchmarks_path, model_dir)
        ledger_path = os.path.join(dir_path, "run_ledger.csv")
        
        if os.path.isdir(dir_path) and os.path.isfile(ledger_path):
            try:
                df = pd.read_csv(ledger_path)
                if df.empty: continue
                
                # Take the latest run entry
                latest_run = df.iloc[-1].to_dict()
                latest_run["Model"] = model_dir
                all_data.append(latest_run)
            except Exception as e:
                print(f"Warning: Could not parse {ledger_path}: {e}")

    if not all_data:
        print("No run data found to aggregate.")
        return

    # Create Master Dataframe
    master_df = pd.DataFrame(all_data)
    
    # Sort by Accuracy for better presentation
    master_df = master_df.sort_values(by="Top1_Acc", ascending=False)
    
    # 1. Output Master JSON
    json_path = "master_benchmarks.json"
    master_df.to_json(json_path, orient="records", indent=4)
    print(f"\n[1] Master benchmarks saved to: {json_path}")

    # 2. Output LaTeX Table
    print("\n[2] Master Thesis Table (LaTeX Formatting):")
    print("-" * 50)
    
    # Simple LaTeX table generator
    latex_lines = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\begin{tabular}{l|cccc|c}",
        "\\hline",
        "Model Architecture & Acc (\%) & SWE $\downarrow$ & ECE $\downarrow$ & GFLOPs & Status \\\\",
        "\\hline"
    ]
    
    for _, row in master_df.iterrows():
        model_name = row["Model"].replace("_", "\\_")
        acc = row["Top1_Acc"] * 100
        swe = row["SWE"]
        ece = row["ECE"]
        flops = row["GFLOPs"]
        
        line = f"{model_name} & {acc:.2f} & {swe:.4f} & {ece:.4f} & {flops:.2f} & \\checkmark \\\\"
        latex_lines.append(line)
        
    latex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\caption{Comprehensive Performance and Trust Benchmarks Across Architectural Variants.}",
        "\\label{tab:master_benchmarks}",
        "\\end{table}"
    ])
    
    print("\n".join(latex_lines))
    print("-" * 50)

if __name__ == "__main__":
    aggregate_results()
