import os
import sys
import argparse
import tarfile
import tempfile
from pathlib import Path
from lightning_sdk import Studio

# --- CONFIGURATION ---
STUDIO_NAME = "scratch-studio-devbox"
TEAMSPACE = "realtime-model-reliability-assessment-project"
USERNAME = "ucefhesham"
INCLUDE_EXTS = {".py", ".yaml", ".sh", ".ps1", ".md", ".txt", ".csv"}
IGNORE_DIRS = {".git", "venv", "__pycache__", "wandb", "checkpoints", "data", "artifacts", "outputs", "multirun"}

# Environment variables to pass (SENSITIVE)
ENV_VARS = ["WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"]

def download_results():
    print(f"Connecting to Studio: {STUDIO_NAME}...")
    try:
        studio = Studio(name=STUDIO_NAME, teamspace=TEAMSPACE, user=USERNAME)
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    if str(studio.status).lower() != "running":
        print(f"Starting Studio (Current status: {studio.status})...")
        studio.start()
        import time
        while str(studio.status).lower() != "running":
            print(f"Waiting for Studio to start (Current: {studio.status})...")
            time.sleep(10)

    print("Bundling remote results (benchmarks, outputs, logs)...")
    # Tar up the important folders
    studio.run("tar -cf results_bundle.tar benchmarks/ outputs/ logs/ wandb/ 2>/dev/null || true")
    
    print("Downloading bundle (this might take a moment)...")
    studio.download_file("results_bundle.tar", "results_bundle.tar")
    
    print("Extracting results locally...")
    if os.path.exists("results_bundle.tar"):
        with tarfile.open("results_bundle.tar", "r") as tar:
            tar.extractall(".")
        os.remove("results_bundle.tar")
        print("Cleaning up remote bundle...")
        studio.run("rm results_bundle.tar")
        print("Success! Results are now in your local 'benchmarks' and 'outputs' folders.")
    else:
        print("Error: Bundle not found after download attempt.")

def sync_and_exec(command: str):
    print(f"Connecting to Studio: {STUDIO_NAME}...")
    try:
        studio = Studio(name=STUDIO_NAME, teamspace=TEAMSPACE, user=USERNAME)
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # Ensure we use the correct status string and refresh if possible
    print(f"Current Studio status: {studio.status}")
    if str(studio.status).lower() != "running":
        print(f"Starting Studio (Current status: {studio.status})...")
        studio.start()
        import time
        while str(studio.status).lower() != "running":
            print(f"Waiting for Studio to start (Current: {studio.status})...")
            time.sleep(10)
    
    print("Studio is Running. Proceeding to sync...")

    # 0. Check Environment
    if "WANDB_API_KEY" not in os.environ:
        print("WANDB_API_KEY not found. Training will crash without auth.")
        return

    # 1. Sync Phase
    print("Packaging project...")
    local_root = Path(".").absolute()
    fd, tar_path = tempfile.mkstemp(suffix=".tar")
    os.close(fd)
    
    try:
        with tarfile.open(tar_path, "w") as tar:
            for file_path in local_root.rglob("*"):
                if any(part in IGNORE_DIRS for part in file_path.parts): continue
                if file_path.is_file() and file_path.suffix in INCLUDE_EXTS:
                    tar.add(file_path, arcname=file_path.relative_to(local_root).as_posix())
        
        print("Cleaning & Uploading...")
        # Clear old debris selectively to protect benchmarks/ and data/
        # Also delete "Pollution" files (flat files with literal backslashes in names)
        # Using a pattern that avoids deleting hidden files or protected directories
        studio.run("find . -maxdepth 1 -name '*\\\\*' -delete 2>/dev/null || true")
        studio.run("rm -rf src scripts tools tests configs wandb logs artifacts sync_bundle.tar 2>/dev/null || true")
        studio.upload_file(tar_path, "sync_bundle.tar")
        
        print("Verifying upload...")
        import time
        time.sleep(5) # Give the cloud a moment to persist the file
        
        # Check if file exists remotely
        ls_out = studio.run("ls -lh sync_bundle.tar || echo 'MISSING'")
        if "MISSING" in ls_out:
            print("Upload failed to persist. Retrying upload once...")
            studio.upload_file(tar_path, "sync_bundle.tar")
            time.sleep(5)

        print("Extracting...")
        studio.run("tar -xf sync_bundle.tar && chmod +x scripts/*.sh && rm sync_bundle.tar")
    finally:
        if os.path.exists(tar_path): os.remove(tar_path)

    # 2. Setup Phase (Immediate feedback)
    print("Installing remote dependencies (this may take 1-2 mins)...")
    setup_out = studio.run("pip install -r requirements.txt --quiet")
    print("Remote environment ready.")

    # 3. Execution Phase
    if command:
        print(f"Starting remote execution: {command}")
        print("Note: Training takes hours. You can watch real-time logs on your WandB dashboard!")
        
        # Prepare env vars (MASKED for local logs)
        env_dict = {v: os.environ[v] for v in ENV_VARS if v in os.environ}
        env_dict["HYDRA_FULL_ERROR"] = "1"
        env_str = " ".join([f"{k}='{v}'" for k, v in env_dict.items()])
        
        # Use nohup for more robust background execution in Cloud Studios
        # Redirect all output to logs/eval_cloud.log for debugging
        nohup_cmd = f"mkdir -p logs; nohup bash -c '{env_str} {command}' > logs/eval_cloud.log 2>&1 &"
        print("Launching detached nohup session on Cloud...")
        studio.run(nohup_cmd)
        print("Success! Your execution is now running in the background.")
        print("Cloud Logs: You can track raw output in 'logs/eval_cloud.log' after downloading.")
        print("YOU CAN NOW SAFELY CLOSE YOUR LAPTOP.")
        print("Check WandB for real-time progress.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", nargs="?", default="bash scripts/train_all.sh", help="Command to run")
    parser.add_argument("--download", action="store_true", help="Download results from Studio instead of running")
    args = parser.parse_args()
    
    if args.download:
        download_results()
    else:
        sync_and_exec(args.cmd)
