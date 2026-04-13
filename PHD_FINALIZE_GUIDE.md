# 🎓 Ph.D. Thesis: Finalization & Result Retrieval Guide

This guide ensures you can safely close your IDE while the cloud handles the heavy lifting, and provides the exact commands to reclaim your research evidence later.

## 🟢 1. 100% Persistence Mode (Lid-Closed Safe)
I have updated `lightning_bridge.py` to use **TMUX**. This means the training runs in a "bubble" on the cloud that does NOT depend on your laptop connection.

- **To activate**: Restart your bridge one last time: `.\venv\Scripts\python.exe scripts\lightning_bridge.py`
- **When to close**: As soon as you see **"YOU CAN NOW SAFELY CLOSE YOUR LAPTOP,"** you are done. The cloud is now autonomous.

---

## 2. Monitoring Progress (The "WandB" Watch)
Even with your laptop closed, you can check progress from your phone or any other browser:
- **Success Signal**: When the last run (likely `convnext_tiny_evidential_research`) moves from "Running" to "Finished" on your **Weights & Biases Dashboard**.
- **Pro-Tip**: If you want to "peak" at the cloud terminal logs again, run:
  `.\venv\Scripts\python.exe -c "from lightning_sdk import Studio; s = Studio(name='scratch-studio-devbox', teamspace='realtime-model-reliability-assessment-project', user='ucefhesham'); print(s.run('tmux capture-pane -pt thesis_session'))"`

---

## 3. Powering Down (Saving Credits)
Once WandB shows the evaluation is complete, shut down the Studio to stop credit consumption.

**Run this command:**
```powershell
.\venv\Scripts\python.exe -c "from lightning_sdk import Studio; s = Studio(name='scratch-studio-devbox', teamspace='realtime-model-reliability-assessment-project', user='ucefhesham'); s.stop(); print('Studio Powered Down. All CU consumption stopped.')"
```

---

## 4. Fetching Your Evidence (Thesis Proof)
Run the "Download" mode to pull all your PNG plots and CSV tables back to your local PC.

**Run this command:**
```powershell
.\venv\Scripts\python.exe scripts\lightning_bridge.py --download
```

**Safe travels! Your T4 GPU is now 100% autonomous.**
# *.\venv\Scripts\python.exe scripts\lightning_bridge.py* This's what I was running