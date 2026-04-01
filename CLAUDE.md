# TINAS-ShipDet: Experiment Execution Guide

**Project:** Deployment-Faithful Hardware-Aware NAS for SAR Ship Detection
**Target Paper:** IEEE Transactions on Geoscience and Remote Sensing (TGRS)
**Hardware:** Laptop GPU (training) + Jetson Orin Nano (profiling)

---

## 1. HARDWARE SETUP

### 1.1 Laptop Requirements
- NVIDIA GPU with ≥8GB VRAM (RTX 4080 / 3090 / A5000 or similar)
- Ubuntu 22.04 or Windows 11 with WSL2
- Python 3.10+, CUDA 11.8+, PyTorch 2.0+
- ~200GB free disk space (weights, datasets, search outputs)

### 1.2 Jetson Orin Nano Setup

This is critical. The Jetson measurements go directly into the paper.

```bash
# 1. Flash JetPack 6.x (if not already done)
#    Use NVIDIA SDK Manager on a host PC.

# 2. After boot, install Python packages:
pip3 install ultralytics psutil pyyaml numpy opencv-python

# 3. Set power mode to MAXN (15W for Orin Nano 8GB):
sudo nvpmodel -m 0

# 4. Lock GPU and CPU clocks (prevents frequency scaling):
sudo jetson_clocks

# 5. Verify:
nvpmodel -q         # should show "MAXN" or "Mode: 0"
cat /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq  # GPU freq

# 6. Wait 5 minutes for thermal steady state before any profiling.

# 7. Check no other GPU workloads are running:
sudo tegrastats    # should show low GPU usage before experiments start
# Press Ctrl+C to stop tegrastats
```

**IMPORTANT:** Every time you reboot the Jetson, repeat steps 3-6.

### 1.3 Network Setup (for SSH profiling)
- Jetson and laptop must be on the same network.
- Set up SSH key-based authentication:
```bash
# On laptop:
ssh-keygen -t rsa    # if you don't have a key yet
ssh-copy-id pretha@<JETSON_IP>

# Test:
ssh pretha@<JETSON_IP> "echo connected"
```
- Copy the profiler script to Jetson:
```bash
scp scripts/jetson_profiler.py pretha@<JETSON_IP>:~/
```

---

## 2. SOFTWARE SETUP

### 2.1 Laptop
```bash
# Clone/extract the project
cd ~/projects
tar -xzf tinas_shipdet_v2.tar.gz
cd tinas_shipdet

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify Ultralytics version:
python -c "import ultralytics; print(ultralytics.__version__)"
# Need ≥ 8.1.0
```

### 2.2 Dataset Preparation

You need two datasets in YOLO format. Each dataset directory must have this structure:

```
data/
├── hrsid/
│   ├── data.yaml
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
└── ssdd/
    ├── data.yaml
    ├── train/images/  ...
    ├── val/images/    ...
    └── test/images/   ...
```

Each `data.yaml` must contain:
```yaml
path: /absolute/path/to/data/hrsid
train: train/images
val: val/images
test: test/images
nc: 1
names: ['ship']
```

**Label format:** YOLO format. Each `.txt` file has one line per ship:
```
0 0.5123 0.3456 0.0800 0.0400
```
(class_id center_x center_y width height, all normalised to [0,1])

**SAR note:** Images are grayscale but should be saved as 3-channel files
(OpenCV reads grayscale JPGs as 3-channel by default — this is fine).

**HRSID:** Download from the official source, convert annotations to YOLO format.
**SSDD:** Download official release, convert to YOLO format.

Use the official train/val/test splits. If only train/test are provided,
split the training set 90/10 for train/val and keep test untouched.

### 2.3 Update Config Paths

Edit `configs/search.yaml` — update these paths to match your machine:

```yaml
datasets:
  hrsid:
    data_yaml: "/home/yourname/data/hrsid/data.yaml"   # ← CHANGE THIS
  ssdd:
    data_yaml: "/home/yourname/data/ssdd/data.yaml"     # ← CHANGE THIS

laptop:
  device: 0    # GPU index. Use 'nvidia-smi' to check.

jetson:
  host: "192.168.1.XXX"    # ← CHANGE to your Jetson's IP
  user: "pretha"           # ← CHANGE to your Jetson username
```

---

## 3. EXPERIMENT EXECUTION (STRICT ORDER)

### Step 1: Smoke Test (10-15 minutes) DONE

```bash
python scripts/smoke_test.py --device 0
```

**Expected output:** `ALL X TESTS PASSED ✓`

If any test fails, STOP and fix the issue. Common failures:
- `architecture build+forward FAIL` → Ultralytics version issue, check error message
- `training completes FAIL` → CUDA/GPU issue, try `--device cpu`
- Import failures → missing package, `pip install <package>`

**Do NOT proceed until smoke test passes.**

### Step 2: Validate Search Space (2-3 minutes) DONE

```bash
python scripts/validate_search_space.py --imgsz 640
```

This tests every combination of backbone block × stride schedule × bottom-up
to ensure the model YAML is correctly wired. All must pass.

If some configurations fail,
fix the arch_builder.py before proceeding.

### Step 3: Rank Correlation Experiment (~18-24 hours GPU) DONE

```bash
python scripts/rank_correlation.py \
    --config configs/search.yaml \
    --n-archs 15 \
    --proxy-epochs 30 \
    --full-epochs 100 \
    --device 0
```

**What this does:** Trains 15 random architectures twice — once for 30 epochs
(proxy) and once for 100 epochs (full). Computes Spearman rank correlation.

**What to record:**
- Spearman ρ value (printed at the end)
- If ρ ≥ 0.80: proxy budget is sufficient. Proceed.
- If 0.60 ≤ ρ < 0.80: Tell Pretha. May need to increase proxy_epochs to 50.
- If ρ < 0.60: STOP. Tell Pretha. Proxy is unreliable.

**Output files:** `runs/tinas_search/rank_correlation/`

### Step 4: Development Search Run (~3-6 hours GPU)

Quick test that the full NAS pipeline works end-to-end:

```bash
python scripts/run_search.py \
    --config configs/search.yaml \
    --local-profiling \
    --override-generations 3 \
    --override-pop-size 6 \
    --override-proxy-epochs 10
```

**What to check:**
- Does it complete without errors?
- Are there output files in `runs/tinas_search/checkpoints/`?
- Does `gen_*_summary.json` contain hypervolume values?
- Does the Pareto front have at least 2-3 candidates?

If this works, proceed to the real search.

### Step 5: Full NAS Search (~3-5 days GPU)

**With Jetson profiling** (preferred for paper):
First edit `configs/search.yaml`:
```yaml
jetson:
  mode: "ssh"
  host: "192.168.1.XXX"
  energy:
    enabled: true
```
Then:
```bash
python scripts/run_search.py --config configs/search.yaml
```

**Monitoring:** The script logs progress to `tinas_search.log`.
Check periodically:
```bash
# Watch live progress:
tail -f tinas_search.log

# Check latest generation:
ls -la runs/tinas_search/checkpoints/
cat runs/tinas_search/checkpoints/gen_*_summary.json | python -m json.tool | tail -30
```

**If it crashes mid-search:** The population is saved after every generation.
Currently there is no resume — you would restart from scratch.
Notify Pretha if this happens and we can add resume logic post debug.

### Step 6: Final Retraining (~5-7 days GPU) in Pretha/Desktop

After search completes:

```bash
# Without knowledge distillation:
python scripts/train_final.py \
    --config configs/search.yaml \
    --pareto-dir runs/tinas_search/checkpoints \
    --n-models 5

# With knowledge distillation (recommended):
python scripts/train_final.py \
    --config configs/search.yaml \
    --pareto-dir runs/tinas_search/checkpoints \
    --n-models 5 \
    --use-kd
```

This retrains the top-5 Pareto models with:
- 200 epochs (full budget)
- 5 random seeds (for mean ± std reporting)
- Both HRSID and SSDD datasets
- Total: 5 models × 5 seeds × 2 datasets = 50 training runs

### Step 7: Final Jetson Profiling

Profile the final retrained models on Jetson (optional as already done during search):

```bash
# On Jetson, for each final model:
python3 jetson_profiler.py \
    --weights /path/to/best.pt \
    --data /path/to/data.yaml \
    --imgsz 800 \
    --runtime tensorrt_fp16 \
    --cold-start \
    --output results_model_X.json
```

This is the most important measurement — these numbers go in the paper tables.

### Step 8: Generate Paper Figures

```bash
python scripts/analyze_results.py \
    --results-dir runs/tinas_search \
    --output-dir paper_figures
```

**Output files:**
- `paper_figures/pareto_front.pdf`
- `paper_figures/latency_distributions.pdf`
- `paper_figures/hypervolume_convergence.pdf`
- `paper_figures/architecture_analysis.json`
- `paper_figures/main_results_table.tex`

---

## 4. WHAT TO RECORD AND REPORT

### For every experiment, save:
1. The exact command you ran
2. The config file used (copy it to the output dir)
3. Start time and end time
4. Any errors or warnings
5. GPU temperature during long runs (check with `nvidia-smi`)
6. Jetson temperature during profiling (`cat /sys/class/thermal/thermal_zone*/temp`)

### Key numbers the paper needs:
1. **Rank correlation:** Spearman ρ value
2. **Search cost:** total wall-clock hours (printed at search end)
3. **Hypervolume convergence:** does it plateau? (check the plot)
4. **Pareto front:** how many models? What are their mAP and p95 latency?
5. **Knee point:** which architecture? What's its genome?
6. **Latency diagnostics:** CV and p99/p50 ratio for each model
7. **Final mAP:** mean ± std across 5 seeds for top models
8. **Jetson profiling:** p50/p95/p99 latency, peak GPU memory, energy per inference

### File structure to preserve:
```
runs/
├── tinas_search/
│   ├── checkpoints/          # ← generation summaries, Pareto fronts
│   ├── gen_000/ ... gen_014/ # ← per-candidate training artifacts
│   ├── config_used.yaml      # ← exact config
│   └── final/                # ← retrained models + multi-seed results
├── baselines/                # ← if you run baselines
└── rank_correlation/         # ← proxy vs full experiment
```

**Back up the entire `runs/` directory after each major experiment.**

---

## 5. TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| CUDA out of memory during training | Reduce `proxy_batch_size` from 16 to 8 in config |
| CUDA OOM during profiling | This shouldn't happen (batch=1). If it does, the model is too large — the feasibility filter should have caught it. Report to Pretha. |
| Jetson SSH connection refused | Check: Jetson powered on? Same network? SSH key copied? `ping <JETSON_IP>` |
| Jetson profiling very slow | Is TensorRT installed? JetPack includes it. Check `python3 -c "import tensorrt"` on Jetson. |
| `C2fGhost not found` warning | Normal — it falls back to C2f. Not an error. |
| Training mAP = 0.0 for all models | Dataset issue. Check labels exist and are in correct YOLO format. |
| Hypervolume = 0 every generation | All candidates have identical objectives (likely all failing). Check training logs. |
| `forward pass failed` in validate | Architecture wiring issue. Tell Pretha with the exact error. |
| Search takes > 7 days | Consider reducing `generations` from 15 to 10. |

---

## 6. CHANGE

- **Do NOT change:** search space options, augmentation settings, profiling protocol
  parameters, or evaluation metrics without approval
- **OK to change:** file paths, GPU device index, number of workers

---

## 7. TIMELINE ESTIMATE

| Step | Duration | Notes |
|------|----------|-------|
| Setup + smoke test | 2-3 hours | One-time|DONE(Pretha/Desktop)
| Rank correlation | 18-24 hours | DONE(Pretha/Desktop) |
| Dev search run | 3-6 hours | Verify pipeline |
| Full NAS search | 3-5 days | Main experiment |
| Final retraining | 5-7 days | 50 training runs in Pretha's 4080 |
| Jetson profiling | 3-4 hours | Final models only |
| **Total** | **~10-14 days** 
