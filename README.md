# TINAS-ShipDet

**Tiny Neural Architectures for Onboard SAR Ship Detection via Deployment-Faithful, Hardware-Aware NAS on Jetson Orin Nano**

## Architecture

```
tinas_shipdet/
├── configs/
│   └── search.yaml          # All hyperparameters (the "reproducibility contract")
├── tinas/
│   ├── search_space.py       # Concrete backbone+neck+head search space
│   ├── arch_builder.py       # Genome → Ultralytics YAML → YOLO model
│   ├── risk.py               # Risk-aware objectives (p95/p99/CVaR + diagnostics)
│   ├── profiler.py           # Unified profiler (laptop proxy + Jetson faithful)
│   ├── dominance.py          # Probabilistic dominance + entropy niching
│   ├── nsga2.py              # PRB-NSGA-II main engine (Algorithm 1)
│   ├── export.py             # ONNX + TensorRT FP16/INT8 export
│   └── distill.py            # Knowledge distillation post-search
├── scripts/
│   ├── run_search.py         # Main NAS entry point (laptop side)
│   ├── jetson_profiler.py    # Standalone Jetson profiler
│   ├── benchmark_baselines.py # Baseline evaluation under unified protocol
│   ├── train_final.py        # Full-budget retraining + multi-seed
│   └── analyze_results.py    # Pareto plots, architecture analysis, LaTeX tables
└── requirements.txt
```

## Split-Compute Workflow

The key design: **train fast on laptop GPU, profile faithfully on Jetson Orin Nano.**

```
┌─────────────────────────────────────────────────────┐
│  LAPTOP GPU (e.g., RTX 3090 / A100)                 │
│                                                      │
│  1. Initialise population (feasibility-filtered)     │
│  2. Train each candidate (30 epochs proxy budget)    │
│  3. Quick proxy latency screening (reject slow ones) │
│  4. Export weights                                    │
│           │                                          │
│           │  SCP weights to Jetson                   │
│           ▼                                          │
│  ┌─────────────────────────────────────┐             │
│  │  JETSON ORIN NANO                   │             │
│  │                                     │             │
│  │  5. Build TensorRT FP16 engine      │             │
│  │  6. Run full profiling protocol:    │             │
│  │     - 50 warmup runs                │             │
│  │     - 300 timed runs (batch=1)      │             │
│  │     - CUDA sync, INA3221 power      │             │
│  │  7. Return JSON with samples        │             │
│  └──────────────┬──────────────────────┘             │
│                 │  SCP results back                   │
│                 ▼                                     │
│  8. Compute risk-aware objectives (p95, CVaR, energy)│
│  9. PRB-NSGA-II: probabilistic dominance ranking     │
│  10. Budget-adaptive resampling if uncertain          │
│  11. Environmental selection + offspring generation   │
│  12. Repeat from step 2                              │
│                                                      │
│  FINAL:                                              │
│  13. Extract Pareto front + knee point               │
│  14. Retrain top-k with full budget (200 epochs)     │
│  15. Optional: Knowledge distillation refinement     │
│  16. Profile final models on Jetson                  │
│  17. Generate paper tables and figures               │
└─────────────────────────────────────────────────────┘
```

## Quick Start — Correct Execution Order

### Step 0: Setup

```bash
pip install -r requirements.txt

# Prepare datasets in Ultralytics format:
# data/hrsid/data.yaml  →  train/val/test paths, nc: 1, names: ['ship']
# data/ssdd/data.yaml    →  same format
# SAR images: grayscale, replicated to 3 channels by Ultralytics automatically.
```

### Step 1: VALIDATE SEARCH SPACE FIRST (non-negotiable, ~2 min)

```bash
python scripts/validate_search_space.py --imgsz 640
# Must pass ALL configurations before proceeding.
# If any fail, fix arch_builder.py before wasting GPU time.
```

### Step 2: Rank Correlation Experiment (~1 day GPU, critical for paper)

```bash
python scripts/rank_correlation.py --config configs/search.yaml --n-archs 15
# Validates that 30-epoch proxy rankings correlate with 100-epoch full rankings.
# Need Spearman ρ > 0.8. If < 0.7, increase proxy_epochs in config.
# This result goes in the paper (Section V methodology validation).
```

### Step 3: Benchmark Baselines (can run in parallel with Step 2)

```bash
python scripts/benchmark_baselines.py --config configs/search.yaml
# Trains YOLOv8n/s, YOLOv11n/s, RT-DETR-l with SAME SAR augmentations.
# Uses COCO-pretrained init (documented in paper).
# Profiles under unified protocol.
```

### Step 4: NAS Search — development mode

```bash
python scripts/run_search.py --config configs/search.yaml \
    --local-profiling --override-generations 3 \
    --override-pop-size 8 --override-proxy-epochs 10
# Quick test run (~few hours). Verify everything works end-to-end.
```

### Step 5: NAS Search — paper experiments

```bash
# On Jetson: sudo nvpmodel -m 0 && sudo jetson_clocks
# Edit configs/search.yaml: jetson.mode: "ssh", jetson.energy.enabled: true
python scripts/run_search.py --config configs/search.yaml
# ~3-5 days on 4080. Evaluates ~300 candidates.
```

### Step 6: Final Retraining + KD

```bash
python scripts/train_final.py \
    --config configs/search.yaml \
    --pareto-dir runs/tinas_search/checkpoints --use-kd
# 200 epochs × 5 seeds × top-5 models × 2 datasets. Budget ~1 week.
```

### Step 7: Generate Paper Tables and Figures

```bash
python scripts/analyze_results.py --results-dir runs/tinas_search --output-dir paper_figures
# Pareto front plots, latency distributions, architecture analysis, LaTeX tables.
```

## Search Space (Section IV-B)

| Dimension | Options | Count |
|-----------|---------|-------|
| Backbone block | C2f, C3, C2fGhost | 3 |
| Width multiplier | 0.25, 0.33, 0.50, 0.67, 0.75 | 5 |
| Depth per stage (×4) | 1, 2, 3 | 3⁴=81 |
| Downsample op | Conv, DWConv | 2 |
| Downsample kernel | 3, 5 | 2 |
| Stride schedule | P3-P5, P2-P5 | 2 |
| Neck block | C2f, C3 | 2 |
| Neck depth | 1, 2 | 2 |
| Fusion op | concat, add | 2 |
| Bottom-up PAN | yes, no | 2 |
| Head type | decoupled, coupled | 2 |
| Head width ratio | 0.5, 1.0 | 2 |
| Head depth | 1, 2 | 2 |
| **Total |A|** | | **~4.98M** |

SAR-tailored choices:
- **P2-P5 stride schedule**: preserves stride-4 features for few-pixel ship targets.
- **C2fGhost**: GhostConv bottleneck halves parameters, critical for 8GB Jetson.
- **Searchable bottom-up PAN**: can be disabled to save latency when FPN alone suffices.

## Profiling Protocol (Section III-H)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch size | 1 | Onboard inference is always batch-1 |
| Warmup runs (Nw) | 50 | Thermal + JIT stabilisation |
| Timed runs (Nt) | 300 | Statistical power for p95/p99 |
| Cold-start repeats (Nc) | 5 | Model load + first inference |
| End-to-end definition | pre + inf + post | Complete pipeline, not forward-only |
| Synchronisation | CUDA sync before/after each block | Prevents async timing bias |
| Power mode | MAXN (15W) | Fixed, disclosed |
| Clock mode | jetson_clocks ON | Frequency locked |
| Runtime | TensorRT FP16 | Deployment-faithful |

## All Issues Fixed

| # | Issue | Fix |
|---|-------|-----|
| 1 | YAML template hardcoded indices | `arch_builder.py` rewritten: indices computed dynamically as layers are built. `validate_search_space.py` tests all combos. |
| 2 | C2fGhost may not exist in Ultralytics | `resolve_block()` validates at runtime, falls back to C2f. `try_add_ghost_blocks()` adds C3Ghost if available. |
| 3 | Single-channel SAR not handled | Documented: SAR images replicated to 3-ch for pretrained backbone compatibility (standard practice). Config has explicit note. |
| 4 | Augmentations not controlled | `SAR_AUGMENTATION` dict in nsga2.py: hsv_h=0, hsv_s=0, no rotation. Same dict used in baselines. |
| 5 | Probabilistic dominance slow | Acceptable at P=20 (~30s/gen). Noted in code. Real risk is it may not change decisions — ablation will show. |
| 6 | Val/test split discipline | Search uses `split="val"`. Final reporting on `split="test"`. Enforced in code. |
| 7 | Pretrained weights unfair | `pretrained_backbone: true` in config. NAS models get COCO init transfer. Baselines use official pretrained. Both documented. |
| 8 | Memory objective uninformative | Memory removed from objectives → hard constraint (`max_gpu_mb: 512`). 3 objectives: ferr, flat, feng. |
| 9 | Multi-dataset not implemented | Alternates datasets each generation: gen 0→HRSID, gen 1→SSDD, gen 2→HRSID... |
| 10 | Unified pipeline not enforced | Same `SAR_AUGMENTATION` dict for NAS + baselines. SAR competitors documented as "their training, our profiling." |
| +  | Proxy ranking never validated | `rank_correlation.py`: trains 15 archs at 30ep and 100ep, computes Spearman ρ. |
| +  | No pre-flight testing | `validate_search_space.py`: builds+forward-tests all major configurations in ~2min. |

## Jetson Orin Nano Setup Checklist (Appendix A)

```bash
# 1. Flash JetPack 6.x (includes CUDA, TensorRT, cuDNN)
# 2. Install Python packages
pip3 install ultralytics psutil pyyaml numpy opencv-python

# 3. Set power mode
sudo nvpmodel -m 0    # MAXN = 15W

# 4. Lock clocks
sudo jetson_clocks

# 5. Verify
nvpmodel -q
tegrastats  # check GPU/CPU frequencies are at max

# 6. Wait for thermal steady state (~5 min after boot)

# 7. Copy jetson_profiler.py to Jetson
scp scripts/jetson_profiler.py pretha@jetson-orin.local:~/

# 8. Run standalone profiling
python3 jetson_profiler.py \
    --weights model.pt \
    --data data.yaml \
    --imgsz 800 \
    --runtime tensorrt_fp16 \
    --cold-start
```
#   N A S - H I L  
 