#!/usr/bin/env python3
"""
Baseline Benchmarking Under Unified Protocol (Section V-B).

Trains and profiles ALL baselines and SAR competitors under identical:
  - Dataset splits and image sizes
  - Augmentation and training budget
  - Evaluation metrics (COCO mAP)
  - Profiling protocol (batch-1, warmup, timed runs, p50/p95/p99)

Baselines (Section V-B.1):
  - SSDLite-MobileNetV3
  - NanoDet-Plus
  - YOLOv8n, YOLOv8s
  - YOLOv11n, YOLOv11s
  - FCOS (ResNet-18-FPN)
  - RT-DETR-l

SAR competitors (Section V-B.2):
  Listed but require manual setup of their repos.

Usage:
  python scripts/benchmark_baselines.py --config configs/search.yaml
  python scripts/benchmark_baselines.py --config configs/search.yaml --models yolov8n yolov11n
"""
import argparse
import json
import logging
import sys
import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinas.profiler import load_profile_images, run_profiling_protocol
from tinas.risk import compute_objectives, latency_diagnostics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("benchmark")


# =============================================================================
# Baseline Model Definitions
# =============================================================================

BASELINES = {
    # --- Ultralytics-native models ---
    "yolov8n": {"type": "ultralytics", "model": "yolov8n.pt"},
    "yolov8s": {"type": "ultralytics", "model": "yolov8s.pt"},
    "yolov11n": {"type": "ultralytics", "model": "yolo11n.pt"},
    "yolov11s": {"type": "ultralytics", "model": "yolo11s.pt"},
    "rtdetr-l": {"type": "ultralytics", "model": "rtdetr-l.pt"},

    # --- Requires separate setup ---
    "nanodet-plus": {
        "type": "custom",
        "repo": "https://github.com/RangiLyu/nanodet",
        "notes": "Clone repo, adapt dataloader to YOLO format, train with their config.",
    },
    "ssdlite-mobilenetv3": {
        "type": "torchvision",
        "model": "ssdlite320_mobilenet_v3_large",
        "notes": "Use torchvision; needs custom COCO-to-SAR adapter.",
    },
    "fcos-r18": {
        "type": "custom",
        "repo": "https://github.com/tianzhi0549/FCOS",
        "notes": "FCOS with ResNet-18 + FPN. Use mmdetection implementation.",
    },
}

SAR_COMPETITORS = {
    "OptiSAR-Net": {
        "repo": "https://github.com/SCNU-RISLAB/OptiSAR-Net",
        "notes": "Clone, adapt dataloader, train with their default config.",
    },
    "LiteSAR-Net": {
        "repo": "https://github.com/ZYMCCX/LiteSAR-Net",
        "notes": "Clone, adapt dataloader.",
    },
    "ESarDet": {
        "repo": "https://github.com/ZYMCCX/ESarDet",
        "notes": "Clone, adapt dataloader.",
    },
    "NRENet": {
        "repo": "https://github.com/Xidian-AIGroup190726/NRENet",
        "notes": "Clone, adapt dataloader.",
    },
    "FBGBNet": {
        "repo": "https://github.com/Xidian-AIGroup190726/FBGBNet",
        "notes": "Clone, adapt dataloader.",
    },
}


    # Fix #4/#10: same SAR augmentation dict used by NAS candidates.
    # This ensures the "unified pipeline" claim holds for baselines too.
SAR_AUG_BASELINES = {
    "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0,
    "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.4,
    "flipud": 0.5, "fliplr": 0.5, "degrees": 0.0,
    "translate": 0.1, "scale": 0.5, "erasing": 0.0,
}


def train_and_evaluate_ultralytics(
    model_key: str,
    model_path: str,
    data_yaml: str,
    imgsz: int,
    epochs: int,
    batch_size: int,
    device: str,
    output_dir: Path,
    seed: int = 42,
    workers: int = 4,
) -> dict:
    """
    Train and evaluate an Ultralytics-native baseline.

    Fix #7: Baselines start from COCO-pretrained weights (e.g., yolov8n.pt).
            This is documented in the paper: "All baselines use their official
            pretrained initialisation." NAS models also use pretrained init
            when config.pretrained_backbone=true.
    Fix #4: Uses identical SAR augmentation dict as NAS candidates.
    """
    from ultralytics import YOLO

    logger.info(f"Training {model_key} (pretrained={model_path})...")
    model = YOLO(model_path)

    t0 = time.time()
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        workers=workers,
        project=str(output_dir),
        name=model_key,
        patience=30,
        seed=seed,
        verbose=True,
        **SAR_AUG_BASELINES,  # Fix #4: unified augmentation
    )
    train_time = time.time() - t0

    # Find best weights
    weights = output_dir / model_key / "weights" / "best.pt"
    if not weights.exists():
        weights = output_dir / model_key / "weights" / "last.pt"

    # Validate
    val_model = YOLO(str(weights))
    val = val_model.val(data=data_yaml, imgsz=imgsz, device=device, verbose=False)

    mAP = float(getattr(getattr(val, "box", None), "map", -1.0))
    mAP50 = float(getattr(getattr(val, "box", None), "map50", -1.0))
    mAP75 = float(getattr(getattr(val, "box", None), "map75", -1.0))

    n_params = sum(p.numel() for p in val_model.model.parameters()) / 1e6

    return {
        "model": model_key,
        "weights": str(weights),
        "mAP": mAP,
        "mAP50": mAP50,
        "mAP75": mAP75,
        "params_M": n_params,
        "train_time_s": train_time,
    }


def profile_baseline(
    model_key: str,
    weights_path: str,
    data_yaml: str,
    imgsz: int,
    config: dict,
    device: str,
) -> dict:
    """Profile a baseline under the unified protocol."""
    from ultralytics import YOLO

    prof_cfg = config["profiling"]
    sources = load_profile_images(
        data_yaml, n_images=prof_cfg["num_profile_images"], seed=42)

    model = YOLO(weights_path)
    samples = run_profiling_protocol(
        model=model,
        sources=sources,
        imgsz=imgsz,
        device=device,
        warmup_runs=prof_cfg["warmup_runs"],
        timed_runs=prof_cfg["timed_runs"],
        conf=prof_cfg["conf_threshold"],
        iou=prof_cfg["iou_threshold"],
        half=prof_cfg.get("use_half", True),
    )

    te2e = [s["t_e2e_ms"] for s in samples]
    diag = latency_diagnostics(te2e, label=model_key)

    return {
        "model": model_key,
        "mean_ms": float(np.mean(te2e)),
        "std_ms": float(np.std(te2e)),
        "p50_ms": float(np.percentile(te2e, 50)),
        "p95_ms": float(np.percentile(te2e, 95)),
        "p99_ms": float(np.percentile(te2e, 99)),
        "cv": diag.get("cv", -1),
        "p99_p50_ratio": diag.get("p99_p50_ratio", -1),
        "peak_gpu_mb": float(max(s["gpu_alloc_mb"] for s in samples)),
        "mean_pre_ms": float(np.mean([s["t_pre_ms"] for s in samples])),
        "mean_inf_ms": float(np.mean([s["t_inf_ms"] for s in samples])),
        "mean_post_ms": float(np.mean([s["t_post_ms"] for s in samples])),
        "diagnostics": diag,
        "n_samples": len(samples),
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline Benchmarking")
    parser.add_argument("--config", type=str, default="configs/search.yaml")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific models to benchmark (default: all Ultralytics)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only profile existing weights")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device or str(config["laptop"]["device"])
    epochs = args.epochs or config["nas"]["full_epochs"]

    # Select models
    model_keys = args.models or [k for k, v in BASELINES.items()
                                  if v["type"] == "ultralytics"]

    output_dir = Path(config["output"]["base_dir"]) / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for ds_name in config["active_datasets"]:
        ds = config["datasets"][ds_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {ds_name} (imgsz={ds['imgsz']})")
        logger.info(f"{'='*60}")

        for model_key in model_keys:
            bdef = BASELINES.get(model_key)
            if not bdef:
                logger.warning(f"Unknown model: {model_key}")
                continue
            if bdef["type"] != "ultralytics":
                logger.info(f"Skipping {model_key} (requires manual setup: {bdef.get('notes', '')})")
                continue

            ds_out = output_dir / ds_name
            ds_out.mkdir(parents=True, exist_ok=True)

            # Train
            if not args.skip_training:
                train_result = train_and_evaluate_ultralytics(
                    model_key, bdef["model"], ds["data_yaml"], ds["imgsz"],
                    epochs, config["nas"]["full_batch_size"], device, ds_out,
                )
            else:
                w = ds_out / model_key / "weights" / "best.pt"
                train_result = {"model": model_key, "weights": str(w)}

            # Profile
            if Path(train_result.get("weights", "")).exists():
                prof_result = profile_baseline(
                    model_key, train_result["weights"], ds["data_yaml"],
                    ds["imgsz"], config, device,
                )
                result = {**train_result, **prof_result, "dataset": ds_name}
                all_results.append(result)

                logger.info(f"  {model_key}: mAP={train_result.get('mAP', '?'):.4f}, "
                             f"p95={prof_result['p95_ms']:.1f}ms, "
                             f"GPU={prof_result['peak_gpu_mb']:.0f}MB, "
                             f"CV={prof_result['cv']:.4f}")

    # Save results
    results_path = output_dir / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {results_path}")

    # Print summary table
    logger.info(f"\n{'='*100}")
    logger.info(f"{'Model':<20} {'Dataset':<8} {'mAP':<8} {'mAP50':<8} "
                f"{'p50ms':<8} {'p95ms':<8} {'p99ms':<8} {'CV':<8} {'GPU_MB':<8}")
    logger.info(f"{'='*100}")
    for r in all_results:
        logger.info(
            f"{r['model']:<20} {r.get('dataset','?'):<8} "
            f"{r.get('mAP', -1):<8.4f} {r.get('mAP50', -1):<8.4f} "
            f"{r.get('p50_ms', -1):<8.1f} {r.get('p95_ms', -1):<8.1f} "
            f"{r.get('p99_ms', -1):<8.1f} {r.get('cv', -1):<8.4f} "
            f"{r.get('peak_gpu_mb', -1):<8.0f}"
        )


if __name__ == "__main__":
    main()
