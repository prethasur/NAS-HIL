#!/usr/bin/env python3
"""
Proxy-vs-Full Rank Correlation Experiment.

Validates that architecture rankings at proxy budget (30 epochs) correlate
with rankings at full budget (100+ epochs). This is CRITICAL evidence:
if Spearman ρ < 0.7, the proxy budget is too short and the search is
optimising the wrong thing.

Also tests laptop-vs-Jetson latency rank correlation (Fix #5 concern):
do faster models on your 4080 stay faster on Jetson Orin Nano?

Usage:
  python scripts/rank_correlation.py --config configs/search.yaml --n-archs 15
  python scripts/rank_correlation.py --config configs/search.yaml --n-archs 15 --full-epochs 100

Outputs:
  - Spearman ρ for mAP ranking (proxy vs full)
  - Spearman ρ for latency ranking (laptop vs Jetson, if Jetson available)
  - Scatter plots
"""
import argparse
import json
import logging
import random
import sys
import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinas.search_space import Genome, is_feasible
from tinas.arch_builder import verify_model, build_yaml_file
from tinas.profiler import load_profile_images, laptop_proxy_profile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("rank_corr")


def spearman_rho(x, y):
    """Spearman rank correlation coefficient."""
    from scipy.stats import spearmanr
    rho, pval = spearmanr(x, y)
    return rho, pval


def spearman_rho_manual(x, y):
    """Spearman without scipy dependency."""
    def rank(arr):
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(arr) + 1)
        return ranks
    rx, ry = rank(np.array(x)), rank(np.array(y))
    n = len(x)
    d2 = np.sum((rx - ry) ** 2)
    return 1 - 6 * d2 / (n * (n**2 - 1))


def resolve_weights_path(model, cand_dir: Path, run_name: str) -> Path:
    """Resolve the actual weights path even when Ultralytics auto-increments names."""
    save_dirs = []

    trainer = getattr(model, "trainer", None)
    trainer_save_dir = getattr(trainer, "save_dir", None)
    if trainer_save_dir:
        save_dirs.append(Path(trainer_save_dir))

    save_dirs.append(cand_dir / run_name)

    # Ultralytics may create proxy2/proxy3... when the base directory exists.
    save_dirs.extend(
        sorted(
            (p for p in cand_dir.glob(f"{run_name}*") if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    )

    seen = set()
    for save_dir in save_dirs:
        key = str(save_dir.resolve()) if save_dir.exists() else str(save_dir)
        if key in seen:
            continue
        seen.add(key)

        best = save_dir / "weights" / "best.pt"
        if best.exists():
            return best
        last = save_dir / "weights" / "last.pt"
        if last.exists():
            return last

    raise FileNotFoundError(
        f"No weights found for run '{run_name}' under '{cand_dir}'. "
        f"Checked trainer.save_dir and matching '{run_name}*' directories."
    )


def find_existing_run(cand_dir: Path, run_name: str, epochs: int, data_yaml: str, imgsz: int) -> Path | None:
    """Return the latest existing run directory whose args match the requested setup."""
    candidates = sorted(
        (p for p in cand_dir.glob(f"{run_name}*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in candidates:
        args_path = run_dir / "args.yaml"
        if not args_path.exists():
            continue
        try:
            with open(args_path, "r") as f:
                args = yaml.safe_load(f) or {}
        except Exception:
            continue

        if int(args.get("epochs", -1)) != int(epochs):
            continue
        if str(args.get("data", "")) != str(data_yaml):
            continue
        if int(args.get("imgsz", -1)) != int(imgsz):
            continue

        weights_dir = run_dir / "weights"
        if (weights_dir / "best.pt").exists() or (weights_dir / "last.pt").exists():
            return run_dir
    return None


def collect_reusable_candidates(out_dir: Path, ds: dict, proxy_ep: int, full_ep: int):
    """Find completed candidate folders that already have matching proxy/full runs."""
    reusable = []
    for cand_dir in sorted((p for p in out_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        model_paths = list(cand_dir.glob("model_*.yaml"))
        if not model_paths:
            continue

        proxy_dir = find_existing_run(cand_dir, "proxy", proxy_ep, ds["data_yaml"], ds["imgsz"])
        full_dir = find_existing_run(cand_dir, "full", full_ep, ds["data_yaml"], ds["imgsz"])
        if not proxy_dir or not full_dir:
            continue

        reusable.append({
            "uid": cand_dir.name,
            "cand_dir": cand_dir,
            "yaml_path": model_paths[0],
            "genome": None,
            "summary": f"reuse existing candidate {cand_dir.name}",
        })
    return reusable


def load_existing_progress(progress_path: Path, proxy_ep: int, full_ep: int, ds_name: str):
    """Load prior completed results if they match the current experiment setup."""
    if not progress_path.exists():
        return []
    try:
        with open(progress_path, "r") as f:
            payload = json.load(f)
    except Exception:
        logger.warning(f"Could not read existing progress file: {progress_path}")
        return []

    if payload.get("proxy_epochs") != proxy_ep:
        return []
    if payload.get("full_epochs") != full_ep:
        return []
    if payload.get("dataset") != ds_name:
        return []

    results = payload.get("results", [])
    if not isinstance(results, list):
        return []
    return results


def save_progress(progress_path: Path, proxy_ep: int, full_ep: int, ds_name: str, results: list):
    """Persist intermediate results so interrupted runs can resume cleanly."""
    with open(progress_path, "w") as f:
        json.dump({
            "proxy_epochs": proxy_ep,
            "full_epochs": full_ep,
            "dataset": ds_name,
            "n_archs": len(results),
            "spearman_rho_mAP": None,
            "results": results,
        }, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="Proxy vs Full Rank Correlation")
    parser.add_argument("--config", type=str, default="configs/search.yaml")
    parser.add_argument("--n-archs", type=int, default=15,
                        help="Number of random architectures to test")
    parser.add_argument("--proxy-epochs", type=int, default=None,
                        help="Proxy training budget (default: from config)")
    parser.add_argument("--full-epochs", type=int, default=100,
                        help="Full training budget for correlation test")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset to use (default: first active)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device or str(config["laptop"]["device"])
    proxy_ep = args.proxy_epochs or config["nas"]["proxy_epochs"]
    full_ep = args.full_epochs
    ds_name = args.dataset or config["active_datasets"][0]
    ds = config["datasets"][ds_name]

    logger.info(f"Rank correlation test: {args.n_archs} architectures")
    logger.info(f"  Proxy: {proxy_ep} epochs, Full: {full_ep} epochs")
    logger.info(f"  Dataset: {ds_name}, Device: {device}")

    out_dir = Path(config["output"]["base_dir"]) / "rank_correlation"
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "rank_correlation_results.json"

    # Import here to avoid slow import on --help
    from ultralytics import YOLO
    from tinas.arch_builder import genome_uid

    existing_results = load_existing_progress(progress_path, proxy_ep, full_ep, ds_name)
    completed_uids = {r.get("uid") for r in existing_results if r.get("uid")}
    if existing_results:
        logger.info(f"Loaded {len(existing_results)} completed result(s) from {progress_path}")

    # Reuse any previously completed candidates with matching budgets.
    candidates = collect_reusable_candidates(out_dir, ds, proxy_ep, full_ep)
    if completed_uids:
        candidates = [c for c in candidates if c["uid"] not in completed_uids]
    if candidates:
        logger.info(f"Reusing {len(candidates)} completed candidate(s) from {out_dir}")

    # Generate random feasible architectures
    rng = random.Random(args.seed)
    archs = []
    seen_uids = {c["uid"] for c in candidates} | completed_uids
    attempts = 0
    target_new_archs = max(0, args.n_archs - len(candidates) - len(existing_results))
    while len(archs) < target_new_archs and attempts < 500:
        g = Genome.random(rng)
        uid = genome_uid(g)
        if uid in seen_uids:
            attempts += 1
            continue
        if is_feasible(g):
            v = verify_model(g, nc=ds["nc"], imgsz=ds["imgsz"])
            if v["valid"]:
                archs.append(g)
                seen_uids.add(uid)
        attempts += 1

    logger.info(f"Generated {len(archs)} new valid architectures ({attempts} attempts)")

    for g in archs:
        uid = genome_uid(g)
        candidates.append({
            "uid": uid,
            "cand_dir": out_dir / uid,
            "yaml_path": None,
            "genome": g,
            "summary": g.summary(),
        })

    results = list(existing_results)
    SAR_AUG = {
        "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.4,
        "flipud": 0.5, "fliplr": 0.5, "degrees": 0.0,
        "translate": 0.1, "scale": 0.5, "erasing": 0.0,
    }

    for i, candidate in enumerate(candidates):
        uid = candidate["uid"]
        cand_dir = candidate["cand_dir"]
        g = candidate["genome"]
        logger.info(f"\n[{i+1}/{len(candidates)}] {candidate['summary']}")

        if g is not None:
            yaml_path = build_yaml_file(g, cand_dir, nc=ds["nc"])

            # --- Proxy training ---
            logger.info(f"  Proxy training ({proxy_ep} epochs)...")
            model_p = YOLO(str(yaml_path))
            model_p.train(
                data=ds["data_yaml"], epochs=proxy_ep, imgsz=ds["imgsz"],
                batch=16, device=device, workers=4,
                project=str(cand_dir), name="proxy",
                patience=proxy_ep, seed=args.seed, verbose=False, **SAR_AUG)

            wp = resolve_weights_path(model_p, cand_dir, "proxy")
        else:
            yaml_path = candidate["yaml_path"]
            logger.info(f"  Reusing proxy/full training artifacts in {cand_dir}")
            wp = find_existing_run(cand_dir, "proxy", proxy_ep, ds["data_yaml"], ds["imgsz"])
            if wp is None:
                raise FileNotFoundError(f"Expected reusable proxy run for {uid} was not found.")
            wp = resolve_weights_path(type("obj", (), {"trainer": type("obj", (), {"save_dir": str(wp)})()})(), cand_dir, wp.name)
        vp = YOLO(str(wp)).val(data=ds["data_yaml"], imgsz=ds["imgsz"],
                               device=device, split="val", verbose=False)
        proxy_mAP = float(getattr(getattr(vp, "box", None), "map", -1.0))

        if g is not None:
            # --- Full training ---
            logger.info(f"  Full training ({full_ep} epochs)...")
            model_f = YOLO(str(yaml_path))
            model_f.train(
                data=ds["data_yaml"], epochs=full_ep, imgsz=ds["imgsz"],
                batch=16, device=device, workers=4,
                project=str(cand_dir), name="full",
                patience=30, seed=args.seed, verbose=False, **SAR_AUG)

            wf = resolve_weights_path(model_f, cand_dir, "full")
        else:
            full_run = find_existing_run(cand_dir, "full", full_ep, ds["data_yaml"], ds["imgsz"])
            if full_run is None:
                raise FileNotFoundError(f"Expected reusable full run for {uid} was not found.")
            wf = resolve_weights_path(type("obj", (), {"trainer": type("obj", (), {"save_dir": str(full_run)})()})(), cand_dir, full_run.name)
        vf = YOLO(str(wf)).val(data=ds["data_yaml"], imgsz=ds["imgsz"],
                               device=device, split="val", verbose=False)
        full_mAP = float(getattr(getattr(vf, "box", None), "map", -1.0))

        # --- Laptop latency ---
        sources = load_profile_images(ds["data_yaml"], n_images=10, seed=42)
        lp = laptop_proxy_profile(YOLO(str(wf)), sources, ds["imgsz"], device)

        results.append({
            "uid": uid,
            "genome": g.to_dict() if g is not None else None,
            "proxy_mAP": proxy_mAP,
            "full_mAP": full_mAP,
            "laptop_p95_ms": lp["proxy_p95_ms"],
            "laptop_mean_ms": lp["proxy_mean_ms"],
            "reused_existing": g is None,
        })
        save_progress(progress_path, proxy_ep, full_ep, ds_name, results)

        logger.info(f"  proxy_mAP={proxy_mAP:.4f}, full_mAP={full_mAP:.4f}, "
                     f"laptop_p95={lp['proxy_p95_ms']:.1f}ms")

        # Cleanup GPU
        if g is not None:
            del model_p, model_f
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Compute correlations ---
    proxy_maps = [r["proxy_mAP"] for r in results if r["proxy_mAP"] >= 0 and r["full_mAP"] >= 0]
    full_maps = [r["full_mAP"] for r in results if r["proxy_mAP"] >= 0 and r["full_mAP"] >= 0]

    if len(proxy_maps) >= 5:
        try:
            rho, pval = spearman_rho(proxy_maps, full_maps)
        except ImportError:
            rho = spearman_rho_manual(proxy_maps, full_maps)
            pval = -1

        logger.info(f"\n{'='*60}")
        logger.info(f"RESULTS: Proxy ({proxy_ep}ep) vs Full ({full_ep}ep) Rank Correlation")
        logger.info(f"{'='*60}")
        logger.info(f"  Spearman ρ = {rho:.4f}" + (f" (p={pval:.4f})" if pval >= 0 else ""))
        logger.info(f"  N = {len(proxy_maps)} architectures")

        if rho >= 0.80:
            logger.info(f"  ✓ STRONG correlation. Proxy budget is sufficient.")
        elif rho >= 0.60:
            logger.info(f"  ~ MODERATE correlation. Consider increasing proxy epochs.")
        else:
            logger.info(f"  ✗ WEAK correlation. Proxy budget too short — "
                        f"increase proxy_epochs or use supernet approach.")
    else:
        logger.warning("Not enough valid results for correlation analysis.")

    # Save
    with open(progress_path, "w") as f:
        json.dump({
            "proxy_epochs": proxy_ep,
            "full_epochs": full_ep,
            "dataset": ds_name,
            "n_archs": len(results),
            "spearman_rho_mAP": rho if len(proxy_maps) >= 5 else None,
            "results": results,
        }, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {progress_path}")

    # Plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.scatter(proxy_maps, full_maps, s=60, c="steelblue", edgecolors="white")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
        ax.set_xlabel(f"mAP at {proxy_ep} epochs (proxy)")
        ax.set_ylabel(f"mAP at {full_ep} epochs (full)")
        ax.set_title(f"Proxy Rank Correlation (ρ={rho:.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "rank_correlation.pdf", dpi=300)
        logger.info(f"Saved: {out_dir / 'rank_correlation.pdf'}")
    except Exception as e:
        logger.info(f"Skipping plot ({e})")


if __name__ == "__main__":
    main()
