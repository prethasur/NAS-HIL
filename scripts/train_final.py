#!/usr/bin/env python3
"""
Final Model Retraining (Section V-C + IV-G).

After NAS search completes, retrain selected Pareto/knee models with:
  - Full training budget (200 epochs)
  - Multiple seeds for mean±std reporting
  - Optional knowledge distillation
  - All active datasets

Usage:
  python scripts/train_final.py --config configs/search.yaml --pareto-dir runs/tinas_search/checkpoints
  python scripts/train_final.py --config configs/search.yaml --pareto-dir runs/tinas_search/checkpoints --use-kd
"""
import argparse
import json
import logging
import sys
import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinas.distill import train_final_models

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("train_final")


def main():
    parser = argparse.ArgumentParser(description="Final Model Retraining")
    parser.add_argument("--config", type=str, default="configs/search.yaml")
    parser.add_argument("--pareto-dir", type=str, required=True,
                        help="Directory with Pareto front checkpoint JSON")
    parser.add_argument("--n-models", type=int, default=5,
                        help="Number of Pareto models to retrain")
    parser.add_argument("--use-kd", action="store_true",
                        help="Enable knowledge distillation")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.use_kd:
        config["distillation"]["enabled"] = True
    else:
        config["distillation"]["enabled"] = False

    # Load Pareto front
    pareto_dir = Path(args.pareto_dir)
    pareto_files = sorted(pareto_dir.glob("*_pareto_raw.json"), reverse=True)
    if not pareto_files:
        logger.error(f"No Pareto front files found in {pareto_dir}")
        sys.exit(1)

    with open(pareto_files[0]) as f:
        pareto_records = json.load(f)

    logger.info(f"Loaded {len(pareto_records)} Pareto models from {pareto_files[0]}")

    # Retrain
    results = train_final_models(pareto_records, config, n_models=args.n_models)

    # Save results
    output_dir = Path(config["output"]["base_dir"]) / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "final_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Compute mean±std per model per dataset
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        key = (r["uid"], r["dataset"])
        grouped[key].append(r.get("mAP", r.get("base_mAP", -1)))

    logger.info(f"\n{'='*60}")
    logger.info(f"{'Model':<15} {'Dataset':<10} {'mAP (mean±std)':<20}")
    logger.info(f"{'='*60}")
    for (uid, ds), maps in grouped.items():
        maps = [m for m in maps if m >= 0]
        if maps:
            logger.info(f"{uid:<15} {ds:<10} {np.mean(maps):.4f} ± {np.std(maps):.4f}")

    logger.info(f"\nResults saved to: {output_dir / 'final_results.json'}")


if __name__ == "__main__":
    main()
