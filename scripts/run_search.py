#!/usr/bin/env python3
"""
TINAS-ShipDet: Main NAS Search Script (runs on laptop GPU).

Usage:
  python scripts/run_search.py --config configs/search.yaml

Workflow:
  1. Load config
  2. Run PRB-NSGA-II search (train on laptop GPU, profile on Jetson or locally)
  3. Save Pareto front + knee point
  4. Print search cost summary

For Jetson profiling:
  - Set jetson.mode="ssh" in config + provide SSH credentials.
  - Or run on Jetson directly with jetson.mode="local".
  - Or do laptop-only mode for development (profiles on laptop GPU).
"""
import argparse
import json
import logging
import sys
import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinas.nsga2 import run_search


def main():
    parser = argparse.ArgumentParser(description="TINAS-ShipDet NAS Search")
    parser.add_argument("--config", type=str, default="configs/search.yaml",
                        help="Path to search configuration YAML")
    parser.add_argument("--override-device", type=str, default=None,
                        help="Override laptop GPU device (e.g., '0', 'cpu')")
    parser.add_argument("--override-generations", type=int, default=None,
                        help="Override number of generations (for quick test)")
    parser.add_argument("--override-pop-size", type=int, default=None,
                        help="Override population size (for quick test)")
    parser.add_argument("--override-proxy-epochs", type=int, default=None,
                        help="Override proxy training epochs")
    parser.add_argument("--local-profiling", action="store_true",
                        help="Profile on laptop GPU instead of Jetson (for development)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("tinas_search.log"),
        ],
    )
    logger = logging.getLogger("tinas")

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply overrides
    if args.override_device is not None:
        config["laptop"]["device"] = args.override_device
    if args.override_generations is not None:
        config["nas"]["generations"] = args.override_generations
    if args.override_pop_size is not None:
        config["nas"]["population_size"] = args.override_pop_size
    if args.override_proxy_epochs is not None:
        config["nas"]["proxy_epochs"] = args.override_proxy_epochs
    if args.local_profiling:
        config["jetson"]["mode"] = "local"

    # Print config summary
    logger.info("=" * 60)
    logger.info("TINAS-ShipDet NAS Search")
    logger.info("=" * 60)
    logger.info(f"Config: {config_path}")
    logger.info(f"Datasets: {config['active_datasets']}")
    logger.info(f"Population: {config['nas']['population_size']}")
    logger.info(f"Generations: {config['nas']['generations']}")
    logger.info(f"Proxy epochs: {config['nas']['proxy_epochs']}")
    logger.info(f"Objectives: {config['objectives']['active']}")
    logger.info(f"Risk mode: {config['objectives']['risk_mode']} (τ={config['objectives']['tau']})")
    logger.info(f"Laptop device: {config['laptop']['device']}")
    logger.info(f"Jetson mode: {config['jetson']['mode']}")
    logger.info(f"Output: {config['output']['base_dir']}")
    logger.info("=" * 60)

    # Run search
    results = run_search(config)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SEARCH RESULTS SUMMARY")
    logger.info("=" * 60)

    pareto = results["pareto_front"]
    knee = results["knee"]
    cost = results["cost_log"]

    logger.info(f"Pareto front size: {len(pareto)}")
    logger.info(f"Total wall-clock: {cost['total_wall_clock_h']:.2f} hours")
    logger.info(f"  Training: {cost['total_train_s']/3600:.2f}h")
    logger.info(f"  Profiling: {cost['total_profile_s']/3600:.2f}h")

    if knee:
        logger.info(f"\nKnee point: {knee['uid']}")
        logger.info(f"  Genome: {Genome.from_dict(knee['genome']).summary()}")
        for obj, val in knee["objectives"].items():
            logger.info(f"  {obj}: {val:.4f}")

    logger.info("\nPareto front:")
    for i, c in enumerate(pareto):
        logger.info(f"  [{i}] {c['uid']}: mAP={c['aux'].get('mAP', '?'):.4f}, "
                     f"p95={c['aux'].get('p95_ms', '?'):.1f}ms, "
                     f"mem={c['aux'].get('peak_gpu_mb', '?'):.0f}MB")

    # Save final config used
    out_dir = Path(config["output"]["base_dir"])
    with open(out_dir / "config_used.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"\nAll results saved to: {out_dir}")


if __name__ == "__main__":
    from tinas.search_space import Genome
    main()
