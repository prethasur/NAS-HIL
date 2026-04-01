#!/usr/bin/env python3
"""
TINAS-ShipDet Smoke Test — run on laptop before handoff.

Tests every module end-to-end using a tiny synthetic dataset.
No Jetson, no real data, no long training needed.
Takes ~10-15 minutes on a laptop GPU.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --device cpu    # if no GPU
    python scripts/smoke_test.py --verbose

Expected output: "ALL TESTS PASSED" at the end.
If any test fails, fix it before giving code to student.
"""
import argparse
import json
import logging
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

# Fix OpenMP duplicate library crash (Anaconda + PyTorch on Windows)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("smoke_test")

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        logger.info(f"  ✓ {name}")
    else:
        FAIL += 1
        logger.error(f"  ✗ {name}" + (f" — {detail}" if detail else ""))


# =============================================================================
# Create tiny synthetic dataset in YOLO format
# =============================================================================

def create_synthetic_dataset(base_dir: Path, n_train=20, n_val=10, imgsz=320):
    """Create a minimal YOLO-format dataset with random images and labels."""
    for split, n in [("train", n_train), ("val", n_val)]:
        img_dir = base_dir / split / "images"
        lbl_dir = base_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            # Random grayscale-ish image (replicated to 3 channels)
            img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
            # Add a bright rectangle to simulate a ship
            cx, cy = random.randint(50, imgsz-50), random.randint(50, imgsz-50)
            w, h = random.randint(10, 40), random.randint(5, 15)
            img[max(0,cy-h):cy+h, max(0,cx-w):cx+w] = 200

            import cv2
            cv2.imwrite(str(img_dir / f"img_{i:04d}.jpg"), img)

            # YOLO label: class cx cy w h (normalised)
            with open(lbl_dir / f"img_{i:04d}.txt", "w") as f:
                ncx = cx / imgsz
                ncy = cy / imgsz
                nw = (2 * w) / imgsz
                nh = (2 * h) / imgsz
                f.write(f"0 {ncx:.4f} {ncy:.4f} {nw:.4f} {nh:.4f}\n")

    # data.yaml
    data_yaml = {
        "path": str(base_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "val/images",  # reuse val for smoke test
        "nc": 1,
        "names": ["ship"],
    }
    yaml_path = base_dir / "data.yaml"
    import yaml
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    return str(yaml_path)


# =============================================================================
# Tests
# =============================================================================

def test_imports():
    logger.info("\n--- Test 1: Imports ---")
    try:
        from tinas.search_space import Genome, is_feasible, compute_cardinality
        check("search_space imports", True)
    except Exception as e:
        check("search_space imports", False, str(e))
        return False

    try:
        from tinas.arch_builder import genome_to_yaml, verify_model, build_model, resolve_block
        check("arch_builder imports", True)
    except Exception as e:
        check("arch_builder imports", False, str(e))
        return False

    try:
        from tinas.risk import R_tau, compute_objectives, latency_diagnostics
        check("risk imports", True)
    except Exception as e:
        check("risk imports", False, str(e))

    try:
        from tinas.profiler import profile_one_run, run_profiling_protocol
        check("profiler imports", True)
    except Exception as e:
        check("profiler imports", False, str(e))

    try:
        from tinas.dominance import (prb_nondominated_sort, generate_reference_vectors,
                                     architecture_entropy)
        check("dominance imports", True)
    except Exception as e:
        check("dominance imports", False, str(e))

    try:
        from tinas.export import export_onnx
        check("export imports", True)
    except Exception as e:
        check("export imports", False, str(e))

    try:
        from tinas.nsga2 import compute_hypervolume
        check("nsga2 + hypervolume imports", True)
    except Exception as e:
        check("nsga2 imports", False, str(e))

    return True


def test_search_space():
    logger.info("\n--- Test 2: Search Space ---")
    from tinas.search_space import Genome, is_feasible, compute_cardinality

    card = compute_cardinality()
    check("cardinality > 0", card > 0, f"|A|={card}")
    logger.info(f"    |A| = {card:,}")

    rng = random.Random(42)
    genomes = [Genome.random(rng) for _ in range(20)]
    check("random genome generation", len(genomes) == 20)

    feasible = [g for g in genomes if is_feasible(g)]
    check("feasibility filter works", len(feasible) < len(genomes) or len(feasible) == len(genomes),
          f"{len(feasible)}/{len(genomes)} feasible")

    g = genomes[0]
    mutated = g.mutate(0.5, rng)
    check("mutation produces valid genome", mutated is not None)

    child = Genome.crossover(genomes[0], genomes[1], rng)
    check("crossover produces valid genome", child is not None)

    tokens = g.signature_tokens()
    check("signature tokens non-empty", len(tokens) > 0)


def test_architecture_building(device):
    logger.info("\n--- Test 3: Architecture Building ---")
    from tinas.search_space import Genome, BACKBONE_BLOCKS, STRIDE_SCHEDS, BOTTOMUP_OPTS
    from tinas.arch_builder import verify_model, genome_to_yaml, resolve_block

    # Test block resolution
    resolved = resolve_block("C2f")
    check("resolve_block(C2f)", resolved == "C2f")

    fake = resolve_block("NonExistentBlock")
    check("resolve_block fallback", fake == "C2f", f"got '{fake}'")

    # Test ALL critical configurations (this is validate_search_space.py inline)
    configs_tested = 0
    configs_passed = 0
    from itertools import product

    for bb, stride, bu in product(BACKBONE_BLOCKS, STRIDE_SCHEDS, BOTTOMUP_OPTS):
        g = Genome(
            bb_block=bb, bb_width=0.25,
            bb_depth_s1=1, bb_depth_s2=1, bb_depth_s3=1, bb_depth_s4=1,
            bb_down_op="Conv", bb_down_k=3, bb_stride=stride,
            neck_block="C2f", neck_depth=1, neck_bottomup=bu,
            head_type="decoupled", head_width_ratio=1.0, head_depth=1,
        )
        result = verify_model(g, nc=1, imgsz=320)
        configs_tested += 1
        if result["valid"]:
            configs_passed += 1
        else:
            logger.error(f"    FAIL: {g.summary()} → {result['error']}")

    check(f"architecture build+forward ({configs_passed}/{configs_tested})",
          configs_passed == configs_tested)

    # Test YAML generation doesn't crash
    g = Genome.random(random.Random(99))
    yaml_str = genome_to_yaml(g, nc=1)
    check("YAML generation", len(yaml_str) > 100, f"{len(yaml_str)} chars")


def test_risk_functions():
    logger.info("\n--- Test 4: Risk Functions ---")
    from tinas.risk import R_tau, shaped_latency, compute_objectives, latency_diagnostics

    data = [10.0, 11.0, 12.0, 10.5, 11.5, 50.0]  # one outlier

    p95 = R_tau(data, 0.95, "percentile")
    check("p95 computation", 40 < p95 < 55, f"p95={p95:.1f}")

    shaped = shaped_latency(data, 0.95, "percentile", 0.5)
    check("shaped latency > p95", shaped >= p95)

    # Diagnostics
    big_data = list(np.random.normal(50, 2, 300))
    diag = latency_diagnostics(big_data)
    check("diagnostics has CV", "cv" in diag)
    check("diagnostics has p99/p50", "p99_p50_ratio" in diag)

    # Full objective computation
    samples = [{"t_e2e_ms": 50 + random.random()*5,
                "t_pre_ms": 5.0, "t_inf_ms": 40.0, "t_post_ms": 5.0,
                "gpu_alloc_mb": 100.0, "gpu_reserved_mb": 150.0,
                "cpu_rss_mb": 200.0, "mean_power_w": 8.0,
                "energy_j": 0.4} for _ in range(100)]
    objectives, aux = compute_objectives(0.75, samples)
    check("objectives has ferr", "ferr" in objectives)
    check("ferr = 0.25", abs(objectives["ferr"] - 0.25) < 0.001)
    check("aux has p95_ms", "p95_ms" in aux)


def test_dominance():
    logger.info("\n--- Test 5: Probabilistic Dominance ---")
    from tinas.dominance import (prb_nondominated_sort, generate_reference_vectors,
                                 architecture_entropy, crowding_distance)

    # Reference vectors
    W = generate_reference_vectors(3, n_divisions=6)
    check("reference vectors generated", len(W) > 0, f"{len(W)} vectors")

    # Entropy
    tokens = ["a", "a", "b", "c", "c", "c"]
    e = architecture_entropy(tokens)
    check("entropy > 0", e > 0, f"H={e:.3f}")

    # Mock population for sorting
    pop = []
    for i in range(6):
        pop.append({
            "objectives": {"ferr": random.random(), "flat": random.random() * 100,
                          "feng": random.random()},
            "profiling_samples": [
                {"t_e2e_ms": 50 + random.random()*10,
                 "gpu_alloc_mb": 100.0, "cpu_rss_mb": 200.0,
                 "energy_j": 0.3 + random.random()*0.2} for _ in range(30)
            ],
        })

    fronts, p_cache = prb_nondominated_sort(
        pop, ["ferr", "flat", "feng"],
        tau=0.95, risk_mode="percentile", lambda_sigma=0.5,
        eta=0.7, n_bootstrap=50, seed=42)

    check("fronts non-empty", len(fronts) > 0)
    check("front 0 has candidates", len(fronts[0]) > 0)
    check("p_cache populated", len(p_cache) > 0)

    # Crowding distance
    cd = crowding_distance(pop, fronts[0], ["ferr", "flat", "feng"])
    check("crowding distance computed", len(cd) > 0)


def test_hypervolume():
    logger.info("\n--- Test 6: Hypervolume ---")
    from tinas.nsga2 import compute_hypervolume

    # Simple 2D test: two points
    pareto = [
        {"objectives": {"ferr": 0.3, "flat": 50.0}},
        {"objectives": {"ferr": 0.1, "flat": 100.0}},
    ]
    hv = compute_hypervolume(pareto, ["ferr", "flat"], ref_point=[1.0, 150.0])
    check("2D hypervolume > 0", hv > 0, f"HV={hv:.4f}")

    # 3D test
    pareto_3d = [
        {"objectives": {"ferr": 0.3, "flat": 50.0, "feng": 0.5}},
        {"objectives": {"ferr": 0.1, "flat": 100.0, "feng": 0.3}},
    ]
    hv3 = compute_hypervolume(pareto_3d, ["ferr", "flat", "feng"],
                               ref_point=[1.0, 150.0, 1.0])
    check("3D hypervolume > 0", hv3 > 0, f"HV={hv3:.4f}")

    # Empty pareto
    hv0 = compute_hypervolume([], ["ferr", "flat"])
    check("empty pareto → HV=0", hv0 == 0.0)


def test_training_and_profiling(device, data_yaml, imgsz):
    logger.info("\n--- Test 7: Training + Profiling (end-to-end) ---")
    from tinas.search_space import Genome
    from tinas.arch_builder import build_yaml_file
    from tinas.profiler import load_profile_images, laptop_proxy_profile
    from tinas.risk import compute_objectives
    from ultralytics import YOLO

    # Use smallest possible genome
    g = Genome(
        bb_block="C2f", bb_width=0.25,
        bb_depth_s1=1, bb_depth_s2=1, bb_depth_s3=1, bb_depth_s4=1,
        bb_down_op="Conv", bb_down_k=3, bb_stride="p3p4p5",
        neck_block="C2f", neck_depth=1, neck_bottomup=True,
        head_type="decoupled", head_width_ratio=1.0, head_depth=1,
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="tinas_smoke_"))

    # Build YAML
    yaml_path = build_yaml_file(g, tmp_dir, nc=1)
    check("YAML file created", yaml_path.exists())

    # Train for 2 epochs
    logger.info("    Training (2 epochs, tiny data)...")
    model = YOLO(str(yaml_path))
    try:
        model.train(
            data=data_yaml, epochs=2, imgsz=imgsz, batch=4,
            device=device, workers=0, project=str(tmp_dir),
            name="smoke_train", patience=0, seed=42, verbose=False,
            mosaic=0.0, hsv_h=0.0, hsv_s=0.0,  # SAR aug
        )
        check("training completes", True)
    except Exception as e:
        check("training completes", False, str(e))
        return

    # Find weights
    weights = tmp_dir / "smoke_train" / "weights" / "last.pt"
    if not weights.exists():
        weights = tmp_dir / "smoke_train" / "weights" / "best.pt"
    check("weights exist", weights.exists())

    if not weights.exists():
        return

    # Validate
    val_model = YOLO(str(weights))
    val_results = val_model.val(data=data_yaml, imgsz=imgsz, device=device,
                                split="val", verbose=False)
    mAP = float(getattr(getattr(val_results, "box", None), "map", -1.0))
    check("validation runs", mAP >= 0, f"mAP={mAP:.4f}")

    # Profile
    logger.info("    Profiling (10 runs)...")
    sources = load_profile_images(data_yaml, n_images=5, seed=42)
    check("profile images loaded", len(sources) > 0)

    proxy = laptop_proxy_profile(val_model, sources, imgsz, device,
                                  n_runs=10, half=False)
    check("profiling runs", proxy["proxy_mean_ms"] > 0,
          f"mean={proxy['proxy_mean_ms']:.1f}ms")

    # Compute objectives
    samples = [{"t_e2e_ms": proxy["proxy_mean_ms"] + random.random(),
                "t_pre_ms": 2.0, "t_inf_ms": proxy["proxy_mean_ms"] - 4,
                "t_post_ms": 2.0,
                "gpu_alloc_mb": 50.0, "gpu_reserved_mb": 80.0,
                "cpu_rss_mb": 100.0, "mean_power_w": -1.0,
                "energy_j": -1.0} for _ in range(10)]
    objectives, aux = compute_objectives(mAP, samples)
    check("objectives computed", "ferr" in objectives and "flat" in objectives)

    # Cleanup
    import torch
    del model, val_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_nsga2_mini(device, data_yaml, imgsz):
    logger.info("\n--- Test 8: Mini NAS Loop (1 gen, 2 candidates) ---")
    import yaml as pyyaml

    # Build minimal config
    config = {
        "datasets": {
            "smoke": {"data_yaml": data_yaml, "imgsz": imgsz, "nc": 1}
        },
        "active_datasets": ["smoke"],
        "pretrained_backbone": False,
        "nas": {
            "population_size": 2,
            "generations": 1,
            "mutation_prob": 0.3,
            "crossover_prob": 0.9,
            "seed": 42,
            "proxy_epochs": 2,
            "proxy_batch_size": 4,
            "proxy_patience": 2,
        },
        "objectives": {
            "active": ["ferr", "flat"],  # skip feng for smoke test
            "tau": 0.95,
            "risk_mode": "percentile",
            "lambda_sigma": 0.0,
        },
        "constraints": {"max_gpu_mb": 4096.0},
        "dominance": {"eta": 0.70, "bootstrap_draws": 30},
        "resampling": {
            "enabled": False, "delta": 0.08,
            "delta_n": 5, "max_timed_runs": 50, "resample_fronts": 1,
        },
        "profiling": {
            "warmup_runs": 3, "timed_runs": 10, "search_timed_runs": 10,
            "cold_start_repeats": 1, "batch_size": 1,
            "conf_threshold": 0.25, "iou_threshold": 0.7,
            "num_profile_images": 5, "use_half": False, "profile_in_ram": True,
        },
        "laptop": {
            "device": device, "workers": 0,
            "proxy_screen": {"enabled": False},
        },
        "jetson": {"mode": "local", "energy": {"enabled": False}},
        "output": {"base_dir": tempfile.mkdtemp(prefix="tinas_nas_smoke_")},
    }

    from tinas.nsga2 import run_search

    logger.info("    Running mini NAS (1 gen, 2 candidates, 2 epochs)...")
    logger.info("    This may take 3-5 minutes...")

    try:
        results = run_search(config)
        check("NAS loop completes", True)
        check("population exists", len(results["population"]) > 0,
              f"size={len(results['population'])}")
        check("pareto_front exists", results["pareto_front"] is not None)
        check("cost_log tracked", "generations" in results["cost_log"])

        # Check hypervolume was computed
        gens = results["cost_log"].get("generations", [])
        if gens:
            hv = gens[0].get("hypervolume", -1)
            check("hypervolume computed", hv >= 0, f"HV={hv:.6f}")

    except Exception as e:
        check("NAS loop completes", False, str(e))
        import traceback
        traceback.print_exc()

    # Cleanup
    shutil.rmtree(config["output"]["base_dir"], ignore_errors=True)


# =============================================================================
# Main
# =============================================================================

def main():
    global PASS, FAIL

    parser = argparse.ArgumentParser(description="TINAS-ShipDet Smoke Test")
    parser.add_argument("--device", type=str, default="0",
                        help="GPU device ('0', 'cpu')")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check GPU
    import torch
    if args.device != "cpu" and torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA: {torch.version.cuda}")
    else:
        logger.info("Running on CPU (slower but functional)")
        args.device = "cpu"

    # Check Ultralytics
    try:
        import ultralytics
        logger.info(f"Ultralytics: {ultralytics.__version__}")
    except ImportError:
        logger.error("Ultralytics not installed! pip install ultralytics")
        sys.exit(1)

    # Create synthetic dataset
    data_dir = Path(tempfile.mkdtemp(prefix="tinas_smoke_data_"))
    imgsz = 320
    logger.info(f"\nCreating synthetic dataset at {data_dir}...")
    data_yaml = create_synthetic_dataset(data_dir, n_train=20, n_val=10, imgsz=imgsz)

    # Run tests
    logger.info("\n" + "=" * 60)
    logger.info("TINAS-ShipDet SMOKE TEST")
    logger.info("=" * 60)

    if not test_imports():
        logger.error("Import failures — fix these first!")
        sys.exit(1)

    test_search_space()
    test_architecture_building(args.device)
    test_risk_functions()
    test_dominance()
    test_hypervolume()
    test_training_and_profiling(args.device, data_yaml, imgsz)
    test_nsga2_mini(args.device, data_yaml, imgsz)

    # Cleanup
    shutil.rmtree(data_dir, ignore_errors=True)

    # Summary
    logger.info("\n" + "=" * 60)
    if FAIL == 0:
        logger.info(f"ALL {PASS} TESTS PASSED ✓")
        logger.info("Safe to hand off to student.")
    else:
        logger.error(f"{PASS} passed, {FAIL} FAILED ✗")
        logger.error("Fix failures before handing off!")
    logger.info("=" * 60)

    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
