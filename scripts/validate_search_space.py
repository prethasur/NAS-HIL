#!/usr/bin/env python3
"""
Pre-Flight Validation: test every major configuration before search.

RUN THIS FIRST. It builds one genome from every critical combination
and verifies the forward pass succeeds. Takes ~2 minutes, saves days.

Usage:
  python scripts/validate_search_space.py
  python scripts/validate_search_space.py --imgsz 640 --verbose
"""
import argparse
import sys
import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinas.search_space import (Genome, BACKBONE_BLOCKS, DOWNSAMPLE_OPS,
                                 DOWNSAMPLE_KS, STRIDE_SCHEDS, NECK_BLOCKS,
                                 NECK_DEPTHS, BOTTOMUP_OPTS, WIDTH_MULTS)
from tinas.arch_builder import verify_model, genome_uid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("TINAS-ShipDet Pre-Flight Validation")
    print("=" * 60)
    print(f"Testing all critical configurations at imgsz={args.imgsz}...\n")

    # Test every combination of:
    #   backbone_block × stride × bottomup × width (a few representative)
    # Other genes use fixed safe defaults — they don't affect layer wiring.

    test_widths = [0.25, 0.50]  # just 2 widths to keep it fast
    test_depths = [1, 2]

    configs = list(product(
        BACKBONE_BLOCKS,
        STRIDE_SCHEDS,
        BOTTOMUP_OPTS,
        test_widths,
        DOWNSAMPLE_OPS,
        NECK_BLOCKS,
    ))

    passed = 0
    failed = 0
    failures = []

    for bb, stride, bu, w, down, neck in configs:
        g = Genome(
            bb_block=bb, bb_width=w,
            bb_depth_s1=1, bb_depth_s2=2, bb_depth_s3=2, bb_depth_s4=1,
            bb_down_op=down, bb_down_k=3,
            bb_stride=stride,
            neck_block=neck, neck_depth=1,
            neck_bottomup=bu,
            head_type="decoupled", head_width_ratio=1.0, head_depth=1,
        )

        result = verify_model(g, nc=1, imgsz=args.imgsz)
        uid = result.get("uid", "?")

        if result["valid"]:
            passed += 1
            if args.verbose:
                print(f"  OK  {g.summary()} → {result['params_M']:.2f}M")
        else:
            failed += 1
            failures.append((g.summary(), result["error"]))
            print(f"  FAIL  {g.summary()}")
            print(f"        Error: {result['error']}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed} configs")
    print(f"{'='*60}")

    if failures:
        print(f"\nFailed configurations:")
        for desc, err in failures:
            print(f"  {desc}")
            print(f"    → {err}")
        print(f"\nFix these before running search!")
        sys.exit(1)
    else:
        print(f"\nAll configurations valid. Safe to start search.")

    # Also test a few random genomes
    print(f"\nBonus: testing 20 random genomes...")
    import random
    rng = random.Random(42)
    rand_pass = 0
    for _ in range(20):
        g = Genome.random(rng)
        r = verify_model(g, nc=1, imgsz=args.imgsz)
        if r["valid"]:
            rand_pass += 1
        else:
            print(f"  Random FAIL: {g.summary()} → {r['error']}")
    print(f"  {rand_pass}/20 random genomes passed")


if __name__ == "__main__":
    main()
