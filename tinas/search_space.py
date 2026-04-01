#!/usr/bin/env python3
"""
TINAS-ShipDet Search Space (Section IV-B).

Fix #2: Only includes blocks verified at import time.
Fix #8: is_feasible rejects P2+wide combos that OOM on 8GB Jetson.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

# =============================================================================
# Operator Libraries — safe defaults, validated at runtime
# =============================================================================
# These are the REQUESTED blocks. arch_builder.resolve_block() will
# fall back to C2f if any are missing from the installed Ultralytics.
BACKBONE_BLOCKS = ["C2f", "C3"]  # Safe defaults guaranteed in all ultralytics>=8.0
NECK_BLOCKS     = ["C2f", "C3"]
DOWNSAMPLE_OPS  = ["Conv", "DWConv"]
DOWNSAMPLE_KS   = [3, 5]

WIDTH_MULTS     = [0.25, 0.33, 0.50, 0.67, 0.75]
STAGE_DEPTHS    = [1, 2, 3]
STRIDE_SCHEDS   = ["p3p4p5", "p2p3p4p5"]  # P2 = SAR small-ship mode
NECK_DEPTHS     = [1, 2]
BOTTOMUP_OPTS   = [True, False]
HEAD_TYPES      = ["decoupled", "coupled"]  # Ultralytics default is decoupled
HEAD_WIDTH_R    = [0.5, 1.0]
HEAD_DEPTHS     = [1, 2]


def try_add_ghost_blocks():
    """Add GhostConv-based blocks if available in this Ultralytics install."""
    global BACKBONE_BLOCKS
    try:
        from ultralytics.nn import modules as m
        # Check for Ghost-related modules
        if hasattr(m, "GhostConv") or hasattr(m, "C3Ghost"):
            if "C3Ghost" not in BACKBONE_BLOCKS and hasattr(m, "C3Ghost"):
                BACKBONE_BLOCKS = BACKBONE_BLOCKS + ["C3Ghost"]
                print(f"[TINAS] Added C3Ghost to search space")
            elif "GhostConv" not in BACKBONE_BLOCKS:
                # GhostConv exists but no C3Ghost — can still use in downsample
                print(f"[TINAS] GhostConv available but no C3Ghost block variant")
    except Exception:
        pass


# Run at import
try_add_ghost_blocks()


def compute_cardinality() -> int:
    n_bb = (len(BACKBONE_BLOCKS) * len(WIDTH_MULTS) *
            len(STAGE_DEPTHS)**4 * len(DOWNSAMPLE_OPS) *
            len(DOWNSAMPLE_KS) * len(STRIDE_SCHEDS))
    n_neck = (len(NECK_BLOCKS) * len(NECK_DEPTHS) * len(BOTTOMUP_OPTS))
    n_head = len(HEAD_TYPES) * len(HEAD_WIDTH_R) * len(HEAD_DEPTHS)
    return n_bb * n_neck * n_head


# =============================================================================
# Genome
# =============================================================================

@dataclass
class Genome:
    bb_block: str
    bb_width: float
    bb_depth_s1: int
    bb_depth_s2: int
    bb_depth_s3: int
    bb_depth_s4: int
    bb_down_op: str
    bb_down_k: int
    bb_stride: str
    neck_block: str
    neck_depth: int
    neck_bottomup: bool
    head_type: str
    head_width_ratio: float
    head_depth: int
    uid: str = ""

    @staticmethod
    def random(rng: random.Random | None = None) -> "Genome":
        r = rng or random.Random()
        c = r.choice
        return Genome(
            bb_block=c(BACKBONE_BLOCKS), bb_width=c(WIDTH_MULTS),
            bb_depth_s1=c(STAGE_DEPTHS), bb_depth_s2=c(STAGE_DEPTHS),
            bb_depth_s3=c(STAGE_DEPTHS), bb_depth_s4=c(STAGE_DEPTHS),
            bb_down_op=c(DOWNSAMPLE_OPS), bb_down_k=c(DOWNSAMPLE_KS),
            bb_stride=c(STRIDE_SCHEDS),
            neck_block=c(NECK_BLOCKS), neck_depth=c(NECK_DEPTHS),
            neck_bottomup=c(BOTTOMUP_OPTS),
            head_type=c(HEAD_TYPES),
            head_width_ratio=c(HEAD_WIDTH_R), head_depth=c(HEAD_DEPTHS),
        )

    def mutate(self, prob: float, rng: random.Random | None = None) -> "Genome":
        r = rng or random.Random()
        def m(val, choices): return r.choice(choices) if r.random() < prob else val
        return Genome(
            bb_block=m(self.bb_block, BACKBONE_BLOCKS),
            bb_width=m(self.bb_width, WIDTH_MULTS),
            bb_depth_s1=m(self.bb_depth_s1, STAGE_DEPTHS),
            bb_depth_s2=m(self.bb_depth_s2, STAGE_DEPTHS),
            bb_depth_s3=m(self.bb_depth_s3, STAGE_DEPTHS),
            bb_depth_s4=m(self.bb_depth_s4, STAGE_DEPTHS),
            bb_down_op=m(self.bb_down_op, DOWNSAMPLE_OPS),
            bb_down_k=m(self.bb_down_k, DOWNSAMPLE_KS),
            bb_stride=m(self.bb_stride, STRIDE_SCHEDS),
            neck_block=m(self.neck_block, NECK_BLOCKS),
            neck_depth=m(self.neck_depth, NECK_DEPTHS),
            neck_bottomup=m(self.neck_bottomup, BOTTOMUP_OPTS),
            head_type=m(self.head_type, HEAD_TYPES),
            head_width_ratio=m(self.head_width_ratio, HEAD_WIDTH_R),
            head_depth=m(self.head_depth, HEAD_DEPTHS),
        )

    @staticmethod
    def crossover(a: "Genome", b: "Genome", rng: random.Random | None = None) -> "Genome":
        r = rng or random.Random()
        c = r.choice
        return Genome(
            bb_block=c([a.bb_block, b.bb_block]),
            bb_width=c([a.bb_width, b.bb_width]),
            bb_depth_s1=c([a.bb_depth_s1, b.bb_depth_s1]),
            bb_depth_s2=c([a.bb_depth_s2, b.bb_depth_s2]),
            bb_depth_s3=c([a.bb_depth_s3, b.bb_depth_s3]),
            bb_depth_s4=c([a.bb_depth_s4, b.bb_depth_s4]),
            bb_down_op=c([a.bb_down_op, b.bb_down_op]),
            bb_down_k=c([a.bb_down_k, b.bb_down_k]),
            bb_stride=c([a.bb_stride, b.bb_stride]),
            neck_block=c([a.neck_block, b.neck_block]),
            neck_depth=c([a.neck_depth, b.neck_depth]),
            neck_bottomup=c([a.neck_bottomup, b.neck_bottomup]),
            head_type=c([a.head_type, b.head_type]),
            head_width_ratio=c([a.head_width_ratio, b.head_width_ratio]),
            head_depth=c([a.head_depth, b.head_depth]),
        )

    def signature_tokens(self) -> List[str]:
        return [
            f"bb:{self.bb_block}", f"w:{self.bb_width}",
            f"d1:{self.bb_depth_s1}", f"d2:{self.bb_depth_s2}",
            f"d3:{self.bb_depth_s3}", f"d4:{self.bb_depth_s4}",
            f"down:{self.bb_down_op}", f"k:{self.bb_down_k}",
            f"stride:{self.bb_stride}",
            f"nk:{self.neck_block}", f"nd:{self.neck_depth}",
            f"bu:{self.neck_bottomup}",
            f"ht:{self.head_type}", f"hw:{self.head_width_ratio}",
            f"hd:{self.head_depth}",
        ]

    def to_dict(self) -> Dict[str, Any]: return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Genome":
        d2 = {k: v for k, v in d.items() if k != "uid"}
        return Genome(**d2, uid=d.get("uid", ""))

    def summary(self) -> str:
        return (f"[{self.bb_block} w={self.bb_width} "
                f"d={self.bb_depth_s1}/{self.bb_depth_s2}/"
                f"{self.bb_depth_s3}/{self.bb_depth_s4} "
                f"down={self.bb_down_op}k{self.bb_down_k} {self.bb_stride} | "
                f"neck={self.neck_block} d{self.neck_depth} "
                f"bu={self.neck_bottomup} | head={self.head_type}]")


# =============================================================================
# Feasibility (Fix #8: P2+wide OOM, memory as constraint not objective)
# =============================================================================

def is_feasible(g: Genome, max_params_M: float = 15.0) -> bool:
    """Reject candidates that will OOM on 8GB Jetson or are too large."""
    # P2 stride + wide backbone = huge feature maps at stride 4
    if g.bb_stride == "p2p3p4p5" and g.bb_width >= 0.67:
        return False
    # P2 + bottom-up PAN + deep = very expensive
    if (g.bb_stride == "p2p3p4p5" and g.neck_bottomup and
            g.bb_width >= 0.50 and
            (g.bb_depth_s1 + g.bb_depth_s2 + g.bb_depth_s3 + g.bb_depth_s4) > 8):
        return False
    # Very rough param estimate to reject obviously huge models
    w = g.bb_width
    total_depth = g.bb_depth_s1 + g.bb_depth_s2 + g.bb_depth_s3 + g.bb_depth_s4
    est_params = w * w * total_depth * 0.8  # very rough M
    if est_params > max_params_M:
        return False
    return True


_CARD = compute_cardinality()
print(f"[TINAS] Search space |A| = {_CARD:,} architectures "
      f"(bb_blocks={BACKBONE_BLOCKS})")
