#!/usr/bin/env python3
"""
Architecture Builder: Genome → Ultralytics YAML → YOLO model.

CRITICAL DESIGN: Layer indices are computed dynamically, not hardcoded.
Every configuration (P3P5/P2P5, bottomup on/off) produces correct Detect
layer references because we track indices as we build.

Also handles:
  - Block registry validation (falls back if C2fGhost missing)
  - Forward pass verification for every new configuration
"""
from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch

logger = logging.getLogger("tinas.arch_builder")

# =============================================================================
# Block Registry Validation (Fix #2)
# =============================================================================

_VALIDATED_BLOCKS: Optional[set] = None


def get_valid_blocks() -> set:
    global _VALIDATED_BLOCKS
    if _VALIDATED_BLOCKS is not None:
        return _VALIDATED_BLOCKS
    _VALIDATED_BLOCKS = set()
    try:
        from ultralytics.nn import modules as nn_modules
        for name in dir(nn_modules):
            obj = getattr(nn_modules, name, None)
            if isinstance(obj, type):
                _VALIDATED_BLOCKS.add(name)
    except Exception as e:
        logger.warning(f"Could not introspect Ultralytics modules: {e}")
        _VALIDATED_BLOCKS = {"Conv", "DWConv", "C2f", "C3", "SPPF",
                             "Concat", "Detect"}
    # These are always valid (nn builtins handled specially by Ultralytics)
    _VALIDATED_BLOCKS.add("nn.Upsample")
    logger.info(f"Validated {len(_VALIDATED_BLOCKS)} Ultralytics blocks. "
                f"Key blocks present: C2f={'C2f' in _VALIDATED_BLOCKS}, "
                f"C3={'C3' in _VALIDATED_BLOCKS}, "
                f"GhostConv={'GhostConv' in _VALIDATED_BLOCKS}")
    return _VALIDATED_BLOCKS


def resolve_block(requested: str, fallback: str = "C2f") -> str:
    valid = get_valid_blocks()
    if requested in valid:
        return requested
    logger.warning(f"Block '{requested}' not in registry → using '{fallback}'")
    return fallback


# =============================================================================
# Genome UID
# =============================================================================

def genome_uid(g) -> str:
    s = json.dumps(g.to_dict(), sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:10]


# =============================================================================
# Dynamic YAML Builder (Fix #1 — no hardcoded indices)
# =============================================================================

def _build_backbone(g, nc: int) -> Tuple[List, Dict[str, int]]:
    """Build backbone layers; return (layers, tap_indices)."""
    bb = resolve_block(g.bb_block)
    dn = resolve_block(g.bb_down_op)
    dk = g.bb_down_k
    layers = []
    taps = {}
    idx = -1  # will increment before first use

    # Stem
    layers.append([-1, 1, "Conv", [64, 3, 2]]); idx += 1

    # Stage 1 → P2/4
    layers.append([-1, 1, dn, [128, dk, 2]]); idx += 1
    layers.append([-1, g.bb_depth_s1, bb, [128, True]]); idx += 1
    taps["p2"] = idx

    # Stage 2 → P3/8
    layers.append([-1, 1, dn, [256, dk, 2]]); idx += 1
    layers.append([-1, g.bb_depth_s2, bb, [256, True]]); idx += 1
    taps["p3"] = idx

    # Stage 3 → P4/16
    layers.append([-1, 1, dn, [512, dk, 2]]); idx += 1
    layers.append([-1, g.bb_depth_s3, bb, [512, True]]); idx += 1
    taps["p4"] = idx

    # Stage 4 → P5/32
    layers.append([-1, 1, dn, [1024, dk, 2]]); idx += 1
    layers.append([-1, g.bb_depth_s4, bb, [1024, True]]); idx += 1
    layers.append([-1, 1, "SPPF", [1024, 5]]); idx += 1
    taps["p5"] = idx

    return layers, taps


def _build_head_p3p5(g, taps, start_idx):
    """3-scale head (P3/P4/P5). Returns (layers, det_indices)."""
    nk = resolve_block(g.neck_block)
    dn = resolve_block(g.bb_down_op)
    dk = g.bb_down_k
    nd = g.neck_depth
    L = []
    i = start_idx - 1  # will increment

    # FPN top-down
    L.append([-1, 1, "Conv", [512, 1, 1]]); i += 1
    L.append([-1, 1, "nn.Upsample", [None, 2, "nearest"]]); i += 1
    L.append([[-1, taps["p4"]], 1, "Concat", [1]]); i += 1
    L.append([-1, nd, nk, [512]]); i += 1; fpn_p4 = i

    L.append([-1, 1, "Conv", [256, 1, 1]]); i += 1
    L.append([-1, 1, "nn.Upsample", [None, 2, "nearest"]]); i += 1
    L.append([[-1, taps["p3"]], 1, "Concat", [1]]); i += 1
    L.append([-1, nd, nk, [256]]); i += 1; fpn_p3 = i

    if g.neck_bottomup:
        L.append([-1, 1, dn, [512, dk, 2]]); i += 1
        L.append([[-1, fpn_p4], 1, "Concat", [1]]); i += 1
        L.append([-1, nd, nk, [512]]); i += 1; pan_p4 = i

        L.append([-1, 1, dn, [1024, dk, 2]]); i += 1
        L.append([[-1, taps["p5"]], 1, "Concat", [1]]); i += 1
        L.append([-1, nd, nk, [1024]]); i += 1; pan_p5 = i

        det = [fpn_p3, pan_p4, pan_p5]
    else:
        det = [fpn_p3, fpn_p4, taps["p5"]]

    return L, det


def _build_head_p2p5(g, taps, start_idx):
    """4-scale head (P2/P3/P4/P5) for small-ship SAR mode."""
    nk = resolve_block(g.neck_block)
    dn = resolve_block(g.bb_down_op)
    dk = g.bb_down_k
    nd = g.neck_depth
    L = []
    i = start_idx - 1

    # FPN top-down: P5→P4
    L.append([-1, 1, "Conv", [512, 1, 1]]); i += 1
    L.append([-1, 1, "nn.Upsample", [None, 2, "nearest"]]); i += 1
    L.append([[-1, taps["p4"]], 1, "Concat", [1]]); i += 1
    L.append([-1, nd, nk, [512]]); i += 1; fpn_p4 = i

    # P4→P3
    L.append([-1, 1, "Conv", [256, 1, 1]]); i += 1
    L.append([-1, 1, "nn.Upsample", [None, 2, "nearest"]]); i += 1
    L.append([[-1, taps["p3"]], 1, "Concat", [1]]); i += 1
    L.append([-1, nd, nk, [256]]); i += 1; fpn_p3 = i

    # P3→P2 (SAR small-ship extra stage)
    L.append([-1, 1, "Conv", [128, 1, 1]]); i += 1
    L.append([-1, 1, "nn.Upsample", [None, 2, "nearest"]]); i += 1
    L.append([[-1, taps["p2"]], 1, "Concat", [1]]); i += 1
    L.append([-1, nd, nk, [128]]); i += 1; fpn_p2 = i

    if g.neck_bottomup:
        L.append([-1, 1, dn, [256, dk, 2]]); i += 1
        L.append([[-1, fpn_p3], 1, "Concat", [1]]); i += 1
        L.append([-1, nd, nk, [256]]); i += 1; pan_p3 = i

        L.append([-1, 1, dn, [512, dk, 2]]); i += 1
        L.append([[-1, fpn_p4], 1, "Concat", [1]]); i += 1
        L.append([-1, nd, nk, [512]]); i += 1; pan_p4 = i

        L.append([-1, 1, dn, [1024, dk, 2]]); i += 1
        L.append([[-1, taps["p5"]], 1, "Concat", [1]]); i += 1
        L.append([-1, nd, nk, [1024]]); i += 1; pan_p5 = i

        det = [fpn_p2, pan_p3, pan_p4, pan_p5]
    else:
        det = [fpn_p2, fpn_p3, fpn_p4, taps["p5"]]

    return L, det


def genome_to_yaml(g, nc: int = 1) -> str:
    """Convert Genome to YAML string with dynamically correct indices."""
    bb_layers, taps = _build_backbone(g, nc)
    bb_count = len(bb_layers)

    if g.bb_stride == "p2p3p4p5":
        head_layers, det = _build_head_p2p5(g, taps, start_idx=bb_count)
    else:
        head_layers, det = _build_head_p3p5(g, taps, start_idx=bb_count)

    head_layers.append([det, 1, "Detect", [nc]])

    lines = [
        f"# TINAS-ShipDet | UID: {genome_uid(g)}",
        f"nc: {nc}",
        f"depth_multiple: 1.0",
        f"width_multiple: {g.bb_width}",
        "", "backbone:",
    ]
    for layer in bb_layers:
        lines.append(f"  - {layer}")
    lines += ["", "head:"]
    for layer in head_layers:
        lines.append(f"  - {layer}")

    return "\n".join(lines) + "\n"


def build_yaml_file(g, out_dir: Path, nc: int = 1) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    uid = genome_uid(g)
    g.uid = uid
    path = out_dir / f"model_{uid}.yaml"
    path.write_text(genome_to_yaml(g, nc=nc), encoding="utf-8")
    return path


def build_model(g, nc: int = 1, verbose: bool = False):
    from ultralytics import YOLO
    yaml_str = genome_to_yaml(g, nc=nc)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                     delete=False) as f:
        f.write(yaml_str)
        f.flush()
        model = YOLO(f.name, verbose=verbose)
    return model


def verify_model(g, nc: int = 1, imgsz: int = 640) -> Dict[str, Any]:
    """Build + forward pass test. Run this before committing to training."""
    try:
        model = build_model(g, nc=nc, verbose=False)
    except Exception as e:
        return {"valid": False, "error": f"Build failed: {e}", "uid": genome_uid(g)}

    dummy = torch.randn(1, 3, imgsz, imgsz)
    try:
        inner = model.model
        inner.eval()
        with torch.no_grad():
            output = inner(dummy)
    except Exception as e:
        return {"valid": False, "error": f"Forward failed: {e}", "uid": genome_uid(g)}

    n_params = sum(p.numel() for p in inner.parameters())
    return {
        "valid": True,
        "params_M": n_params / 1e6,
        "n_layers": len(list(inner.modules())),
        "uid": genome_uid(g),
    }


def get_model_info(model) -> Dict[str, Any]:
    info = {}
    try:
        info["params_M"] = sum(p.numel() for p in model.model.parameters()) / 1e6
    except Exception:
        info["params_M"] = -1.0
    try:
        info["n_layers"] = len(list(model.model.modules()))
    except Exception:
        info["n_layers"] = -1
    return info
