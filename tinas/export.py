#!/usr/bin/env python3
"""
Model Export: PyTorch → ONNX → TensorRT FP16/INT8.

For deployment-faithful profiling, models MUST be exported to the same
runtime format used in deployment. On Jetson Orin Nano, TensorRT FP16
is the standard inference path.

Reference: Paper Section V-E (deployment device software stack).
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("tinas.export")


def export_onnx(
    weights_path: str,
    imgsz: int = 640,
    opset: int = 17,
    simplify: bool = True,
    half: bool = False,
    output_path: Optional[str] = None,
) -> str:
    """
    Export Ultralytics model to ONNX.

    Args:
        weights_path: path to .pt weights file.
        imgsz: input image size.
        opset: ONNX opset version.
        simplify: run onnx-simplifier.
        half: export with FP16 weights (note: TRT handles precision separately).
        output_path: custom output path (default: same dir as weights).

    Returns: path to exported .onnx file.
    """
    from ultralytics import YOLO

    model = YOLO(weights_path)
    result = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        half=half,
    )

    onnx_path = str(result) if result else weights_path.replace(".pt", ".onnx")

    if output_path:
        import shutil
        shutil.move(onnx_path, output_path)
        onnx_path = output_path

    logger.info(f"Exported ONNX: {onnx_path}")
    return onnx_path


def export_tensorrt(
    weights_path: str,
    imgsz: int = 640,
    half: bool = True,
    int8: bool = False,
    workspace_gb: int = 2,
    output_path: Optional[str] = None,
) -> str:
    """
    Export to TensorRT engine (requires TensorRT installed).

    For Jetson Orin Nano:
      - FP16 is the standard deployment mode (half=True).
      - INT8 requires a calibration dataset and is optional.

    Can also be done via trtexec on the Jetson directly:
      trtexec --onnx=model.onnx --saveEngine=model.engine --fp16

    Returns: path to .engine file.
    """
    from ultralytics import YOLO

    model = YOLO(weights_path)
    result = model.export(
        format="engine",
        imgsz=imgsz,
        half=half,
        int8=int8,
        workspace=workspace_gb,
    )

    engine_path = str(result) if result else weights_path.replace(".pt", ".engine")

    if output_path:
        import shutil
        shutil.move(engine_path, output_path)
        engine_path = output_path

    logger.info(f"Exported TensorRT{'(FP16)' if half else ''}{'(INT8)' if int8 else ''}: "
                f"{engine_path}")
    return engine_path


def export_for_jetson(
    weights_path: str,
    imgsz: int,
    runtime: str = "tensorrt_fp16",
    workspace_gb: int = 2,
) -> str:
    """
    High-level export for Jetson deployment.

    Args:
        runtime: one of "pytorch", "onnx", "tensorrt_fp16", "tensorrt_int8"

    Note: TensorRT engines are platform-specific. If exporting on laptop,
    you'll need to re-export on Jetson. For SSH workflow, the Jetson
    profiler script handles export on-device.
    """
    if runtime == "pytorch":
        return weights_path
    elif runtime == "onnx":
        return export_onnx(weights_path, imgsz=imgsz)
    elif runtime == "tensorrt_fp16":
        return export_tensorrt(weights_path, imgsz=imgsz, half=True,
                               workspace_gb=workspace_gb)
    elif runtime == "tensorrt_int8":
        return export_tensorrt(weights_path, imgsz=imgsz, half=False, int8=True,
                               workspace_gb=workspace_gb)
    else:
        raise ValueError(f"Unknown runtime: {runtime}")


def trtexec_build(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    int8: bool = False,
    workspace_mb: int = 2048,
    calib_cache: Optional[str] = None,
) -> bool:
    """
    Build TensorRT engine using trtexec (available on Jetson).
    This is the recommended way to build engines ON the target device.
    """
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace_mb}",
    ]
    if fp16:
        cmd.append("--fp16")
    if int8:
        cmd.append("--int8")
        if calib_cache:
            cmd.append(f"--calib={calib_cache}")

    logger.info(f"Building TRT engine: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"trtexec failed:\n{result.stderr}")
        return False

    logger.info(f"TRT engine built: {engine_path}")
    return True
