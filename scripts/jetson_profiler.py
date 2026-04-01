#!/usr/bin/env python3
"""
Jetson Orin Nano Standalone Profiler.

This script runs ON the Jetson device. It is called either:
  (a) Via SSH from the laptop during NAS search, or
  (b) Directly on Jetson for final model validation.

It handles:
  1. Loading model (PyTorch / ONNX / TensorRT)
  2. Optional on-device TensorRT engine build
  3. Full profiling protocol (Section III-H)
  4. Power/energy measurement via INA3221
  5. Cold-start measurement
  6. Output: JSON file with all measurements

Prerequisites on Jetson:
  pip install ultralytics psutil numpy pyyaml
  # TensorRT is pre-installed on JetPack

Usage:
  python jetson_profiler.py --config profile_config.json --output results.json
  # Or standalone:
  python jetson_profiler.py --weights model.pt --data data.yaml --imgsz 800

Hardware checklist (Appendix A):
  1. Set power mode:  sudo nvpmodel -m 0   (MAXN=15W on Orin Nano 8GB)
  2. Lock clocks:     sudo jetson_clocks
  3. Ensure thermal steady state (~5min idle after boot)
  4. No other GPU workloads running
"""
from __future__ import annotations

import argparse
import json
import logging
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import psutil
import torch
import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("jetson_profiler")


# =============================================================================
# Hardware State Verification
# =============================================================================

def verify_jetson_state() -> Dict[str, Any]:
    """
    Verify and log Jetson hardware state for reproducibility.
    This information goes into the paper (Section V-E).
    """
    state = {}

    # Power mode
    try:
        result = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True)
        state["power_mode"] = result.stdout.strip()
    except Exception:
        state["power_mode"] = "unknown"

    # Jetson clocks status
    try:
        # Check if GPU clock is at max (indicator of jetson_clocks)
        gpu_freq_path = "/sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq"
        if Path(gpu_freq_path).exists():
            state["gpu_freq_hz"] = int(Path(gpu_freq_path).read_text().strip())
        else:
            # Fallback for different Jetson models
            state["gpu_freq_hz"] = "check_manually"
    except Exception:
        state["gpu_freq_hz"] = "unknown"

    # Temperature
    try:
        temps = {}
        for zone in Path("/sys/class/thermal/").glob("thermal_zone*"):
            try:
                temp = int((zone / "temp").read_text().strip()) / 1000.0
                ttype = (zone / "type").read_text().strip()
                temps[ttype] = temp
            except Exception:
                pass
        state["temperatures_C"] = temps
    except Exception:
        state["temperatures_C"] = {}

    # JetPack version
    try:
        result = subprocess.run(["dpkg", "-l", "nvidia-jetpack"],
                                capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if "nvidia-jetpack" in line:
                state["jetpack_version"] = line.split()[2]
                break
    except Exception:
        state["jetpack_version"] = "unknown"

    # CUDA version
    state["cuda_version"] = torch.version.cuda or "unknown"
    state["pytorch_version"] = torch.__version__
    state["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    return state


# =============================================================================
# Power Sampling (INA3221)
# =============================================================================

class JetsonPowerSampler:
    """
    Sample power from INA3221 rails on Jetson Orin Nano.

    Common sysfs paths for Orin Nano:
      VDD_IN (total board power):
        /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*/in1_input  (voltage mV)
        /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*/curr1_input (current mA)
      Or power directly:
        /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*/power1_input (uW)

    Auto-detects available paths at init.
    """

    def __init__(self, power_paths: List[str] = None, period_s: float = 0.01):
        self.period_s = period_s
        self._stop = threading.Event()
        self._thread = None
        self.samples_w: List[float] = []
        self.timestamps_s: List[float] = []

        # Auto-detect power paths if not provided
        if power_paths:
            self.power_paths = [Path(p) for p in power_paths]
        else:
            self.power_paths = self._auto_detect_paths()

        logger.info(f"Power paths: {[str(p) for p in self.power_paths]}")

    def _auto_detect_paths(self) -> List[Path]:
        """Auto-detect INA3221 power measurement paths."""
        candidates = [
            "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon",
            "/sys/bus/i2c/drivers/ina3221/7-0040/hwmon",
        ]
        for base in candidates:
            bp = Path(base)
            if bp.exists():
                for hwmon in bp.iterdir():
                    # Look for VDD_IN power or current*voltage
                    power = hwmon / "power1_input"
                    if power.exists():
                        return [power]
                    curr = hwmon / "curr1_input"
                    volt = hwmon / "in1_input"
                    if curr.exists() and volt.exists():
                        return [curr, volt]  # will compute power = V * I
        logger.warning("No INA3221 power paths found. Energy will not be measured.")
        return []

    def _read_power_w(self) -> Optional[float]:
        if not self.power_paths:
            return None
        try:
            if len(self.power_paths) == 1:
                # Direct power reading (microwatts)
                raw = float(self.power_paths[0].read_text().strip())
                if raw > 1e6:
                    return raw / 1e6
                elif raw > 1e3:
                    return raw / 1e3
                return raw
            elif len(self.power_paths) == 2:
                # Current (mA) * Voltage (mV) → Power (W)
                curr_mA = float(self.power_paths[0].read_text().strip())
                volt_mV = float(self.power_paths[1].read_text().strip())
                return (curr_mA * volt_mV) / 1e6  # mA * mV = µW, / 1e6 = W
        except Exception:
            return None

    def start(self):
        self.samples_w.clear()
        self.timestamps_s.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def mean_w(self) -> float:
        return float(np.mean(self.samples_w)) if self.samples_w else -1.0

    def _loop(self):
        while not self._stop.is_set():
            p = self._read_power_w()
            if p is not None and p > 0:
                self.samples_w.append(p)
                self.timestamps_s.append(time.monotonic())
            time.sleep(self.period_s)


# =============================================================================
# Profiling Core
# =============================================================================

def load_images(data_yaml: str, split: str = "val",
                n_images: int = 50, seed: int = 42) -> List[Any]:
    """Load profiling images into RAM."""
    with open(data_yaml) as f:
        data = yaml.safe_load(f)

    base = Path(data_yaml).parent
    sv = data.get(split, data.get("val", ""))
    if isinstance(sv, list):
        sv = sv[0]
    sp = (base / sv).resolve()

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if sp.is_dir():
        imgs = [p for p in sp.rglob("*") if p.suffix.lower() in exts]
    elif sp.suffix == ".txt":
        imgs = [Path(l.strip()) for l in sp.read_text().splitlines() if l.strip()]
    else:
        imgs = [sp]
    imgs = [p for p in imgs if p.exists()]
    random.Random(seed).shuffle(imgs)
    imgs = imgs[:n_images]

    try:
        import cv2
        loaded = []
        for p in imgs:
            im = cv2.imread(str(p))
            if im is not None:
                loaded.append(im)
        return loaded if loaded else [str(p) for p in imgs]
    except ImportError:
        return [str(p) for p in imgs]


def profile_run(model, source, imgsz, device, conf, iou, half, ps=None):
    """Single batch-1 run with CUDA sync."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    rss0 = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    if ps:
        ps.start()

    t0 = time.perf_counter()
    results = model.predict(source=source, imgsz=imgsz, device=device,
                            conf=conf, iou=iou, half=half,
                            verbose=False, save=False, stream=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    if ps:
        ps.stop()

    wall_ms = (t1 - t0) * 1000.0
    pre = inf = post = 0.0
    try:
        sp = getattr(results[0], "speed", {}) or {}
        pre = float(sp.get("preprocess", 0.0))
        inf = float(sp.get("inference", 0.0))
        post = float(sp.get("postprocess", 0.0))
    except Exception:
        pass
    te2e = (pre + inf + post) if (pre + inf + post) > 0 else wall_ms

    ga = gr = 0.0
    if torch.cuda.is_available():
        ga = torch.cuda.max_memory_allocated() / (1024 ** 2)
        gr = torch.cuda.max_memory_reserved() / (1024 ** 2)

    rss1 = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

    return {
        "t_pre_ms": pre, "t_inf_ms": inf, "t_post_ms": post,
        "t_e2e_ms": te2e, "wall_ms": wall_ms,
        "gpu_alloc_mb": ga, "gpu_reserved_mb": gr,
        "cpu_rss_mb": max(rss0, rss1),
        "mean_power_w": ps.mean_w() if ps else -1.0,
        "energy_j": (ps.mean_w() * wall_ms / 1000.0) if (ps and ps.mean_w() > 0) else -1.0,
    }


def run_full_protocol(model, sources, imgsz, device, conf, iou, half,
                      warmup, timed, power_sampler=None):
    """Full profiling protocol: warmup + timed runs."""
    n = len(sources)

    logger.info(f"Warmup: {warmup} runs...")
    for i in range(warmup):
        profile_run(model, sources[i % n], imgsz, device, conf, iou, half)

    logger.info(f"Timed: {timed} runs...")
    samples = []
    for i in range(timed):
        s = profile_run(model, sources[i % n], imgsz, device, conf, iou, half, power_sampler)
        samples.append(s)
        if (i + 1) % 50 == 0:
            te2e = [x["t_e2e_ms"] for x in samples]
            logger.info(f"  Run {i+1}/{timed}: mean={np.mean(te2e):.1f}ms, "
                        f"p95={np.percentile(te2e, 95):.1f}ms")

    return samples


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Jetson Orin Nano Profiler")
    parser.add_argument("--config", type=str, help="JSON config from SSH workflow")
    parser.add_argument("--output", type=str, default="profile_results.json")
    parser.add_argument("--weights", type=str, help="Model weights path")
    parser.add_argument("--data", type=str, help="Data YAML path")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--timed", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--half", action="store_true", default=True)
    parser.add_argument("--runtime", type=str, default="pytorch",
                        choices=["pytorch", "tensorrt_fp16", "tensorrt_int8"])
    parser.add_argument("--no-energy", action="store_true")
    parser.add_argument("--cold-start", action="store_true")
    args = parser.parse_args()

    # Load config if provided (SSH workflow)
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        weights = cfg["weights"]
        data_yaml = cfg.get("data_yaml", args.data)
        imgsz = cfg.get("imgsz", args.imgsz)
        warmup = cfg.get("warmup_runs", args.warmup)
        timed = cfg.get("timed_runs", args.timed)
        conf = cfg.get("conf", args.conf)
        iou = cfg.get("iou", args.iou)
        half = cfg.get("half", args.half)
        runtime = cfg.get("runtime", args.runtime)
        energy_enabled = cfg.get("energy_enabled", not args.no_energy)
        power_paths = cfg.get("power_paths", [])
    else:
        weights = args.weights
        data_yaml = args.data
        imgsz = args.imgsz
        warmup = args.warmup
        timed = args.timed
        conf = args.conf
        iou = args.iou
        half = args.half
        runtime = args.runtime
        energy_enabled = not args.no_energy
        power_paths = []

    if not weights:
        logger.error("No weights specified. Use --weights or --config.")
        return

    # Verify hardware state
    hw_state = verify_jetson_state()
    logger.info(f"Hardware state: {json.dumps(hw_state, indent=2)}")

    # Load model
    from ultralytics import YOLO

    if runtime == "tensorrt_fp16":
        # Export to TensorRT on-device if not already done
        engine_path = weights.replace(".pt", ".engine")
        if not Path(engine_path).exists():
            logger.info("Building TensorRT FP16 engine on-device...")
            model = YOLO(weights)
            model.export(format="engine", imgsz=imgsz, half=True)
        model = YOLO(engine_path)
    elif runtime == "tensorrt_int8":
        engine_path = weights.replace(".pt", "_int8.engine")
        if not Path(engine_path).exists():
            model = YOLO(weights)
            model.export(format="engine", imgsz=imgsz, int8=True)
        model = YOLO(engine_path)
    else:
        model = YOLO(weights)

    # Load images
    sources = load_images(data_yaml, n_images=50, seed=42)
    logger.info(f"Loaded {len(sources)} profiling images")

    # Power sampler
    ps = None
    if energy_enabled:
        ps = JetsonPowerSampler(power_paths or None)

    # Run profiling
    device = "0" if torch.cuda.is_available() else "cpu"
    samples = run_full_protocol(
        model, sources, imgsz, device, conf, iou, half,
        warmup, timed, ps,
    )

    # Cold start (optional)
    cold_start = {}
    if args.cold_start:
        logger.info("Measuring cold start...")
        cold_times = []
        for _ in range(5):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            m = YOLO(weights)
            m.predict(source=sources[0], imgsz=imgsz, device=device,
                     verbose=False, save=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            cold_times.append((time.perf_counter() - t0) * 1000.0)
            del m
            torch.cuda.empty_cache()
        cold_start = {
            "median_ms": float(np.median(cold_times)),
            "min_ms": float(min(cold_times)),
            "max_ms": float(max(cold_times)),
        }

    # Summary statistics
    te2e = [s["t_e2e_ms"] for s in samples]
    summary = {
        "mean_ms": float(np.mean(te2e)),
        "std_ms": float(np.std(te2e)),
        "p50_ms": float(np.percentile(te2e, 50)),
        "p95_ms": float(np.percentile(te2e, 95)),
        "p99_ms": float(np.percentile(te2e, 99)),
        "cv": float(np.std(te2e) / np.mean(te2e)) if np.mean(te2e) > 0 else 0,
        "p99_p50_ratio": float(np.percentile(te2e, 99) / np.percentile(te2e, 50))
            if np.percentile(te2e, 50) > 0 else 0,
        "peak_gpu_mb": float(max(s["gpu_alloc_mb"] for s in samples)),
        "mean_energy_j": float(np.mean([s["energy_j"] for s in samples
                                         if s["energy_j"] >= 0])) or -1,
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"RESULTS: mean={summary['mean_ms']:.1f}ms, "
                f"p95={summary['p95_ms']:.1f}ms, p99={summary['p99_ms']:.1f}ms")
    logger.info(f"  CV={summary['cv']:.4f}, p99/p50={summary['p99_p50_ratio']:.3f}")
    logger.info(f"  Peak GPU: {summary['peak_gpu_mb']:.0f}MB")
    logger.info(f"  Energy: {summary['mean_energy_j']:.4f} J/inference")
    logger.info(f"{'='*50}")

    # Save results
    output = {
        "hardware_state": hw_state,
        "config": {
            "weights": weights, "imgsz": imgsz, "runtime": runtime,
            "warmup": warmup, "timed": timed, "half": half,
        },
        "summary": summary,
        "cold_start": cold_start,
        "samples": samples,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
