#!/usr/bin/env python3
"""
Deployment-Faithful Profiler (Section III-B, III-H).

Two modes:
  1. Laptop profiler: quick proxy latency screening on training GPU.
     Used during search to reject obviously-slow candidates before
     expensive Jetson transfer.
  2. Jetson profiler: batch-1, end-to-end, synchronized timing with
     warmup, tail-risk stats, energy measurement via INA3221 rails.
     This is the deployment-faithful measurement that goes in the paper.

Both modes measure pre+inf+post as separate timed blocks with CUDA sync.

Energy measurement on Jetson:
  We sample INA3221 power rails at ~100Hz in a background thread.
  Per-inference energy = mean_power × wall_time (Eq. 10).
  Temporal resolution caveat: at ~10ms sampling with ~30ms inference,
  we get ~3 power samples per inference. For loops of Nt=300 runs, the
  aggregate statistics converge, but individual per-run energy is noisy.
  This is documented as a limitation (addresses reviewer concern).
"""
from __future__ import annotations

import logging
import os
import random
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import psutil
import torch
import yaml

logger = logging.getLogger("tinas.profiler")

# Optional imports
try:
    import pynvml
    NVML_OK = True
except ImportError:
    NVML_OK = False

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False


# =============================================================================
# Power Sampling
# =============================================================================

class PowerSampler:
    """Base class for background power sampling."""

    def __init__(self, period_s: float = 0.01):
        self.period_s = period_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples_w: List[float] = []
        self.timestamps: List[float] = []

    def read_power_w(self) -> Optional[float]:
        raise NotImplementedError

    def start(self):
        self.samples_w.clear()
        self.timestamps.clear()
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
            p = self.read_power_w()
            if p is not None and p > 0:
                self.samples_w.append(float(p))
                self.timestamps.append(time.monotonic())
            time.sleep(self.period_s)


class JetsonINA3221Sampler(PowerSampler):
    """
    Read power from Jetson INA3221 power rails via sysfs.
    Handles microwatt/milliwatt/watt auto-scaling.

    Typical path on Orin Nano:
      /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*/in1_input
    """

    def __init__(self, power_paths: List[str], period_s: float = 0.01):
        super().__init__(period_s)
        self.power_paths = [Path(p) for p in power_paths]
        # Validate at init
        for p in self.power_paths:
            if not p.exists():
                logger.warning(f"Power rail path not found: {p}")

    def read_power_w(self) -> Optional[float]:
        total = 0.0
        for p in self.power_paths:
            try:
                raw = float(p.read_text().strip())
                # Jetson sysfs reports in milliwatts typically
                if raw > 1e6:
                    total += raw / 1e6  # microwatts → watts
                elif raw > 1e3:
                    total += raw / 1e3  # milliwatts → watts
                else:
                    total += raw        # already watts
            except Exception:
                return None
        return total if total > 0 else None


class NVMLSampler(PowerSampler):
    """Desktop GPU power via NVML (for laptop proxy)."""

    def __init__(self, gpu_index: int = 0, period_s: float = 0.02):
        super().__init__(period_s)
        if not NVML_OK:
            raise RuntimeError("pynvml not installed: pip install nvidia-ml-py3")
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    def read_power_w(self) -> Optional[float]:
        try:
            return float(pynvml.nvmlDeviceGetPowerUsage(self.handle)) / 1000.0
        except Exception:
            return None


# =============================================================================
# Image Loading
# =============================================================================

def load_profile_images(data_yaml: str, split: str = "val",
                        n_images: int = 50, seed: int = 42,
                        in_ram: bool = True) -> List[Any]:
    """
    Load a fixed set of images for profiling.
    Returns list of numpy arrays (if in_ram) or file paths.
    """
    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)

    yaml_dir = Path(data_yaml).resolve().parent
    split_val = data.get(split, data.get("val", ""))
    if isinstance(split_val, list):
        split_val = split_val[0]

    def _resolve_split_path(split_entry: str) -> Path:
        split_path = Path(split_entry)
        if split_path.is_absolute():
            return split_path

        candidate_roots = []
        data_root = data.get("path")
        if data_root:
            data_root_path = Path(data_root)
            candidate_roots.append(
                data_root_path if data_root_path.is_absolute() else (yaml_dir / data_root_path)
            )
        candidate_roots.append(yaml_dir)

        candidates = []
        for root in candidate_roots:
            candidates.append((root / split_path).resolve())

            # Some dataset YAMLs incorrectly prefix paths with ../ even though
            # the split folders live under the dataset root.
            cleaned_parts = [part for part in split_path.parts if part not in ("..", ".")]
            if cleaned_parts and cleaned_parts != list(split_path.parts):
                candidates.append((root.joinpath(*cleaned_parts)).resolve())

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]

    split_path = _resolve_split_path(str(split_val))
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    if split_path.is_dir():
        imgs = [p for p in split_path.rglob("*") if p.suffix.lower() in exts]
    elif split_path.suffix == ".txt":
        imgs = []
        for line in split_path.read_text().splitlines():
            entry = line.strip()
            if not entry:
                continue
            entry_path = Path(entry)
            if not entry_path.is_absolute():
                entry_path = (split_path.parent / entry_path).resolve()
            imgs.append(entry_path)
    else:
        imgs = [split_path]

    imgs = [p for p in imgs if p.exists()]
    random.Random(seed).shuffle(imgs)
    imgs = imgs[:n_images]

    if not imgs:
        logger.warning(f"No profiling images found for split='{split}' from {data_yaml}")
        return []

    if in_ram and CV2_OK:
        loaded = []
        for p in imgs:
            im = cv2.imread(str(p))
            if im is not None:
                loaded.append(im)
        return loaded if loaded else [str(p) for p in imgs]

    return [str(p) for p in imgs]


# =============================================================================
# Single-Run Profiling
# =============================================================================

def _gpu_peak_mb() -> tuple:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    a = torch.cuda.max_memory_allocated() / (1024 ** 2)
    r = torch.cuda.max_memory_reserved() / (1024 ** 2)
    return float(a), float(r)


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def profile_one_run(
    model,
    source: Any,
    imgsz: int,
    device: str,
    conf: float = 0.25,
    iou: float = 0.7,
    half: bool = True,
    power_sampler: Optional[PowerSampler] = None,
) -> Dict[str, float]:
    """
    One batch-1 inference run with synchronized timing.

    Returns dict with:
      t_pre_ms, t_inf_ms, t_post_ms, t_e2e_ms, wall_ms,
      gpu_alloc_mb, gpu_reserved_mb, cpu_rss_mb,
      mean_power_w, energy_j
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    rss_before = _rss_mb()
    if power_sampler:
        power_sampler.start()

    t0 = time.perf_counter()

    # Ultralytics predict() includes pre+inf+post
    results = model.predict(
        source=source, imgsz=imgsz, device=device,
        conf=conf, iou=iou, half=half,
        verbose=False, save=False, stream=False,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    if power_sampler:
        power_sampler.stop()

    wall_ms = (t1 - t0) * 1000.0

    # Extract Ultralytics speed breakdown
    pre = inf = post = 0.0
    try:
        sp = getattr(results[0], "speed", {}) or {}
        pre = float(sp.get("preprocess", 0.0))
        inf = float(sp.get("inference", 0.0))
        post = float(sp.get("postprocess", 0.0))
    except Exception:
        pass

    te2e = (pre + inf + post) if (pre + inf + post) > 0 else wall_ms

    ga, gr = _gpu_peak_mb()
    rss_after = _rss_mb()

    mean_pw = power_sampler.mean_w() if power_sampler else -1.0
    energy = mean_pw * (wall_ms / 1000.0) if mean_pw > 0 else -1.0

    return {
        "t_pre_ms": pre,
        "t_inf_ms": inf,
        "t_post_ms": post,
        "t_e2e_ms": te2e,
        "wall_ms": wall_ms,
        "gpu_alloc_mb": ga,
        "gpu_reserved_mb": gr,
        "cpu_rss_mb": max(rss_before, rss_after),
        "mean_power_w": mean_pw,
        "energy_j": energy,
    }


# =============================================================================
# Full Profiling Protocol (Section III-H)
# =============================================================================

def run_profiling_protocol(
    model,
    sources: List[Any],
    imgsz: int,
    device: str,
    warmup_runs: int = 50,
    timed_runs: int = 300,
    conf: float = 0.25,
    iou: float = 0.7,
    half: bool = True,
    power_sampler: Optional[PowerSampler] = None,
) -> List[Dict[str, float]]:
    """
    Execute the full profiling protocol (Section III-H):
      1. Nw warmup runs (discarded)
      2. Nt timed runs (kept)
      All at batch=1 with CUDA sync.

    Returns: list of Nt per-run measurement dicts.
    """
    n_src = len(sources)
    logger.info(f"Profiling: {warmup_runs} warmup + {timed_runs} timed runs "
                f"(batch=1, {n_src} images, {'FP16' if half else 'FP32'})")

    # Warmup (discard results)
    for i in range(warmup_runs):
        _ = profile_one_run(
            model, sources[i % n_src], imgsz, device,
            conf=conf, iou=iou, half=half, power_sampler=None,
        )

    # Timed runs
    samples = []
    for i in range(timed_runs):
        s = profile_one_run(
            model, sources[i % n_src], imgsz, device,
            conf=conf, iou=iou, half=half, power_sampler=power_sampler,
        )
        samples.append(s)

    return samples


def measure_cold_start(
    weights_path: str,
    source: Any,
    imgsz: int,
    device: str,
    n_repeats: int = 5,
) -> Dict[str, float]:
    """
    Cold-start latency: model load + first inference in a fresh YOLO instance.
    Repeated n_repeats times, reporting median and range.
    """
    from ultralytics import YOLO

    times = []
    for _ in range(n_repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        m = YOLO(weights_path)
        _ = m.predict(source=source, imgsz=imgsz, device=device,
                      verbose=False, save=False, stream=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "cold_start_median_ms": float(np.median(times)),
        "cold_start_min_ms": float(min(times)),
        "cold_start_max_ms": float(max(times)),
        "cold_start_n": n_repeats,
    }


# =============================================================================
# Laptop Proxy Profiler (Quick Screening)
# =============================================================================

def laptop_proxy_profile(
    model,
    sources: List[Any],
    imgsz: int,
    device: str,
    n_runs: int = 30,
    half: bool = True,
) -> Dict[str, float]:
    """
    Quick latency check on laptop GPU. Used to pre-filter candidates
    that are obviously too slow before expensive Jetson transfer.
    Returns mean and p95 latency.
    """
    if not sources:
        raise ValueError("No profiling images were loaded for laptop_proxy_profile().")

    # Small warmup
    for i in range(min(5, len(sources))):
        _ = profile_one_run(model, sources[i % len(sources)], imgsz, device,
                           half=half, power_sampler=None)

    times = []
    for i in range(n_runs):
        s = profile_one_run(model, sources[i % len(sources)], imgsz, device,
                           half=half, power_sampler=None)
        times.append(s["t_e2e_ms"])

    return {
        "proxy_mean_ms": float(np.mean(times)),
        "proxy_p95_ms": float(np.percentile(times, 95)),
        "proxy_n_runs": n_runs,
    }


# =============================================================================
# Jetson Remote Profiling via SSH
# =============================================================================

def profile_on_jetson_ssh(
    weights_path: str,
    data_yaml: str,
    config: Dict[str, Any],
    imgsz: int = 640,
) -> List[Dict[str, float]]:
    """
    Transfer model to Jetson, run profiling script remotely, retrieve results.

    Workflow:
      1. SFTP weights + config to Jetson (via paramiko — supports password auth)
      2. SSH: run jetson_profiler.py on the device
      3. SFTP results back
      4. Parse and return samples

    Uses paramiko so that password-based auth works without sshpass.
    Falls back to key-based auth if no password is set in config.
    """
    import json
    import tempfile

    try:
        import paramiko
    except ImportError:
        raise RuntimeError("paramiko not installed: pip install paramiko")

    jetson = config["jetson"]
    host = jetson["host"]
    user = jetson["user"]
    password = jetson.get("password")
    key_file = os.path.expanduser(jetson.get("key_file", "~/.ssh/id_rsa"))
    remote_dir = jetson["remote_workdir"]

    # Build SSH client
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connect_kwargs: Dict[str, Any] = {"username": user, "timeout": 30}
    if password:
        connect_kwargs.update({
            "password": password,
            "look_for_keys": False,
            "allow_agent": False,
        })
    elif os.path.exists(key_file):
        connect_kwargs["key_filename"] = key_file

    logger.info(f"Connecting to Jetson at {host} as {user} ...")
    try:
        client.connect(host, **connect_kwargs)
    except paramiko.AuthenticationException as e:
        logger.error(f"Jetson SSH auth failed: {e}")
        return []
    except Exception as e:
        logger.error(f"Jetson SSH connection failed: {e}")
        return []

    try:
        sftp = client.open_sftp()

        # Create remote workdir
        stdin, stdout, stderr = client.exec_command(f"mkdir -p {remote_dir}")
        stdout.channel.recv_exit_status()

        # Transfer weights
        remote_weights = f"{remote_dir}/model.pt"
        logger.info(f"Uploading weights → Jetson:{remote_weights} ...")
        sftp.put(weights_path, remote_weights)

        # Transfer jetson_profiler.py to remote workdir
        local_profiler = str(
            Path(__file__).resolve().parent.parent / "scripts" / "jetson_profiler.py"
        )
        remote_profiler = f"{remote_dir}/jetson_profiler.py"
        sftp.put(local_profiler, remote_profiler)

        # ------------------------------------------------------------------ #
        # Upload val images from laptop to Jetson (once; skip if present).   #
        # The laptop data_yaml path doesn't exist on Jetson, so we mirror    #
        # the val split and write a matching remote data.yaml.                #
        # ------------------------------------------------------------------ #
        local_data_yaml = Path(data_yaml)
        import yaml as _yaml
        with open(local_data_yaml) as _f:
            _ds_cfg = _yaml.safe_load(_f)

        laptop_ds_root = Path(_ds_cfg.get("path", local_data_yaml.parent))
        if not laptop_ds_root.is_absolute():
            laptop_ds_root = (local_data_yaml.parent / laptop_ds_root).resolve()

        val_rel = _ds_cfg.get("val", "val/images")
        local_val_dir = (laptop_ds_root / val_rel).resolve()

        # Derive a stable remote dataset name from the local root folder name
        ds_name = laptop_ds_root.name  # e.g. "HRSID.v1i.yolov11"
        remote_ds_root = f"{remote_dir}/datasets/{ds_name}"
        remote_val_dir = f"{remote_ds_root}/val/images"

        # Check if already uploaded (count remote files)
        stdin_c, stdout_c, _ = client.exec_command(
            f"ls {remote_val_dir} 2>/dev/null | wc -l"
        )
        remote_count = int(stdout_c.read().decode().strip() or "0")

        local_val_imgs = sorted(local_val_dir.glob("*")) if local_val_dir.is_dir() else []
        local_val_imgs = [p for p in local_val_imgs
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

        if remote_count < len(local_val_imgs):
            logger.info(f"Uploading {len(local_val_imgs)} val images → Jetson:{remote_val_dir} ...")
            stdin_m, stdout_m, _ = client.exec_command(f"mkdir -p {remote_val_dir}")
            stdout_m.channel.recv_exit_status()
            for img_path in local_val_imgs:
                sftp.put(str(img_path), f"{remote_val_dir}/{img_path.name}")
            logger.info("Val images upload complete.")
        else:
            logger.info(f"Val images already on Jetson ({remote_count} files), skipping upload.")

        # Write remote data.yaml
        remote_data_yaml = f"{remote_ds_root}/data.yaml"
        remote_yaml_content = (
            f"path: {remote_ds_root}\n"
            f"train: train/images\n"
            f"val: val/images\n"
            f"test: test/images\n"
            f"nc: {_ds_cfg.get('nc', 1)}\n"
            f"names: {_ds_cfg.get('names', ['ship'])}\n"
        )
        stdin_y, stdout_y, _ = client.exec_command(
            f"mkdir -p {remote_ds_root} && cat > {remote_data_yaml} << 'EOYAML'\n"
            f"{remote_yaml_content}EOYAML"
        )
        stdout_y.channel.recv_exit_status()

        # Use the venv python that has ultralytics + torch
        python_bin = jetson.get("python_bin", "/home/iisri/venv310/bin/python3")

        # Build and upload profiling config
        prof_config = {
            "weights": remote_weights,
            "data_yaml": remote_data_yaml,
            "imgsz": imgsz,
            "warmup_runs": config["profiling"]["warmup_runs"],
            "timed_runs": config["profiling"]["search_timed_runs"],
            "conf": config["profiling"]["conf_threshold"],
            "iou": config["profiling"]["iou_threshold"],
            "half": config["profiling"].get("use_half", True),
            "runtime": jetson.get("runtime", "pytorch"),
            "energy_enabled": jetson["energy"]["enabled"],
            "power_paths": jetson["energy"].get("power_paths", []),
            "power_period_ms": jetson["energy"].get("sample_period_ms", 10),
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(prof_config, f)
            local_config_path = f.name

        remote_config = f"{remote_dir}/profile_config.json"
        sftp.put(local_config_path, remote_config)
        os.unlink(local_config_path)

        # Run remote profiler using the venv python
        remote_results = f"{remote_dir}/profile_results.json"
        cmd = (f"cd {remote_dir} && "
               f"{python_bin} {remote_profiler} --config {remote_config} --output {remote_results}")
        logger.info(f"Running Jetson profiler via SSH ...")
        stdin, stdout, stderr = client.exec_command(cmd, timeout=600)
        exit_code = stdout.channel.recv_exit_status()

        if exit_code != 0:
            err_out = stderr.read().decode(errors="replace")
            logger.error(f"Jetson profiling failed (exit {exit_code}):\n{err_out}")
            return []

        # Retrieve results
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False) as f:
            local_results_path = f.name
        sftp.get(remote_results, local_results_path)

        with open(local_results_path, "r") as f:
            samples = json.load(f)

        os.unlink(local_results_path)
        sftp.close()
        return samples

    finally:
        client.close()
