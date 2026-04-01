#!/usr/bin/env python3
"""
PRB-NSGA-II Engine (Algorithm 1, Section IV-F).

Fixes applied:
  #4  Explicit SAR augmentations (no hsv_h/s, flips, controlled mosaic)
  #6  Val for search selection, test reserved for final reporting
  #7  Pretrained backbone option (COCO init when available)
  #8  Memory as hard constraint, not wasted objective dimension
  #9  Multi-dataset: alternate datasets across generations
  #10 Unified pipeline: same augmentation dict for all candidates
"""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .search_space import Genome, is_feasible
from .arch_builder import (build_yaml_file, genome_uid, get_model_info,
                           verify_model)
from .risk import compute_objectives, latency_diagnostics
from .profiler import (load_profile_images, run_profiling_protocol,
                       laptop_proxy_profile, profile_on_jetson_ssh,
                       JetsonINA3221Sampler)
from .dominance import (prb_nondominated_sort, generate_reference_vectors,
                        assign_niches, architecture_entropy,
                        identify_uncertain_candidates, crowding_distance)

logger = logging.getLogger("tinas.nsga2")


# =============================================================================
# Fix #4: SAR-specific augmentation dict (unified across ALL candidates)
# =============================================================================
SAR_AUGMENTATION = {
    "augment": True,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "hsv_h": 0.0,     # SAR is grayscale — no hue augmentation
    "hsv_s": 0.0,     # SAR is grayscale — no saturation augmentation
    "hsv_v": 0.4,     # brightness variation only
    "flipud": 0.5,    # ships can appear flipped
    "fliplr": 0.5,
    "degrees": 0.0,   # no rotation (ships have canonical orientation)
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "erasing": 0.0,   # no random erasing for small targets
}


# =============================================================================
# Fix #8: Memory as hard constraint
# =============================================================================
def passes_memory_constraint(samples: List[Dict], max_gpu_mb: float = 512.0) -> bool:
    """Check if peak GPU memory is within Jetson budget."""
    if not samples:
        return True
    peak = max(s.get("gpu_alloc_mb", 0) for s in samples)
    return peak <= max_gpu_mb


# =============================================================================
# Candidate record
# =============================================================================

def make_record(genome: Genome) -> Dict[str, Any]:
    return {
        "genome": genome.to_dict(),
        "uid": genome_uid(genome),
        "objectives": {},
        "aux": {},
        "profiling_samples": [],
        "model_info": {},
        "weights_path": "",
        "yaml_path": "",
        "rank": 999,
        "niche": -1,
        "crowding": 0.0,
        "entropy": architecture_entropy(genome.signature_tokens()),
        "proxy_latency": {},
        "train_time_s": 0.0,
        "profile_time_s": 0.0,
        "dataset_trained_on": "",
    }


# =============================================================================
# Fix #1 pre-flight: verify genome builds before training
# =============================================================================

def preflight_check(genome: Genome, nc: int, imgsz: int) -> bool:
    """Quick build+forward test. Rejects broken YAML before wasting GPU time."""
    result = verify_model(genome, nc=nc, imgsz=imgsz)
    if not result["valid"]:
        logger.warning(f"Preflight FAILED for {genome.summary()}: {result['error']}")
        return False
    logger.debug(f"Preflight OK: {result['uid']} ({result['params_M']:.2f}M params)")
    return True


# =============================================================================
# Training (Fix #4 augmentation, Fix #7 pretrained, Fix #6 val split)
# =============================================================================

def train_candidate(
    record: Dict[str, Any],
    data_yaml: str,
    imgsz: int,
    epochs: int,
    batch_size: int,
    device: str,
    out_dir: Path,
    patience: int = 10,
    seed: int = 42,
    workers: int = 4,
    nc: int = 1,
    pretrained_backbone: bool = False,
) -> Dict[str, Any]:
    """
    Train candidate with:
      - Explicit SAR augmentations (Fix #4)
      - Optional COCO pretrained backbone (Fix #7)
      - Validation on 'val' split only (Fix #6: test reserved for final)
    """
    from ultralytics import YOLO

    genome = Genome.from_dict(record["genome"])
    cand_dir = out_dir / record["uid"]

    yaml_path = build_yaml_file(genome, cand_dir, nc=nc)
    record["yaml_path"] = str(yaml_path)

    t0 = time.time()
    model = YOLO(str(yaml_path))

    # Fix #7: load pretrained backbone weights if requested
    # This is fair because baselines also start from COCO pretrained
    if pretrained_backbone:
        try:
            # Load a pretrained model and transfer matching backbone weights
            pretrained = YOLO("yolov8n.pt")  # small pretrained model
            state = pretrained.model.state_dict()
            model_state = model.model.state_dict()
            transferred = 0
            for k, v in state.items():
                if k in model_state and v.shape == model_state[k].shape:
                    model_state[k] = v
                    transferred += 1
            model.model.load_state_dict(model_state, strict=False)
            logger.debug(f"  Transferred {transferred} pretrained layers")
            del pretrained
        except Exception as e:
            logger.debug(f"  Pretrained transfer failed (OK, training from scratch): {e}")

    # Train with unified SAR augmentations (Fix #4)
    train_args = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch_size,
        "device": device,
        "workers": workers,
        "project": str(cand_dir),
        "name": "train",
        "patience": patience,
        "seed": seed,
        "verbose": False,
        "val": True,       # validate on val split during training
        **SAR_AUGMENTATION,  # Fix #4: explicit augmentations
    }
    model.train(**train_args)

    record["train_time_s"] = time.time() - t0

    weights = cand_dir / "train" / "weights" / "best.pt"
    if not weights.exists():
        weights = cand_dir / "train" / "weights" / "last.pt"
    if not weights.exists():
        logger.error(f"No weights for {record['uid']}")
        record["objectives"]["ferr"] = 1.0
        return record

    record["weights_path"] = str(weights)

    # Fix #6: validate on 'val' split (test reserved for final reporting)
    val_model = YOLO(str(weights))
    val_results = val_model.val(
        data=data_yaml, imgsz=imgsz, device=device,
        split="val", workers=workers, verbose=False,
    )

    mAP = -1.0
    try:
        mAP = float(getattr(getattr(val_results, "box", None), "map", -1.0))
    except Exception:
        pass
    if mAP < 0:
        try:
            mAP = float(val_results.results_dict.get("metrics/mAP50-95(B)", -1.0))
        except Exception:
            pass

    record["aux"]["mAP"] = mAP
    record["objectives"]["ferr"] = 1.0 - max(0.0, mAP)

    try:
        record["model_info"] = get_model_info(val_model)
    except Exception:
        pass

    logger.info(f"  Trained {record['uid']}: mAP={mAP:.4f}, "
                f"params={record['model_info'].get('params_M', '?'):.2f}M, "
                f"{record['train_time_s']:.0f}s")

    del val_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


# =============================================================================
# Profiling
# =============================================================================

def profile_candidate(record, data_yaml, imgsz, config, device):
    """Profile locally with memory constraint check (Fix #8)."""
    from ultralytics import YOLO

    if not record["weights_path"]:
        return record

    prof_cfg = config["profiling"]
    sources = load_profile_images(
        data_yaml, split="val",
        n_images=prof_cfg["num_profile_images"],
        seed=42, in_ram=prof_cfg.get("profile_in_ram", True))

    power_sampler = None
    j_cfg = config.get("jetson", {}).get("energy", {})
    if j_cfg.get("enabled") and j_cfg.get("power_paths"):
        pp = j_cfg["power_paths"]
        if Path(pp[0]).exists():
            power_sampler = JetsonINA3221Sampler(pp, j_cfg.get("sample_period_ms", 10)/1000)

    model = YOLO(record["weights_path"])
    t0 = time.time()

    # Use search-phase timed runs (fewer than final protocol)
    timed = prof_cfg.get("search_timed_runs", prof_cfg["timed_runs"])

    samples = run_profiling_protocol(
        model, sources, imgsz, device,
        warmup_runs=prof_cfg["warmup_runs"],
        timed_runs=timed,
        conf=prof_cfg["conf_threshold"],
        iou=prof_cfg["iou_threshold"],
        half=prof_cfg.get("use_half", True),
        power_sampler=power_sampler)

    record["profiling_samples"] = samples
    record["profile_time_s"] = time.time() - t0

    # Fix #8: memory constraint check
    max_mem = config.get("constraints", {}).get("max_gpu_mb", 512.0)
    if not passes_memory_constraint(samples, max_mem):
        logger.info(f"    REJECTED: GPU memory exceeds {max_mem}MB")
        record["objectives"] = {"ferr": 1.0, "flat": float("inf"), "feng": float("inf")}
        del model
        return record

    # Compute objectives (Fix #8: memory NOT in objective vector)
    obj_cfg = config["objectives"]
    objectives, aux = compute_objectives(
        mAP=record["aux"].get("mAP", 0.0),
        samples=samples,
        tau=obj_cfg["tau"],
        risk_mode=obj_cfg["risk_mode"],
        lambda_sigma=obj_cfg["lambda_sigma"],
        mem_mode="gpu_alloc",
        energy_mode="risk" if "feng" in obj_cfg["active"] else "mean")
    record["objectives"] = objectives
    record["aux"].update(aux)

    te2e = [s["t_e2e_ms"] for s in samples]
    record["aux"]["latency_diagnostics"] = latency_diagnostics(te2e, record["uid"])

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def add_resampling(record, data_yaml, imgsz, config, device, n_add=30):
    from ultralytics import YOLO
    if not record["weights_path"]: return record

    sources = load_profile_images(data_yaml, n_images=30, seed=42, in_ram=True)
    model = YOLO(record["weights_path"])
    from .profiler import profile_one_run

    for i in range(min(3, len(sources))):
        model.predict(source=sources[i], imgsz=imgsz, device=device,
                     verbose=False, save=False)

    for i in range(n_add):
        s = profile_one_run(model, sources[i % len(sources)], imgsz, device,
                           half=config["profiling"].get("use_half", True))
        record["profiling_samples"].append(s)

    obj_cfg = config["objectives"]
    objectives, aux = compute_objectives(
        record["aux"].get("mAP", 0.0), record["profiling_samples"],
        tau=obj_cfg["tau"], risk_mode=obj_cfg["risk_mode"],
        lambda_sigma=obj_cfg["lambda_sigma"])
    record["objectives"] = objectives
    record["aux"].update(aux)

    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return record


# =============================================================================
# Selection & offspring
# =============================================================================

def tournament(pop, k=2, rng=None):
    r = rng or random.Random()
    cs = r.sample(pop, min(k, len(pop)))
    return min(cs, key=lambda c: (c.get("rank", 999), -c.get("crowding", 0)))


def gen_offspring(survivors, n, mut_p, xo_p, rng=None, retries=20):
    r = rng or random.Random()
    out = []
    for _ in range(n):
        for _ in range(retries):
            if r.random() < xo_p and len(survivors) >= 2:
                p1 = Genome.from_dict(tournament(survivors, rng=r)["genome"])
                p2 = Genome.from_dict(tournament(survivors, rng=r)["genome"])
                child = Genome.crossover(p1, p2, rng=r).mutate(mut_p, rng=r)
            else:
                child = Genome.from_dict(tournament(survivors, rng=r)["genome"]).mutate(mut_p, rng=r)
            if is_feasible(child):
                out.append(child); break
        else:
            out.append(Genome.random(rng=r))
    return out


# =============================================================================
# Hypervolume tracker (convergence evidence for paper)
# =============================================================================

def compute_hypervolume(pareto: List[Dict], objectives: List[str],
                        ref_point: List[float] | None = None) -> float:
    """
    Compute hypervolume indicator of the Pareto front.

    This is THE convergence evidence: if HV plateaus across generations,
    the search has adequately explored the space.  Plot HV vs generation
    in the paper (Figure X) to answer "300 out of 5M is enough?"

    Uses the inclusion-exclusion exact algorithm for 2-3 objectives.
    For >3 objectives, falls back to Monte Carlo approximation.
    """
    if not pareto:
        return 0.0

    # Extract objective vectors (all minimised)
    points = []
    for c in pareto:
        vec = [c["objectives"].get(o, float("inf")) for o in objectives]
        if any(v == float("inf") for v in vec):
            continue
        points.append(vec)

    if not points:
        return 0.0

    points = np.array(points)
    n_obj = points.shape[1]

    # Default reference point: worst value per objective × 1.1
    if ref_point is None:
        ref_point = (points.max(axis=0) * 1.1).tolist()
    ref = np.array(ref_point)

    # Filter dominated-by-reference
    points = points[np.all(points < ref, axis=1)]
    if len(points) == 0:
        return 0.0

    if n_obj == 2:
        # Exact 2D hypervolume
        sorted_pts = points[points[:, 0].argsort()]
        hv = 0.0
        prev_y = ref[1]
        for x, y in sorted_pts:
            if y < prev_y:
                hv += (ref[0] - x) * (prev_y - y)
                prev_y = y
        return float(hv)

    elif n_obj == 3:
        # Exact 3D via slicing (simple O(n^2) implementation)
        sorted_pts = points[points[:, 2].argsort()]
        hv = 0.0
        prev_z = ref[2]
        for k in range(len(sorted_pts)):
            z = sorted_pts[k, 2]
            # 2D hypervolume of points[:k+1] projected onto first 2 objectives
            proj = sorted_pts[:k + 1, :2]
            proj_sorted = proj[proj[:, 0].argsort()]
            hv_2d = 0.0
            prev_y = ref[1]
            for x, y in proj_sorted:
                if y < prev_y:
                    hv_2d += (ref[0] - x) * (prev_y - y)
                    prev_y = y
            hv += hv_2d * (prev_z - z)
            prev_z = z
        return float(hv)

    else:
        # Monte Carlo approximation for >3 objectives
        n_samples = 100000
        rng = np.random.RandomState(42)
        mins = points.min(axis=0)
        vol_box = np.prod(ref - mins)
        samples = rng.uniform(mins, ref, size=(n_samples, n_obj))
        dominated = np.zeros(n_samples, dtype=bool)
        for p in points:
            dominated |= np.all(samples >= p, axis=1)
        return float(vol_box * dominated.mean())


def knee_point(pareto, objectives):
    if len(pareto) <= 2: return pareto[0] if pareto else None
    M = np.array([[c["objectives"].get(o, 1e9) for o in objectives] for c in pareto])
    mins, maxs = M.min(0), M.max(0)
    rng = maxs - mins; rng[rng == 0] = 1.0
    N = (M - mins) / rng
    ideal, nadir = N.min(0), N.max(0)
    line = nadir - ideal; ll = np.linalg.norm(line)
    if ll == 0: return pareto[0]
    dists = [np.linalg.norm(N[i] - ideal - np.dot(N[i]-ideal, line)/ll * line/ll)
             for i in range(len(N))]
    return pareto[int(np.argmax(dists))]


# =============================================================================
# Main search (Algorithm 1)
# =============================================================================

def run_search(config: Dict[str, Any]) -> Dict[str, Any]:
    nas = config["nas"]
    obj_cfg = config["objectives"]
    dom_cfg = config["dominance"]
    res_cfg = config["resampling"]

    base_dir = Path(config["output"]["base_dir"])
    base_dir.mkdir(parents=True, exist_ok=True)

    search_start = time.time()
    cost = {"generations": [], "total_train_s": 0, "total_profile_s": 0}

    # Fix #9: multi-dataset alternation
    all_ds = config["active_datasets"]
    rng = random.Random(nas["seed"])
    np.random.seed(nas["seed"])

    objectives = obj_cfg["active"]
    # Fix #8: remove fmem from objectives (it's a constraint now)
    objectives = [o for o in objectives if o != "fmem"]
    logger.info(f"Optimising objectives: {objectives} (memory is a hard constraint)")

    ref_vectors = generate_reference_vectors(len(objectives), n_divisions=12)

    # Initialise population
    logger.info(f"Initialising P={nas['population_size']}")
    population: List[Dict] = []
    init_genomes = []
    att = 0
    while len(init_genomes) < nas["population_size"] and att < 2000:
        g = Genome.random(rng)
        if is_feasible(g):
            init_genomes.append(g)
        att += 1

    candidate_genomes = init_genomes

    for gen in range(nas["generations"]):
        gen_start = time.time()

        # Fix #9: alternate dataset each generation
        ds_name = all_ds[gen % len(all_ds)]
        ds = config["datasets"][ds_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Generation {gen+1}/{nas['generations']} — dataset: {ds_name}")
        logger.info(f"{'='*60}")

        for i, genome in enumerate(candidate_genomes):
            logger.info(f"  [{gen+1}] Candidate {i+1}/{len(candidate_genomes)}: "
                        f"{genome.summary()}")

            # Fix #1: preflight check before training
            if not preflight_check(genome, nc=ds["nc"], imgsz=ds["imgsz"]):
                rec = make_record(genome)
                rec["objectives"] = {"ferr": 1.0, "flat": float("inf"), "feng": float("inf")}
                population.append(rec)
                continue

            rec = make_record(genome)
            rec["dataset_trained_on"] = ds_name

            rec = train_candidate(
                rec, ds["data_yaml"], ds["imgsz"],
                epochs=nas["proxy_epochs"],
                batch_size=nas["proxy_batch_size"],
                device=str(config["laptop"]["device"]),
                out_dir=base_dir / f"gen_{gen:03d}",
                patience=nas["proxy_patience"],
                seed=nas["seed"],
                workers=config["laptop"]["workers"],
                nc=ds["nc"],
                pretrained_backbone=config.get("pretrained_backbone", False),
            )

            # Proxy screening
            proxy_cfg = config["laptop"].get("proxy_screen", {})
            if proxy_cfg.get("enabled") and rec["weights_path"]:
                from ultralytics import YOLO
                srcs = load_profile_images(ds["data_yaml"], n_images=10, seed=42)
                pm = YOLO(rec["weights_path"])
                proxy = laptop_proxy_profile(pm, srcs, ds["imgsz"],
                                            str(config["laptop"]["device"]))
                rec["proxy_latency"] = proxy
                del pm; torch.cuda.empty_cache() if torch.cuda.is_available() else None
                if proxy["proxy_p95_ms"] > proxy_cfg["max_latency_ms"]:
                    logger.info(f"    REJECTED: proxy p95={proxy['proxy_p95_ms']:.0f}ms")
                    rec["objectives"] = {"ferr": 1.0, "flat": float("inf"), "feng": float("inf")}
                    population.append(rec)
                    continue

            # Profile
            j_cfg = config.get("jetson", {})
            if j_cfg.get("mode") == "ssh" and rec["weights_path"]:
                samples = profile_on_jetson_ssh(rec["weights_path"], ds["data_yaml"], config, imgsz=ds["imgsz"])
                if samples:
                    rec["profiling_samples"] = samples
                    objs, aux = compute_objectives(
                        rec["aux"].get("mAP", 0.0), samples,
                        tau=obj_cfg["tau"], risk_mode=obj_cfg["risk_mode"],
                        lambda_sigma=obj_cfg["lambda_sigma"])
                    rec["objectives"] = objs; rec["aux"].update(aux)
            else:
                rec = profile_candidate(rec, ds["data_yaml"], ds["imgsz"],
                                       config, str(config["laptop"]["device"]))

            population.append(rec)
            cost["total_train_s"] += rec.get("train_time_s", 0)
            cost["total_profile_s"] += rec.get("profile_time_s", 0)

        # Ranking
        logger.info(f"  Ranking {len(population)} candidates...")
        fronts, p_cache = prb_nondominated_sort(
            population, objectives,
            tau=obj_cfg["tau"], risk_mode=obj_cfg["risk_mode"],
            lambda_sigma=obj_cfg["lambda_sigma"],
            eta=dom_cfg["eta"], n_bootstrap=dom_cfg["bootstrap_draws"],
            seed=nas["seed"] + gen * 999)

        for rank, front in enumerate(fronts):
            for idx in front:
                population[idx]["rank"] = rank

        # Resampling
        if res_cfg["enabled"]:
            for fi, fidx in enumerate(fronts[:res_cfg["resample_fronts"]]):
                for idx in identify_uncertain_candidates(fidx, p_cache, res_cfg["delta"]):
                    ns = len(population[idx].get("profiling_samples", []))
                    if ns >= res_cfg["max_timed_runs"]: continue
                    na = min(res_cfg["delta_n"], res_cfg["max_timed_runs"] - ns)
                    logger.info(f"    Resample {population[idx]['uid']} +{na} runs")
                    population[idx] = add_resampling(
                        population[idx], ds["data_yaml"], ds["imgsz"],
                        config, str(config["laptop"]["device"]), na)

            fronts, p_cache = prb_nondominated_sort(
                population, objectives,
                tau=obj_cfg["tau"], risk_mode=obj_cfg["risk_mode"],
                lambda_sigma=obj_cfg["lambda_sigma"],
                eta=dom_cfg["eta"], n_bootstrap=dom_cfg["bootstrap_draws"],
                seed=nas["seed"] + gen * 999 + 777)
            for rank, front in enumerate(fronts):
                for idx in front: population[idx]["rank"] = rank

        # Niching
        niche_ids = assign_niches(population, objectives, ref_vectors)
        for i, nid in enumerate(niche_ids):
            population[i]["niche"] = nid

        # Environmental selection
        P = nas["population_size"]
        surv_idx = []
        for fidx in fronts:
            if len(surv_idx) + len(fidx) <= P:
                surv_idx.extend(fidx)
            else:
                rem = P - len(surv_idx)
                cd = crowding_distance(population, fidx, objectives)
                scored = sorted(fidx, key=lambda x: (cd.get(x, 0),
                                population[x].get("entropy", 0)), reverse=True)
                surv_idx.extend(scored[:rem])
                break
        survivors = [population[i] for i in surv_idx]

        pareto = [population[i] for i in fronts[0]] if fronts else []
        kn = knee_point(pareto, objectives)

        # Hypervolume convergence tracking (evidence for "300/5M is enough")
        hv = compute_hypervolume(pareto, objectives)

        gen_time = time.time() - gen_start
        cost["generations"].append({
            "gen": gen + 1, "pop_size": len(population),
            "pareto_size": len(pareto), "gen_time_s": gen_time,
            "dataset": ds_name,
            "hypervolume": hv,
            "knee_uid": kn["uid"] if kn else None,
            "knee_obj": kn["objectives"] if kn else None,
        })

        logger.info(f"  Gen {gen+1}: {len(pareto)} Pareto, HV={hv:.6f}, {gen_time:.0f}s")
        if kn:
            logger.info(f"  Knee: {kn['uid']} mAP={kn['aux'].get('mAP','?'):.4f} "
                         f"p95={kn['aux'].get('p95_ms','?'):.1f}ms")

        _save(base_dir, gen, population, pareto, kn, cost)

        candidate_genomes = gen_offspring(
            survivors, P, nas["mutation_prob"],
            nas.get("crossover_prob", 0.9), rng)

    total = time.time() - search_start
    cost["total_wall_clock_s"] = total
    cost["total_wall_clock_h"] = total / 3600

    logger.info(f"\nSearch complete: {total/3600:.1f}h wall-clock, "
                f"{len(pareto)} Pareto-optimal")
    _save(base_dir, "final", population, pareto, kn, cost)

    return {"pareto_front": pareto, "knee": kn,
            "population": population, "cost_log": cost}


def _save(base_dir, gen, population, pareto, knee, cost):
    out = base_dir / "checkpoints"; out.mkdir(exist_ok=True)
    def ser(obj):
        if isinstance(obj, dict):
            return {k: (f"[{len(v)} samples]" if k == "profiling_samples"
                        else ser(v)) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [ser(x) for x in obj]
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    summary = {"generation": gen, "pareto_front": ser(pareto),
               "knee": ser(knee), "cost_log": ser(cost),
               "population_size": len(population)}
    with open(out / f"gen_{gen}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
