#!/usr/bin/env python3
"""
Results Analysis & Paper Figures (Section VI).

Generates:
  1. Pareto front plots (mAP vs p95 latency, mAP vs memory, etc.)
  2. Latency distribution histograms + tail diagnostics
  3. Discovered architecture analysis (what operators/widths were preferred)
  4. Ablation comparison tables (mean vs tail-risk, point vs probabilistic dominance)
  5. Baseline comparison tables in LaTeX format
  6. Search cost breakdown

Usage:
  python scripts/analyze_results.py --results-dir runs/tinas_search --output-dir paper_figures
"""
import argparse
import json
import logging
import sys
import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analyze")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not available. Skipping figure generation.")


def load_results(results_dir: Path) -> dict:
    """Load all checkpoint data."""
    data = {}

    # Pareto front
    pareto_files = sorted(results_dir.glob("checkpoints/*_pareto_raw.json"), reverse=True)
    if pareto_files:
        with open(pareto_files[0]) as f:
            data["pareto"] = json.load(f)

    # Summary
    summary_files = sorted(results_dir.glob("checkpoints/*_summary.json"), reverse=True)
    if summary_files:
        with open(summary_files[0]) as f:
            data["summary"] = json.load(f)

    # Baselines
    baseline_path = results_dir / "baselines" / "baseline_results.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            data["baselines"] = json.load(f)

    return data


# =============================================================================
# 1. Pareto Front Plots
# =============================================================================

def plot_pareto_front(pareto, baselines=None, output_dir=None):
    """Plot mAP vs p95 latency Pareto front with baselines."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # mAP vs p95 latency
    ax = axes[0]
    if pareto:
        mAPs = [1 - c["objectives"]["ferr"] for c in pareto]
        p95s = [c["aux"].get("p95_ms", c["objectives"].get("flat", 0)) for c in pareto]
        ax.scatter(p95s, mAPs, c="red", s=80, zorder=5, label="TINAS-ShipDet (Pareto)")

    if baselines:
        for b in baselines:
            ax.scatter(b.get("p95_ms", 0), b.get("mAP", 0),
                       s=60, marker="^", zorder=4, label=b["model"])

    ax.set_xlabel("p95 End-to-End Latency (ms)")
    ax.set_ylabel("mAP [0.50:0.95]")
    ax.set_title("Accuracy vs Tail Latency")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # mAP vs peak GPU memory
    ax = axes[1]
    if pareto:
        mems = [c["aux"].get("peak_gpu_mb", 0) for c in pareto]
        ax.scatter(mems, mAPs, c="red", s=80, zorder=5, label="TINAS-ShipDet")

    if baselines:
        for b in baselines:
            ax.scatter(b.get("peak_gpu_mb", 0), b.get("mAP", 0),
                       s=60, marker="^", label=b["model"])

    ax.set_xlabel("Peak GPU Memory (MB)")
    ax.set_ylabel("mAP [0.50:0.95]")
    ax.set_title("Accuracy vs Memory")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # mAP vs energy
    ax = axes[2]
    if pareto:
        energies = [c["aux"].get("mean_energy_j", -1) for c in pareto]
        valid = [(e, m) for e, m in zip(energies, mAPs) if e > 0]
        if valid:
            e_vals, m_vals = zip(*valid)
            ax.scatter(e_vals, m_vals, c="red", s=80, zorder=5, label="TINAS-ShipDet")

    ax.set_xlabel("Energy per Inference (J)")
    ax.set_ylabel("mAP [0.50:0.95]")
    ax.set_title("Accuracy vs Energy")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / "pareto_front.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(output_dir / "pareto_front.png", dpi=150, bbox_inches="tight")
    logger.info("Saved: pareto_front.pdf")


# =============================================================================
# 2. Latency Distribution Analysis (for "is tail-risk justified?" argument)
# =============================================================================

def plot_latency_distributions(pareto, baselines=None, output_dir=None):
    """Plot latency histograms for Pareto models and baselines."""
    if not HAS_MPL:
        return

    # Collect models with raw sample data
    models = []
    if pareto:
        for c in pareto[:3]:  # top-3 Pareto models
            samples = c.get("profiling_samples", [])
            if samples and isinstance(samples[0], dict):
                te2e = [s["t_e2e_ms"] for s in samples]
                models.append((f"TINAS-{c['uid'][:6]}", te2e))

    n = len(models)
    if n == 0:
        logger.warning("No raw profiling data available for distribution plots.")
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (label, te2e) in zip(axes, models):
        te2e = np.array(te2e)
        ax.hist(te2e, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
        ax.axvline(np.percentile(te2e, 50), color="green", ls="--", label=f"p50={np.percentile(te2e,50):.1f}")
        ax.axvline(np.percentile(te2e, 95), color="orange", ls="--", label=f"p95={np.percentile(te2e,95):.1f}")
        ax.axvline(np.percentile(te2e, 99), color="red", ls="--", label=f"p99={np.percentile(te2e,99):.1f}")
        ax.set_xlabel("End-to-End Latency (ms)")
        ax.set_ylabel("Count")
        ax.set_title(f"{label}\nCV={np.std(te2e)/np.mean(te2e):.4f}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / "latency_distributions.pdf", dpi=300, bbox_inches="tight")
    logger.info("Saved: latency_distributions.pdf")


# =============================================================================
# 3. Discovered Architecture Analysis (addresses "no architecture narrative")
# =============================================================================

def analyze_discovered_architectures(pareto, output_dir=None):
    """
    Analyze what the search found: which operators, widths, depths were
    preferred on the Pareto front. This gives transferable insight.
    """
    if not pareto:
        return

    logger.info("\n" + "=" * 60)
    logger.info("DISCOVERED ARCHITECTURE ANALYSIS")
    logger.info("=" * 60)

    # Collect genome stats
    bb_blocks = []
    widths = []
    depths = []
    strides = []
    down_ops = []
    neck_blocks = []
    head_types = []
    bottomup = []

    for c in pareto:
        g = c.get("genome", {})
        bb_blocks.append(g.get("bb_block", "?"))
        widths.append(g.get("bb_width", 0))
        depths.append(sum([
            g.get("bb_depth_s1", 0), g.get("bb_depth_s2", 0),
            g.get("bb_depth_s3", 0), g.get("bb_depth_s4", 0),
        ]))
        strides.append(g.get("bb_stride", "?"))
        down_ops.append(g.get("bb_down_op", "?"))
        neck_blocks.append(g.get("neck_block", "?"))
        head_types.append(g.get("head_type", "?"))
        bottomup.append(g.get("neck_bottomup", "?"))

    logger.info(f"Pareto front size: {len(pareto)}")
    logger.info(f"\nBackbone block distribution: {dict(Counter(bb_blocks))}")
    logger.info(f"Width multiplier: mean={np.mean(widths):.2f}, values={Counter(widths)}")
    logger.info(f"Total backbone depth: mean={np.mean(depths):.1f}, range=[{min(depths)}, {max(depths)}]")
    logger.info(f"Stride schedule: {dict(Counter(strides))}")
    logger.info(f"Downsample op: {dict(Counter(down_ops))}")
    logger.info(f"Neck block: {dict(Counter(neck_blocks))}")
    logger.info(f"Head type: {dict(Counter(head_types))}")
    logger.info(f"Bottom-up PAN enabled: {dict(Counter(bottomup))}")

    # Key insight extraction
    logger.info("\n--- KEY INSIGHTS ---")
    if Counter(strides).get("p2p3p4p5", 0) > len(pareto) * 0.5:
        logger.info("→ P2-P5 stride (small-ship mode) preferred on >50% of Pareto front.")
    if Counter(bb_blocks).most_common(1)[0][0] == "C2fGhost":
        logger.info("→ GhostConv backbone preferred: parameter efficiency dominates.")
    if Counter(bottomup).get(False, 0) > Counter(bottomup).get(True, 0):
        logger.info("→ Disabling bottom-up PAN saves latency without large accuracy loss.")

    # Save analysis
    if output_dir:
        analysis = {
            "pareto_size": len(pareto),
            "bb_block_dist": dict(Counter(bb_blocks)),
            "width_dist": dict(Counter(widths)),
            "depth_stats": {"mean": float(np.mean(depths)), "min": int(min(depths)), "max": int(max(depths))},
            "stride_dist": dict(Counter(strides)),
            "down_op_dist": dict(Counter(down_ops)),
            "neck_block_dist": dict(Counter(neck_blocks)),
            "head_type_dist": dict(Counter(head_types)),
            "bottomup_dist": dict(Counter(bottomup)),
        }
        with open(output_dir / "architecture_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        logger.info(f"Saved: architecture_analysis.json")


# =============================================================================
# 4. LaTeX Table Generation
# =============================================================================

def generate_latex_table(pareto, baselines, output_dir=None):
    """Generate LaTeX comparison table for the paper."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comparison of TINAS-ShipDet against baselines and SAR competitors "
        r"under the unified deployment-faithful protocol (batch-1, Jetson Orin Nano).}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Method & Params (M) & mAP$_{50{:}95}$ & AP$_{50}$ & "
        r"p50 (ms) & p95 (ms) & p99 (ms) & GPU (MB) & Energy (J) \\",
        r"\midrule",
    ]

    if baselines:
        lines.append(r"\multicolumn{9}{l}{\textit{Real-time detection baselines}} \\")
        for b in baselines:
            lines.append(
                f"{b['model']} & {b.get('params_M', '—'):.1f} & "
                f"{b.get('mAP', 0):.3f} & {b.get('mAP50', 0):.3f} & "
                f"{b.get('p50_ms', 0):.1f} & {b.get('p95_ms', 0):.1f} & "
                f"{b.get('p99_ms', 0):.1f} & {b.get('peak_gpu_mb', 0):.0f} & "
                f"— \\\\"
            )

    if pareto:
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{9}{l}{\textit{TINAS-ShipDet (Pareto models)}} \\")
        for c in pareto[:5]:
            g = c.get("genome", {})
            uid = c.get("uid", "?")[:8]
            lines.append(
                f"TINAS-{uid} & {c.get('model_info', {}).get('params_M', '—'):.1f} & "
                f"{c.get('aux', {}).get('mAP', 0):.3f} & — & "
                f"{c.get('aux', {}).get('p50_ms', 0):.1f} & "
                f"{c.get('aux', {}).get('p95_ms', 0):.1f} & "
                f"{c.get('aux', {}).get('p99_ms', 0):.1f} & "
                f"{c.get('aux', {}).get('peak_gpu_mb', 0):.0f} & "
                f"{c.get('aux', {}).get('mean_energy_j', 0):.3f} \\\\"
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    latex = "\n".join(lines)
    if output_dir:
        with open(output_dir / "main_results_table.tex", "w") as f:
            f.write(latex)
        logger.info("Saved: main_results_table.tex")

    return latex


# =============================================================================
# 5. Search Cost Summary
# =============================================================================

def print_search_cost(summary, output_dir=None):
    """Print search cost breakdown + hypervolume convergence."""
    cost = summary.get("cost_log", {})
    logger.info("\n" + "=" * 40)
    logger.info("SEARCH COST BREAKDOWN")
    logger.info("=" * 40)
    logger.info(f"Total wall-clock: {cost.get('total_wall_clock_h', '?'):.2f} hours")
    logger.info(f"  Training time: {cost.get('total_train_s', 0) / 3600:.2f} hours")
    logger.info(f"  Profiling time: {cost.get('total_profile_s', 0) / 3600:.2f} hours")

    gens = cost.get("generations", [])
    if gens:
        gen_times = [g["gen_time_s"] for g in gens]
        logger.info(f"  Avg time/generation: {np.mean(gen_times) / 60:.1f} min")

        hvs = [g.get("hypervolume", 0) for g in gens]
        if any(h > 0 for h in hvs):
            logger.info(f"\n  Hypervolume convergence:")
            for g in gens:
                logger.info(f"    Gen {g['gen']:2d}: HV={g.get('hypervolume',0):.6f} "
                            f"(Pareto={g['pareto_size']})")
            if len(hvs) >= 5:
                max_hv = max(hvs)
                if max_hv > 0 and all(h >= 0.98 * max_hv for h in hvs[-3:]):
                    logger.info(f"    → CONVERGED (last 3 gens within 2% of peak)")
                else:
                    logger.info(f"    → NOT converged — consider more generations")


def plot_hypervolume_convergence(summary, output_dir=None):
    """
    Hypervolume vs generation plot — THE convergence evidence.
    Answers: "300 out of 5M is enough?"
    """
    if not HAS_MPL:
        return
    cost = summary.get("cost_log", {})
    gens = cost.get("generations", [])
    hvs = [g.get("hypervolume", 0) for g in gens]
    if not gens or not any(h > 0 for h in hvs):
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    gen_nums = [g["gen"] for g in gens]
    ax1.plot(gen_nums, hvs, "o-", color="steelblue", linewidth=2, markersize=6)
    ax1.set_xlabel("Generation", fontsize=12)
    ax1.set_ylabel("Hypervolume Indicator", color="steelblue", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(gen_nums, [g["pareto_size"] for g in gens],
             "s--", color="coral", linewidth=1.5, markersize=5)
    ax2.set_ylabel("Pareto Front Size", color="coral", fontsize=12)
    ax1.set_title("Search Convergence", fontsize=13)
    fig.tight_layout()

    if output_dir:
        plt.savefig(output_dir / "hypervolume_convergence.pdf", dpi=300, bbox_inches="tight")
    logger.info("Saved: hypervolume_convergence.pdf")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Results Analysis")
    parser.add_argument("--results-dir", type=str, default="runs/tinas_search")
    parser.add_argument("--output-dir", type=str, default="paper_figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_results(results_dir)

    pareto = data.get("pareto", [])
    baselines = data.get("baselines", [])
    summary = data.get("summary", {})

    # Generate all analyses
    plot_pareto_front(pareto, baselines, output_dir)
    plot_latency_distributions(pareto, baselines, output_dir)
    plot_hypervolume_convergence(summary, output_dir)
    analyze_discovered_architectures(pareto, output_dir)
    generate_latex_table(pareto, baselines, output_dir)
    print_search_cost(summary, output_dir)

    logger.info(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
