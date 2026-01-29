"""
Plotting utilities for sweep analysis and visualization.

Generates plots in structure:
    plots/
        depth_d18/
            loss_curve_variants.png
            linear_probe_auroc_variants.png
            ...
        depth_d34/
        depth_d50/
        scaling_*.png
        *_summary.md
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Lazy imports for matplotlib
_plt = None
_matplotlib = None


def _get_matplotlib():
    """Lazy import matplotlib with Agg backend."""
    global _plt, _matplotlib
    if _plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _matplotlib = matplotlib
        _plt = plt
    return _plt


# Variant styling
VARIANT_COLORS = {
    "A": "#1f77b4",  # Blue - Baseline (ResNet-MLP + MLP)
    "B": "#ff7f0e",  # Orange - KAN Head (ResNet-MLP + ChebyKAN)
    "C": "#2ca02c",  # Green - KAN Backbone (ResNet-KAN + MLP)
    "D": "#d62728",  # Red - Full KAN (ResNet-KAN + ChebyKAN)
}

VARIANT_LABELS = {
    "A": "ResNet-MLP + MLP (Baseline)",
    "B": "ResNet-MLP + ChebyKAN",
    "C": "ResNet-KAN + MLP",
    "D": "ResNet-KAN + ChebyKAN",
}

VARIANT_MARKERS = {"A": "o", "B": "s", "C": "^", "D": "D"}
VARIANT_LINESTYLES = {"A": "-", "B": "--", "C": "-.", "D": ":"}


def load_metrics_jsonl(path: Path) -> List[dict]:
    """Load metrics from JSONL file."""
    if not path.exists():
        return []
    metrics = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))
    return metrics


def collect_sweep_data(sweep_dir: Path, variants: List[str], depths: List[int]) -> Dict:
    """
    Collect all metrics from sweep directory.
    
    Returns:
        dict mapping (variant, depth) -> list of epoch metrics
    """
    all_data = {}
    
    for variant in variants:
        for depth in depths:
            run_dir = sweep_dir / variant / f"d{depth}"
            metrics_path = run_dir / "metrics.jsonl"
            
            if not metrics_path.exists():
                # Try .json format for backwards compatibility
                metrics_path = run_dir / "metrics.json"
                if metrics_path.exists():
                    with open(metrics_path, "r") as f:
                        metrics_list = json.load(f)
                else:
                    continue
            else:
                metrics_list = load_metrics_jsonl(metrics_path)
            
            if metrics_list:
                all_data[(variant, depth)] = metrics_list
    
    return all_data


def _plot_metric_curve(ax, all_data: Dict, depth: int, metric_key: str, 
                       ylabel: str, title: str, variants: List[str]):
    """Plot a single metric curve for all variants at a given depth."""
    for variant in variants:
        key = (variant, depth)
        if key not in all_data:
            continue
        
        metrics_list = all_data[key]
        
        # Extract epochs and values
        epochs = []
        values = []
        for m in metrics_list:
            if metric_key in m:
                epochs.append(m.get("epoch", len(epochs) + 1))
                values.append(m[metric_key])
        
        if epochs:
            ax.plot(epochs, values, 
                    color=VARIANT_COLORS.get(variant, "gray"),
                    linestyle=VARIANT_LINESTYLES.get(variant, "-"),
                    marker=VARIANT_MARKERS.get(variant, "o"),
                    markersize=4, markevery=max(1, len(epochs)//10),
                    label=f"{variant}: {VARIANT_LABELS.get(variant, variant)[:20]}",
                    linewidth=1.5)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_depth_comparison(all_data: Dict, depth: int, output_dir: Path, 
                          variants: List[str] = None):
    """
    Generate all comparison plots for a single depth.
    
    Creates:
        depth_d{N}/
            loss_curve_variants.png
            linear_probe_auroc_variants.png
            alignment_variants.png
            uniformity_variants.png
            ap_curve_variants.png
            lr_curve_variants.png
            parameters.md
    """
    plt = _get_matplotlib()
    
    if variants is None:
        variants = ["A", "B", "C", "D"]
    
    depth_dir = output_dir / f"depth_d{depth}"
    depth_dir.mkdir(parents=True, exist_ok=True)
    
    # Define metrics to plot
    metrics_config = [
        ("train_loss", "Training Loss", "Loss Curve - All Variants"),
        ("linear_probe_auroc", "Linear Probe AUROC", "Linear Probe AUROC - All Variants"),
        ("alignment", "Alignment", "Alignment (↓ better) - All Variants"),
        ("uniformity", "Uniformity", "Uniformity (↓ better) - All Variants"),
        ("linear_probe_ap", "Average Precision", "Average Precision - All Variants"),
        ("lr", "Learning Rate", "Learning Rate Schedule"),
    ]
    
    for metric_key, ylabel, title in metrics_config:
        # Check if any variant has this metric
        has_data = False
        for variant in variants:
            key = (variant, depth)
            if key in all_data:
                if any(metric_key in m for m in all_data[key]):
                    has_data = True
                    break
        
        if not has_data:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        _plot_metric_curve(ax, all_data, depth, metric_key, ylabel, 
                          f"{title} (Depth {depth})", variants)
        
        # Filename
        filename = f"{metric_key.replace('_', '_')}_variants.png"
        if metric_key == "train_loss":
            filename = "loss_curve_variants.png"
        elif metric_key == "linear_probe_auroc":
            filename = "linear_probe_auroc_variants.png"
        elif metric_key == "linear_probe_ap":
            filename = "ap_curve_variants.png"
        elif metric_key == "lr":
            filename = "lr_curve_variants.png"
        elif metric_key == "alignment":
            filename = "alignment_variants.png"
        elif metric_key == "uniformity":
            filename = "uniformity_variants.png"
        
        plt.tight_layout()
        plt.savefig(depth_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
    
    # Generate parameters.md
    _generate_parameters_md(all_data, depth, depth_dir, variants)
    
    print(f"  Generated plots for depth {depth}")


def _generate_parameters_md(all_data: Dict, depth: int, output_dir: Path, 
                            variants: List[str]):
    """Generate parameters comparison markdown table."""
    lines = [
        f"# Parameter Comparison - Depth {depth}\n",
        "| Variant | Backbone | Head | Encoder Params | Projector Params | Total Params |",
        "|---------|----------|------|----------------|------------------|--------------|",
    ]
    
    variant_info = {
        "A": ("ResNet-MLP", "MLP"),
        "B": ("ResNet-MLP", "ChebyKAN"),
        "C": ("ResNet-KAN", "MLP"),
        "D": ("ResNet-KAN", "ChebyKAN"),
    }
    
    for variant in variants:
        key = (variant, depth)
        if key not in all_data:
            continue
        
        metrics = all_data[key]
        # Try to find param counts in metrics
        encoder_params = 0
        projector_params = 0
        total_params = 0
        
        for m in metrics:
            if "param_counts" in m:
                pc = m["param_counts"]
                encoder_params = pc.get("encoder", 0)
                projector_params = pc.get("projector", 0)
                total_params = pc.get("total", 0)
                break
        
        backbone, head = variant_info.get(variant, ("?", "?"))
        lines.append(
            f"| {variant} | {backbone} | {head} | "
            f"{encoder_params:,} | {projector_params:,} | {total_params:,} |"
        )
    
    with open(output_dir / "parameters.md", "w") as f:
        f.write("\n".join(lines))


def plot_scaling_analysis(all_data: Dict, output_dir: Path, depths: List[int],
                          variants: List[str] = None):
    """
    Generate scaling analysis plots (cross-depth comparisons).
    
    Creates:
        scaling_auc_vs_params.png
        scaling_auc_vs_time.png
        param_counts_by_variant.png
        efficiency_by_variant.png
        auc_by_variant.png
    """
    plt = _get_matplotlib()
    
    if variants is None:
        variants = ["A", "B", "C", "D"]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect final metrics for each (variant, depth)
    summary_data = []
    for variant in variants:
        for depth in depths:
            key = (variant, depth)
            if key not in all_data:
                continue
            
            metrics = all_data[key]
            if not metrics:
                continue
            
            # Get final training entry
            train_entries = [m for m in metrics if "train_loss" in m]
            eval_entries = [m for m in metrics if "linear_probe_auroc" in m]
            
            if not train_entries:
                continue
            
            final_train = train_entries[-1]
            final_eval = eval_entries[-1] if eval_entries else {}
            
            total_time = sum(m.get("epoch_time_sec", 0) for m in metrics)
            
            # Get param counts
            total_params = 0
            for m in metrics:
                if "param_counts" in m:
                    total_params = m["param_counts"].get("total", 0)
                    break
            
            summary_data.append({
                "variant": variant,
                "depth": depth,
                "total_params": total_params,
                "total_time_min": total_time / 60,
                "final_loss": final_train.get("train_loss", 0),
                "auroc": final_eval.get("linear_probe_auroc", 0),
                "ap": final_eval.get("linear_probe_ap", 0),
                "alignment": final_eval.get("alignment", 0),
                "uniformity": final_eval.get("uniformity", 0),
            })
    
    if not summary_data:
        print("  No summary data available for scaling plots")
        return
    
    # 1. AUROC vs Parameters
    fig, ax = plt.subplots(figsize=(10, 7))
    for variant in variants:
        vdata = [s for s in summary_data if s["variant"] == variant]
        if vdata:
            params = [s["total_params"] / 1e6 for s in vdata]
            aucs = [s["auroc"] for s in vdata]
            ax.scatter(params, aucs, 
                      color=VARIANT_COLORS.get(variant, "gray"),
                      marker=VARIANT_MARKERS.get(variant, "o"),
                      s=100, label=VARIANT_LABELS.get(variant, variant))
            ax.plot(params, aucs, 
                   color=VARIANT_COLORS.get(variant, "gray"),
                   linestyle="--", alpha=0.5)
            # Annotate depths
            for s in vdata:
                ax.annotate(f"d{s['depth']}", 
                           (s["total_params"]/1e6, s["auroc"]),
                           textcoords="offset points", xytext=(5,5), fontsize=8)
    
    ax.set_xlabel("Total Parameters (M)")
    ax.set_ylabel("Linear Probe AUROC")
    ax.set_title("Scaling: AUROC vs Model Size")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "scaling_auc_vs_params.png", dpi=150)
    plt.close()
    
    # 2. AUROC vs Time
    fig, ax = plt.subplots(figsize=(10, 7))
    for variant in variants:
        vdata = [s for s in summary_data if s["variant"] == variant]
        if vdata:
            times = [s["total_time_min"] for s in vdata]
            aucs = [s["auroc"] for s in vdata]
            ax.scatter(times, aucs,
                      color=VARIANT_COLORS.get(variant, "gray"),
                      marker=VARIANT_MARKERS.get(variant, "o"),
                      s=100, label=VARIANT_LABELS.get(variant, variant))
            ax.plot(times, aucs,
                   color=VARIANT_COLORS.get(variant, "gray"),
                   linestyle="--", alpha=0.5)
            for s in vdata:
                ax.annotate(f"d{s['depth']}",
                           (s["total_time_min"], s["auroc"]),
                           textcoords="offset points", xytext=(5,5), fontsize=8)
    
    ax.set_xlabel("Training Time (minutes)")
    ax.set_ylabel("Linear Probe AUROC")
    ax.set_title("Scaling: AUROC vs Training Time")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "scaling_auc_vs_time.png", dpi=150)
    plt.close()
    
    # 3. Bar chart: Final AUROC by variant
    fig, ax = plt.subplots(figsize=(12, 6))
    x_labels = []
    aucs = []
    colors = []
    
    for variant in variants:
        for depth in depths:
            vdata = [s for s in summary_data if s["variant"] == variant and s["depth"] == depth]
            if vdata:
                x_labels.append(f"{variant}/d{depth}")
                aucs.append(vdata[0]["auroc"])
                colors.append(VARIANT_COLORS.get(variant, "gray"))
    
    x = np.arange(len(x_labels))
    ax.bar(x, aucs, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("Linear Probe AUROC")
    ax.set_title("Final AUROC by Variant and Depth")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "auc_by_variant.png", dpi=150)
    plt.close()
    
    # 4. Efficiency plot (AUROC / params)
    fig, ax = plt.subplots(figsize=(12, 6))
    x_labels = []
    efficiency = []
    colors = []
    
    for variant in variants:
        for depth in depths:
            vdata = [s for s in summary_data if s["variant"] == variant and s["depth"] == depth]
            if vdata and vdata[0]["total_params"] > 0:
                x_labels.append(f"{variant}/d{depth}")
                eff = vdata[0]["auroc"] / (vdata[0]["total_params"] / 1e6)  # AUROC per million params
                efficiency.append(eff)
                colors.append(VARIANT_COLORS.get(variant, "gray"))
    
    x = np.arange(len(x_labels))
    ax.bar(x, efficiency, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("AUROC / Parameters (M)")
    ax.set_title("Parameter Efficiency by Variant")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "efficiency_by_variant.png", dpi=150)
    plt.close()
    
    # 5. Param counts by variant
    fig, ax = plt.subplots(figsize=(12, 6))
    x_labels = []
    params = []
    colors = []
    
    for variant in variants:
        for depth in depths:
            vdata = [s for s in summary_data if s["variant"] == variant and s["depth"] == depth]
            if vdata:
                x_labels.append(f"{variant}/d{depth}")
                params.append(vdata[0]["total_params"] / 1e6)
                colors.append(VARIANT_COLORS.get(variant, "gray"))
    
    x = np.arange(len(x_labels))
    ax.bar(x, params, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("Total Parameters (M)")
    ax.set_title("Model Size by Variant and Depth")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "param_counts_by_variant.png", dpi=150)
    plt.close()
    
    # Generate scaling_summary.md
    _generate_scaling_summary_md(summary_data, output_dir, variants, depths)
    
    # Generate plot_summary.md
    _generate_plot_summary_md(output_dir, depths)
    
    print(f"  Generated scaling analysis plots")


def _generate_scaling_summary_md(summary_data: List[dict], output_dir: Path,
                                  variants: List[str], depths: List[int]):
    """Generate comprehensive summary markdown."""
    lines = [
        "# Sweep Results Summary\n",
        "## Final Metrics by Variant and Depth\n",
        "| Variant | Depth | Params (M) | Time (min) | Loss | AUROC | AP | Align | Uniform |",
        "|---------|-------|------------|------------|------|-------|-----|-------|---------|",
    ]
    
    for variant in variants:
        for depth in depths:
            vdata = [s for s in summary_data if s["variant"] == variant and s["depth"] == depth]
            if not vdata:
                continue
            s = vdata[0]
            lines.append(
                f"| {variant} | {depth} | {s['total_params']/1e6:.2f} | "
                f"{s['total_time_min']:.1f} | {s['final_loss']:.4f} | "
                f"{s['auroc']:.4f} | {s['ap']:.4f} | "
                f"{s['alignment']:.4f} | {s['uniformity']:.4f} |"
            )
    
    lines.extend([
        "",
        "## Variant Descriptions",
        "- **A**: ResNet-MLP backbone + MLP projection head (Baseline)",
        "- **B**: ResNet-MLP backbone + ChebyKAN projection head",
        "- **C**: ResNet-KAN backbone + MLP projection head",
        "- **D**: ResNet-KAN backbone + ChebyKAN projection head",
    ])
    
    with open(output_dir / "scaling_summary.md", "w") as f:
        f.write("\n".join(lines))


def _generate_plot_summary_md(output_dir: Path, depths: List[int]):
    """Generate plot descriptions markdown."""
    lines = [
        "# Plot Summary\n",
        "## Per-Depth Comparison Plots",
        "",
    ]
    
    for depth in depths:
        lines.extend([
            f"### Depth {depth} (`depth_d{depth}/`)",
            f"- `loss_curve_variants.png`: Training loss over epochs for all variants",
            f"- `linear_probe_auroc_variants.png`: Linear evaluation AUROC over epochs",
            f"- `alignment_variants.png`: Representation alignment (lower = better)",
            f"- `uniformity_variants.png`: Representation uniformity (lower = better)",
            f"- `ap_curve_variants.png`: Average precision over epochs",
            f"- `lr_curve_variants.png`: Learning rate schedule",
            f"- `parameters.md`: Parameter count comparison table",
            "",
        ])
    
    lines.extend([
        "## Scaling Analysis Plots",
        "- `scaling_auc_vs_params.png`: AUROC vs model size (scatter with depth annotations)",
        "- `scaling_auc_vs_time.png`: AUROC vs training time",
        "- `auc_by_variant.png`: Bar chart of final AUROC",
        "- `efficiency_by_variant.png`: AUROC per million parameters",
        "- `param_counts_by_variant.png`: Model sizes comparison",
        "- `scaling_summary.md`: Full results table",
    ])
    
    with open(output_dir / "plot_summary.md", "w") as f:
        f.write("\n".join(lines))


def generate_sweep_readme(sweep_dir: Path, variants: List[str], depths: List[int],
                          epochs: int, eval_every: int):
    """Generate README.md for sweep directory."""
    lines = [
        f"# Sweep Experiment Results\n",
        f"**Generated**: {sweep_dir.name.replace('_sweep_', '')}",
        "",
        "## Configuration",
        f"- **Variants**: {', '.join(variants)}",
        f"- **Depths**: {', '.join(map(str, depths))}",
        f"- **Epochs**: {epochs}",
        f"- **Eval Every**: {eval_every} epochs",
        "",
        "## Variants",
        "| Variant | Backbone | Projection Head |",
        "|---------|----------|-----------------|",
        "| A | ResNet-MLP | MLP |",
        "| B | ResNet-MLP | ChebyKAN |",
        "| C | ResNet-KAN | MLP |",
        "| D | ResNet-KAN | ChebyKAN |",
        "",
        "## Directory Structure",
        "```",
        f"{sweep_dir.name}/",
        "├── README.md",
        "├── plots/",
    ]
    
    for depth in depths:
        lines.append(f"│   ├── depth_d{depth}/")
    
    lines.extend([
        "│   ├── scaling_*.png",
        "│   └── *_summary.md",
    ])
    
    for variant in variants:
        lines.append(f"├── {variant}/")
        for depth in depths:
            lines.append(f"│   └── d{depth}/")
    
    lines.extend([
        "```",
        "",
        "## Results",
        "See `plots/scaling_summary.md` for full results table.",
    ])
    
    with open(sweep_dir / "README.md", "w") as f:
        f.write("\n".join(lines))


def generate_all_sweep_plots(sweep_dir: Path, variants: List[str] = None,
                              depths: List[int] = None):
    """
    Generate all sweep plots from collected metrics.
    
    Args:
        sweep_dir: Path to sweep directory (e.g., results/_sweep_timestamp)
        variants: List of variants to include (default: A, B, C, D)
        depths: List of depths to include (default: auto-detect)
    """
    if variants is None:
        variants = ["A", "B", "C", "D"]
    
    # Auto-detect depths if not provided
    if depths is None:
        depths = []
        for v in variants:
            variant_dir = sweep_dir / v
            if variant_dir.exists():
                for d in variant_dir.iterdir():
                    if d.is_dir() and d.name.startswith("d"):
                        try:
                            depth = int(d.name[1:])
                            if depth not in depths:
                                depths.append(depth)
                        except ValueError:
                            pass
        depths = sorted(depths)
    
    if not depths:
        print("No depths found in sweep directory")
        return
    
    print(f"Generating plots for variants {variants}, depths {depths}")
    
    # Collect all data
    all_data = collect_sweep_data(sweep_dir, variants, depths)
    
    if not all_data:
        print("No data found in sweep directory")
        return
    
    plots_dir = sweep_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate per-depth comparison plots
    for depth in depths:
        plot_depth_comparison(all_data, depth, plots_dir, variants)
    
    # Generate scaling analysis plots
    plot_scaling_analysis(all_data, plots_dir, depths, variants)
    
    print(f"All plots saved to {plots_dir}")
