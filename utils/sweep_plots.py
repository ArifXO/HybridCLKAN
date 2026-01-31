"""
Sweep plot generation utilities.

Handles:
- Per-depth comparison plots (loss, AUROC, alignment, uniformity)
- Scaling analysis plots (AUROC vs params, AUROC vs time, etc.)
- Summary markdown generation

Supports:
- Single-seed mode: sweep_dir/{variant}/d{depth}/
- Multi-seed mode: sweep_dir/{variant}/d{depth}/seed_*/
  - Creates plots/seed_{N}/ for individual seed plots
  - Creates plots/combined/ for aggregated mean±std plots
- Hidden-dim mode: sweep_dir/{variant}/d{depth}/h{hidden_dim}/
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Lazy imports for matplotlib
_plt = None


def _get_matplotlib():
    """Lazy import matplotlib with Agg backend."""
    global _plt
    if _plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
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
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return metrics


def _aggregate_seed_metrics(seed_metrics_list: List[List[dict]]) -> List[dict]:
    """
    Aggregate metrics across multiple seeds by epoch.
    
    Args:
        seed_metrics_list: List of metrics lists, one per seed
        
    Returns:
        Aggregated metrics list with mean and std for each metric
    """
    if not seed_metrics_list:
        return []
    if len(seed_metrics_list) == 1:
        return seed_metrics_list[0]
    
    # Collect all metrics by epoch
    epoch_data = {}
    for seed_metrics in seed_metrics_list:
        for m in seed_metrics:
            epoch = m.get("epoch", 0)
            if epoch not in epoch_data:
                epoch_data[epoch] = []
            epoch_data[epoch].append(m)
    
    # Aggregate each epoch
    aggregated = []
    numeric_keys = ["train_loss", "lr", "epoch_time_sec", 
                    "linear_probe_auroc", "linear_probe_ap", 
                    "alignment", "uniformity"]
    
    for epoch in sorted(epoch_data.keys()):
        epoch_entries = epoch_data[epoch]
        agg_entry = {"epoch": epoch, "n_seeds": len(epoch_entries)}
        
        # Copy param_counts from first entry (same for all seeds)
        if "param_counts" in epoch_entries[0]:
            agg_entry["param_counts"] = epoch_entries[0]["param_counts"]
        if "projector_build_info" in epoch_entries[0]:
            agg_entry["projector_build_info"] = epoch_entries[0]["projector_build_info"]
        
        # Aggregate numeric metrics
        for key in numeric_keys:
            values = [e.get(key) for e in epoch_entries if key in e]
            if values:
                agg_entry[key] = float(np.mean(values))
                agg_entry[f"{key}_std"] = float(np.std(values))
        
        aggregated.append(agg_entry)
    
    return aggregated


def detect_sweep_mode(sweep_dir: Path, variants: List[str], depths: List[int]) -> Tuple[str, List[str]]:
    """
    Detect the sweep mode based on directory structure.
    
    Returns:
        tuple: (mode, seeds_or_hidden_dims)
            - ("single", []) 
            - ("multi_seed", ["seed_41", "seed_42", ...])
            - ("hidden_dim", [256, 512, ...])
    """
    for variant in variants:
        for depth in depths:
            run_dir = sweep_dir / variant / f"d{depth}"
            if not run_dir.exists():
                continue
            
            # Check for seed_* directories
            seed_dirs = sorted([d.name for d in run_dir.glob("seed_*") if d.is_dir()])
            if seed_dirs:
                return "multi_seed", seed_dirs
            
            # Check for h* directories
            hidden_dirs = sorted(run_dir.glob("h*"))
            if hidden_dirs:
                hidden_dims = []
                for hd in hidden_dirs:
                    try:
                        hidden_dims.append(int(hd.name[1:]))
                    except ValueError:
                        pass
                if hidden_dims:
                    return "hidden_dim", sorted(hidden_dims)
            
            # Check for direct metrics.jsonl
            if (run_dir / "metrics.jsonl").exists():
                return "single", []
    
    return "single", []


def collect_sweep_data_single(sweep_dir: Path, variants: List[str], depths: List[int]) -> Dict:
    """Collect metrics for single-seed mode."""
    all_data = {}
    for variant in variants:
        for depth in depths:
            metrics_path = sweep_dir / variant / f"d{depth}" / "metrics.jsonl"
            if metrics_path.exists():
                all_data[(variant, depth)] = load_metrics_jsonl(metrics_path)
    return all_data


def collect_sweep_data_per_seed(sweep_dir: Path, variants: List[str], depths: List[int], 
                                 seed_name: str) -> Dict:
    """Collect metrics for a specific seed in multi-seed mode."""
    all_data = {}
    for variant in variants:
        for depth in depths:
            metrics_path = sweep_dir / variant / f"d{depth}" / seed_name / "metrics.jsonl"
            if metrics_path.exists():
                all_data[(variant, depth)] = load_metrics_jsonl(metrics_path)
    return all_data


def collect_sweep_data_combined(sweep_dir: Path, variants: List[str], depths: List[int],
                                 seed_names: List[str]) -> Dict:
    """Collect and aggregate metrics across all seeds."""
    all_data = {}
    for variant in variants:
        for depth in depths:
            seed_metrics_list = []
            for seed_name in seed_names:
                metrics_path = sweep_dir / variant / f"d{depth}" / seed_name / "metrics.jsonl"
                if metrics_path.exists():
                    seed_metrics_list.append(load_metrics_jsonl(metrics_path))
            
            if seed_metrics_list:
                all_data[(variant, depth)] = _aggregate_seed_metrics(seed_metrics_list)
    return all_data


def collect_sweep_data_hidden_dim(sweep_dir: Path, variants: List[str], depths: List[int],
                                   hidden_dims: List[int]) -> Dict:
    """Collect metrics for hidden-dim sweep mode."""
    all_data = {}
    for variant in variants:
        for depth in depths:
            for hidden_dim in hidden_dims:
                metrics_path = sweep_dir / variant / f"d{depth}" / f"h{hidden_dim}" / "metrics.jsonl"
                if metrics_path.exists():
                    all_data[(variant, depth, hidden_dim)] = load_metrics_jsonl(metrics_path)
    return all_data


# Keep backward-compatible function
def collect_sweep_data(sweep_dir: Path, variants: List[str], depths: List[int]) -> Tuple[Dict, str]:
    """
    Collect all metrics from sweep directory.
    
    Returns:
        tuple: (data_dict, sweep_mode)
    """
    sweep_mode, extra = detect_sweep_mode(sweep_dir, variants, depths)
    
    if sweep_mode == "multi_seed":
        # Return aggregated data for backward compatibility
        all_data = collect_sweep_data_combined(sweep_dir, variants, depths, extra)
        return all_data, sweep_mode
    elif sweep_mode == "hidden_dim":
        all_data = collect_sweep_data_hidden_dim(sweep_dir, variants, depths, extra)
        return all_data, sweep_mode
    else:
        all_data = collect_sweep_data_single(sweep_dir, variants, depths)
        return all_data, sweep_mode


def _plot_metric_curve(ax, all_data: Dict, depth: int, metric_key: str, 
                       ylabel: str, title: str, variants: List[str],
                       hidden_dim: int = None, show_std: bool = True):
    """Plot a single metric curve for all variants at a given depth."""
    for variant in variants:
        # Build key based on whether hidden_dim is specified
        if hidden_dim is not None:
            key = (variant, depth, hidden_dim)
        else:
            key = (variant, depth)
        
        if key not in all_data:
            continue
        
        metrics_list = all_data[key]
        
        # Extract epochs and values
        epochs = []
        values = []
        stds = []
        std_key = f"{metric_key}_std"
        
        for m in metrics_list:
            if metric_key in m:
                epochs.append(m.get("epoch", len(epochs) + 1))
                values.append(m[metric_key])
                stds.append(m.get(std_key, 0))
        
        if epochs:
            color = VARIANT_COLORS.get(variant, "gray")
            epochs = np.array(epochs)
            values = np.array(values)
            stds = np.array(stds)
            
            # Plot mean line
            ax.plot(epochs, values, 
                    color=color,
                    linestyle=VARIANT_LINESTYLES.get(variant, "-"),
                    marker=VARIANT_MARKERS.get(variant, "o"),
                    markersize=4, markevery=max(1, len(epochs)//10),
                    label=f"{variant}: {VARIANT_LABELS.get(variant, variant)[:20]}",
                    linewidth=1.5)
            
            # Plot shaded std band if multi-seed and show_std enabled
            if show_std and np.any(stds > 0):
                ax.fill_between(epochs, values - stds, values + stds,
                               color=color, alpha=0.2)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_depth_comparison(all_data: Dict, depth: int, output_dir: Path, 
                          variants: List[str], sweep_mode: str = "single",
                          hidden_dim: int = None, show_std: bool = True,
                          title_suffix: str = ""):
    """
    Generate all comparison plots for a single depth.
    
    Creates:
        depth_d{N}/ (or depth_d{N}_h{H}/)
            loss_curve_variants.png
            linear_probe_auroc_variants.png
            alignment_variants.png
            uniformity_variants.png
            ap_curve_variants.png
            lr_curve_variants.png
            parameters.md
    """
    plt = _get_matplotlib()
    
    # Create output directory name
    if hidden_dim is not None:
        depth_dir = output_dir / f"depth_d{depth}_h{hidden_dim}"
    else:
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
            if hidden_dim is not None:
                key = (variant, depth, hidden_dim)
            else:
                key = (variant, depth)
            if key in all_data:
                if any(metric_key in m for m in all_data[key]):
                    has_data = True
                    break
        
        if not has_data:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        full_title_suffix = f" (Depth {depth})"
        if hidden_dim is not None:
            full_title_suffix = f" (Depth {depth}, Hidden {hidden_dim})"
        if title_suffix:
            full_title_suffix += f" {title_suffix}"
        
        _plot_metric_curve(ax, all_data, depth, metric_key, ylabel, 
                          f"{title}{full_title_suffix}", variants, hidden_dim, show_std)
        
        # Filename
        filename = f"{metric_key}_variants.png"
        if metric_key == "train_loss":
            filename = "loss_curve_variants.png"
        elif metric_key == "linear_probe_auroc":
            filename = "linear_probe_auroc_variants.png"
        elif metric_key == "linear_probe_ap":
            filename = "ap_curve_variants.png"
        elif metric_key == "lr":
            filename = "lr_curve_variants.png"
        
        plt.tight_layout()
        plt.savefig(depth_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
    
    # Generate parameters.md
    _generate_parameters_md(all_data, depth, depth_dir, variants, hidden_dim)
    
    suffix = f" (h{hidden_dim})" if hidden_dim else ""
    print(f"    Generated plots for depth {depth}{suffix}")


def _generate_parameters_md(all_data: Dict, depth: int, output_dir: Path, 
                            variants: List[str], hidden_dim: int = None):
    """Generate parameters comparison markdown table."""
    title = f"# Parameter Comparison - Depth {depth}"
    if hidden_dim is not None:
        title = f"# Parameter Comparison - Depth {depth}, Hidden {hidden_dim}"
    
    lines = [
        f"{title}\n",
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
        if hidden_dim is not None:
            key = (variant, depth, hidden_dim)
        else:
            key = (variant, depth)
        
        if key not in all_data:
            continue
        
        metrics = all_data[key]
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
    
    with open(output_dir / "parameters.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_scaling_analysis(all_data: Dict, output_dir: Path, depths: List[int],
                          variants: List[str], sweep_mode: str = "single",
                          hidden_dims: List[int] = None, title_suffix: str = ""):
    """
    Generate scaling analysis plots (cross-depth comparisons).
    """
    plt = _get_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect summary data
    summary_data = []
    
    if sweep_mode == "hidden_dim" and hidden_dims:
        for variant in variants:
            for depth in depths:
                for hidden_dim in hidden_dims:
                    key = (variant, depth, hidden_dim)
                    if key not in all_data:
                        continue
                    entry = _extract_summary_entry(all_data[key], variant, depth)
                    if entry:
                        entry["hidden_dim"] = hidden_dim
                        summary_data.append(entry)
    else:
        for variant in variants:
            for depth in depths:
                key = (variant, depth)
                if key not in all_data:
                    continue
                entry = _extract_summary_entry(all_data[key], variant, depth)
                if entry:
                    summary_data.append(entry)
    
    if not summary_data:
        print("    No summary data available for scaling plots")
        return
    
    has_multi_seed = any(s.get('n_seeds', 1) > 1 for s in summary_data)
    
    # Generate plots
    _plot_auc_vs_params(summary_data, output_dir, variants, has_multi_seed, title_suffix)
    _plot_auc_vs_time(summary_data, output_dir, variants, has_multi_seed, title_suffix)
    _plot_auc_bar_chart(summary_data, output_dir, variants, depths, has_multi_seed, sweep_mode, hidden_dims, title_suffix)
    _plot_efficiency_chart(summary_data, output_dir, variants, depths, sweep_mode, hidden_dims)
    _plot_param_counts(summary_data, output_dir, variants, depths, sweep_mode, hidden_dims)
    
    # Generate summary markdown
    _generate_scaling_summary_md(summary_data, output_dir, variants, depths, sweep_mode, hidden_dims)
    _generate_plot_summary_md(output_dir, depths, sweep_mode, hidden_dims)
    
    print(f"    Generated scaling analysis plots")


def _extract_summary_entry(metrics: List[dict], variant: str, depth: int) -> Optional[dict]:
    """Extract summary statistics from metrics list."""
    if not metrics:
        return None
    
    train_entries = [m for m in metrics if "train_loss" in m]
    eval_entries = [m for m in metrics if "linear_probe_auroc" in m]
    
    if not train_entries:
        return None
    
    final_train = train_entries[-1]
    final_eval = eval_entries[-1] if eval_entries else {}
    
    total_time = sum(m.get("epoch_time_sec", 0) for m in metrics)
    
    total_params = 0
    for m in metrics:
        if "param_counts" in m:
            total_params = m["param_counts"].get("total", 0)
            break
    
    return {
        "variant": variant,
        "depth": depth,
        "n_seeds": final_train.get("n_seeds", 1),
        "total_params": total_params,
        "total_time_min": total_time / 60,
        "final_loss": final_train.get("train_loss", 0),
        "final_loss_std": final_train.get("train_loss_std", 0),
        "auroc": final_eval.get("linear_probe_auroc", 0),
        "auroc_std": final_eval.get("linear_probe_auroc_std", 0),
        "ap": final_eval.get("linear_probe_ap", 0),
        "ap_std": final_eval.get("linear_probe_ap_std", 0),
        "alignment": final_eval.get("alignment", 0),
        "alignment_std": final_eval.get("alignment_std", 0),
        "uniformity": final_eval.get("uniformity", 0),
        "uniformity_std": final_eval.get("uniformity_std", 0),
    }


def _plot_auc_vs_params(summary_data: List[dict], output_dir: Path, 
                        variants: List[str], has_multi_seed: bool, title_suffix: str = ""):
    """Plot AUROC vs Parameters."""
    plt = _get_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for variant in variants:
        vdata = [s for s in summary_data if s["variant"] == variant]
        if vdata:
            params = [s["total_params"] / 1e6 for s in vdata]
            aucs = [s["auroc"] for s in vdata]
            auc_stds = [s.get("auroc_std", 0) for s in vdata]
            
            ax.scatter(params, aucs, 
                      color=VARIANT_COLORS.get(variant, "gray"),
                      marker=VARIANT_MARKERS.get(variant, "o"),
                      s=100, label=VARIANT_LABELS.get(variant, variant))
            ax.plot(params, aucs, 
                   color=VARIANT_COLORS.get(variant, "gray"),
                   linestyle="--", alpha=0.5)
            
            if has_multi_seed and any(s > 0 for s in auc_stds):
                ax.errorbar(params, aucs, yerr=auc_stds, 
                           color=VARIANT_COLORS.get(variant, "gray"),
                           fmt='none', capsize=3, alpha=0.7)
            
            for s in vdata:
                label = f"d{s['depth']}"
                if "hidden_dim" in s:
                    label = f"d{s['depth']}/h{s['hidden_dim']}"
                ax.annotate(label, 
                           (s["total_params"]/1e6, s["auroc"]),
                           textcoords="offset points", xytext=(5,5), fontsize=8)
    
    ax.set_xlabel("Total Parameters (M)")
    ax.set_ylabel("Linear Probe AUROC")
    title = "Scaling: AUROC vs Model Size"
    if has_multi_seed:
        title += " (mean±std)"
    if title_suffix:
        title += f" {title_suffix}"
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "scaling_auc_vs_params.png", dpi=150)
    plt.close()


def _plot_auc_vs_time(summary_data: List[dict], output_dir: Path, 
                      variants: List[str], has_multi_seed: bool, title_suffix: str = ""):
    """Plot AUROC vs Training Time."""
    plt = _get_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for variant in variants:
        vdata = [s for s in summary_data if s["variant"] == variant]
        if vdata:
            times = [s["total_time_min"] for s in vdata]
            aucs = [s["auroc"] for s in vdata]
            auc_stds = [s.get("auroc_std", 0) for s in vdata]
            
            ax.scatter(times, aucs,
                      color=VARIANT_COLORS.get(variant, "gray"),
                      marker=VARIANT_MARKERS.get(variant, "o"),
                      s=100, label=VARIANT_LABELS.get(variant, variant))
            ax.plot(times, aucs,
                   color=VARIANT_COLORS.get(variant, "gray"),
                   linestyle="--", alpha=0.5)
            
            if has_multi_seed and any(s > 0 for s in auc_stds):
                ax.errorbar(times, aucs, yerr=auc_stds,
                           color=VARIANT_COLORS.get(variant, "gray"),
                           fmt='none', capsize=3, alpha=0.7)
            
            for s in vdata:
                label = f"d{s['depth']}"
                if "hidden_dim" in s:
                    label = f"d{s['depth']}/h{s['hidden_dim']}"
                ax.annotate(label,
                           (s["total_time_min"], s["auroc"]),
                           textcoords="offset points", xytext=(5,5), fontsize=8)
    
    ax.set_xlabel("Training Time (minutes)")
    ax.set_ylabel("Linear Probe AUROC")
    title = "Scaling: AUROC vs Training Time"
    if has_multi_seed:
        title += " (mean±std)"
    if title_suffix:
        title += f" {title_suffix}"
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "scaling_auc_vs_time.png", dpi=150)
    plt.close()


def _plot_auc_bar_chart(summary_data: List[dict], output_dir: Path, 
                        variants: List[str], depths: List[int],
                        has_multi_seed: bool, sweep_mode: str,
                        hidden_dims: List[int] = None, title_suffix: str = ""):
    """Bar chart of final AUROC by variant."""
    plt = _get_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_labels = []
    aucs = []
    auc_stds = []
    colors = []
    
    if sweep_mode == "hidden_dim" and hidden_dims:
        for variant in variants:
            for depth in depths:
                for hidden_dim in hidden_dims:
                    vdata = [s for s in summary_data 
                            if s["variant"] == variant and s["depth"] == depth 
                            and s.get("hidden_dim") == hidden_dim]
                    if vdata:
                        x_labels.append(f"{variant}/d{depth}/h{hidden_dim}")
                        aucs.append(vdata[0]["auroc"])
                        auc_stds.append(vdata[0].get("auroc_std", 0))
                        colors.append(VARIANT_COLORS.get(variant, "gray"))
    else:
        for variant in variants:
            for depth in depths:
                vdata = [s for s in summary_data if s["variant"] == variant and s["depth"] == depth]
                if vdata:
                    x_labels.append(f"{variant}/d{depth}")
                    aucs.append(vdata[0]["auroc"])
                    auc_stds.append(vdata[0].get("auroc_std", 0))
                    colors.append(VARIANT_COLORS.get(variant, "gray"))
    
    x = np.arange(len(x_labels))
    yerr = auc_stds if has_multi_seed and any(s > 0 for s in auc_stds) else None
    ax.bar(x, aucs, color=colors, yerr=yerr, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("Linear Probe AUROC")
    title = "Final AUROC by Variant and Depth"
    if has_multi_seed:
        title += " (mean±std)"
    if title_suffix:
        title += f" {title_suffix}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "auc_by_variant.png", dpi=150)
    plt.close()


def _plot_efficiency_chart(summary_data: List[dict], output_dir: Path, 
                           variants: List[str], depths: List[int],
                           sweep_mode: str, hidden_dims: List[int] = None):
    """Plot parameter efficiency (AUROC / params)."""
    plt = _get_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_labels = []
    efficiency = []
    colors = []
    
    if sweep_mode == "hidden_dim" and hidden_dims:
        for variant in variants:
            for depth in depths:
                for hidden_dim in hidden_dims:
                    vdata = [s for s in summary_data 
                            if s["variant"] == variant and s["depth"] == depth 
                            and s.get("hidden_dim") == hidden_dim]
                    if vdata and vdata[0]["total_params"] > 0:
                        x_labels.append(f"{variant}/d{depth}/h{hidden_dim}")
                        eff = vdata[0]["auroc"] / (vdata[0]["total_params"] / 1e6)
                        efficiency.append(eff)
                        colors.append(VARIANT_COLORS.get(variant, "gray"))
    else:
        for variant in variants:
            for depth in depths:
                vdata = [s for s in summary_data if s["variant"] == variant and s["depth"] == depth]
                if vdata and vdata[0]["total_params"] > 0:
                    x_labels.append(f"{variant}/d{depth}")
                    eff = vdata[0]["auroc"] / (vdata[0]["total_params"] / 1e6)
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


def _plot_param_counts(summary_data: List[dict], output_dir: Path, 
                       variants: List[str], depths: List[int],
                       sweep_mode: str, hidden_dims: List[int] = None):
    """Plot model sizes."""
    plt = _get_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_labels = []
    params = []
    colors = []
    
    if sweep_mode == "hidden_dim" and hidden_dims:
        for variant in variants:
            for depth in depths:
                for hidden_dim in hidden_dims:
                    vdata = [s for s in summary_data 
                            if s["variant"] == variant and s["depth"] == depth 
                            and s.get("hidden_dim") == hidden_dim]
                    if vdata:
                        x_labels.append(f"{variant}/d{depth}/h{hidden_dim}")
                        params.append(vdata[0]["total_params"] / 1e6)
                        colors.append(VARIANT_COLORS.get(variant, "gray"))
    else:
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


def _generate_scaling_summary_md(summary_data: List[dict], output_dir: Path,
                                  variants: List[str], depths: List[int],
                                  sweep_mode: str, hidden_dims: List[int] = None):
    """Generate comprehensive summary markdown."""
    has_multi_seed = any(s.get('n_seeds', 1) > 1 for s in summary_data)
    has_hidden_dim = sweep_mode == "hidden_dim"
    
    lines = [
        "# Sweep Results Summary\n",
        "## Final Metrics by Variant and Depth\n",
    ]
    
    # Build header
    if has_hidden_dim:
        if has_multi_seed:
            lines.extend([
                "| Variant | Depth | Hidden | Seeds | Params (M) | Time (min) | Loss | AUROC | AP |",
                "|---------|-------|--------|-------|------------|------------|------|-------|-----|",
            ])
        else:
            lines.extend([
                "| Variant | Depth | Hidden | Params (M) | Time (min) | Loss | AUROC | AP |",
                "|---------|-------|--------|------------|------------|------|-------|-----|",
            ])
    else:
        if has_multi_seed:
            lines.extend([
                "| Variant | Depth | Seeds | Params (M) | Time (min) | Loss | AUROC | AP |",
                "|---------|-------|-------|------------|------------|------|-------|-----|",
            ])
        else:
            lines.extend([
                "| Variant | Depth | Params (M) | Time (min) | Loss | AUROC | AP |",
                "|---------|-------|------------|------------|------|-------|-----|",
            ])
    
    def fmt_metric(val, std=0, precision=4):
        if std > 0:
            return f"{val:.{precision}f}±{std:.{precision}f}"
        return f"{val:.{precision}f}"
    
    # Build rows
    if has_hidden_dim and hidden_dims:
        for variant in variants:
            for depth in depths:
                for hidden_dim in hidden_dims:
                    vdata = [s for s in summary_data 
                            if s["variant"] == variant and s["depth"] == depth 
                            and s.get("hidden_dim") == hidden_dim]
                    if not vdata:
                        continue
                    s = vdata[0]
                    if has_multi_seed:
                        lines.append(
                            f"| {variant} | {depth} | {hidden_dim} | {s.get('n_seeds', 1)} | "
                            f"{s['total_params']/1e6:.2f} | {s['total_time_min']:.1f} | "
                            f"{fmt_metric(s['final_loss'], s.get('final_loss_std', 0))} | "
                            f"{fmt_metric(s['auroc'], s.get('auroc_std', 0))} | "
                            f"{fmt_metric(s['ap'], s.get('ap_std', 0))} |"
                        )
                    else:
                        lines.append(
                            f"| {variant} | {depth} | {hidden_dim} | "
                            f"{s['total_params']/1e6:.2f} | {s['total_time_min']:.1f} | "
                            f"{s['final_loss']:.4f} | {s['auroc']:.4f} | {s['ap']:.4f} |"
                        )
    else:
        for variant in variants:
            for depth in depths:
                vdata = [s for s in summary_data if s["variant"] == variant and s["depth"] == depth]
                if not vdata:
                    continue
                s = vdata[0]
                if has_multi_seed:
                    lines.append(
                        f"| {variant} | {depth} | {s.get('n_seeds', 1)} | "
                        f"{s['total_params']/1e6:.2f} | {s['total_time_min']:.1f} | "
                        f"{fmt_metric(s['final_loss'], s.get('final_loss_std', 0))} | "
                        f"{fmt_metric(s['auroc'], s.get('auroc_std', 0))} | "
                        f"{fmt_metric(s['ap'], s.get('ap_std', 0))} |"
                    )
                else:
                    lines.append(
                        f"| {variant} | {depth} | "
                        f"{s['total_params']/1e6:.2f} | {s['total_time_min']:.1f} | "
                        f"{s['final_loss']:.4f} | {s['auroc']:.4f} | {s['ap']:.4f} |"
                    )
    
    lines.extend([
        "",
        "## Variant Descriptions",
        "- **A**: ResNet-MLP backbone + MLP projection head (Baseline)",
        "- **B**: ResNet-MLP backbone + ChebyKAN projection head",
        "- **C**: ResNet-KAN backbone + MLP projection head",
        "- **D**: ResNet-KAN backbone + ChebyKAN projection head",
    ])
    
    with open(output_dir / "scaling_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _generate_plot_summary_md(output_dir: Path, depths: List[int],
                               sweep_mode: str, hidden_dims: List[int] = None):
    """Generate plot descriptions markdown."""
    lines = [
        "# Plot Summary\n",
        "## Per-Depth Comparison Plots",
        "",
    ]
    
    if sweep_mode == "hidden_dim" and hidden_dims:
        for depth in depths:
            for hidden_dim in hidden_dims:
                lines.extend([
                    f"### Depth {depth}, Hidden {hidden_dim} (`depth_d{depth}_h{hidden_dim}/`)",
                    f"- `loss_curve_variants.png`: Training loss over epochs",
                    f"- `linear_probe_auroc_variants.png`: Linear evaluation AUROC",
                    "",
                ])
    else:
        for depth in depths:
            lines.extend([
                f"### Depth {depth} (`depth_d{depth}/`)",
                f"- `loss_curve_variants.png`: Training loss over epochs for all variants",
                f"- `linear_probe_auroc_variants.png`: Linear evaluation AUROC over epochs",
                f"- `alignment_variants.png`: Representation alignment (lower = better)",
                f"- `uniformity_variants.png`: Representation uniformity (lower = better)",
                "",
            ])
    
    lines.extend([
        "## Scaling Analysis Plots",
        "- `scaling_auc_vs_params.png`: AUROC vs model size",
        "- `scaling_auc_vs_time.png`: AUROC vs training time",
        "- `auc_by_variant.png`: Bar chart of final AUROC",
        "- `efficiency_by_variant.png`: AUROC per million parameters",
        "- `param_counts_by_variant.png`: Model sizes comparison",
        "- `scaling_summary.md`: Full results table",
    ])
    
    with open(output_dir / "plot_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_all_sweep_plots(sweep_dir: Path, variants: List[str] = None,
                              depths: List[int] = None, hidden_dims: List[int] = None):
    """
    Generate all sweep plots from collected metrics.
    
    For multi-seed mode, creates:
        plots/seed_41/ - Individual seed plots
        plots/seed_42/ - Individual seed plots
        plots/combined/ - Aggregated mean±std plots
    
    For single-seed and hidden-dim modes, creates:
        plots/depth_d18/, plots/depth_d34/, etc.
    """
    if variants is None:
        variants = ["A", "B", "C", "D"]
    
    # Auto-detect depths if not provided
    if depths is None:
        depths = set()
        for v in variants:
            variant_dir = sweep_dir / v
            if variant_dir.exists():
                for d in variant_dir.iterdir():
                    if d.is_dir() and d.name.startswith("d"):
                        try:
                            depth = int(d.name[1:])
                            depths.add(depth)
                        except ValueError:
                            pass
        depths = sorted(depths)
    
    if not depths:
        print("No depths found in sweep directory")
        return
    
    print(f"Generating plots for variants {variants}, depths {depths}")
    
    # Detect mode
    sweep_mode, extra = detect_sweep_mode(sweep_dir, variants, depths)
    print(f"  Detected sweep mode: {sweep_mode}")
    
    plots_dir = sweep_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    if sweep_mode == "multi_seed":
        seed_names = extra
        print(f"  Seeds found: {seed_names}")
        
        # Generate individual seed plots
        for seed_name in seed_names:
            print(f"\n  Generating plots for {seed_name}...")
            seed_dir = plots_dir / seed_name
            seed_dir.mkdir(parents=True, exist_ok=True)
            
            # Collect data for this seed
            seed_data = collect_sweep_data_per_seed(sweep_dir, variants, depths, seed_name)
            
            if seed_data:
                # Generate depth comparison plots
                for depth in depths:
                    plot_depth_comparison(seed_data, depth, seed_dir, variants, 
                                         sweep_mode="single", show_std=False,
                                         title_suffix=f"[{seed_name}]")
                
                # Generate scaling analysis
                plot_scaling_analysis(seed_data, seed_dir, depths, variants, 
                                     sweep_mode="single", title_suffix=f"[{seed_name}]")
        
        # Generate combined (aggregated) plots
        print(f"\n  Generating combined plots (mean±std across seeds)...")
        combined_dir = plots_dir / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        
        combined_data = collect_sweep_data_combined(sweep_dir, variants, depths, seed_names)
        
        if combined_data:
            for depth in depths:
                plot_depth_comparison(combined_data, depth, combined_dir, variants,
                                     sweep_mode="multi_seed", show_std=True,
                                     title_suffix="[Combined]")
            
            plot_scaling_analysis(combined_data, combined_dir, depths, variants,
                                 sweep_mode="multi_seed", title_suffix="[Combined]")
    
    elif sweep_mode == "hidden_dim":
        hidden_dims_detected = extra if hidden_dims is None else hidden_dims
        print(f"  Hidden dims: {hidden_dims_detected}")
        
        all_data = collect_sweep_data_hidden_dim(sweep_dir, variants, depths, hidden_dims_detected)
        
        if all_data:
            for depth in depths:
                for hidden_dim in hidden_dims_detected:
                    plot_depth_comparison(all_data, depth, plots_dir, variants,
                                         sweep_mode="hidden_dim", hidden_dim=hidden_dim)
            
            plot_scaling_analysis(all_data, plots_dir, depths, variants,
                                 sweep_mode="hidden_dim", hidden_dims=hidden_dims_detected)
    
    else:  # single mode
        all_data = collect_sweep_data_single(sweep_dir, variants, depths)
        
        if all_data:
            for depth in depths:
                plot_depth_comparison(all_data, depth, plots_dir, variants, sweep_mode="single")
            
            plot_scaling_analysis(all_data, plots_dir, depths, variants, sweep_mode="single")
    
    print(f"\nAll plots saved to {plots_dir}")
