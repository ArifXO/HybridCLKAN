"""
Plotting utilities for sweep analysis and visualization.

This module provides a unified interface for all plotting functions.
Implementation is split across:
- utils/sweep_plots.py: Metric curves, scaling analysis plots
- utils/umap_plots.py: UMAP visualizations
- utils/sweep_readme.py: README generation

Usage:
    from utils.plotting import generate_all_sweep_plots, generate_sweep_readme
"""

from pathlib import Path
from typing import List, Optional

# Re-export from sweep_plots
from utils.sweep_plots import (
    VARIANT_COLORS,
    VARIANT_LABELS,
    VARIANT_MARKERS,
    VARIANT_LINESTYLES,
    load_metrics_jsonl,
    collect_sweep_data,
    detect_sweep_mode,
    plot_depth_comparison,
    plot_scaling_analysis,
    generate_all_sweep_plots as _generate_sweep_plots,
)

# Re-export from sweep_readme
from utils.sweep_readme import generate_sweep_readme

# Re-export from umap_plots
from utils.umap_plots import generate_umap_plots


def generate_all_sweep_plots(sweep_dir: Path, variants: List[str] = None,
                              depths: List[int] = None, hidden_dims: List[int] = None,
                              skip_umap: bool = False):
    """
    Generate all sweep plots from collected metrics.
    
    Args:
        sweep_dir: Path to sweep directory
        variants: List of variants (default: A, B, C, D)
        depths: List of depths (default: auto-detect)
        hidden_dims: List of hidden dims for hidden_dim mode (default: auto-detect)
        skip_umap: Skip UMAP generation (faster)
    """
    # Generate metric curves and scaling plots
    _generate_sweep_plots(sweep_dir, variants, depths, hidden_dims)
    
    # Generate UMAP plots (optional, can be slow)
    if not skip_umap:
        if variants is None:
            variants = ["A", "B", "C", "D"]
        
        # Auto-detect depths
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
        
        if depths:
            generate_umap_plots(sweep_dir, variants, depths)


# For backwards compatibility
__all__ = [
    "VARIANT_COLORS",
    "VARIANT_LABELS", 
    "VARIANT_MARKERS",
    "VARIANT_LINESTYLES",
    "load_metrics_jsonl",
    "collect_sweep_data",
    "detect_sweep_mode",
    "plot_depth_comparison",
    "plot_scaling_analysis",
    "generate_all_sweep_plots",
    "generate_sweep_readme",
    "generate_umap_plots",
]
