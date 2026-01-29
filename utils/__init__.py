"""
Utils package for SimCLR training pipeline.

Modules:
    - core: Training utilities (checkpointing, metrics, timing)
    - plotting: Sweep visualization and plot generation
"""

from utils.core import (
    seed_everything,
    make_run_dir,
    save_config,
    save_json_metrics_append,
    load_json_metrics,
    get_rng_states,
    set_rng_states,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    Timer,
    AverageMeter,
)

from utils.plotting import (
    VARIANT_COLORS,
    VARIANT_LABELS,
    VARIANT_MARKERS,
    collect_sweep_data,
    plot_depth_comparison,
    plot_scaling_analysis,
    generate_all_sweep_plots,
    generate_sweep_readme,
)

__all__ = [
    # Core utilities
    "seed_everything",
    "make_run_dir",
    "save_config",
    "save_json_metrics_append",
    "load_json_metrics",
    "get_rng_states",
    "set_rng_states",
    "save_checkpoint",
    "load_checkpoint",
    "count_parameters",
    "Timer",
    "AverageMeter",
    # Plotting utilities
    "VARIANT_COLORS",
    "VARIANT_LABELS",
    "VARIANT_MARKERS",
    "collect_sweep_data",
    "plot_depth_comparison",
    "plot_scaling_analysis",
    "generate_all_sweep_plots",
    "generate_sweep_readme",
]
