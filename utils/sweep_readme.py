"""
Sweep README and documentation generation utilities.
"""

from pathlib import Path
from typing import List


def generate_sweep_readme(sweep_dir: Path, variants: List[str], depths: List[int],
                          epochs: int, eval_every: int, seeds: List[int] = None,
                          hidden_dims: List[int] = None):
    """Generate README.md for sweep directory."""
    if seeds is None:
        seeds = [42]
    
    multi_seed = len(seeds) > 1
    hidden_dim_mode = hidden_dims is not None and len(hidden_dims) > 0
    
    lines = [
        f"# Sweep Experiment Results\n",
        f"**Generated**: {sweep_dir.name.replace('_sweep_', '')}",
        "",
        "## Configuration",
        f"- **Variants**: {', '.join(variants)}",
        f"- **Depths**: {', '.join(map(str, depths))}",
    ]
    
    if hidden_dim_mode:
        lines.append(f"- **Hidden Dims**: {', '.join(map(str, hidden_dims))}")
    
    lines.extend([
        f"- **Seeds**: {', '.join(map(str, seeds))}",
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
    ])
    
    for depth in depths:
        if hidden_dim_mode:
            for hd in hidden_dims:
                lines.append(f"│   ├── depth_d{depth}_h{hd}/")
        else:
            lines.append(f"│   ├── depth_d{depth}/")
    
    lines.extend([
        "│   ├── scaling_*.png",
        "│   └── *_summary.md",
    ])
    
    for variant in variants:
        lines.append(f"├── {variant}/")
        for depth in depths:
            if hidden_dim_mode:
                lines.append(f"│   └── d{depth}/")
                for hd in hidden_dims:
                    lines.append(f"│       └── h{hd}/")
            elif multi_seed:
                lines.append(f"│   └── d{depth}/")
                for seed in seeds:
                    lines.append(f"│       └── seed_{seed}/")
            else:
                lines.append(f"│   └── d{depth}/")
    
    lines.extend([
        "```",
        "",
        "## Results",
        "See `plots/scaling_summary.md` for full results table.",
    ])
    
    if multi_seed:
        lines.extend([
            "",
            "## Multi-Seed Notes",
            f"- This sweep ran with {len(seeds)} seeds: {seeds}",
            "- Metrics are aggregated as mean ± standard deviation",
            "- Error bands in plots show ±1 std",
        ])
    
    if hidden_dim_mode:
        lines.extend([
            "",
            "## Hidden Dimension Sweep Notes",
            f"- This sweep tested {len(hidden_dims)} projector hidden dimensions: {hidden_dims}",
            "- Each combination of variant/depth runs with different hidden dimensions",
        ])
    
    with open(sweep_dir / "README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
