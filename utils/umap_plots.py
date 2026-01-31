"""
UMAP visualization utilities for sweep analysis.

Generates UMAP plots from trained model checkpoints.
"""

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Lazy imports
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


# Variant styling (duplicated for standalone use)
VARIANT_LABELS = {
    "A": "ResNet-MLP + MLP (Baseline)",
    "B": "ResNet-MLP + ChebyKAN",
    "C": "ResNet-KAN + MLP",
    "D": "ResNet-KAN + ChebyKAN",
}

# ChestMNIST class names
CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia"
]


def generate_umap_plots(sweep_dir: Path, variants: List[str], depths: List[int],
                        device: str = None):
    """
    Generate UMAP visualizations for each variant/depth from saved checkpoints.
    
    Creates:
        plots/umap/
            umap_A_d18.png
            umap_B_d18.png
            ...
            umap_combined_d18.png
    """
    import torch
    import yaml
    
    plt = _get_matplotlib()
    
    # Add project root for imports
    project_root = sweep_dir.parent.parent if sweep_dir.name.startswith("_sweep") else sweep_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Check for UMAP
    try:
        from umap import UMAP
    except ImportError:
        print("  UMAP not installed. Skipping UMAP plots. Install with: pip install umap-learn")
        return
    
    # Import project modules
    try:
        from models.simclr import build_simclr
        from data.chestmnist import get_dataloaders
    except ImportError as e:
        print(f"  Could not import project modules: {e}")
        return
    
    umap_dir = sweep_dir / "plots" / "umap"
    umap_dir.mkdir(parents=True, exist_ok=True)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    for depth in depths:
        for variant in variants:
            # Find run directory (handle single-seed, multi-seed, hidden-dim modes)
            run_dir = _find_run_dir(sweep_dir, variant, depth)
            
            if run_dir is None:
                print(f"  Skipping UMAP for {variant}/d{depth} - run directory not found")
                continue
            
            ckpt_path = run_dir / "best.ckpt"
            config_path = run_dir / "config_resolved.yaml"
            
            if not ckpt_path.exists():
                ckpt_path = run_dir / "last.ckpt"
            
            if not ckpt_path.exists() or not config_path.exists():
                print(f"  Skipping UMAP for {variant}/d{depth} - checkpoint not found")
                continue
            
            print(f"  Generating UMAP for {variant}/d{depth}...")
            
            try:
                # Load config
                with open(config_path, "r") as f:
                    cfg = yaml.safe_load(f)
                
                # Build model - build_simclr returns (model, projector_build_info)
                model, _ = build_simclr(cfg)
                
                # Load checkpoint
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                model = model.to(device)
                model.eval()
                
                # Get test data (use num_workers=0 for Windows compatibility)
                cfg_copy = dict(cfg)
                cfg_copy["data"] = dict(cfg.get("data", {}))
                cfg_copy["data"]["num_workers"] = 0
                loaders = get_dataloaders(cfg_copy, splits=("test",), as_simclr=False)
                test_loader = loaders["test"]
                
                # Extract features
                all_features = []
                all_labels = []
                
                with torch.no_grad():
                    for images, labels in test_loader:
                        images = images.to(device)
                        features = model.encode(images)
                        all_features.append(features.cpu().numpy())
                        all_labels.append(labels.numpy())
                
                features = np.concatenate(all_features, axis=0)
                labels = np.concatenate(all_labels, axis=0)
                
                # Run UMAP
                reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
                embedding = reducer.fit_transform(features)
                
                # Get primary label for each sample
                primary_labels = []
                for lbl in labels:
                    pos_idx = np.where(lbl == 1)[0]
                    primary_labels.append(pos_idx[0] if len(pos_idx) > 0 else -1)
                primary_labels = np.array(primary_labels)
                
                # Plot
                fig, ax = plt.subplots(figsize=(12, 10))
                cmap = plt.cm.get_cmap('tab20', 14)
                
                for i in range(14):
                    mask = primary_labels == i
                    if mask.sum() > 0:
                        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                                  c=[cmap(i)], label=CLASS_NAMES[i], alpha=0.6, s=10)
                
                ax.set_title(f"UMAP - Variant {variant} (d{depth})\n{VARIANT_LABELS.get(variant, variant)}")
                ax.set_xlabel("UMAP 1")
                ax.set_ylabel("UMAP 2")
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
                
                plt.tight_layout()
                plt.savefig(umap_dir / f"umap_{variant}_d{depth}.png", dpi=150, bbox_inches="tight")
                plt.close()
                
                # Clean up
                del model, features, embedding
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  Error generating UMAP for {variant}/d{depth}: {e}")
                continue
    
    # Generate combined UMAP comparison plot per depth
    for depth in depths:
        _generate_combined_umap(sweep_dir, variants, depth, umap_dir)
    
    print(f"  UMAP plots saved to {umap_dir}")


def _find_run_dir(sweep_dir: Path, variant: str, depth: int) -> Optional[Path]:
    """
    Find the run directory for a given variant/depth.
    
    Handles:
        - Single-seed: sweep_dir/{variant}/d{depth}/
        - Multi-seed: sweep_dir/{variant}/d{depth}/seed_*/ (returns first seed)
        - Hidden-dim: sweep_dir/{variant}/d{depth}/h*/ (returns first hidden_dim)
    """
    base_dir = sweep_dir / variant / f"d{depth}"
    
    if not base_dir.exists():
        return None
    
    # Check for direct metrics (single-seed)
    if (base_dir / "metrics.jsonl").exists() or (base_dir / "config_resolved.yaml").exists():
        return base_dir
    
    # Check for seed subdirectories
    seed_dirs = sorted(base_dir.glob("seed_*"))
    if seed_dirs:
        return seed_dirs[0]
    
    # Check for hidden_dim subdirectories
    hidden_dirs = sorted(base_dir.glob("h*"))
    if hidden_dirs:
        return hidden_dirs[0]
    
    return None


def _generate_combined_umap(sweep_dir: Path, variants: List[str], depth: int, umap_dir: Path):
    """Generate a combined 2x2 UMAP plot for all variants at a given depth."""
    plt = _get_matplotlib()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    has_any = False
    for idx, variant in enumerate(variants[:4]):
        umap_path = umap_dir / f"umap_{variant}_d{depth}.png"
        if umap_path.exists():
            img = plt.imread(str(umap_path))
            axes[idx].imshow(img)
            axes[idx].axis('off')
            has_any = True
        else:
            axes[idx].text(0.5, 0.5, f"No data for {variant}/d{depth}", 
                          ha='center', va='center', fontsize=14)
            axes[idx].axis('off')
    
    if has_any:
        fig.suptitle(f"UMAP Comparison - Depth {depth}", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(umap_dir / f"umap_combined_d{depth}.png", dpi=150, bbox_inches="tight")
    plt.close()
