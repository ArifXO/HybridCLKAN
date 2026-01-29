"""
Scaling Sweep: Run grid of experiments with periodic evaluation and generate combined plots.

Creates folder structure:
    results/_sweep_<timestamp>/
        README.md
        plots/
            depth_d18/, depth_d34/, depth_d50/
            scaling_*.png
            *_summary.md
        A/d18/, A/d34/, A/d50/
        B/d18/, ...
        C/d18/, ...
        D/d18/, ...

Usage:
    python scripts/sweep_scaling.py
    python scripts/sweep_scaling.py --variants A B C D --depths 18 34 --epochs 50 --eval_every 10
    python scripts/sweep_scaling.py --skip_training --sweep_dir results/_sweep_2026-01-29
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.core import seed_everything
from utils.plotting import generate_all_sweep_plots, generate_sweep_readme


def run_single_experiment(variant: str, depth: int, epochs: int, batch_size: int,
                          sweep_dir: Path, base_config: Path, eval_every: int = 10,
                          seed: int = 42, device: str = None):
    """
    Run a single pretrain experiment.
    
    Args:
        variant: A, B, C, or D
        depth: ResNet depth (18, 34, 50)
        epochs: number of training epochs
        batch_size: batch size
        sweep_dir: parent sweep directory
        base_config: path to base config file
        eval_every: run evaluation every N epochs
        seed: random seed
        device: cuda/cpu
    
    Returns:
        run_dir: path to run directory
        success: bool
    """
    # Create run directory: sweep_dir/variant/d{depth}
    run_name = f"d{depth}"
    variant_dir = sweep_dir / variant
    run_dir = variant_dir / run_name
    
    # Build pretrain command
    pretrain_cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "pretrain.py"),
        "--config", str(base_config),
        "--run_name", run_name,
        "--variant", variant,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--resnet_depth", str(depth),
        "--eval_every", str(eval_every),
        "--output_dir", str(variant_dir),
        "--seed", str(seed),
    ]
    
    if device:
        pretrain_cmd.extend(["--device", device])
    
    print(f"\n{'='*60}")
    print(f"Running: Variant {variant}, Depth {depth}")
    print(f"Output: {run_dir}")
    print(f"{'='*60}")
    
    # Run pretraining
    try:
        result = subprocess.run(pretrain_cmd, cwd=PROJECT_ROOT)
        success = result.returncode == 0
    except Exception as e:
        print(f"Error running experiment: {e}")
        success = False
    
    if success:
        print(f"Completed: {variant}/d{depth}")
    else:
        print(f"FAILED: {variant}/d{depth}")
    
    return run_dir, success


def main():
    parser = argparse.ArgumentParser(description="Scaling Sweep with Periodic Evaluation")
    parser.add_argument("--config", type=str, default="config/pretrain.yaml", help="Base config")
    parser.add_argument("--variants", nargs="+", default=["A", "B", "C", "D"], help="Variants to run")
    parser.add_argument("--depths", nargs="+", type=int, default=[18], help="ResNet depths")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs per run")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--eval_every", type=int, default=10, help="Evaluate every N epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--skip_training", action="store_true", help="Skip training, only generate plots")
    parser.add_argument("--sweep_dir", type=str, default=None, help="Existing sweep dir (for skip_training)")
    args = parser.parse_args()
    
    # Set seed
    seed_everything(args.seed)
    
    # Create or use sweep directory
    if args.sweep_dir:
        sweep_dir = Path(args.sweep_dir)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sweep_dir = PROJECT_ROOT / "results" / f"_sweep_{timestamp}"
    
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sweep directory: {sweep_dir}")
    
    variants = args.variants
    depths = args.depths
    
    print(f"\n{'='*60}")
    print(f"Running sweep: {len(variants)} variants x {len(depths)} depths")
    print(f"Variants: {variants}")
    print(f"Depths: {depths}")
    print(f"Epochs: {args.epochs}, Eval every: {args.eval_every}")
    print(f"{'='*60}")
    
    # Run experiments
    if not args.skip_training:
        base_config = PROJECT_ROOT / args.config
        
        successful = []
        failed = []
        
        for depth in depths:
            for variant in variants:
                run_dir, success = run_single_experiment(
                    variant=variant,
                    depth=depth,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    sweep_dir=sweep_dir,
                    base_config=base_config,
                    eval_every=args.eval_every,
                    seed=args.seed,
                    device=args.device,
                )
                
                if success:
                    successful.append((variant, depth))
                else:
                    failed.append((variant, depth))
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Successful: {len(successful)}/{len(variants)*len(depths)}")
        if failed:
            print(f"Failed: {failed}")
        print(f"{'='*60}")
    
    # Generate README
    generate_sweep_readme(sweep_dir, variants, depths, args.epochs, args.eval_every)
    
    # Generate plots
    print("\nGenerating plots...")
    generate_all_sweep_plots(sweep_dir, variants, depths)
    
    print(f"\n{'='*60}")
    print(f"Sweep complete!")
    print(f"Results: {sweep_dir}")
    print(f"Plots: {sweep_dir / 'plots'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
