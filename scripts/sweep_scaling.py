"""
Scaling Sweep: Run grid of experiments with periodic evaluation and generate combined plots.

Creates folder structure:
    Single-seed mode (--seed):
        results/_sweep_<timestamp>/
            A/d18/, A/d34/, ...
            B/d18/, ...
    
    Multi-seed mode (--seeds):
        results/_sweep_<timestamp>/
            A/d18/seed_41/, A/d18/seed_42/, A/d18/seed_43/
            A/d34/seed_41/, ...
            B/d18/seed_41/, ...

Usage:
    python scripts/sweep_scaling.py
    python scripts/sweep_scaling.py --variants A B C D --depths 18 34 --epochs 50 --eval_every 10
    python scripts/sweep_scaling.py --seeds 41 42 43 --variants A B --depths 18
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
                          seed: int = 42, device: str = None,
                          chebykan_match_mlp_params: bool = False,
                          multi_seed_mode: bool = False,
                          resume_from: str = None,
                          hidden_dim: int = None,
                          hidden_dim_sweep_mode: bool = False):
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
        chebykan_match_mlp_params: match ChebyKAN params to MLP params
        multi_seed_mode: if True, output to sweep_dir/variant/d{depth}/seed_{seed}/
        resume_from: path to checkpoint to resume from
        hidden_dim: projector hidden dimension
        hidden_dim_sweep_mode: if True, output to sweep_dir/variant/d{depth}/h{hidden_dim}/
    
    Returns:
        run_dir: path to run directory
        success: bool
    """
    # Create run directory
    # Single-seed:         sweep_dir/variant/d{depth}
    # Multi-seed:          sweep_dir/variant/d{depth}/seed_{seed}
    # Hidden-dim sweep:    sweep_dir/variant/d{depth}/h{hidden_dim}
    if hidden_dim_sweep_mode and hidden_dim is not None:
        run_name = f"h{hidden_dim}"
        output_dir = sweep_dir / variant / f"d{depth}"
    elif multi_seed_mode:
        run_name = f"seed_{seed}"
        output_dir = sweep_dir / variant / f"d{depth}"
    else:
        run_name = f"d{depth}"
        output_dir = sweep_dir / variant
    
    run_dir = output_dir / run_name
    
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
        "--output_dir", str(output_dir),
        "--seed", str(seed),
    ]
    
    if device:
        pretrain_cmd.extend(["--device", device])
    
    if chebykan_match_mlp_params:
        pretrain_cmd.append("--chebykan_match_mlp_params")
    
    if resume_from:
        pretrain_cmd.extend(["--resume", str(resume_from)])
    
    if hidden_dim is not None:
        pretrain_cmd.extend(["--projection_hidden_dim", str(hidden_dim)])
    
    print(f"\n{'='*60}")
    print(f"Running: Variant {variant}, Depth {depth}, Seed {seed}")
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
        print(f"Completed: {variant}/d{depth}/seed_{seed}" if multi_seed_mode else f"Completed: {variant}/d{depth}")
    else:
        print(f"FAILED: {variant}/d{depth}/seed_{seed}" if multi_seed_mode else f"FAILED: {variant}/d{depth}")
    
    return run_dir, success


def main():
    parser = argparse.ArgumentParser(description="Scaling Sweep with Periodic Evaluation")
    parser.add_argument("--config", type=str, default="config/pretrain.yaml", help="Base config")
    parser.add_argument("--variants", nargs="+", default=["A", "B", "C", "D"], help="Variants to run")
    parser.add_argument("--depths", nargs="+", type=int, default=[18], help="ResNet depths")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs per run")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--eval_every", type=int, default=10, help="Evaluate every N epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (single-seed mode)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, 
                        help="Multiple seeds for multi-seed mode (e.g., --seeds 41 42 43)")
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=None,
                        help="Projector hidden dimensions to sweep (e.g., --hidden_dims 256 512 1024 2048)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--skip_training", action="store_true", help="Skip training, only generate plots")
    parser.add_argument("--sweep_dir", type=str, default=None, help="Existing sweep dir (for skip_training)")
    parser.add_argument("--chebykan_match_mlp_params", action="store_true",
                        help="Match ChebyKAN projector params to MLP projector params")
    args = parser.parse_args()
    
    # Determine seed mode
    if args.seeds is not None:
        seeds = args.seeds
        multi_seed_mode = True
    else:
        seeds = [args.seed]
        multi_seed_mode = False
    
    # Determine hidden_dim sweep mode
    if args.hidden_dims is not None:
        hidden_dims = args.hidden_dims
        hidden_dim_sweep_mode = True
    else:
        hidden_dims = [None]  # None means use config default
        hidden_dim_sweep_mode = False
    
    # Set initial seed for reproducibility
    seed_everything(seeds[0])
    
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
    if hidden_dim_sweep_mode:
        print(f"Running hidden-dim sweep: {len(variants)} variants x {len(depths)} depths x {len(hidden_dims)} hidden_dims")
        print(f"Hidden dims: {hidden_dims}")
    elif multi_seed_mode:
        print(f"Running sweep: {len(variants)} variants x {len(depths)} depths x {len(seeds)} seeds")
        print(f"Seeds: {seeds}")
    else:
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
        skipped = []
        
        # Calculate total runs based on mode
        if hidden_dim_sweep_mode:
            total_runs = len(depths) * len(variants) * len(hidden_dims)
        else:
            total_runs = len(depths) * len(variants) * len(seeds)
        run_count = 0
        
        # Order depends on mode:
        # - hidden_dim_sweep: hidden_dim → depth → variant
        # - multi_seed: seed → depth → variant
        # - single_seed: depth → variant
        
        if hidden_dim_sweep_mode:
            # Hidden-dim sweep mode: iterate over hidden_dims
            for hidden_dim in hidden_dims:
                for depth in depths:
                    for variant in variants:
                        run_count += 1
                        
                        run_dir = sweep_dir / variant / f"d{depth}" / f"h{hidden_dim}"
                        metrics_file = run_dir / "metrics.jsonl"
                        resume_checkpoint = None
                        
                        if metrics_file.exists():
                            with open(metrics_file, 'r') as f:
                                n_epochs = sum(1 for _ in f)
                            if n_epochs >= args.epochs:
                                print(f"\n[{run_count}/{total_runs}] SKIPPING (already complete): {variant}/d{depth}/h{hidden_dim}")
                                skipped.append((variant, depth, hidden_dim))
                                continue
                            else:
                                checkpoints = list(run_dir.glob("*.ckpt"))
                                if checkpoints:
                                    last_ckpt = run_dir / "last.ckpt"
                                    if last_ckpt.exists():
                                        resume_checkpoint = last_ckpt
                                    else:
                                        resume_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                                    print(f"\n[{run_count}/{total_runs}] Found partial run ({n_epochs}/{args.epochs} epochs), resuming from {resume_checkpoint.name}...")
                                else:
                                    print(f"\n[{run_count}/{total_runs}] Found partial run ({n_epochs}/{args.epochs} epochs) but no checkpoint, starting fresh...")
                                    metrics_file.unlink()
                        
                        print(f"\n[{run_count}/{total_runs}]")
                        
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
                            chebykan_match_mlp_params=args.chebykan_match_mlp_params,
                            multi_seed_mode=False,
                            resume_from=resume_checkpoint,
                            hidden_dim=hidden_dim,
                            hidden_dim_sweep_mode=True,
                        )
                        
                        if success:
                            successful.append((variant, depth, hidden_dim))
                        else:
                            failed.append((variant, depth, hidden_dim))
        else:
            # Seed mode (single or multi)
            for seed in seeds:
                for depth in depths:
                    for variant in variants:
                        run_count += 1
                        
                        # Check if run already exists (has metrics.jsonl)
                        if multi_seed_mode:
                            run_dir = sweep_dir / variant / f"d{depth}" / f"seed_{seed}"
                        else:
                            run_dir = sweep_dir / variant / f"d{depth}"
                        
                        metrics_file = run_dir / "metrics.jsonl"
                        resume_checkpoint = None
                        
                        if metrics_file.exists():
                            # Check if it has enough epochs (at least args.epochs lines)
                            with open(metrics_file, 'r') as f:
                                n_epochs = sum(1 for _ in f)
                            if n_epochs >= args.epochs:
                                print(f"\n[{run_count}/{total_runs}] SKIPPING (already complete): {variant}/d{depth}/seed_{seed}" if multi_seed_mode else f"\n[{run_count}/{total_runs}] SKIPPING (already complete): {variant}/d{depth}")
                                skipped.append((variant, depth, seed))
                                continue
                            else:
                                # Find the latest checkpoint to resume from
                                checkpoints = list(run_dir.glob("*.ckpt"))
                                if checkpoints:
                                    # Prefer last.ckpt, otherwise use the most recent
                                    last_ckpt = run_dir / "last.ckpt"
                                    if last_ckpt.exists():
                                        resume_checkpoint = last_ckpt
                                    else:
                                        # Sort by modification time
                                        resume_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                                    print(f"\n[{run_count}/{total_runs}] Found partial run ({n_epochs}/{args.epochs} epochs), resuming from {resume_checkpoint.name}...")
                                else:
                                    # No checkpoint, need to start fresh (delete corrupted metrics)
                                    print(f"\n[{run_count}/{total_runs}] Found partial run ({n_epochs}/{args.epochs} epochs) but no checkpoint, starting fresh...")
                                    metrics_file.unlink()
                        
                        print(f"\n[{run_count}/{total_runs}]")
                        
                        run_dir, success = run_single_experiment(
                            variant=variant,
                            depth=depth,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            sweep_dir=sweep_dir,
                            base_config=base_config,
                            eval_every=args.eval_every,
                            seed=seed,
                            device=args.device,
                            chebykan_match_mlp_params=args.chebykan_match_mlp_params,
                            multi_seed_mode=multi_seed_mode,
                            resume_from=resume_checkpoint,
                        )
                        
                        if success:
                            successful.append((variant, depth, seed))
                        else:
                            failed.append((variant, depth, seed))
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Successful: {len(successful)}, Skipped: {len(skipped)}, Failed: {len(failed)}")
        print(f"Total: {len(successful) + len(skipped)}/{total_runs}")
        if failed:
            print(f"Failed: {failed}")
        print(f"{'='*60}")
    
    # Generate README
    generate_sweep_readme(
        sweep_dir, variants, depths, args.epochs, args.eval_every, seeds,
        hidden_dims=hidden_dims if hidden_dim_sweep_mode else None
    )
    
    # Generate plots
    print("\nGenerating plots...")
    generate_all_sweep_plots(
        sweep_dir, variants, depths,
        hidden_dims=hidden_dims if hidden_dim_sweep_mode else None,
        skip_umap=hidden_dim_sweep_mode  # Skip UMAP for hidden_dim sweeps (too many)
    )
    
    print(f"\n{'='*60}")
    print(f"Sweep complete!")
    print(f"Results: {sweep_dir}")
    print(f"Plots: {sweep_dir / 'plots'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
