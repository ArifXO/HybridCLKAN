"""
SimCLR Pretraining Script with periodic evaluation.

Usage:
    python scripts/pretrain.py
    python scripts/pretrain.py --run_name simclr_B --variant B
    python scripts/pretrain.py --resume results/simclr_A/last.ckpt
    python scripts/pretrain.py --eval_every 10  # Run eval every 10 epochs
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.chestmnist import get_dataloaders
from losses.ntxent import NTXentLoss, alignment, uniformity
from models.simclr import build_simclr
from utils.core import (
    seed_everything,
    make_run_dir,
    save_config,
    save_json_metrics_append,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    Timer,
    AverageMeter,
)


def load_config(config_path, overrides=None):
    """Load YAML config and apply CLI overrides."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                # Handle nested keys like "training.lr"
                keys = key.split(".")
                d = cfg
                for k in keys[:-1]:
                    d = d.setdefault(k, {})
                d[keys[-1]] = value
    
    # Apply variant mapping: A/B/C/D -> backbone + head settings
    # A: resnet_mlp + mlp
    # B: resnet_mlp + chebykan
    # C: resnet_kan + mlp
    # D: resnet_kan + chebykan
    variant = cfg.get("variant")
    if variant:
        variant_map = {
            "A": {"backbone": "resnet_mlp", "head": "mlp"},
            "B": {"backbone": "resnet_mlp", "head": "chebykan"},
            "C": {"backbone": "resnet_kan", "head": "mlp"},
            "D": {"backbone": "resnet_kan", "head": "chebykan"},
        }
        if variant in variant_map:
            cfg.setdefault("model", {})
            cfg["model"]["backbone"] = variant_map[variant]["backbone"]
            cfg["model"]["head"] = variant_map[variant]["head"]
    
    return cfg


def build_optimizer(model, cfg):
    """Build optimizer from config."""
    training_cfg = cfg["training"]
    opt_name = training_cfg.get("optimizer", "adam").lower()
    lr = float(training_cfg["lr"])
    wd = float(training_cfg.get("weight_decay", 1e-4))
    
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(optimizer, cfg, steps_per_epoch):
    """Build LR scheduler from config."""
    training_cfg = cfg["training"]
    sched_name = training_cfg.get("scheduler", "cosine").lower()
    epochs = training_cfg["epochs"]
    warmup_epochs = training_cfg.get("warmup_epochs", 10)
    
    if sched_name == "none":
        return None
    
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    
    if sched_name == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    
    for view1, view2, _ in loader:
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # Forward pass
            z1 = model(view1)
            z2 = model(view2)
            
            # Compute loss
            loss = criterion(z1, z2)
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        batch_size = view1.size(0)
        loss_meter.update(loss.item(), batch_size)
    
    return {
        "train_loss": loss_meter.avg,
    }


def extract_features(model, loader, device):
    """Extract encoder features and labels from dataset."""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            features = model.encode(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    return features, labels


def quick_linear_probe(train_features, train_labels, test_features, test_labels, device, epochs=50):
    """Quick linear probe evaluation (fewer epochs for periodic eval).
    
    Returns:
        dict with auroc and average_precision
    """
    from sklearn.metrics import average_precision_score
    
    input_dim = train_features.shape[1]
    num_classes = train_labels.shape[1]
    
    probe = nn.Linear(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    train_x = torch.from_numpy(train_features).float().to(device)
    train_y = torch.from_numpy(train_labels).float().to(device)
    test_x = torch.from_numpy(test_features).float().to(device)
    test_y_np = test_labels
    
    # Train
    probe.train()
    batch_size = 256
    n_train = len(train_x)
    
    for _ in range(epochs):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            loss = criterion(probe(train_x[idx]), train_y[idx])
            loss.backward()
            optimizer.step()
    
    # Evaluate
    probe.eval()
    with torch.no_grad():
        logits = probe(test_x)
        probs = torch.sigmoid(logits).cpu().numpy()
    
    try:
        auc = roc_auc_score(test_y_np, probs, average="macro")
    except ValueError:
        auc = 0.0
    
    try:
        ap = average_precision_score(test_y_np, probs, average="macro")
    except ValueError:
        ap = 0.0
    
    return {"auroc": auc, "ap": ap}


def run_periodic_eval(model, cfg, device):
    """Run evaluation (linear probe + alignment/uniformity) and return metrics."""
    # Get eval dataloaders
    eval_loaders = get_dataloaders(cfg, splits=("train", "test"), as_simclr=False)
    
    # Extract features
    train_features, train_labels = extract_features(model, eval_loaders["train"], device)
    test_features, test_labels = extract_features(model, eval_loaders["test"], device)
    
    # Linear probe (quick version)
    linear_results = quick_linear_probe(train_features, train_labels, test_features, test_labels, device, epochs=50)
    
    # Calculate alignment and uniformity on test set using SimCLR loader
    simclr_loader = get_dataloaders(cfg, splits=("test",), as_simclr=True)["test"]
    align_val, uniform_val = compute_alignment_uniformity(model, simclr_loader, device)
    
    return {
        "linear_probe_auc": linear_results["auroc"],
        "linear_probe_ap": linear_results["ap"],
        "alignment": align_val,
        "uniformity": uniform_val,
    }


def compute_alignment_uniformity(model, loader, device):
    """Compute alignment and uniformity metrics on a dataset."""
    model.eval()
    
    align_meter = AverageMeter()
    all_z = []
    
    with torch.no_grad():
        for view1, view2, _ in loader:
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            
            z1 = model(view1)
            z2 = model(view2)
            
            # Alignment: mean distance between positive pairs
            align_val = alignment(z1, z2).item()
            align_meter.update(align_val, view1.size(0))
            
            # Collect all embeddings for uniformity
            all_z.append(z1)
            all_z.append(z2)
    
    # Uniformity: computed on all embeddings
    all_z = torch.cat(all_z, dim=0)
    # Sample if too many (for memory)
    if len(all_z) > 5000:
        idx = torch.randperm(len(all_z))[:5000]
        all_z = all_z[idx]
    uniform_val = uniformity(all_z).item()
    
    return align_meter.avg, uniform_val


def main():
    parser = argparse.ArgumentParser(description="SimCLR Pretraining")
    parser.add_argument("--config", type=str, default="config/pretrain.yaml", help="Path to config file")
    parser.add_argument("--run_name", type=str, default=None, help="Run name (overrides config)")
    parser.add_argument("--variant", type=str, choices=["A", "B", "C", "D"], default=None, help="Model variant")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    # Model architecture args
    parser.add_argument("--resnet_depth", type=int, default=None, help="ResNet depth (18, 34, 50)")
    parser.add_argument("--embedding_dim", type=int, default=None, help="Encoder embedding dimension")
    parser.add_argument("--projection_dim", type=int, default=None, help="Projector output dimension")
    parser.add_argument("--projection_hidden_dim", type=int, default=None, help="Projector hidden dimension")
    # Periodic evaluation
    parser.add_argument("--eval_every", type=int, default=0, help="Run eval every N epochs (0=disabled)")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()
    
    # Load config
    config_path = PROJECT_ROOT / args.config
    overrides = {
        "run_name": args.run_name,
        "variant": args.variant,
        "training.epochs": args.epochs,
        "training.batch_size": args.batch_size,
        "training.lr": args.lr,
        "seed": args.seed,
        # Model architecture overrides
        "model.resnet_depth": args.resnet_depth,
        "model.embedding_dim": args.embedding_dim,
        "model.projection_dim": args.projection_dim,
        "model.projection_hidden_dim": args.projection_hidden_dim,
    }
    # Filter None values
    overrides = {k: v for k, v in overrides.items() if v is not None}
    
    cfg = load_config(config_path, overrides)
    
    # Handle resume path override
    if args.resume:
        cfg["resume"] = args.resume
    
    # Handle output_dir override
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    
    # Setup
    seed = cfg.get("seed", 42)
    seed_everything(seed)
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Create run directory
    run_name = cfg.get("run_name", "simclr_run")
    output_dir = cfg.get("output_dir", "results")
    run_dir = make_run_dir(output_dir, run_name)
    print(f"Run directory: {run_dir}")
    
    # Save resolved config
    save_config(run_dir, cfg)
    
    # Build data loaders
    print("Loading data...")
    loaders = get_dataloaders(cfg, splits=("train",), as_simclr=True)
    train_loader = loaders["train"]
    print(f"Train samples: {len(train_loader.dataset)}")
    
    # Build model
    print("Building model...")
    model_cfg = cfg["model"]
    print(f"Model config: depth={model_cfg.get('resnet_depth', 18)}, "
          f"embed_dim={model_cfg.get('embedding_dim', 512)}, "
          f"proj_hidden={model_cfg.get('projection_hidden_dim', 512)}, "
          f"proj_out={model_cfg.get('projection_dim', 128)}")
    model = build_simclr(cfg)
    model = model.to(device)
    
    # Count parameters
    param_counts = count_parameters(model)
    print(f"Parameters - Encoder: {param_counts['encoder']:,}, Projector: {param_counts['projector']:,}, Total: {param_counts['total']:,}")
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    
    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None
    
    # Loss function
    temperature = cfg.get("simclr", {}).get("temperature", 0.5)
    criterion = NTXentLoss(temperature=temperature)
    
    # Resume if specified
    start_epoch = 0
    best_loss = float("inf")
    
    resume_path = cfg.get("resume")
    if resume_path and Path(resume_path).exists():
        print(f"Resuming from {resume_path}")
        resume_info = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device
        )
        start_epoch = resume_info["epoch"]
        best_loss = resume_info.get("best_metric", float("inf")) or float("inf")
        print(f"Resumed at epoch {start_epoch}, best_loss: {best_loss:.4f}")
    
    # Training loop
    epochs = cfg["training"]["epochs"]
    metrics_path = run_dir / "metrics.jsonl"
    eval_every = args.eval_every
    
    print(f"\nStarting training from epoch {start_epoch} to {epochs}")
    if eval_every > 0:
        print(f"Periodic evaluation every {eval_every} epochs")
    print("-" * 60)
    
    for epoch in range(start_epoch, epochs):
        with Timer() as epoch_timer:
            metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, scaler, device
            )
        
        # Build metrics dict in consistent order matching expected format
        epoch_metrics = {
            "epoch": epoch + 1,  # 1-indexed for display
            "train_loss": metrics["train_loss"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_timer.elapsed,
        }
        
        # Save param_counts on first epoch only
        if epoch == start_epoch:
            epoch_metrics["param_counts"] = param_counts
        
        # Periodic evaluation (includes alignment/uniformity)
        if eval_every > 0 and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1):
            print(f"  Running periodic evaluation...")
            eval_metrics = run_periodic_eval(model, cfg, device)
            # Add eval metrics in expected order
            epoch_metrics["linear_probe_auroc"] = eval_metrics.get("linear_probe_auc", 0.0)
            epoch_metrics["linear_probe_ap"] = eval_metrics.get("linear_probe_ap", 0.0)
            epoch_metrics["alignment"] = eval_metrics.get("alignment", 0.0)
            epoch_metrics["uniformity"] = eval_metrics.get("uniformity", 0.0)
            print(f"  Linear AUC: {epoch_metrics['linear_probe_auroc']:.4f}, AP: {epoch_metrics['linear_probe_ap']:.4f}")
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {epoch_metrics['train_loss']:.4f} | "
              f"LR: {epoch_metrics['lr']:.6f} | "
              f"Time: {epoch_metrics['epoch_time_sec']:.1f}s")
        
        # Save metrics
        save_json_metrics_append(metrics_path, epoch_metrics)
        
        # Save last checkpoint
        save_checkpoint(
            run_dir / "last.ckpt",
            model, optimizer, scheduler, scaler, epoch, best_loss
        )
        
        # Save best checkpoint
        if metrics["train_loss"] < best_loss:
            best_loss = metrics["train_loss"]
            save_checkpoint(
                run_dir / "best.ckpt",
                model, optimizer, scheduler, scaler, epoch, best_loss
            )
            print(f"  -> New best loss: {best_loss:.4f}")
    
    print("-" * 60)
    print(f"Training complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {run_dir}")


if __name__ == "__main__":
    main()
