"""
Core utility functions for training: seeding, checkpointing, metrics, timing.
"""

import os
import json
import random
import time
from pathlib import Path

import torch
import yaml


def seed_everything(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: random seed integer
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_run_dir(output_dir, run_name):
    """
    Create run directory structure: results/<run_name>/ and plots/ subdirectory.
    
    Args:
        output_dir: base output directory (e.g., "results")
        run_name: name of the run
    
    Returns:
        Path to run directory
    """
    run_dir = Path(output_dir) / run_name
    plots_dir = run_dir / "plots"
    
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir


def save_config(run_dir, cfg):
    """
    Save resolved config to run directory.
    
    Args:
        run_dir: path to run directory
        cfg: config dict
    """
    config_path = Path(run_dir) / "config_resolved.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def save_json_metrics_append(path, metrics_dict):
    """
    Append metrics dict to a JSONL file (one JSON object per line).
    
    Args:
        path: path to metrics.jsonl file
        metrics_dict: dict of metrics for current epoch
    """
    path = Path(path)
    
    with open(path, "a") as f:
        f.write(json.dumps(metrics_dict) + "\n")


def truncate_metrics_for_resume(path, resume_epoch):
    """
    Truncate metrics.jsonl to only contain entries for epochs < resume_epoch.
    
    This prevents duplicate/out-of-order entries when resuming from checkpoint.
    
    Args:
        path: path to metrics.jsonl file
        resume_epoch: the epoch we're resuming from (1-indexed display epoch)
    
    Returns:
        int: number of entries kept
    """
    path = Path(path)
    if not path.exists():
        return 0
    
    # Read all entries and keep only those before resume_epoch
    kept_entries = []
    with open(path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                # Keep entries with epoch < resume_epoch (epochs are 1-indexed in metrics)
                if entry.get("epoch", 0) < resume_epoch:
                    kept_entries.append(line)
            except json.JSONDecodeError:
                continue
    
    # Rewrite file with only kept entries
    with open(path, "w") as f:
        f.writelines(kept_entries)
    
    return len(kept_entries)


def load_json_metrics(path):
    """
    Load metrics list from JSONL file (one JSON object per line).
    
    Args:
        path: path to metrics.jsonl file
    
    Returns:
        list of metrics dicts
    """
    path = Path(path)
    if path.exists():
        metrics_list = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    metrics_list.append(json.loads(line))
        return metrics_list
    return []


def get_rng_states():
    """Get current RNG states for reproducible resume."""
    rng_states = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_states["cuda"] = torch.cuda.get_rng_state_all()
    return rng_states


def set_rng_states(rng_states):
    """Restore RNG states for reproducible resume."""
    random.setstate(rng_states["python"])
    # Ensure torch RNG state is on CPU and is ByteTensor
    torch_rng = rng_states["torch"]
    if hasattr(torch_rng, 'cpu'):
        torch_rng = torch_rng.cpu()
    if torch_rng.dtype != torch.uint8:
        torch_rng = torch_rng.to(torch.uint8)
    torch.set_rng_state(torch_rng)
    if torch.cuda.is_available() and "cuda" in rng_states:
        cuda_states = rng_states["cuda"]
        # Convert each CUDA RNG state to ByteTensor if needed
        converted_states = []
        for state in cuda_states:
            if hasattr(state, 'cpu'):
                state = state.cpu()
            if state.dtype != torch.uint8:
                state = state.to(torch.uint8)
            converted_states.append(state)
        torch.cuda.set_rng_state_all(converted_states)


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_metric=None):
    """
    Save full training state checkpoint.
    
    Args:
        path: path to save checkpoint
        model: model (or model.state_dict())
        optimizer: optimizer
        scheduler: lr scheduler (can be None)
        scaler: GradScaler for AMP (can be None)
        epoch: current epoch (0-indexed, completed)
        best_metric: best metric value so far (optional)
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict() if hasattr(model, "state_dict") else model,
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_states": get_rng_states(),
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    
    if best_metric is not None:
        checkpoint["best_metric"] = best_metric
    
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
    """
    Load full training state from checkpoint.
    
    Args:
        path: path to checkpoint
        model: model to load state into
        optimizer: optimizer (optional)
        scheduler: lr scheduler (optional)
        scaler: GradScaler (optional)
        device: device to map tensors to
    
    Returns:
        dict with 'epoch' (int, next epoch to run), 'best_metric' (float or None)
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    if "rng_states" in checkpoint:
        set_rng_states(checkpoint["rng_states"])
    
    return {
        "epoch": checkpoint["epoch"] + 1,  # Next epoch to run
        "best_metric": checkpoint.get("best_metric", None)
    }


def count_parameters(model):
    """
    Count parameters in model, split by encoder/projector/total.
    
    Args:
        model: SimCLR model with encoder and projector attributes
    
    Returns:
        dict with 'encoder', 'projector', 'total' param counts
    """
    if hasattr(model, "param_counts"):
        return model.param_counts()
    
    # Fallback for generic models
    encoder_params = 0
    projector_params = 0
    
    if hasattr(model, "encoder"):
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
    
    if hasattr(model, "projector"):
        projector_params = sum(p.numel() for p in model.projector.parameters())
    
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        "encoder": encoder_params,
        "projector": projector_params,
        "total": total_params
    }


class Timer:
    """
    Simple timer as context manager.
    
    Usage:
        with Timer() as t:
            # code to time
        print(f"Elapsed: {t.elapsed:.2f}s")
    """
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0
    
    def start(self):
        self.start_time = time.perf_counter()
        return self
    
    def stop(self):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
