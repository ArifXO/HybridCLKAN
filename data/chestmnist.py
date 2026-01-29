"""
ChestMNIST dataset loading and SimCLR augmentations.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import medmnist
from medmnist import ChestMNIST
import numpy as np


class SimCLRAugmentation:
    """Returns two augmented views of the same image for SimCLR."""
    
    def __init__(self, cfg):
        aug_cfg = cfg["augmentations"]
        size = cfg["data"]["image_size"]
        s = aug_cfg.get("jitter_strength", 0.5)
        
        transform_list = []
        
        # Random resized crop
        if aug_cfg.get("random_crop", True):
            scale = tuple(aug_cfg.get("crop_scale", [0.2, 1.0]))
            transform_list.append(
                transforms.RandomResizedCrop(size, scale=scale)
            )
        
        # Horizontal flip
        if aug_cfg.get("horizontal_flip", True):
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        # Color jitter
        if aug_cfg.get("color_jitter", True):
            transform_list.append(
                transforms.RandomApply([
                    transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
                ], p=0.8)
            )
            transform_list.append(transforms.RandomGrayscale(p=0.2))
        
        # Gaussian blur (typically off for small images like 28x28)
        if aug_cfg.get("gaussian_blur", False):
            kernel_size = aug_cfg.get("blur_kernel_size", 3)
            transform_list.append(
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size)
                ], p=0.5)
            )
        
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        
        self.transform = transforms.Compose(transform_list)
    
    def __call__(self, x):
        return self.transform(x), self.transform(x)


class SimCLRDataset(Dataset):
    """Wraps a dataset to apply SimCLR augmentation and handle grayscale."""
    
    def __init__(self, base_dataset, augmentation, grayscale_mode="repeat"):
        self.base_dataset = base_dataset
        self.augmentation = augmentation
        self.grayscale_mode = grayscale_mode
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        # img is PIL Image from medmnist
        
        # Apply SimCLR augmentation (returns two views)
        view1, view2 = self.augmentation(img)
        
        # Handle grayscale mode
        if self.grayscale_mode == "repeat" and view1.shape[0] == 1:
            view1 = view1.repeat(3, 1, 1)
            view2 = view2.repeat(3, 1, 1)
        # If grayscale_mode == "adapt", keep 1 channel (model stem will handle it)
        
        # label is numpy array for multilabel
        label = torch.from_numpy(label).float().squeeze()
        
        return view1, view2, label


class EvalDataset(Dataset):
    """Dataset for evaluation (single view, with labels)."""
    
    def __init__(self, base_dataset, image_size, grayscale_mode="repeat"):
        self.base_dataset = base_dataset
        self.grayscale_mode = grayscale_mode
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        img = self.transform(img)
        
        if self.grayscale_mode == "repeat" and img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        
        label = torch.from_numpy(label).float().squeeze()
        
        return img, label


def get_chestmnist_datasets(cfg, split="train", as_simclr=True):
    """
    Load ChestMNIST dataset.
    
    Args:
        cfg: config dict
        split: "train", "val", or "test"
        as_simclr: if True, returns SimCLRDataset with two views; else EvalDataset
    
    Returns:
        Dataset
    """
    data_cfg = cfg["data"]
    size = data_cfg["image_size"]
    grayscale_mode = data_cfg.get("grayscale_mode", "repeat")
    
    # Download and load ChestMNIST
    # medmnist returns PIL images when transform=None
    base_dataset = ChestMNIST(
        split=split,
        transform=None,
        download=True,
        as_rgb=False,  # Keep as grayscale, we handle conversion
        size=size
    )
    
    if as_simclr:
        augmentation = SimCLRAugmentation(cfg)
        return SimCLRDataset(base_dataset, augmentation, grayscale_mode)
    else:
        return EvalDataset(base_dataset, size, grayscale_mode)


def get_dataloaders(cfg, splits=("train",), as_simclr=True):
    """
    Get DataLoaders for specified splits.
    
    Args:
        cfg: config dict
        splits: tuple of split names
        as_simclr: if True, SimCLR augmentation; else eval mode
    
    Returns:
        dict of DataLoaders
    """
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]
    
    batch_size = training_cfg["batch_size"]
    num_workers = data_cfg.get("num_workers", 4)
    
    loaders = {}
    for split in splits:
        dataset = get_chestmnist_datasets(cfg, split=split, as_simclr=as_simclr)
        shuffle = (split == "train")
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train" and as_simclr)
        )
    
    return loaders
