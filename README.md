# SimCLR with KAN: Contrastive Learning for Medical Imaging

A research project exploring **Kolmogorov-Arnold Networks (KAN)** as replacements for MLPs in self-supervised contrastive learning pipelines. This repository implements SimCLR pretraining on medical imaging data (ChestMNIST) with four model variants comparing standard ResNet/MLP architectures against KAN-based alternatives.

## 🎯 Research Objective

Investigate whether KAN-based architectures can improve representation learning in self-supervised contrastive learning frameworks by:

1. **Replacing the projection head**: Substituting the standard MLP projection head with a ChebyshevKAN head
2. **Replacing the backbone**: Using ResNet-KAN (RKANet) instead of standard ResNet
3. **Comparing all combinations** across different model scales (ResNet-18/34/50)

## 📊 Model Variants

| Variant | Encoder (Backbone) | Projector (Head) | Description |
|---------|-------------------|------------------|-------------|
| **A** | ResNet-MLP | MLP | Baseline: Standard SimCLR |
| **B** | ResNet-MLP | ChebyKAN | KAN projection head only |
| **C** | ResNet-KAN | MLP | KAN backbone only |
| **D** | ResNet-KAN | ChebyKAN | Full KAN: backbone + head |

## 🏗️ Repository Structure

```
├── config/
│   └── pretrain.yaml          # Main configuration file
├── data/
│   └── chestmnist.py          # Data loading, SimCLR augmentations
├── losses/
│   └── ntxent.py              # NT-Xent contrastive loss, alignment/uniformity
├── models/
│   ├── encoders.py            # ResNet-MLP and ResNet-KAN encoders
│   ├── projectors.py          # MLP and ChebyKAN projection heads
│   └── simclr.py              # SimCLR model wrapper
├── scripts/
│   ├── pretrain.py            # Main training script with periodic eval
│   └── sweep_scaling.py       # Grid experiments across variants/depths
├── utils/                     # Utilities package
│   ├── __init__.py
│   ├── core.py                # Core utilities (checkpointing, metrics)
│   └── plotting.py            # Sweep plotting and visualization
├── Third_party/
│   ├── KAN_Conv/              # ChebyshevKAN implementation
│   │   ├── chebyshevkan.py    # ChebyshevKANLinear layer
│   │   └── ...
│   └── residual_networks/     # ResNet-KAN implementation
│       └── RKAN_ResNet.py     # RKANet (ResNet with KAN layers)
└── results/                   # Training outputs, checkpoints, metrics
```

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd VV

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install medmnist scikit-learn matplotlib umap-learn pyyaml tqdm
```

## 📖 Usage

### Single Pretraining Run

```bash
# Train variant A (baseline) with ResNet-18
python scripts/pretrain.py --run_name simclr_A_d18 --variant A --resnet_depth 18 --epochs 100

# Train variant B (KAN projection head) with ResNet-34
python scripts/pretrain.py --run_name simclr_B_d34 --variant B --resnet_depth 34 --epochs 100

# Train with periodic evaluation every 10 epochs
python scripts/pretrain.py --run_name simclr_C_d18 --variant C --eval_every 10 --epochs 100
```

### Full Sweep Experiment

Run all 4 variants × 3 depths (12 experiments total):

```bash
python scripts/sweep_scaling.py \
    --variants A B C D \
    --depths 18 34 50 \
    --epochs 50 \
    --batch_size 256 \
    --eval_every 10 \
    --seed 42
```

This creates a structured output:
```
results/_sweep_<timestamp>/
├── README.md                    # Sweep configuration summary
├── plots/
│   ├── depth_d18/               # Per-depth comparison
│   │   ├── loss_curve_variants.png
│   │   ├── linear_probe_auroc_variants.png
│   │   ├── alignment_variants.png
│   │   ├── uniformity_variants.png
│   │   ├── ap_curve_variants.png
│   │   └── parameters.md
│   ├── depth_d34/
│   ├── depth_d50/
│   ├── scaling_auc_vs_params.png
│   ├── scaling_auc_vs_time.png
│   ├── auc_by_variant.png
│   ├── efficiency_by_variant.png
│   ├── param_counts_by_variant.png
│   ├── scaling_summary.md
│   └── plot_summary.md
├── A/d18/, A/d34/, A/d50/       # Variant A results
├── B/d18/, B/d34/, B/d50/       # Variant B results
├── C/d18/, C/d34/, C/d50/       # Variant C results
└── D/d18/, D/d34/, D/d50/       # Variant D results
```

**Training order**: For each depth, all variants run in order A→B→C→D before moving to next depth:
```
A/d18 → B/d18 → C/d18 → D/d18 → A/d34 → B/d34 → C/d34 → D/d34 → A/d50 → B/d50 → C/d50 → D/d50
```

To generate plots from an existing sweep (without retraining):
```bash
python scripts/sweep_scaling.py --skip_training --sweep_dir results/_sweep_2026-01-29_20-03-08
```

### Resume Training

```bash
python scripts/pretrain.py --run_name simclr_A_d18 --resume results/simclr_A_d18/last_model.pt
```

## ⚙️ Configuration

The main configuration file is `config/pretrain.yaml`:

```yaml
# Model variants: A, B, C, D
variant: "A"

# Data settings
data:
  dataset: "chestmnist"      # MedMNIST ChestMNIST (14 pathologies)
  image_size: 28
  grayscale_mode: "repeat"   # Convert 1-channel to 3-channel

# Model architecture
model:
  resnet_depth: 18           # 18, 34, or 50
  embedding_dim: 512         # Encoder output dimension
  projection_dim: 128        # Projector output dimension
  projection_hidden_dim: 512 # Projector hidden dimension
  chebykan_degree: 4         # Chebyshev polynomial degree (for KAN)

# Training
training:
  epochs: 100
  batch_size: 256
  lr: 0.001
  weight_decay: 1e-4
  optimizer: adam
  scheduler: cosine
  warmup_epochs: 10

# SimCLR settings
simclr:
  temperature: 0.5           # NT-Xent temperature

# Augmentations (SimCLR-style)
augmentations:
  random_crop: true
  crop_scale: [0.2, 1.0]
  horizontal_flip: true
  color_jitter: true
  jitter_strength: 0.5
  gaussian_blur: false
```

### CLI Overrides

All config values can be overridden via command line:

```bash
python scripts/pretrain.py \
    --variant B \
    --resnet_depth 34 \
    --epochs 200 \
    --batch_size 512 \
    --lr 0.0005
```

## 📈 Metrics & Outputs

### Training Metrics (saved to `metrics.jsonl`)

Each epoch saves a JSON line with:
- `epoch`: Current epoch number (1-indexed)
- `train_loss`: NT-Xent contrastive loss
- `lr`: Current learning rate
- `epoch_time_sec`: Epoch training time in seconds
- `alignment`: Mean pairwise distance of positive pairs (lower = better)
- `uniformity`: Distribution uniformity on hypersphere (more negative = better)
- `param_counts`: Parameter counts (saved on first epoch only)
  - `encoder`: Encoder/backbone parameters
  - `projector`: Projector/head parameters  
  - `total`: Total model parameters

On evaluation epochs (controlled by `--eval_every`):
- `linear_probe_auroc`: Linear probe evaluation AUROC (multi-label)
- `linear_probe_ap`: Linear probe average precision

### Checkpoints

- `best.ckpt`: Model with lowest training loss
- `last.ckpt`: Most recent model (for resuming)

### Plots (from sweep)

Per-depth plots (`plots/depth_dN/`):
- `loss_curve_variants.png`: Training loss over epochs
- `linear_probe_auroc_variants.png`: Linear probe AUROC over epochs
- `alignment_variants.png`: Alignment metric over epochs
- `uniformity_variants.png`: Uniformity metric over epochs
- `ap_curve_variants.png`: Average precision over epochs
- `lr_curve_variants.png`: Learning rate schedule
- `parameters.md`: Parameter count comparison table

Scaling analysis plots (`plots/`):
- `scaling_auc_vs_params.png`: AUROC vs model size
- `scaling_auc_vs_time.png`: AUROC vs training time
- `auc_by_variant.png`: Final AUROC bar chart
- `efficiency_by_variant.png`: AUROC per million parameters
- `param_counts_by_variant.png`: Model sizes comparison

## 🧠 Technical Details

### SimCLR Framework

1. **Data Augmentation**: Each image generates two augmented views (positive pair)
2. **Encoder**: Extracts representations from augmented views
3. **Projector**: Maps representations to contrastive embedding space
4. **NT-Xent Loss**: Attracts positive pairs, repels negative pairs

### KAN (Kolmogorov-Arnold Networks)

Unlike MLPs that use fixed activation functions, KANs use **learnable activation functions** parameterized by Chebyshev polynomials. This allows:

- More expressive function approximation
- Potentially better feature learning
- Interpretable learned activations

### ChebyshevKAN

The ChebyshevKAN layer replaces standard `Linear + ReLU` with:
```
y = Σ cᵢ · Tᵢ(x)
```
where `Tᵢ` are Chebyshev polynomials of the first kind and `cᵢ` are learnable coefficients.

## 📚 Dataset

**ChestMNIST** from the MedMNIST collection:
- 78,468 training / 11,219 validation / 22,433 test images
- 28×28 grayscale chest X-rays
- 14-class multi-label classification (thoracic diseases)
- Evaluation metric: Area Under ROC Curve (AUC)

## 🔬 Experiment Tracking

Each run saves:
```
results/<run_name>/
├── config_resolved.yaml   # Full resolved configuration
├── metrics.jsonl          # Epoch-by-epoch metrics (JSONL format)
├── best.ckpt              # Best checkpoint
└── last.ckpt              # Latest checkpoint (for resuming)
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{simclr-kan-2026,
  title={SimCLR with KAN: Exploring Kolmogorov-Arnold Networks for Self-Supervised Medical Image Representation Learning},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/your-repo}}
}
```

## 🙏 Acknowledgments

- [SimCLR](https://arxiv.org/abs/2002.05709) by Chen et al.
- [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) by Liu et al.
- [MedMNIST](https://medmnist.com/) dataset collection
- Third-party implementations: KAN_Conv, RKAN_ResNet

## 📄 License

This project is for academic research purposes. See individual third-party licenses in the `Third_party/` directory.
