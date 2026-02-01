# Sweep Experiment Results

**Generated**: 2026-01-30_03-41-21

## Configuration
- **Variants**: A, B, C, D
- **Depths**: 18, 34
- **Seeds**: 42
- **Epochs**: 100
- **Eval Every**: 10 epochs

## Variants
| Variant | Backbone | Projection Head |
|---------|----------|-----------------|
| A | ResNet-MLP | MLP |
| B | ResNet-MLP | ChebyKAN |
| C | ResNet-KAN | MLP |
| D | ResNet-KAN | ChebyKAN |

## Directory Structure
```
_sweep_2026-01-30_03-41-21/
├── README.md
├── plots/
│   ├── depth_d18/
│   ├── depth_d34/
│   ├── scaling_*.png
│   └── *_summary.md
├── A/
│   └── d18/
│   └── d34/
├── B/
│   └── d18/
│   └── d34/
├── C/
│   └── d18/
│   └── d34/
├── D/
│   └── d18/
│   └── d34/
```

## Results
See `plots/scaling_summary.md` for full results table.