# Sweep Experiment Results

**Generated**: hidden_2025-01-31_01-00-00

## Configuration
- **Variants**: A, B, C, D
- **Depths**: 18
- **Hidden Dims**: 1024, 2048
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
_sweep_hidden_2025-01-31_01-00-00/
├── README.md
├── plots/
│   ├── depth_d18_h1024/
│   ├── depth_d18_h2048/
│   ├── scaling_*.png
│   └── *_summary.md
├── A/
│   └── d18/
│       └── h1024/
│       └── h2048/
├── B/
│   └── d18/
│       └── h1024/
│       └── h2048/
├── C/
│   └── d18/
│       └── h1024/
│       └── h2048/
├── D/
│   └── d18/
│       └── h1024/
│       └── h2048/
```

## Results
See `plots/scaling_summary.md` for full results table.

## Hidden Dimension Sweep Notes
- This sweep tested 2 projector hidden dimensions: [1024, 2048]
- Each combination of variant/depth runs with different hidden dimensions