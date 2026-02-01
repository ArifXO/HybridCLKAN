# Sweep Results Summary

## Final Metrics by Variant and Depth

| Variant | Depth | Params (M) | Time (min) | Loss | AUROC | AP |
|---------|-------|------------|------------|------|-------|-----|
| A | 18 | 11.51 | 77.5 | 4.4953 | 0.7259 | 0.1273 |
| A | 34 | 21.61 | 79.1 | 4.4910 | 0.7256 | 0.1284 |
| B | 18 | 11.50 | 78.0 | 4.5105 | 0.7205 | 0.1254 |
| B | 34 | 21.61 | 68.2 | 4.5066 | 0.7205 | 0.1255 |
| C | 18 | 12.03 | 78.2 | 4.4936 | 0.7291 | 0.1313 |
| C | 34 | 22.14 | 72.5 | 4.4883 | 0.7245 | 0.1281 |
| D | 18 | 12.03 | 79.4 | 4.5144 | 0.7264 | 0.1283 |
| D | 34 | 22.14 | 76.7 | 4.5041 | 0.7242 | 0.1269 |

## Variant Descriptions
- **A**: ResNet-MLP backbone + MLP projection head (Baseline)
- **B**: ResNet-MLP backbone + ChebyKAN projection head
- **C**: ResNet-KAN backbone + MLP projection head
- **D**: ResNet-KAN backbone + ChebyKAN projection head