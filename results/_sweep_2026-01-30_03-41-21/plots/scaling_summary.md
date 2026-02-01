# Sweep Results Summary

## Final Metrics by Variant and Depth

| Variant | Depth | Params (M) | Time (min) | Loss | AUROC | AP |
|---------|-------|------------|------------|------|-------|-----|
| A | 18 | 11.51 | 38.8 | 4.5259 | 0.7153 | 0.1214 |
| A | 34 | 21.61 | 39.4 | 4.5231 | 0.7124 | 0.1216 |
| B | 18 | 13.47 | 39.0 | 4.5508 | 0.7135 | 0.1200 |
| B | 34 | 23.58 | 39.5 | 4.5455 | 0.7117 | 0.1189 |
| C | 18 | 12.03 | 39.6 | 4.5248 | 0.7173 | 0.1234 |
| C | 34 | 22.14 | 41.9 | 4.5240 | 0.7191 | 0.1248 |
| D | 18 | 14.00 | 39.6 | 4.5509 | 0.7199 | 0.1217 |
| D | 34 | 24.11 | 42.7 | 4.5549 | 0.7147 | 0.1198 |

## Variant Descriptions
- **A**: ResNet-MLP backbone + MLP projection head (Baseline)
- **B**: ResNet-MLP backbone + ChebyKAN projection head
- **C**: ResNet-KAN backbone + MLP projection head
- **D**: ResNet-KAN backbone + ChebyKAN projection head