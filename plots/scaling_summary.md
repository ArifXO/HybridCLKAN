# Sweep Results Summary

## Final Metrics by Variant and Depth

| Variant | Depth | Params (M) | Time (min) | Loss | AUROC | AP | Align | Uniform |
|---------|-------|------------|------------|------|-------|-----|-------|---------|
| A | 18 | 11.51 | 38.8 | 4.5259 | 0.7153 | 0.1214 | 0.1817 | -3.4107 |
| A | 34 | 21.61 | 39.4 | 4.5231 | 0.7124 | 0.1216 | 0.1770 | -3.4160 |
| B | 18 | 13.47 | 39.0 | 4.5508 | 0.7135 | 0.1200 | 0.1562 | -3.2423 |
| B | 34 | 23.58 | 39.5 | 4.5455 | 0.7117 | 0.1189 | 0.1546 | -3.2439 |
| C | 18 | 12.03 | 39.6 | 4.5248 | 0.7173 | 0.1234 | 0.1811 | -3.4159 |
| C | 34 | 22.14 | 41.9 | 4.5240 | 0.7191 | 0.1248 | 0.1759 | -3.4198 |
| D | 18 | 14.00 | 39.6 | 4.5509 | 0.7199 | 0.1217 | 0.1502 | -3.2163 |
| D | 34 | 24.11 | 42.7 | 4.5549 | 0.7147 | 0.1198 | 0.1461 | -3.1984 |

## Variant Descriptions
- **A**: ResNet-MLP backbone + MLP projection head (Baseline)
- **B**: ResNet-MLP backbone + ChebyKAN projection head
- **C**: ResNet-KAN backbone + MLP projection head
- **D**: ResNet-KAN backbone + ChebyKAN projection head