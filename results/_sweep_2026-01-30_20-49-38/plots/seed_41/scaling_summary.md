# Sweep Results Summary

## Final Metrics by Variant and Depth

| Variant | Depth | Params (M) | Time (min) | Loss | AUROC | AP |
|---------|-------|------------|------------|------|-------|-----|
| A | 18 | 11.51 | 75.9 | 4.4952 | 0.7250 | 0.1295 |
| A | 34 | 21.61 | 79.5 | 4.4900 | 0.7246 | 0.1307 |
| B | 18 | 11.50 | 78.1 | 4.5091 | 0.7215 | 0.1260 |
| B | 34 | 21.61 | 68.9 | 4.5053 | 0.7207 | 0.1269 |
| C | 18 | 12.03 | 78.1 | 4.4946 | 0.7251 | 0.1285 |
| C | 34 | 22.14 | 80.4 | 4.4909 | 0.7256 | 0.1271 |
| D | 18 | 12.03 | 77.5 | 4.5084 | 0.7249 | 0.1269 |
| D | 34 | 22.14 | 76.6 | 4.5085 | 0.7272 | 0.1272 |

## Variant Descriptions
- **A**: ResNet-MLP backbone + MLP projection head (Baseline)
- **B**: ResNet-MLP backbone + ChebyKAN projection head
- **C**: ResNet-KAN backbone + MLP projection head
- **D**: ResNet-KAN backbone + ChebyKAN projection head