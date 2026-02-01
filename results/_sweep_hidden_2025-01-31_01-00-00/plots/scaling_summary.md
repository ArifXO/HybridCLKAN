# Sweep Results Summary

## Final Metrics by Variant and Depth

| Variant | Depth | Hidden | Params (M) | Time (min) | Loss | AUROC | AP |
|---------|-------|--------|------------|------------|------|-------|-----|
| A | 18 | 1024 | 11.84 | 70.0 | 4.4938 | 0.7274 | 0.1274 |
| A | 18 | 2048 | 12.49 | 70.0 | 4.4958 | 0.7250 | 0.1272 |
| B | 18 | 1024 | 11.84 | 70.2 | 4.5141 | 0.7196 | 0.1252 |
| B | 18 | 2048 | 12.49 | 70.2 | 4.5167 | 0.7235 | 0.1259 |
| C | 18 | 1024 | 12.36 | 71.5 | 4.4939 | 0.7263 | 0.1294 |
| C | 18 | 2048 | 13.02 | 69.8 | 4.4958 | 0.7267 | 0.1312 |
| D | 18 | 1024 | 12.36 | 71.2 | 4.5186 | 0.7234 | 0.1271 |
| D | 18 | 2048 | 13.02 | 68.5 | 4.5208 | 0.7249 | 0.1278 |

## Variant Descriptions
- **A**: ResNet-MLP backbone + MLP projection head (Baseline)
- **B**: ResNet-MLP backbone + ChebyKAN projection head
- **C**: ResNet-KAN backbone + MLP projection head
- **D**: ResNet-KAN backbone + ChebyKAN projection head