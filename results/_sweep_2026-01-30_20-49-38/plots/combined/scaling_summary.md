# Sweep Results Summary

## Final Metrics by Variant and Depth

| Variant | Depth | Seeds | Params (M) | Time (min) | Loss | AUROC | AP |
|---------|-------|-------|------------|------------|------|-------|-----|
| A | 18 | 2 | 11.51 | 76.7 | 4.4952±0.0001 | 0.7255±0.0004 | 0.1284±0.0011 |
| A | 34 | 2 | 21.61 | 79.3 | 4.4905±0.0005 | 0.7251±0.0005 | 0.1295±0.0011 |
| B | 18 | 2 | 11.50 | 78.0 | 4.5098±0.0007 | 0.7210±0.0005 | 0.1257±0.0003 |
| B | 34 | 2 | 21.61 | 68.5 | 4.5059±0.0007 | 0.7206±0.0001 | 0.1262±0.0007 |
| C | 18 | 2 | 12.03 | 78.2 | 4.4941±0.0005 | 0.7271±0.0020 | 0.1299±0.0014 |
| C | 34 | 2 | 22.14 | 76.4 | 4.4896±0.0013 | 0.7250±0.0005 | 0.1276±0.0005 |
| D | 18 | 2 | 12.03 | 78.5 | 4.5114±0.0030 | 0.7256±0.0008 | 0.1276±0.0007 |
| D | 34 | 2 | 22.14 | 76.6 | 4.5063±0.0022 | 0.7257±0.0015 | 0.1270±0.0002 |

## Variant Descriptions
- **A**: ResNet-MLP backbone + MLP projection head (Baseline)
- **B**: ResNet-MLP backbone + ChebyKAN projection head
- **C**: ResNet-KAN backbone + MLP projection head
- **D**: ResNet-KAN backbone + ChebyKAN projection head