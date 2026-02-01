# Parameter Comparison - Depth 18, Hidden 1024

| Variant | Backbone | Head | Encoder Params | Projector Params | Total Params |
|---------|----------|------|----------------|------------------|--------------|
| A | ResNet-MLP | MLP | 11,176,512 | 658,560 | 11,835,072 |
| B | ResNet-MLP | ChebyKAN | 11,176,512 | 658,854 | 11,835,366 |
| C | ResNet-KAN | MLP | 11,705,100 | 658,560 | 12,363,660 |
| D | ResNet-KAN | ChebyKAN | 11,705,100 | 658,854 | 12,363,954 |