# Parameter Comparison - Depth 18, Hidden 2048

| Variant | Backbone | Head | Encoder Params | Projector Params | Total Params |
|---------|----------|------|----------------|------------------|--------------|
| A | ResNet-MLP | MLP | 11,176,512 | 1,316,992 | 12,493,504 |
| B | ResNet-MLP | ChebyKAN | 11,176,512 | 1,317,708 | 12,494,220 |
| C | ResNet-KAN | MLP | 11,705,100 | 1,316,992 | 13,022,092 |
| D | ResNet-KAN | ChebyKAN | 11,705,100 | 1,317,708 | 13,022,808 |