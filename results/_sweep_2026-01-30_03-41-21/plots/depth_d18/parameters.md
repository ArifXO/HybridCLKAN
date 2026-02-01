# Parameter Comparison - Depth 18

| Variant | Backbone | Head | Encoder Params | Projector Params | Total Params |
|---------|----------|------|----------------|------------------|--------------|
| A | ResNet-MLP | MLP | 11,176,512 | 329,344 | 11,505,856 |
| B | ResNet-MLP | ChebyKAN | 11,176,512 | 2,294,784 | 13,471,296 |
| C | ResNet-KAN | MLP | 11,705,100 | 329,344 | 12,034,444 |
| D | ResNet-KAN | ChebyKAN | 11,705,100 | 2,294,784 | 13,999,884 |