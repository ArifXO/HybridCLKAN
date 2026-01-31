# Parameter Comparison - Depth 34

| Variant | Backbone | Head | Encoder Params | Projector Params | Total Params |
|---------|----------|------|----------------|------------------|--------------|
| A | ResNet-MLP | MLP | 21,284,672 | 329,344 | 21,614,016 |
| B | ResNet-MLP | ChebyKAN | 21,284,672 | 327,186 | 21,611,858 |
| C | ResNet-KAN | MLP | 21,813,260 | 329,344 | 22,142,604 |
| D | ResNet-KAN | ChebyKAN | 21,813,260 | 327,186 | 22,140,446 |