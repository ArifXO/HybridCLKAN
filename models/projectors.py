"""
Projection heads: MLP and ChebyKAN.
"""

import sys
import os
import torch
import torch.nn as nn

# Add Third_party to path for imports (prefer Third_party, fallback to "Third party")
_base_dir = os.path.dirname(__file__)
_third_party_candidates = [
    os.path.join(_base_dir, "..", "Third_party"),
    os.path.join(_base_dir, "..", "Third party"),
]
THIRD_PARTY_PATH = None
for _candidate in _third_party_candidates:
    if os.path.isdir(_candidate):
        THIRD_PARTY_PATH = os.path.abspath(_candidate)
        break
if THIRD_PARTY_PATH and THIRD_PARTY_PATH not in sys.path:
    sys.path.insert(0, THIRD_PARTY_PATH)
    
from KAN_Conv.chebyshevkan import ChebyshevKANLinear


class MLPProjector(nn.Module):
    """
    Standard MLP projection head for SimCLR.
    Architecture: Linear -> BN -> ReLU -> Linear
    """
    
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projector(x)


class ChebyKANProjector(nn.Module):
    """
    ChebyKAN projection head using ChebyshevKANLinear from Third_party.
    Architecture: ChebyshevKANLinear -> BN -> ChebyshevKANLinear
    """
    
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128, degree=4):
        super().__init__()
        
        
        
        self.layer1 = ChebyshevKANLinear(
            in_features=input_dim,
            out_features=hidden_dim,
            polynomial_degree=degree,
            enable_scaler=True,
            base_activation=nn.SiLU,
            use_linear=True,
            skip_activation=True,
            normalization="tanh",
            polynomial_type="chebyshev",
            use_layernorm=False
        )
        
        self.bn = nn.BatchNorm1d(hidden_dim)
        
        self.layer2 = ChebyshevKANLinear(
            in_features=hidden_dim,
            out_features=output_dim,
            polynomial_degree=degree,
            enable_scaler=True,
            base_activation=nn.SiLU,
            use_linear=True,
            skip_activation=True,
            normalization="tanh",
            polynomial_type="chebyshev",
            use_layernorm=False
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.bn(x)
        x = self.layer2(x)
        return x


def build_projector(head_type, input_dim=512, hidden_dim=512, output_dim=128, chebykan_degree=4):
    """
    Factory function to build projector.
    
    Args:
        head_type: "mlp" or "chebykan"
        input_dim: input dimension (encoder embedding_dim)
        hidden_dim: hidden layer dimension
        output_dim: output dimension (contrastive space)
        chebykan_degree: polynomial degree for ChebyKAN
    
    Returns:
        projector module
    """
    if head_type == "mlp":
        return MLPProjector(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
    elif head_type == "chebykan":
        return ChebyKANProjector(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            degree=chebykan_degree
        )
    else:
        raise ValueError(f"Unknown head type: {head_type}")
