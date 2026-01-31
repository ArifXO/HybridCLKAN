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


# Constants for param matching
_MAX_HIDDEN_SEARCH = 2048
_CHEBYKAN_DEGREE = 4
_CHEBYKAN_USE_LINEAR = True
_CHEBYKAN_ENABLE_SCALER = True


def count_mlp_projector_params(input_dim: int, hidden_dim: int, output_dim: int) -> int:
    """
    Count parameters in MLPProjector: Linear -> BN -> ReLU -> Linear.
    
    Linear1: input_dim * hidden_dim + hidden_dim (weight + bias)
    BN: 2 * hidden_dim (gamma + beta)
    Linear2: hidden_dim * output_dim + output_dim (weight + bias)
    """
    linear1 = input_dim * hidden_dim + hidden_dim
    bn = 2 * hidden_dim  # gamma and beta
    linear2 = hidden_dim * output_dim + output_dim
    return linear1 + bn + linear2


def count_chebykan_projector_params(input_dim: int, hidden_dim: int, output_dim: int,
                                     degree: int = _CHEBYKAN_DEGREE,
                                     use_linear: bool = _CHEBYKAN_USE_LINEAR,
                                     enable_scaler: bool = _CHEBYKAN_ENABLE_SCALER) -> int:
    """
    Count parameters in ChebyKANProjector: ChebyKANLinear -> BN -> ChebyKANLinear.
    
    For each ChebyshevKANLinear:
      - polynomial_weight: out_features * in_features * (degree + 1)
      - base_linear.weight (if use_linear): out_features * in_features (no bias)
      - scaler (if enable_scaler): out_features * in_features
    BN: 2 * hidden_dim (gamma + beta)
    """
    def kan_linear_params(in_f, out_f):
        params = out_f * in_f * (degree + 1)  # polynomial_weight
        if use_linear:
            params += out_f * in_f  # base_linear.weight (no bias)
        if enable_scaler:
            params += out_f * in_f  # scaler
        return params
    
    layer1 = kan_linear_params(input_dim, hidden_dim)
    bn = 2 * hidden_dim  # gamma and beta
    layer2 = kan_linear_params(hidden_dim, output_dim)
    return layer1 + bn + layer2


def solve_hidden_dim_for_target(target_params: int, input_dim: int, output_dim: int,
                                 degree: int = _CHEBYKAN_DEGREE,
                                 use_linear: bool = _CHEBYKAN_USE_LINEAR,
                                 enable_scaler: bool = _CHEBYKAN_ENABLE_SCALER) -> int:
    """
    Find hidden_dim that makes ChebyKAN param count closest to target_params.
    Scans hidden_dim from 1 to _MAX_HIDDEN_SEARCH, picks smallest hidden_dim in ties.
    """
    best_hidden = 1
    best_diff = float('inf')
    
    for h in range(1, _MAX_HIDDEN_SEARCH + 1):
        params = count_chebykan_projector_params(input_dim, h, output_dim, degree, use_linear, enable_scaler)
        diff = abs(params - target_params)
        if diff < best_diff:
            best_diff = diff
            best_hidden = h
    
    return best_hidden


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


def build_projector(head_type, input_dim=512, hidden_dim=512, output_dim=128, chebykan_degree=4,
                    chebykan_match_mlp_params=False):
    """
    Factory function to build projector.
    
    Args:
        head_type: "mlp" or "chebykan"
        input_dim: input dimension (encoder embedding_dim)
        hidden_dim: hidden layer dimension (used directly for MLP, or as reference for param matching)
        output_dim: output dimension (contrastive space)
        chebykan_degree: polynomial degree for ChebyKAN
        chebykan_match_mlp_params: if True and head_type='chebykan', choose hidden_dim
                                   so ChebyKAN params match MLP params
    
    Returns:
        tuple: (projector module, build_info dict)
    """
    build_info = {
        "head_type": head_type,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "chebykan_match_mlp_params": chebykan_match_mlp_params,
    }
    
    if head_type == "mlp":
        build_info["projector_hidden_dim_used"] = hidden_dim
        build_info["projector_params_actual"] = count_mlp_projector_params(input_dim, hidden_dim, output_dim)
        projector = MLPProjector(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        return projector, build_info
    
    elif head_type == "chebykan":
        if chebykan_match_mlp_params:
            # Compute target params from MLP with same hidden_dim
            target_params = count_mlp_projector_params(input_dim, hidden_dim, output_dim)
            # Solve for ChebyKAN hidden_dim
            cheby_hidden = solve_hidden_dim_for_target(target_params, input_dim, output_dim, chebykan_degree)
            actual_params = count_chebykan_projector_params(input_dim, cheby_hidden, output_dim, chebykan_degree)
            
            print(f"Param-match ON: MLP target params={target_params:,}, "
                  f"Cheby hidden_dim={cheby_hidden}, Cheby params={actual_params:,}")
            
            build_info["projector_params_target"] = target_params
            build_info["projector_hidden_dim_used"] = cheby_hidden
            build_info["projector_params_actual"] = actual_params
            build_info["mlp_reference_hidden_dim"] = hidden_dim
        else:
            cheby_hidden = hidden_dim
            build_info["projector_hidden_dim_used"] = cheby_hidden
            build_info["projector_params_actual"] = count_chebykan_projector_params(
                input_dim, cheby_hidden, output_dim, chebykan_degree
            )
        
        projector = ChebyKANProjector(
            input_dim=input_dim,
            hidden_dim=cheby_hidden,
            output_dim=output_dim,
            degree=chebykan_degree
        )
        return projector, build_info
    
    else:
        raise ValueError(f"Unknown head type: {head_type}")
