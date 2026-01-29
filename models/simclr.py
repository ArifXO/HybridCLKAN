"""
SimCLR model: encoder + projector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLR(nn.Module):
    """
    SimCLR model combining encoder (backbone) and projector (head).
    
    Args:
        encoder: backbone network that outputs embeddings
        projector: projection head that maps embeddings to contrastive space
    """
    
    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
    
    def encode(self, x):
        """
        Extract features using encoder only.
        
        Args:
            x: input images [N, C, H, W]
        
        Returns:
            h: encoder embeddings [N, embedding_dim]
        """
        return self.encoder(x)
    
    def project(self, h):
        """
        Project embeddings through projection head.
        
        Args:
            h: encoder embeddings [N, embedding_dim]
        
        Returns:
            z: projected embeddings [N, projection_dim]
        """
        return self.projector(h)
    
    def forward(self, x):
        """
        Full forward pass: encode + project + L2 normalize.
        
        Args:
            x: input images [N, C, H, W]
        
        Returns:
            z: L2-normalized projected embeddings [N, projection_dim]
        """
        h = self.encode(x)
        z = self.project(h)
        z = F.normalize(z, dim=1)
        return z
    
    def get_encoder_params(self):
        """Returns number of parameters in encoder."""
        return sum(p.numel() for p in self.encoder.parameters())
    
    def get_projector_params(self):
        """Returns number of parameters in projector."""
        return sum(p.numel() for p in self.projector.parameters())
    
    def get_total_params(self):
        """Returns total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def param_counts(self):
        """Returns dict with encoder, projector, and total param counts."""
        return {
            "encoder": self.get_encoder_params(),
            "projector": self.get_projector_params(),
            "total": self.get_total_params()
        }


def build_simclr(cfg):
    """
    Build SimCLR model from config.
    
    Args:
        cfg: config dict with model settings
    
    Returns:
        SimCLR model
    """
    from models.encoders import build_encoder
    from models.projectors import build_projector
    
    model_cfg = cfg["model"]
    
    # Determine backbone and head from variant or explicit config
    variant = cfg.get("variant", "A")
    variant_map = {
        "A": ("resnet_mlp", "mlp"),
        "B": ("resnet_mlp", "chebykan"),
        "C": ("resnet_kan", "mlp"),
        "D": ("resnet_kan", "chebykan"),
    }
    
    # Use variant defaults, but allow explicit override
    default_backbone, default_head = variant_map.get(variant, ("resnet_mlp", "mlp"))
    backbone = model_cfg.get("backbone", default_backbone)
    head = model_cfg.get("head", default_head)
    
    # Build encoder
    encoder = build_encoder(
        backbone_type=backbone,
        depth=model_cfg.get("resnet_depth", 18),
        embedding_dim=model_cfg.get("embedding_dim", 512),
        in_channels=1 if cfg["data"].get("grayscale_mode") == "adapt" else 3
    )
    
    # Build projector
    projector = build_projector(
        head_type=head,
        input_dim=model_cfg.get("embedding_dim", 512),
        hidden_dim=model_cfg.get("projection_hidden_dim", 512),
        output_dim=model_cfg.get("projection_dim", 128),
        chebykan_degree=model_cfg.get("chebykan_degree", 4)
    )
    
    return SimCLR(encoder, projector)
