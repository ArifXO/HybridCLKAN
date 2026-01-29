"""
Encoder backbones: ResNet-MLP (standard) and ResNet-KAN (from Third_party).
"""

import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models

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

from residual_networks.RKAN_ResNet import RKANet


class ResNetMLPEncoder(nn.Module):
    """
    Standard ResNet encoder (torchvision) without classification head.
    Outputs fixed-dim embeddings.
    """
    
    def __init__(self, depth=18, embedding_dim=512, in_channels=3, pretrained=False):
        super().__init__()
        
        # Select ResNet variant
        resnet_map = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
        }
        if depth not in resnet_map:
            raise ValueError(f"Unsupported ResNet depth: {depth}. Choose from {list(resnet_map.keys())}")
        
        weights = "DEFAULT" if pretrained else None
        self.resnet = resnet_map[depth](weights=weights)
        
        # Adapt first conv if grayscale (in_channels=1)
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Get feature dimension before FC
        if depth in [18, 34]:
            self.feat_dim = 512
        else:  # 50, 101, 152
            self.feat_dim = 2048
        
        # Remove original FC, add projection to embedding_dim if needed
        self.resnet.fc = nn.Identity()
        
        if self.feat_dim != embedding_dim:
            self.fc = nn.Linear(self.feat_dim, embedding_dim)
        else:
            self.fc = nn.Identity()
    
    def forward(self, x):
        # Standard ResNet forward (without FC returns features)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class ResNetKANEncoder(nn.Module):
    """
    ResNet-KAN encoder using RKANet from Third_party/residual_networks.
    Outputs fixed-dim embeddings.
    """
    
    def __init__(self, depth=18, embedding_dim=512, in_channels=3, pretrained=False):
        super().__init__()
        
        
        
        version_map = {
            18: "resnet18",
            34: "resnet34",
            50: "resnet50",
        }
        if depth not in version_map:
            raise ValueError(f"Unsupported ResNet depth: {depth}. Choose from {list(version_map.keys())}")
        
        version = version_map[depth]
        
        # Create RKANet with dummy num_classes (we'll remove FC)
        self.rkanet = RKANet(
            num_classes=embedding_dim,  # Will be replaced
            version=version,
            kan_type="chebyshev",
            pretrained=pretrained,
            n_convs=1,
            reduce_factor=[2, 2, 2, 2],
            mechanisms=[None, None, None, "addition"],  # KAN only on last stage
            spline_order=(3, 2),
            grid_size=(3, 2),
            inv_bottleneck=False,
            shortcut=False
        )
        
        # Adapt first conv if grayscale
        if in_channels != 3:
            self.rkanet.resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Get feature dimension
        if depth in [18, 34]:
            self.feat_dim = 512
        else:
            self.feat_dim = 2048
        
        # Replace FC with projection to embedding_dim
        self.rkanet.resnet.fc = nn.Identity()
        
        if self.feat_dim != embedding_dim:
            self.fc = nn.Linear(self.feat_dim, embedding_dim)
        else:
            self.fc = nn.Identity()
    
    def forward(self, x):
        # RKANet forward without final FC
        out = self.rkanet.resnet.conv1(x)
        out = self.rkanet.resnet.bn1(out)
        out = self.rkanet.resnet.relu(out)
        out = self.rkanet.resnet.maxpool(out)
        
        layers = [
            self.rkanet.resnet.layer1,
            self.rkanet.resnet.layer2,
            self.rkanet.resnet.layer3,
            self.rkanet.resnet.layer4
        ]
        
        for i, (layer, mechanism) in enumerate(zip(layers, self.rkanet.mechanisms)):
            identity = out
            out = layer(out)
            
            if mechanism is not None:
                residual = self.rkanet.conv_reduce[i](identity)
                residual = self.rkanet.silu(residual)
                residual = self.rkanet.kan_conv1[i](residual)
                residual = self.rkanet.kan_bn[i](residual)
                residual = self.rkanet.conv_expand[i](residual)
                residual = self.rkanet.silu(residual)
                
                if i == len(self.rkanet.mechanisms) - 1:
                    residual = self.rkanet.kan_conv2[i](residual)
                residual = self.rkanet.kan_expand_bn[i](residual)
                
                if self.rkanet.shortcut:
                    shortcut = self.rkanet.conv_shortcut[i](identity)
                    shortcut = self.rkanet.shortcut_bn[i](shortcut)
                    residual = residual + shortcut
                
                out = self.rkanet.apply_mechanism(out, residual, i, mechanism)
        
        out = self.rkanet.resnet.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out


def build_encoder(backbone_type, depth=18, embedding_dim=512, in_channels=3, pretrained=False):
    """
    Factory function to build encoder.
    
    Args:
        backbone_type: "resnet_mlp" or "resnet_kan"
        depth: ResNet depth (18, 34, 50)
        embedding_dim: output embedding dimension
        in_channels: input channels (1 for grayscale adapt, 3 for repeat)
        pretrained: use pretrained weights
    
    Returns:
        encoder module
    """
    if backbone_type == "resnet_mlp":
        return ResNetMLPEncoder(
            depth=depth,
            embedding_dim=embedding_dim,
            in_channels=in_channels,
            pretrained=pretrained
        )
    elif backbone_type == "resnet_kan":
        return ResNetKANEncoder(
            depth=depth,
            embedding_dim=embedding_dim,
            in_channels=in_channels,
            pretrained=pretrained
        )
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
