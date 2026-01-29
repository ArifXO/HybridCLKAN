"""
NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss for SimCLR.
Includes alignment and uniformity metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent loss for SimCLR.
    
    Given a batch of N samples with two views each (2N total),
    treats the two views of the same sample as positives,
    and all other 2(N-1) samples as negatives.
    """
    
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2):
        """
        Args:
            z1: [N, D] embeddings from view 1
            z2: [N, D] embeddings from view 2
        
        Returns:
            loss: scalar NT-Xent loss
        """
        N = z1.size(0)
        device = z1.device
        
        # Normalize embeddings to unit sphere
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate: [z1_0, z1_1, ..., z1_N-1, z2_0, z2_1, ..., z2_N-1]
        z = torch.cat([z1, z2], dim=0)  # [2N, D]
        
        # Cosine similarity matrix [2N, 2N]
        sim = torch.mm(z, z.t()) / self.temperature  # [2N, 2N]
        
        # Create mask to exclude self-similarity (diagonal)
        mask_self = torch.eye(2 * N, dtype=torch.bool, device=device)
        
        # For each sample i, its positive is at index (i + N) % (2N)
        # z1[i] pairs with z2[i], and z2[i] pairs with z1[i]
        # Positive pairs: (0, N), (1, N+1), ..., (N-1, 2N-1), (N, 0), (N+1, 1), ...
        
        # Mask out self-similarity by setting diagonal to large negative
        sim = sim.masked_fill(mask_self, -1e9)
        
        # Positive pair indices
        # For i in [0, N-1]: positive is at i + N
        # For i in [N, 2N-1]: positive is at i - N
        pos_indices = torch.cat([
            torch.arange(N, 2 * N, device=device),  # z1[i] -> z2[i]
            torch.arange(0, N, device=device)       # z2[i] -> z1[i]
        ])  # [2N]
        
        # Gather positive similarities
        pos_sim = sim[torch.arange(2 * N, device=device), pos_indices]  # [2N]
        
        # Log-sum-exp over all negatives (row-wise)
        # logsumexp includes positives too, but we use it as denominator
        logsumexp = torch.logsumexp(sim, dim=1)  # [2N]
        
        # NT-Xent loss: -log(exp(pos) / sum(exp(all))) = -pos + logsumexp
        loss = -pos_sim + logsumexp
        
        return loss.mean()


def alignment(z1, z2):
    """
    Alignment metric: measures closeness of positive pairs.
    Lower is better.
    
    alignment = E[||z1 - z2||^2] where z1, z2 are L2-normalized.
    
    Args:
        z1: [N, D] embeddings from view 1
        z2: [N, D] embeddings from view 2
    
    Returns:
        scalar alignment value
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    return ((z1 - z2) ** 2).sum(dim=1).mean()


def uniformity(z, t=2.0):
    """
    Uniformity metric: measures how uniformly distributed embeddings are on hypersphere.
    Lower is better (more uniform).
    
    uniformity = log E[exp(-t * ||z_i - z_j||^2)]
    
    Args:
        z: [N, D] embeddings (can be concatenation of z1 and z2)
        t: temperature parameter (default 2.0 as in original paper)
    
    Returns:
        scalar uniformity value
    """
    z = F.normalize(z, dim=1)
    # Pairwise squared distances
    sq_pdist = torch.cdist(z, z, p=2).pow(2)  # [N, N]
    
    # Exclude diagonal (self-distances)
    N = z.size(0)
    mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
    sq_pdist = sq_pdist[mask].view(N, N - 1)  # [N, N-1]
    
    # Log-mean-exp of negative distances
    return torch.log(torch.exp(-t * sq_pdist).mean())
