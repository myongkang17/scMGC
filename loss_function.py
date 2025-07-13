import torch
import torch.nn.functional as F
import numpy as np

def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


def binary_cross_entropy(x_pred, x):
    #mask = torch.sign(x)
    return - torch.sum(x * torch.log(x_pred + 1e-8) + (1 - x) * torch.log(1 - x_pred + 1e-8), dim=1)


def reconstruction_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    # x = x.to_dense()
    loss_rec = loss_func(decoded, x)
    return loss_rec


def contrastive_loss(z1, z2, temperature=0.1):
    """
    Compute contrastive loss using InfoNCE
    z1, z2: latent representations from two views
    """
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    
    # Compute similarity matrix
    sim_matrix = torch.exp(torch.mm(z, z.t()) / temperature)
    
    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim_matrix = sim_matrix.masked_fill(mask, 0)
    
    # Positive pairs
    pos_sim = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    
    # Compute loss
    loss = -torch.log(pos_sim / sim_matrix.sum(dim=-1))
    return loss.mean()

def mae_loss(x_hat, x, mask):
    """
    Compute masked reconstruction loss
    """
    # Apply mask
    x_hat = x_hat * mask
    x = x * mask
    
    # Compute MSE loss
    return F.mse_loss(x_hat, x, reduction='mean')

def total_loss(z1, z2, x_hat1, x_hat2, x1, x2, mask1, mask2, alpha=0):
    """
    Combine reconstruction and contrastive losses
    """
    # Reconstruction losses
    rec_loss1 = mae_loss(x_hat1, x1, mask1)
    rec_loss2 = mae_loss(x_hat2, x2, mask2)
    rec_loss = (rec_loss1 + rec_loss2) / 2
    
    # Contrastive loss
    con_loss = contrastive_loss(z1, z2)
    
    # Total loss
    return alpha * rec_loss + (1 - alpha) * con_loss
