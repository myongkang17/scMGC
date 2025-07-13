import torch
import torch.nn as nn
import torch.nn.functional as F
from GAE import GraphConvolution

class MaskedGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dims, z_dim, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = in_dim
        for h_dim in hidden_dims:
            self.layers.append(GraphConvolution(prev_dim, h_dim))
            prev_dim = h_dim
        self.final_layer = GraphConvolution(prev_dim, z_dim)
        self.dropout_rate = dropout_rate
        
    def forward(self, x, adj, mask_ratio=0.25):
        # Apply random masking
        batch_size, num_nodes = x.shape[0], x.shape[1]
        num_mask = int(num_nodes * mask_ratio)
        
        # Generate random mask
        mask = torch.ones(batch_size, num_nodes, device=x.device)
        for i in range(batch_size):
            idx = torch.randperm(num_nodes)[:num_mask]
            mask[i, idx] = 0


        # Apply mask
        x_masked = x * mask
        
        # GCN layers
        h = x_masked
        for layer in self.layers:
            h = F.relu(layer(h, adj))
            h = F.dropout(h, self.dropout_rate, training=self.training)
            
        z = self.final_layer(h, adj)
        # Add L2 normalization for better clustering performance
        z = F.normalize(z, p=2, dim=-1)
        return z, mask

class ContrastiveDecoder(nn.Module):
    def __init__(self, z_dim, hidden_dims, out_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = z_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        self.final_layer = nn.Linear(prev_dim, out_dim)
        
    def forward(self, z):
        h = z
        for layer in self.layers:
            h = F.relu(layer(h))
        x_hat = self.final_layer(h)
        return x_hat

class MAE_Contrastive(nn.Module):
    def __init__(self, in_dim, hidden_dims, z_dim, out_dim, dropout_rate):
        super().__init__()
        self.encoder = MaskedGraphEncoder(in_dim, hidden_dims, z_dim, dropout_rate)
        self.decoder = ContrastiveDecoder(z_dim, hidden_dims[::-1], out_dim)
        
    def forward(self, x, adj, mask_ratio=0.75):
        # Encoding
        z, mask = self.encoder(x, adj, mask_ratio)
        
        # Decoding
        x_hat = self.decoder(z)
        
        # Apply mask for reconstruction
        x_hat = x_hat * mask
        
        return z, x_hat, mask
