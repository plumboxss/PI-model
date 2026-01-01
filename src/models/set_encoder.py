import torch
import torch.nn as nn

class SetEncoder(nn.Module):
    """
    Encodes a set of feedback vectors {h_i}_{i=1..K} into latent distribution (μ, Σ)
    Input: H: (B, K, h_dim) - K comparison feedback vectors
    Output: (μ, log_var): (B, z_dim), (B, z_dim)
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, encoder_type='attention', n_heads=4, n_layers=2):
        super(SetEncoder, self).__init__()
        self.encoder_type = encoder_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        if encoder_type == 'attention':
            # Transformer-based set encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.output_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2)
            )
        else:  # 'deepset' or default
            # DeepSets: mean pooling + MLP
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.output_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2)
            )
        
        # Output heads for mean and log variance
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, H):
        """
        Args:
            H: (B, K, input_dim) - set of K feedback vectors
        Returns:
            mean: (B, latent_dim) - mean of latent distribution
            log_var: (B, latent_dim) - log variance of latent distribution
        """
        if self.encoder_type == 'attention':
            # Project input
            x = self.input_proj(H)  # (B, K, hidden_dim)
            # Apply transformer
            x = self.transformer(x)  # (B, K, hidden_dim)
            # Mean pooling over K (set aggregation)
            x = x.mean(dim=1)  # (B, hidden_dim)
            # Final MLP
            x = self.output_mlp(x)  # (B, hidden_dim)
        else:  # DeepSets
            # Project input
            x = self.input_proj(H)  # (B, K, hidden_dim)
            # Mean pooling (permutation-invariant aggregation)
            x = x.mean(dim=1)  # (B, hidden_dim)
            # Final MLP
            x = self.output_mlp(x)  # (B, hidden_dim)
        
        # Output mean and log variance
        mean = self.FC_mean(x)  # (B, latent_dim)
        log_var = self.FC_var(x)  # (B, latent_dim)
        
        return mean, log_var

