import torch
import torch.nn as nn

class RewardDecoder(nn.Module):
    """
    Decodes (obs, act, z) into reward r_φ(s, a, z)
    Input: obs: (B, T, obs_dim), act: (B, T, act_dim), z: (B, z_dim) or (B, T, z_dim)
    Output: r: (B, T, 1) - reward at each timestep
    """
    def __init__(self, obs_dim, act_dim, latent_dim, hidden_dim, output_dim=1):
        super(RewardDecoder, self).__init__()
        # Input: obs (obs_dim) + act (act_dim) + z (latent_dim)
        input_dim = obs_dim + act_dim + latent_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, obs, act, z):
        """
        Args:
            obs: (B, T, obs_dim) - observations
            act: (B, T, act_dim) - actions
            z: (B, z_dim) or (B, T, z_dim) - latent preference vector
        Returns:
            r: (B, T, 1) - reward at each timestep
        """
        # Broadcast z to match timesteps if needed
        if z.dim() == 2:
            # z: (B, z_dim) -> (B, T, z_dim)
            T = obs.shape[1]
            z = z.unsqueeze(1).expand(-1, T, -1)  # (B, T, z_dim)
        
        # Concatenate obs, act, z
        decoder_input = torch.cat([obs, act, z], dim=-1)  # (B, T, obs_dim + act_dim + latent_dim)
        
        # Decode to reward
        r = self.model(decoder_input)  # (B, T, 1)
        
        return r

