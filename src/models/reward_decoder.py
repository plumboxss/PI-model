import torch
import torch.nn as nn

class RewardDecoder(nn.Module):
    """
    Residual reward: r = g(obs, act) + h(obs, act, z)
    g: common dynamics (shared across users)
    h: personalized residual (captures user-specific preferences)
    """
    def __init__(self, obs_dim, act_dim, latent_dim, hidden_dim, output_dim=1):
        super(RewardDecoder, self).__init__()

        # Common net g(obs, act)
        common_input_dim = obs_dim + act_dim
        common_hidden = 32  # bottleneck to force reliance on personal net
        self.common_norm = nn.LayerNorm(common_input_dim)
        # NOTE: Common net intentionally unused to force reliance on personal net (z-dependent)
        self.common_net = nn.Sequential(
            nn.Linear(common_input_dim, common_hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(common_hidden, common_hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(common_hidden, output_dim),
        )

        # Personal net h(obs, act, z)
        personal_input_dim = obs_dim + act_dim + latent_dim
        self.personal_norm = nn.LayerNorm(personal_input_dim)
        self.personal_net = nn.Sequential(
            nn.Linear(personal_input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim),
        )

        # Initialize personal head to near-zero so g dominates at start
        personal_out = self.personal_net[-1]
        nn.init.constant_(personal_out.weight, 0.0)
        nn.init.constant_(personal_out.bias, 0.0)
        nn.init.constant_(personal_out.weight, 0.0)
        nn.init.constant_(personal_out.bias, 0.0)
    
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
        
        common_input = torch.cat([obs, act], dim=-1)  # (B, T, obs+act)
        personal_input = torch.cat([obs, act, z], dim=-1)  # (B, T, obs+act+z)

        # LayerNorm at the input level to reduce scale disparity (e.g., velocity vs position)
        common_input = self.common_norm(common_input)
        personal_input = self.personal_norm(personal_input)

        r_personal = self.personal_net(personal_input)  # (B, T, 1)

        # Force reliance on personal_net (z-dependent); disable common_net contribution
        # r_common = self.common_net(common_input)   # (B, T, 1)
        r = r_personal
        return r
