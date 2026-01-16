import torch
import torch.nn as nn


class RewardDecoder(nn.Module):
    """
    Dot-product reward: r = <phi(s, a), psi(z)>
    - feature_net: (obs, act) -> hidden_dim -> feature_dim
    - weight_net: z -> hidden_dim -> feature_dim (tanh bounded)
    """

    def __init__(self, obs_dim, act_dim, latent_dim, hidden_dim=32, output_dim=1):
        super(RewardDecoder, self).__init__()
        self.feature_dim = 16

        # Feature Network (phi): (obs+act) -> hidden_dim -> feature_dim
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.feature_dim),
        )

        # Weight Network (psi): z -> hidden_dim -> feature_dim -> BatchNorm
        self.weight_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.feature_dim),
        )
        self.weight_bn = nn.BatchNorm1d(self.feature_dim)

    def forward(self, obs, act, z):
        """
        Args:
            obs: (B, T, obs_dim)
            act: (B, T, act_dim)
            z:   (B, z_dim) or (B, T, z_dim)
        Returns:
            r: (B, T, 1) reward at each timestep
        """
        features = self.feature_net(torch.cat([obs, act], dim=-1))  # (B, T, feature_dim)

        if z.dim() == 2:
            z = z.unsqueeze(1)  # (B, 1, z_dim) broadcasts over T
        weights = self.weight_net(z)  # (B, T or 1, feature_dim)
        # BatchNorm expects (N, C); flatten time for normalization
        if weights.dim() == 3:
            B, T, F = weights.shape
            weights = self.weight_bn(weights.reshape(B * T, F)).reshape(B, T, F)
        else:
            weights = self.weight_bn(weights)

        r = (features * weights).sum(dim=-1, keepdim=True)  # (B, T, 1)
        return r
