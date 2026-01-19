import torch
import torch.nn as nn


class RewardDecoder(nn.Module):
    """
    Dot-product reward: r = <phi(s, a), psi(z)>
    - feature_net: (obs, act) -> hidden_dim -> feature_dim
    - weight_net: z -> hidden_dim -> feature_dim (tanh bounded)
    """

    def __init__(self, obs_dim, act_dim, latent_dim, hidden_dim=32, output_dim=1, feature_dropout: float = 0.1):
        super(RewardDecoder, self).__init__()
        self.feature_dim = 16

        # Feature Network (phi): (obs+act) -> hidden_dim -> feature_dim
        drop = nn.Dropout(p=float(feature_dropout)) if feature_dropout and feature_dropout > 0.0 else nn.Identity()
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            drop,
            nn.Linear(hidden_dim, self.feature_dim),
        )

        # Weight Network (psi): z -> hidden_dim -> feature_dim -> BatchNorm
        self.weight_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.feature_dim),
        )
        # LayerNorm은 배치 통계를 사용하지 않아, 컨텍스트별 z 정보가 배치/시간축 평균으로 씻기는 문제를 줄인다.
        # (BN은 여기서 B*T로 펼쳐 적용되어 z별 차이를 약화시킬 수 있음)
        self.weight_ln = nn.LayerNorm(self.feature_dim)
        # Learnable gain to recover output scale after normalization
        self.weight_gain = nn.Parameter(torch.ones(1, 1, self.feature_dim))

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
        # LayerNorm normalizes over the last dimension (feature_dim)
        weights = self.weight_ln(weights)
        weights = weights * self.weight_gain

        r = (features * weights).sum(dim=-1, keepdim=True)  # (B, T, 1)
        return r
