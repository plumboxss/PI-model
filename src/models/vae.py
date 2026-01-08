import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.trajectory_encoder import TrajectoryEncoder
from src.models.pair_encoder import PairEncoder
from src.models.set_encoder import SetEncoder
from src.models.reward_decoder import RewardDecoder

class VAEModel(nn.Module):
    """
    VAE model for preference-conditioned reward learning.
    
    Architecture:
    1. Trajectory encoder: τ -> e (fixed-length embedding)
    2. Pair encoder: (e1, e2, y) -> h_i
    3. Set encoder: {h_i}_{i=1..K} -> (μ, log_var)
    4. Reward decoder: r_φ(s, a, z)
    
    Forward pass:
    - Context: K comparisons to estimate z
    - Query: 1 comparison for loss computation
    """
    def __init__(
        self,
        obs_dim,
        act_dim,
        latent_dim,
        hidden_dim,
        kl_weight=1.0,
        learned_prior=False,
        annealer=None,
        reward_scaling=1.0,
        trajectory_encoder_type='transformer',
        set_encoder_type='attention',
        n_heads=4,
        n_layers=2,
        trajectory_embedding_dim=None,
        pair_embedding_dim=None,
        free_bits: float = 0.5,
    ):
        super(VAEModel, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.kl_weight = kl_weight
        self.learned_prior = learned_prior
        self.annealer = annealer
        self.scaling = reward_scaling
        # Free bits threshold: KL이 이 값 이하일 때는 벌점을 면제해 posterior collapse 완화
        self.free_bits = free_bits
        
        # Trajectory embedding dimension (default: hidden_dim)
        traj_emb_dim = trajectory_embedding_dim if trajectory_embedding_dim is not None else hidden_dim
        # Pair embedding dimension (default: hidden_dim)
        pair_emb_dim = pair_embedding_dim if pair_embedding_dim is not None else hidden_dim
        
        # 1. Trajectory encoder: τ (B, T, obs+act) -> e (B, traj_emb_dim)
        traj_input_dim = obs_dim + act_dim
        self.trajectory_encoder = TrajectoryEncoder(
            input_dim=traj_input_dim,
            hidden_dim=hidden_dim,
            output_dim=traj_emb_dim,
            encoder_type=trajectory_encoder_type
        )
        
        # 2. Pair encoder: (e1, e2, y) -> h_i
        self.pair_encoder = PairEncoder(
            embedding_dim=traj_emb_dim,
            hidden_dim=hidden_dim,
            output_dim=pair_emb_dim
        )
        
        # 3. Set encoder: {h_i} -> (μ, log_var)
        self.set_encoder = SetEncoder(
            input_dim=pair_emb_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            encoder_type=set_encoder_type,
            n_heads=n_heads,
            n_layers=n_layers
        )
        
        # 4. Reward decoder: (obs, act, z) -> r
        self.reward_decoder = RewardDecoder(
            obs_dim=obs_dim,
            act_dim=act_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=1
        )
        
        # Learned prior parameters (if enabled)
        if learned_prior:
            self.prior_mean = nn.Parameter(torch.zeros(latent_dim))
            self.prior_log_var = nn.Parameter(torch.zeros(latent_dim))
        else:
            self.register_buffer('prior_mean', torch.zeros(latent_dim))
            self.register_buffer('prior_log_var', torch.zeros(latent_dim))
    
    def reparameterization(self, mean, log_var):
        """
        Reparameterization trick: z = μ + σ * ε
        Args:
            mean: (B, latent_dim)
            log_var: (B, latent_dim)
        Returns:
            z: (B, latent_dim)
        """
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z
    
    def encode_context(self, context_s1, context_s2, context_y):
        """
        Encode context comparisons to estimate latent z.
        
        Args:
            context_s1: (B, K, T, D_sa) - K trajectories (preferred)
            context_s2: (B, K, T, D_sa) - K trajectories (non-preferred)
            context_y: (B, K, 1) - preference labels
        Returns:
            mean: (B, latent_dim)
            log_var: (B, latent_dim)
        """
        B, K, T, D_sa = context_s1.shape
        
        # Reshape to process all trajectories: (B*K, T, D_sa)
        s1_flat = context_s1.view(B * K, T, D_sa)
        s2_flat = context_s2.view(B * K, T, D_sa)
        y_flat = context_y.view(B * K, 1)
        
        # 1. Encode trajectories: (B*K, T, D_sa) -> (B*K, traj_emb_dim)
        e1 = self.trajectory_encoder(s1_flat)  # (B*K, traj_emb_dim)
        e2 = self.trajectory_encoder(s2_flat)  # (B*K, traj_emb_dim)
        
        # 2. Encode pairs: (B*K, traj_emb_dim) -> (B*K, pair_emb_dim)
        h_flat = self.pair_encoder(e1, e2, y_flat)  # (B*K, pair_emb_dim)
        
        # 3. Reshape back: (B*K, pair_emb_dim) -> (B, K, pair_emb_dim)
        H = h_flat.view(B, K, -1)  # (B, K, pair_emb_dim)
        
        # 4. Encode set: (B, K, pair_emb_dim) -> (B, latent_dim)
        mean, log_var = self.set_encoder(H)
        
        return mean, log_var
    
    def decode_reward(self, obs, act, z):
        """
        Decode reward for a trajectory given latent z.
        
        Args:
            obs: (B, T, obs_dim)
            act: (B, T, act_dim)
            z: (B, latent_dim) or (B, T, latent_dim)
        Returns:
            r: (B, T, 1) - reward at each timestep
        """
        return self.reward_decoder(obs, act, z)
    
    def forward(self, context_s1, context_s2, context_y, query_s1, query_s2, query_y):
        """
        Forward pass with context-query structure.
        
        Args:
            context_s1: (B, K, T, D_sa) - K context trajectories (preferred)
            context_s2: (B, K, T, D_sa) - K context trajectories (non-preferred)
            context_y: (B, K, 1) - context preference labels
            query_s1: (B, T, D_sa) - query trajectory 1
            query_s2: (B, T, D_sa) - query trajectory 2
            query_y: (B, 1) - query preference label
        Returns:
            loss: scalar tensor
            metrics: dict with loss components and accuracy
        """
        # 1. Encode context to get latent z
        mean, log_var = self.encode_context(context_s1, context_s2, context_y)
        z = self.reparameterization(mean, log_var)  # (B, latent_dim)
        
        # 2. Split query trajectories into obs and act
        query_obs1 = query_s1[..., :-self.act_dim]  # (B, T, obs_dim)
        query_act1 = query_s1[..., -self.act_dim:]  # (B, T, act_dim)
        query_obs2 = query_s2[..., :-self.act_dim]  # (B, T, obs_dim)
        query_act2 = query_s2[..., -self.act_dim:]  # (B, T, act_dim)
        
        # 3. Decode rewards for query trajectories
        r1 = self.decode_reward(query_obs1, query_act1, z)  # (B, T, 1)
        r2 = self.decode_reward(query_obs2, query_act2, z)  # (B, T, 1)
        
        # 4. Sum rewards over trajectory to get total return
        R1 = r1.sum(dim=1) / self.scaling  # (B, 1)
        R2 = r2.sum(dim=1) / self.scaling  # (B, 1)
        
        # 5. Bradley-Terry model: p = sigmoid(R1 - R2)
        p_hat = torch.sigmoid(R1 - R2)  # (B, 1)
        
        # 6. Compute losses
        # Reconstruction loss (BCE)
        reconstruction_loss = F.binary_cross_entropy(
            p_hat.view(-1, 1),
            query_y.view(-1, 1),
            reduction='mean'
        )
        
        # KL divergence loss: KL(q(z|context) || p(z))
        if self.learned_prior:
            prior_mean = self.prior_mean
            prior_log_var = self.prior_log_var
            # KL with learned prior
            kl_loss_raw = -0.5 * torch.sum(
                1 + log_var - prior_log_var
                - ((mean - prior_mean).pow(2) + log_var.exp()) / prior_log_var.exp()
            ) / mean.size(0)
        else:
            # Standard KL: KL(q(z|context) || N(0, I))
            kl_loss_raw = -0.5 * torch.sum(
                1 + log_var - mean.pow(2) - log_var.exp()
            ) / mean.size(0)

        # Free bits 적용: kl_loss_raw가 임계값 이하이면 벌점을 주지 않는다.
        kl_loss = torch.clamp(kl_loss_raw - self.free_bits, min=0.0)

        # Annealed KL weight
        kl_weight = self.annealer.slope() if self.annealer else self.kl_weight
        
        # Total loss
        loss = reconstruction_loss + kl_weight * kl_loss
        
        # Accuracy
        predicted = (p_hat > 0.5).float()
        correct = (predicted == query_y.view(-1, 1)).float()
        accuracy = correct.mean()
        
        metrics = {
            "loss": loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "kld_loss": kl_loss.item(),
            "kld_loss_raw": kl_loss_raw.item(),
            "accuracy": accuracy.item(),
            "kl_weight": kl_weight
        }
        
        return loss, metrics
    
    # Backward compatibility: old interface for single batch
    def forward_legacy(self, s1, s2, y):
        """
        Legacy forward pass for backward compatibility.
        Treats each sample as its own context (K=1) and query.
        """
        B, T, D_sa = s1.shape
        
        # Treat as context with K=1
        context_s1 = s1.unsqueeze(1)  # (B, 1, T, D_sa)
        context_s2 = s2.unsqueeze(1)  # (B, 1, T, D_sa)
        context_y = y.unsqueeze(1)  # (B, 1, 1)
        
        # Use same trajectories as query
        query_s1 = s1  # (B, T, D_sa)
        query_s2 = s2  # (B, T, D_sa)
        query_y = y  # (B, 1)
        
        return self.forward(context_s1, context_s2, context_y, query_s1, query_s2, query_y)
