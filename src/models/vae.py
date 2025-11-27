import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TrajectoryEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.model(x)

class PairEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PairEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)

class SelfAttentionEncoder(nn.Module):
    def __init__(self, traj_dim, hidden_dim, latent_dim, n_heads=4, n_layers=2):
        super(SelfAttentionEncoder, self).__init__()
        self.trajectory_encoder = TrajectoryEncoder(traj_dim, hidden_dim, hidden_dim)
        pair_input_dim = hidden_dim * 2 + 1  # e1, e2, y
        self.pair_encoder = PairEncoder(pair_input_dim, hidden_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        # input_dim = traj_dim * 2 + 1
        
        # Split x into s1, s2, y
        s1 = x[:, :, :self.trajectory_encoder.model[0].in_features]
        s2 = x[:, :, self.trajectory_encoder.model[0].in_features:-1]
        y = x[:, :, -1].unsqueeze(-1)
        
        # Encode trajectories
        e1 = self.trajectory_encoder(s1)
        e2 = self.trajectory_encoder(s2)
        
        # Concatenate embeddings and label
        pair_input = torch.cat([e1, e2, y], dim=-1)
        
        # Encode pair
        h = self.pair_encoder(pair_input)
        
        # Apply Transformer
        transformer_out = self.transformer_encoder(h)
        
        # Aggregate (mean pooling)
        aggregated = transformer_out.mean(dim=1)
        
        h_ = self.output_mlp(aggregated)
        
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        
        return mean, log_var

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.LSTM = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, output_dim)
        self.FC_var = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.LSTM(x)
        # Concat bidirectional hidden states
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        x = self.fc(hidden)
        mean = self.FC_mean(x)
        log_var = self.FC_var(x)
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x_hat = self.model(x)
        return x_hat

class VAEModel(nn.Module):
    def __init__(self, encoder_input_dim, decoder_input_dim, action_dim, latent_dim, hidden_dim, kl_weight=1.0, learned_prior=False, annealer=None, reward_scaling=1.0, encoder_type='attention', n_heads=4, n_layers=2):
        super(VAEModel, self).__init__()
        self.encoder_type = encoder_type
        self.action_dim = action_dim
        
        if self.encoder_type == 'attention':
            traj_dim = (encoder_input_dim - 1) // 2
            self.Encoder = SelfAttentionEncoder(traj_dim=traj_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, n_heads=n_heads, n_layers=n_layers)
        else:
            self.Encoder = Encoder(encoder_input_dim, hidden_dim, latent_dim)
            
        self.Decoder = Decoder(decoder_input_dim, hidden_dim, 1)
        self.latent_dim = latent_dim
        self.mean = torch.nn.Parameter(torch.zeros(latent_dim), requires_grad=learned_prior)
        self.log_var = torch.nn.Parameter(torch.zeros(latent_dim), requires_grad=learned_prior)
        self.learned_prior = learned_prior
        self.kl_weight = kl_weight
        self.annealer = annealer
        self.scaling = reward_scaling

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(var.device)
        z = mean + var * epsilon
        return z

    def encode(self, s1, s2, y):
        if self.encoder_type == 'attention':
            # Combine inputs for attention encoder
            # s1: (B, T, D), s2: (B, T, D), y: (B, T) -> x: (B, T, 2*D + 1)
            
            # Force y to match s1's batch and sequence dimensions, adding a feature dimension at the end
            # s1 shape: (batch_size, seq_len, feature_dim)
            
            batch_size = s1.shape[0]
            seq_len = s1.shape[1]
            
            # Reshape y to (batch_size, 1, 1) first. 
            # Use reshape instead of view to handle non-contiguous tensors if any.
            y_reshaped = y.reshape(batch_size, 1, 1)
            
            # Expand to (batch_size, seq_len, 1)
            y_expanded = y_reshaped.expand(batch_size, seq_len, 1)

            x = torch.cat([s1, s2, y_expanded], dim=-1)
            mean, log_var = self.Encoder(x)
        else:
            # For LSTM encoder, we might stack differently
            x = torch.cat([s1, s2, y.unsqueeze(-1)], dim=-1) 
            mean, log_var = self.Encoder(x)
        return mean, log_var

    def decode(self, obs, act, z):
        # obs: (B, T, O_dim)
        # act: (B, T, A_dim)
        # z: (B, T, Z_dim) or (B, 1, Z_dim) broadcasted
        decoder_input = torch.cat([obs, act, z], dim=-1)
        r = self.Decoder(decoder_input)
        return r

    def forward(self, s1, s2, y):
        # s1, s2: (B, T, D) where D = obs_dim + act_dim
        mean, log_var = self.encode(s1, s2, y)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        
        # Expand z for decoding
        # z: (B, latent_dim) -> (B, T, latent_dim)
        num_contexts = s1.shape[0]
        context_len = s1.shape[1] # Number of pairs in context
        traj_len = s1.shape[2] # This might be confusing if s1 is flattened.
        
        # Assuming s1 passed here is actually (Batch, Sequence_of_Pairs, Trajectory_Features)
        # In train.py, s1 is (B, T, obs+act). Wait, Encoder takes sequence of pairs.
        # Let's trace train.py. 
        # s1: batch['s1'] -> (B, num_pairs, obs_dim+act_dim) if trajectory is length 1?
        # No, in build_preference_dataset, we have full trajectories.
        # But the Encoder structure suggests we are feeding a sequence of (traj1, traj2, label).
        # The TrajectoryEncoder takes 'traj_dim'. 
        
        # Let's assume z corresponds to the preference for the whole batch/context.
        # To decode rewards for individual timesteps in s1/s2, we need z broadcasted.
        
        # Re-checking train.py and dataset structure:
        # s1 is concatenation of obs and act.
        
        # z: (B, latent_dim)
        # We need to decode r for each timestep in s1 and s2.
        # obs1 = s1[..., :-self.action_dim]
        # act1 = s1[..., -self.action_dim:]
        
        # If s1 is (B, Seq_Len, Feat_Dim), then z needs to be (B, Seq_Len, Latent_Dim).
        # Check z dim first
        if z.dim() == 2:
            # z is (B, Latent_Dim)
            # s1 is (B, Num_Pairs, Feat_Dim)
            z_expanded = z.unsqueeze(1).expand(s1.shape[0], s1.shape[1], self.latent_dim)
        else:
            # Assume z is already correct shape or broadcasting works
            z_expanded = z
        
        # obs1 shape: (B, Num_Pairs, obs_dim)
        # act1 shape: (B, Num_Pairs, act_dim)
        obs1 = s1[..., :-self.action_dim]
        act1 = s1[..., -self.action_dim:]
        obs2 = s2[..., :-self.action_dim]
        act2 = s2[..., -self.action_dim:]
        
        r0 = self.decode(obs1, act1, z_expanded)
        r1 = self.decode(obs2, act2, z_expanded)
        
        # Sum rewards over trajectory/segment to get total return
        # r0: (B, Seq_Len, 1) -> sum over Seq_Len -> (B, 1)
        
        # Sum over the sequence dimension (dim 1)
        r_hat1 = r0.sum(dim=1) / self.scaling
        r_hat2 = r1.sum(dim=1) / self.scaling
        
        # Bradley-Terry model
        # p_hat: (B, 1)
        p_hat = torch.sigmoid(r_hat1 - r_hat2)
        
        p_hat = p_hat.view(-1, 1)
        labels = y.view(-1, 1)
        
        reconstruction_loss = self.reconstruction_loss(labels, p_hat)
        accuracy = self.accuracy(labels, p_hat)
        latent_loss = self.latent_loss(mean, log_var)
        
        kl_weight = self.annealer.slope() if self.annealer else self.kl_weight
        loss = reconstruction_loss + kl_weight * latent_loss
        
        metrics = {
            "loss": loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "kld_loss": latent_loss.item(),
            "accuracy": accuracy.item(),
            "kl_weight": kl_weight
        }
        return loss, metrics

    def reconstruction_loss(self, y, p_hat):
        # BCE Loss
        return F.binary_cross_entropy(p_hat, y, reduction="mean")

    def latent_loss(self, mean, log_var):
        # KL Divergence
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return kld / mean.size(0) # Normalize by batch size

    def accuracy(self, y, p_hat):
        predicted = (p_hat > 0.5).float()
        correct = (predicted == y).float()
        return correct.mean()

