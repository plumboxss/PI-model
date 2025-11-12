import os
import math
import pickle

import numpy as np
import torch
import torch.nn as nn
from pref_learn.models.flow import Flow


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h_ = self.model(x)
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)

        return mean, log_var


class SelfAttentionEncoder(nn.Module):
    def __init__(self, pair_dim, hidden_dim, latent_dim, n_heads=4, n_layers=2):
        super(SelfAttentionEncoder, self).__init__()
        
        # 1. Individual pair embedding layer
        self.pair_embed = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2. Transformer Encoder to process the sequence of pairs
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 3. Output layers
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x): # Expected input shape: (batch_size, seq_len, pair_dim)
        # 1. Embed each pair in the sequence
        embeddings = self.pair_embed(x) # (B, S, H)

        # 2. Process sequence with transformer
        transformer_out = self.transformer_encoder(embeddings) # (B, S, H)

        # 3. Aggregate sequence information (mean pooling)
        aggregated = transformer_out.mean(dim=1) # (B, H)
        
        # 4. Final MLP and output heads
        h_ = self.output_mlp(aggregated)
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)

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
    def __init__(
        self,
        encoder_input,
        decoder_input,
        latent_dim,
        hidden_dim,
        annotation_size,
        size_segment,
        kl_weight=1.0,
        learned_prior=False,
        flow_prior=False,
        annealer=None,
        reward_scaling=1.0,
        encoder_type='mlp',
        n_heads=4,
        n_layers=2,
    ):
        super(VAEModel, self).__init__()
        self.encoder_type = encoder_type
        if self.encoder_type == 'attention':
            # For attention, encoder_input is the dimension of ONE pair
            self.Encoder = SelfAttentionEncoder(
                pair_dim=encoder_input,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                n_heads=n_heads,
                n_layers=n_layers
            )
        else: # Default to original MLP encoder
            # For MLP, encoder_input is the flattened dimension of ALL pairs in context
            self.Encoder = Encoder(encoder_input, hidden_dim, latent_dim)

        self.Decoder = Decoder(decoder_input, hidden_dim, 1)
        self.latent_dim = latent_dim
        self.mean = torch.nn.Parameter(
            torch.zeros(latent_dim), requires_grad=learned_prior
        )
        self.log_var = torch.nn.Parameter(
            torch.zeros(latent_dim), requires_grad=learned_prior
        )
        self.annotation_size = annotation_size
        self.size_segment = size_segment
        self.learned_prior = learned_prior

        self.flow_prior = flow_prior
        if flow_prior:
            self.flow = Flow(latent_dim, "radial", 4)

        self.kl_weight = kl_weight
        self.annealer = annealer
        self.scaling = reward_scaling

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(mean.device)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def encode(self, s1, s2, y):
        # s1, s2 shape for MLP: (B, Ann_size, T, State)
        # s1, s2 shape for Attention: (B, Context_len, T, State)
        
        s1_ = s1.view(s1.shape[0], s1.shape[1], -1)
        s2_ = s2.view(s2.shape[0], s2.shape[1], -1)
        y_ = y.view(y.shape[0], y.shape[1], -1)

        # Shape of pair_data is (B, Context_len, Pair_dim)
        pair_data = torch.cat([s1_, s2_, y_], dim=-1)

        if self.encoder_type == 'attention':
            # Input is already in the correct shape for SelfAttentionEncoder
            encoder_input = pair_data
        else: # Original MLP logic
            # Flatten the sequence dimension for MLP
            encoder_input = pair_data.view(s1.shape[0], -1)

        mean, log_var = self.Encoder(encoder_input)
        return mean, log_var

    def decode(self, obs, z):
        r = torch.cat([obs, z], dim=-1)  # Batch x Ann x T x (State + Z)
        r = self.Decoder(r)  # Batch x Ann x T x 1
        return r

    def get_reward(self, r):
        r = self.Decoder(r)  # Batch x Ann x T x 1
        return r

    def transform(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)

        return self.flow(z)

    def reconstruction_loss(self, x, x_hat):
        return nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")

    def accuracy(self, x, x_hat):
        predicted_class = (x_hat > 0.5).float()
        return torch.mean((predicted_class == x).float())

    def latent_loss(self, mean, log_var):
        if self.learned_prior:
            kl = -torch.sum(
                1
                + (log_var - self.log_var)
                - (log_var - self.log_var).exp()
                - (mean - self.mean).pow(2) / (self.log_var.exp())
            )
        else:
            kl = -torch.sum(1.0 + log_var - mean.pow(2) - log_var.exp())
        return kl

    def forward(self, s1, s2, y):  # Batch x Ann x T x State, Batch x Ann x 1
        # import pdb; pdb.set_trace()
        mean, log_var = self.encode(s1, s2, y)

        if self.flow_prior:
            z, log_det = self.transform(mean, log_var)
        else:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # Batch x Z
            log_det = None
        
        # When using attention encoder, z is (B, Z). For decoder, it needs to be broadcasted.
        # When using MLP encoder, z is (B, Z), but context was flattened.
        # The number of segments to decode over is s1.shape[1] (annotation/context size) * s1.shape[2] (traj length)
        # This broadcasting logic seems complex and might need adjustment based on data loader.
        # Let's assume the data loader for attention gives (B, C, T, D) and z is (B, Z).
        # We need z to become (B, C, T, Z) for decoding.
        
        num_contexts = s1.shape[0] # B
        context_len = s1.shape[1]  # C
        traj_len = s1.shape[2]     # T

        # Reshape z from (B, Z) to (B, 1, 1, Z) and then expand
        z_expanded = z.unsqueeze(1).unsqueeze(1).expand(num_contexts, context_len, traj_len, self.latent_dim)

        r0 = self.decode(s1, z_expanded)
        r1 = self.decode(s2, z_expanded)

        r_hat1 = r0.sum(axis=2) / self.scaling
        r_hat2 = r1.sum(axis=2) / self.scaling

        p_hat = torch.nn.functional.sigmoid(r_hat1 - r_hat2)
        
        # The loss should be calculated per pair in the context
        # p_hat is (B, C, 1), y is (B, C, 1)
        # We need to calculate loss over all pairs in all contexts
        p_hat = p_hat.view(-1, 1)
        labels = y.view(-1, 1)

        reconstruction_loss = self.reconstruction_loss(labels, p_hat)
        accuracy = self.accuracy(labels, p_hat)
        latent_loss = self.latent_loss(mean, log_var)

        kl_weight = self.annealer.slope() if self.annealer else self.kl_weight
        loss = reconstruction_loss + kl_weight * latent_loss

        if self.flow_prior:
            loss = loss - torch.sum(log_det)

        metrics = {
            "loss": loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "kld_loss": latent_loss.item(),
            "accuracy": accuracy.item(),
            "kl_weight": kl_weight,
        }

        return loss, metrics

    def sample_prior(self, size):
        z = torch.randn(size, self.latent_dim).to(next(self.parameters()).device)
        if self.learned_prior:
            z = z * torch.exp(0.5 * self.log_var) + self.mean
        elif self.flow_prior:
            z, _ = self.flow(z)
        return z

    def sample_posterior(self, s1, s2, y):
        mean, log_var = self.encode(s1, s2, y)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        return mean, log_var, z

    def update_posteriors(self, posteriors, biased_latents):
        self.posteriors = posteriors
        self.biased_latents = biased_latents


class VAEClassifier(VAEModel):
    def __init__(
        self,
        encoder_input,
        decoder_input,
        latent_dim,
        hidden_dim,
        annotation_size,
        size_segment,
        kl_weight=1.0,
        learned_prior=False,
        flow_prior=False,
        annealer=None,
        reward_scaling=1.0,
        encoder_type='mlp', # Pass through
        n_heads=4,
        n_layers=2,
    ):
        super(VAEClassifier, self).__init__(
            encoder_input,
            decoder_input,
            latent_dim,
            hidden_dim,
            annotation_size,
            size_segment,
            kl_weight,
            learned_prior,
            flow_prior,
            annealer,
            reward_scaling,
            encoder_type,     # Pass through
            n_heads,
            n_layers,
        )

    def forward(self, s1, s2, y):  # Batch x Ann x T x State, Batch x Ann x 1
        # import pdb; pdb.set_trace()
        mean, log_var = self.encode(s1, s2, y)

        if self.flow_prior:
            z, log_det = self.transform(mean, log_var)
        else:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # Batch x Z
            log_det = None

        # Broadcasting z for classifier decoder
        num_contexts = s1.shape[0] # B
        context_len = s1.shape[1]  # C
        traj_len = s1.shape[2]     # T
        z_expanded = z.unsqueeze(1).unsqueeze(1).expand(num_contexts, context_len, traj_len, self.latent_dim)

        p_hat = self.Decoder(torch.cat([s1, s2, z_expanded], dim=-1))
        p_hat = torch.nn.functional.sigmoid(p_hat).view(-1, 1)
        labels = y.view(-1, 1)

        reconstruction_loss = self.reconstruction_loss(labels, p_hat)
        accuracy = self.accuracy(labels, p_hat)
        latent_loss = self.latent_loss(mean, log_var)

        kl_weight = self.annealer.slope() if self.annealer else self.kl_weight
        loss = reconstruction_loss + kl_weight * latent_loss

        if self.flow_prior:
            loss = loss - torch.sum(log_det)

        metrics = {
            "loss": loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "kld_loss": latent_loss.item(),
            "accuracy": accuracy.item(),
            "kl_weight": kl_weight,
        }

        return loss, metrics

    def decode(self, x, y, z):  # B x S, N x S, B x Z
        x = x[:, None].repeat(1, y.shape[0], 1)  # B x N x S
        z = z[:, None].repeat(1, y.shape[0], 1)  # B x N x Z
        y = y[None].repeat(x.shape[0], 1, 1)  # B x N x S
        x = torch.cat([x, y, z], dim=-1)  # B x N x (2S + Z)
        x = torch.nn.functional.sigmoid(self.Decoder(x))  # B x N x 1
        return x[:, :, 0].mean(dim=-1)  # (B, )
