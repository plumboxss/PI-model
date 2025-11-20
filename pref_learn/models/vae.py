"""
파일의 전체적인 목적:
이 파일은 VAE(Variational Autoencoder)를 기반으로 한 선호도 학습 모델을 정의합니다.
모델의 핵심 기능은 여러 궤적 쌍과 선호도 레이블로 구성된 '컨텍스트'를 입력받아,
그 안에 내재된 사용자의 '숨겨진 선호도'를 잠재 벡터 z로 추출(인코딩)하는 것입니다.
이 z 벡터를 사용하면 새로운 궤적의 점수를 매기거나(디코딩) 선호도를 예측할 수 있습니다.
"""
import os
import math
import pickle

import numpy as np
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    간단한 MLP(다층 퍼셉트론) 기반의 인코더입니다.
    여러 선호도 쌍 컨텍스트가 하나의 긴 벡터로 펼쳐진(flattened) 형태로 입력을 받습니다.
    """
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

        # 잠재 공간의 정규분포를 정의하는 평균(mean)과 로그 분산(log_var)을 출력하는 헤드
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h_ = self.model(x)
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)

        return mean, log_var


class TrajectoryEncoder(nn.Module):
    """
    MLP 기반의 궤적 인코더입니다.
    궤적을 입력받아 임베딩 벡터를 출력합니다.
    """
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
    """
    두 궤적 임베딩과 선호도 레이블을 입력받아,
    단일 피드백 벡터 h를 출력하는 MLP입니다.
    """
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
    """
    트랜스포머 아키텍처를 사용하는 고성능 인코더입니다.
    컨텍스트 내의 여러 선호도 쌍들 간의 관계를 학습하여 더 정교한 잠재 벡터 z를 추출합니다.
    """
    def __init__(self, traj_dim, hidden_dim, latent_dim, n_heads=4, n_layers=2):
        super(SelfAttentionEncoder, self).__init__()
        
        # 1. 궤적 인코더: 각 궤적을 임베딩합니다.
        self.trajectory_encoder = TrajectoryEncoder(traj_dim, hidden_dim, hidden_dim)

        # 2. 쌍 인코더: 두 궤적 임베딩과 레이블을 결합하여 피드백 벡터 h를 생성합니다.
        pair_input_dim = hidden_dim * 2 + 1  # e1, e2, y
        self.pair_encoder = PairEncoder(pair_input_dim, hidden_dim, hidden_dim)

        # 3. 트랜스포머 인코더: 피드백 벡터 h 시퀀스를 처리합니다.
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 4. 출력 계층: 트랜스포머의 출력을 받아 최종적인 mean과 log_var를 계산합니다.
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x): # 입력 x는 (s1, s2, y)를 합친 형태입니다.
        # 입력 분리: s1, s2, y
        s1 = x[:, :, :self.trajectory_encoder.model[0].in_features]
        s2 = x[:, :, self.trajectory_encoder.model[0].in_features:-1]
        y = x[:, :, -1].unsqueeze(-1)
        
        # 1. 각 궤적을 임베딩합니다.
        e1 = self.trajectory_encoder(s1)
        e2 = self.trajectory_encoder(s2)

        # 2. 쌍 인코더를 통해 피드백 벡터 h를 생성합니다.
        pair_input = torch.cat([e1, e2, y], dim=-1)
        h = self.pair_encoder(pair_input)

        # 3. 트랜스포머로 시퀀스를 처리합니다.
        transformer_out = self.transformer_encoder(h)

        # 4. 시퀀스 정보 집계 (평균 풀링)
        aggregated = transformer_out.mean(dim=1)
        
        # 5. 최종 MLP 및 출력 헤드
        h_ = self.output_mlp(aggregated)
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)

        return mean, log_var


class Decoder(nn.Module):
    """
    잠재 벡터 z와 특정 상태(state)를 입력받아, 해당 상태의 '점수' 또는 '원시 보상'을 출력하는 MLP 디코더입니다.
    """
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
        # 입력 x는 (상태 벡터, 잠재 벡터 z)가 합쳐진 형태입니다.
        x_hat = self.model(x)
        return x_hat


class VAEModel(nn.Module):
    """
    VAE 모델의 전체 구조와 학습 과정을 정의하는 핵심 클래스입니다.
    인코더, 디코더를 조합하고 VAE의 손실 함수(재구성 손실 + KL 발산)를 계산합니다.
    """
    def __init__(
        self,
        encoder_input_dim,
        decoder_input_dim,
        action_dim,
        latent_dim,
        hidden_dim,
        kl_weight=1.0,
        learned_prior=False,
        annealer=None,
        reward_scaling=1.0,
        encoder_type='attention',
        n_heads=4,
        n_layers=2,
    ):
        super(VAEModel, self).__init__()
        self.encoder_type = encoder_type
        self.action_dim = action_dim
        # `encoder_type` 인자에 따라 MLP 인코더 또는 Self-Attention 인코더를 동적으로 선택합니다.
        if self.encoder_type == 'attention':
            traj_dim = (encoder_input_dim - 1) // 2
            self.Encoder = SelfAttentionEncoder(
                traj_dim=traj_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                n_heads=n_heads,
                n_layers=n_layers
            )
        else: # 기본값은 MLP 인코더
            self.Encoder = Encoder(encoder_input_dim, hidden_dim, latent_dim)

        self.Decoder = Decoder(decoder_input_dim, hidden_dim, 1)
        self.latent_dim = latent_dim
        # VAE의 사전 분포 p(z)를 학습 가능한 파라미터로 설정하는 옵션입니다.
        self.mean = torch.nn.Parameter(
            torch.zeros(latent_dim), requires_grad=learned_prior
        )
        self.log_var = torch.nn.Parameter(
            torch.zeros(latent_dim), requires_grad=learned_prior
        )
        self.learned_prior = learned_prior
        self.kl_weight = kl_weight
        self.annealer = annealer
        self.scaling = reward_scaling

    def reparameterization(self, mean, var):
        """
        재매개변수화 트릭(reparameterization trick)을 구현합니다.
        z = mean + std * epsilon 형태로 샘플링하여, 랜덤 샘플링 과정에 역전파가 가능하도록 합니다.
        """
        epsilon = torch.randn_like(var).to(mean.device)  # epsilon 샘플링
        z = mean + var * epsilon  # 재매개변수화
        return z

    def encode(self, s1, s2, y):
        """
        입력(s1, s2, y)을 인코더에 맞는 형태로 가공하여 mean과 log_var를 반환합니다.
        """
        s1_ = s1.view(s1.shape[0], s1.shape[1], -1)
        s2_ = s2.view(s2.shape[0], s2.shape[1], -1)
        y_ = y.view(y.shape[0], y.shape[1], -1)

        # 쌍 데이터 형태: (B, Context_len, Pair_dim)
        pair_data = torch.cat([s1_, s2_, y_], dim=-1)

        if self.encoder_type == 'attention':
            # SelfAttentionEncoder는 시퀀스 형태의 입력을 그대로 사용합니다.
            encoder_input = pair_data
        else: # 기존 MLP 로직
            # MLP 인코더는 시퀀스 차원을 펼쳐서 하나의 긴 벡터로 만들어 입력합니다.
            encoder_input = pair_data.view(s1.shape[0], -1)

        mean, log_var = self.Encoder(encoder_input)
        return mean, log_var

    def decode(self, obs, act, z):
        """
        상태(obs), 행동(act), 잠재 벡터(z)를 입력받아 디코더를 통해 점수(보상)를 계산합니다.
        """
        decoder_input = torch.cat([obs, act, z], dim=-1)
        r = self.Decoder(decoder_input)
        return r

    def get_reward(self, r):
        r = self.Decoder(r)  # Batch x Ann x T x 1
        return r

    def reconstruction_loss(self, x, x_hat):
        """
        예측된 선호 확률(x_hat)과 실제 레이블(x) 사이의 재구성 손실을 계산합니다.
        Binary Cross-Entropy Loss를 사용합니다.
        """
        return nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")

    def accuracy(self, x, x_hat):
        """
        예측의 정확도를 계산합니다.
        """
        predicted_class = (x_hat > 0.5).float()
        return torch.mean((predicted_class == x).float())

    def latent_loss(self, mean, log_var):
        """
        잠재 손실, 즉 KL 발산(KL-divergence)을 계산합니다.
        이는 인코더가 출력한 사후 분포 q(z|context)와 사전 분포 p(z) 사이의 거리를 측정합니다.
        VAE 손실 함수의 정규화 항으로 작용합니다.
        """
        if self.learned_prior:
            # 사전 분포를 학습하는 경우
            kl = -torch.sum(
                1
                + (log_var - self.log_var)
                - (log_var - self.log_var).exp()
                - (mean - self.mean).pow(2) / (self.log_var.exp())
            )
        else:
            # 사전 분포가 N(0, I)인 경우
            kl = -torch.sum(1.0 + log_var - mean.pow(2) - log_var.exp())
        return kl

    def forward(self, s1, s2, y):  # 입력 형태: Batch x Ann x T x (State + Action), Batch x Ann x 1
        """
        모델의 순전파 및 손실 계산을 수행하는 메인 로직입니다.
        """
        # 1. 인코딩: 컨텍스트로부터 잠재 분포의 파라미터(mean, log_var)를 추출합니다.
        mean, log_var = self.encode(s1, s2, y)

        # 2. 샘플링: 재매개변수화 트릭을 사용해 잠재 벡터 z를 샘플링합니다.
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # Batch x Z
        
        # 3. 브로드캐스팅: 컨텍스트 전체를 대표하는 z를 각 궤적의 모든 타임스텝에 적용할 수 있도록 확장합니다.
        num_contexts = s1.shape[0] # B
        context_len = s1.shape[1]  # C
        traj_len = s1.shape[2]     # T
        z_expanded = z.unsqueeze(1).unsqueeze(1).expand(num_contexts, context_len, traj_len, self.latent_dim)

        # 4. 입력 분리: s1, s2에서 obs와 action을 분리합니다.
        obs1 = s1[..., :-self.action_dim]
        act1 = s1[..., -self.action_dim:]
        obs2 = s2[..., :-self.action_dim]
        act2 = s2[..., -self.action_dim:]

        # 5. 디코딩: 확장된 z를 사용하여 s1과 s2의 모든 상태에 대한 점수(r0, r1)를 계산합니다.
        r0 = self.decode(obs1, act1, z_expanded)
        r1 = self.decode(obs2, act2, z_expanded)

        # 6. 선호도 예측: 각 궤적의 총 점수를 계산하고, 그 차이를 시그모이드 함수에 통과시켜
        #    "s1이 s2보다 선호될 확률" p_hat을 예측합니다.
        r_hat1 = r0.sum(axis=2) / self.scaling
        r_hat2 = r1.sum(axis=2) / self.scaling
        p_hat = torch.nn.functional.sigmoid(r_hat1 - r_hat2)
        
        # 손실 계산을 위해 텐서 형태를 맞춥니다.
        p_hat = p_hat.view(-1, 1)
        labels = y.view(-1, 1)

        # 7. 손실 계산: 재구성 손실과 잠재 손실(KL 발산)을 각각 계산합니다.
        reconstruction_loss = self.reconstruction_loss(labels, p_hat)
        accuracy = self.accuracy(labels, p_hat)
        latent_loss = self.latent_loss(mean, log_var)

        # KL 가중치(annealing에 사용)를 적용하여 두 손실을 합칩니다.
        kl_weight = self.annealer.slope() if self.annealer else self.kl_weight
        loss = reconstruction_loss + kl_weight * latent_loss

        # 로깅을 위한 메트릭을 딕셔너리로 반환합니다.
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
        return z

    def sample_posterior(self, s1, s2, y):
        mean, log_var = self.encode(s1, s2, y)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        return mean, log_var, z

    def update_posteriors(self, posteriors, biased_latents):
        self.posteriors = posteriors
        self.biased_latents = biased_latents


class VAEClassifier(VAEModel):
    """
    VAEModel의 변형 아키텍처입니다.
    주요 차이점은 디코딩 방식으로, 두 궤적과 z를 모두 합쳐 디코더에 한 번에 입력하여
    선호도를 직접 예측(분류)합니다.
    """
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
        # 인코딩 및 샘플링은 VAEModel과 동일합니다.
        mean, log_var = self.encode(s1, s2, y)

        if self.flow_prior:
            z, log_det = self.transform(mean, log_var)
        else:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # Batch x Z
            log_det = None

        # z 브로드캐스팅
        num_contexts = s1.shape[0] # B
        context_len = s1.shape[1]  # C
        traj_len = s1.shape[2]     # T
        z_expanded = z.unsqueeze(1).unsqueeze(1).expand(num_contexts, context_len, traj_len, self.latent_dim)

        # 디코딩 방식의 차이점: s1, s2, z를 모두 합쳐서 디코더에 입력합니다.
        p_hat = self.Decoder(torch.cat([s1, s2, z_expanded], dim=-1))
        p_hat = torch.nn.functional.sigmoid(p_hat).view(-1, 1)
        labels = y.view(-1, 1)

        # 손실 계산은 VAEModel과 동일합니다.
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

