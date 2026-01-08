import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TrajectoryEncoder(nn.Module):
    """
    Encodes a trajectory τ: (B, T, D_sa) into a fixed-length embedding e: (B, d)
    where D_sa = obs_dim + act_dim
    """
    def __init__(self, input_dim, hidden_dim, output_dim, encoder_type='transformer'):
        super(TrajectoryEncoder, self).__init__()
        self.encoder_type = encoder_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        if encoder_type == 'transformer':
            # Transformer-based encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.pos_enc = SinusoidalPositionalEncoding(hidden_dim, max_len=5000)
            self.dropout = nn.Dropout(0.1)
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, output_dim)
            )
        elif encoder_type == 'lstm':
            # LSTM-based encoder
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True
            )
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, output_dim)
            )
        else:  # 'mlp' or default
            # Simple MLP with temporal pooling
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2),
            )
            self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, tau):
        """
        Args:
            tau: (B, T, D_sa) - trajectory with T timesteps
        Returns:
            e: (B, output_dim) - fixed-length trajectory embedding
        """
        if self.encoder_type == 'transformer':
            # Project input to hidden_dim
            x = self.input_proj(tau)  # (B, T, hidden_dim)
            # Add positional encoding
            x = self.pos_enc(x)  # (B, T, hidden_dim)
            # Apply dropout
            x = self.dropout(x)  # (B, T, hidden_dim)
            # Apply transformer
            x = self.transformer(x)  # (B, T, hidden_dim)
            # Max pooling over time to capture salient peaks
            x, _ = torch.max(x, dim=1)  # (B, hidden_dim)
            # Final projection
            e = self.output_proj(x)  # (B, output_dim)
            # Verify output shape
            assert e.shape == (tau.shape[0], self.output_dim), \
                f"Expected output shape ({tau.shape[0]}, {self.output_dim}), got {e.shape}"
        elif self.encoder_type == 'lstm':
            # Apply LSTM
            lstm_out, (hidden, _) = self.lstm(tau)  # lstm_out: (B, T, hidden_dim*2)
            # Use max over time on bidirectional outputs to capture peaks
            x, _ = torch.max(lstm_out, dim=1)  # (B, hidden_dim*2)
            # Final projection
            e = self.output_proj(x)  # (B, output_dim)
        else:  # MLP with temporal pooling
            # Apply MLP to each timestep
            x = self.mlp(tau)  # (B, T, hidden_dim)
            # Max pooling over time to capture salient peaks
            x, _ = torch.max(x, dim=1)  # (B, hidden_dim)
            # Final projection
            e = self.output_proj(x)  # (B, output_dim)
        
        return e

