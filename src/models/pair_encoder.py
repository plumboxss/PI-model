import torch
import torch.nn as nn

class PairEncoder(nn.Module):
    """
    Encodes a preference pair (e1, e2, y) into a feedback vector h_i
    Input: e1: (B, d), e2: (B, d), y: (B, 1)
    Output: h: (B, h_dim)
    """
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(PairEncoder, self).__init__()
        # Input: e1 (d) + e2 (d) + y (1) = 2*d + 1
        input_dim = embedding_dim * 2 + 1
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, e1, e2, y):
        """
        Args:
            e1: (B, embedding_dim) - embedding of trajectory 1
            e2: (B, embedding_dim) - embedding of trajectory 2
            y: (B, 1) - preference label (1 if e1 preferred, 0 if e2 preferred)
        Returns:
            h: (B, output_dim) - feedback vector
        """
        # Concatenate embeddings and label
        pair_input = torch.cat([e1, e2, y], dim=-1)  # (B, 2*embedding_dim + 1)
        h = self.model(pair_input)  # (B, output_dim)
        return h

