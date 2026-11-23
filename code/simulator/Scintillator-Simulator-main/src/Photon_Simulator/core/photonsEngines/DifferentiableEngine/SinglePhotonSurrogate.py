import torch
import torch.nn as nn
import torch.nn.functional as F


class SinglePhotonSurrogate(nn.Module):
    def __init__(
        self,
        photon_dim: int = 7,
        detector_dim: int = 0,
        hidden_dim: int = 128,
        n_time_bins: int = 150,
    ):
        super().__init__()

        input_dim = photon_dim + detector_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
        )

        self.time_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_time_bins),
        )

    def forward(self, photon, detector_params=None):
        if detector_params is not None:
            x = torch.cat([photon, detector_params], dim=-1)
        else:
            x = photon

        h = self.encoder(x)

        state_logits = self.state_head(h)
        time_logits = self.time_head(h)

        state_probs = F.softmax(state_logits, dim=-1)
        time_probs = F.softmax(time_logits, dim=-1)

        p_captured = state_probs[..., 0:1]

        soft_hist = p_captured * time_probs

        return {
            "state_logits": state_logits,
            "state_probs": state_probs,
            "time_logits": time_logits,
            "time_probs": time_probs,
            "soft_hist": soft_hist,
        }