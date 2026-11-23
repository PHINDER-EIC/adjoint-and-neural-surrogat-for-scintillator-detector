from torch import nn
import torch

from Photon_Simulator.core.photonsEngines.DifferentiableEngine.DetectorDecoder import DetectorDecoder
from Photon_Simulator.core.photonsEngines.DifferentiableEngine.TransformerEncoder import PhotonTransformerEncoder


class PhotonTransformerSurrogate(nn.Module):
    def __init__(
        self,
        photon_dim=8,
        detector_dim=0,
        embed_dim=128,
        output_bins=200,
    ):
        super().__init__()

        self.encoder = PhotonTransformerEncoder(
            photon_dim=photon_dim,
            detector_dim=detector_dim,
            embed_dim=embed_dim,
        )

        self.decoder = DetectorDecoder(
            embed_dim=embed_dim,
            output_bins=output_bins,
        )

    def forward(self, photons, mask, detector_params=None):
        event_embedding = self.encoder(
            photons=photons,
            mask=mask,
            detector_params=detector_params,
        )

        output = self.decoder(event_embedding)

        return output

class PhotonDeepSetSurrogate(nn.Module):
    def __init__(self, photon_dim=7, hidden_dim=128, output_bins=150):
        super().__init__()

        self.photon_encoder = nn.Sequential(
            nn.Linear(photon_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.event_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_bins),
        )

    def forward(self, photons, mask=None):
        # photons: [B, N, photon_dim]

        h = self.photon_encoder(photons)  # [B, N, hidden_dim]

        if mask is not None:
            mask = mask.unsqueeze(-1)
            h = h * mask

            count = mask.sum(dim=1).clamp(min=1)
            mean_pool = h.sum(dim=1) / count
            max_pool = h.masked_fill(mask == 0, -1e9).max(dim=1).values
        else:
            mean_pool = h.mean(dim=1)
            max_pool = h.max(dim=1).values

        sum_pool = h.sum(dim=1)

        z = torch.cat([mean_pool, max_pool, sum_pool], dim=-1)

        return self.event_decoder(z)