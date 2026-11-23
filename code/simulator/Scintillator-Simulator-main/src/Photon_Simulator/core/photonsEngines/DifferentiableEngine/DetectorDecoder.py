from torch import nn


class DetectorDecoder(nn.Module):
    def __init__(self, embed_dim=128, output_bins=200):
        super().__init__()

        self.net = nn.Sequential(  # noqa: F821
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, output_bins),
            nn.Softplus(),
        )

    def forward(self, event_embedding):
        return self.net(event_embedding)