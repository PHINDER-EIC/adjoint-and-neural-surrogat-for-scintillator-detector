import torch
import torch.nn as nn

EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
FF_DIM = 256
DROPOUT = 0.1

class PhotonTransformerEncoder(nn.Module):
    """
    This class is implemented to take a list of photons, giving it to the Transformer and get a compact vector of the event
    """
    def __init__(self, photon_dim = 7, detector_dim = 0, embed_dim = EMBED_DIM, num_heads = NUM_HEADS, num_layers = NUM_LAYERS, ff_dim  = FF_DIM, dropout = DROPOUT):
        """
        Parameters
        ---
        photon_dim: int
            Each photon has defined parameters (3D position at which it is generated, time, lambda, ...)
            e.g x, y, z, t, dx, dy, dz
        detector_dim: int
            Detector parameters that affects photon evolution. E.g thickness, n, reflectivity...
        
        embed_dim: int
            interla dimension of the transformer. Each photon goes from `photon_dim` to `embed_dim`

        num_head: int
            A head is an "observation channel" that learns autonomously which input elements should influence each other and to what extent.
        
        num_layers: int
            Number of consecutive tranformes blocks
        ff_dim: int
            feedfoward dimension. Suggestion 4 * d_model
        
        dropout: float
            fraction of neurons shutted down during training to reduce overvfitting
            

        """
        super().__init__()

        #? Each photon goes from photon_dim -> embed_dim
        self.photon_embedding = nn.Linear(photon_dim, embed_dim)

        #? Each photon goes from detector_dim -> embed_dim
        self.detector_embedding = None
        if detector_dim > 0:
            self.detector_embedding = nn.Linear(detector_dim,embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,  #input dimension
            nhead= num_heads, #heads
            dim_feedforward=ff_dim, # middle dimension: we go from embed_dim -> ff_dim -> embed_dim
            dropout=dropout,
            batch_first=True,
            activation="gelu" # GELU(x) = x*(CDG della normale std) per avere una funzione più morbida della relu
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, photons, mask, detector_params = None):
        """
        photons: [batch, max_photons, photon_dim], e.g [21,5000,8] -> 32 events, max 5000 photons per events, 8 features per photon
        mask: [batch, max_photons], True for padding
        detector_params: [batch, detector_dim]
        """

        x = self.photon_embedding(photons)

        if detector_params is not None:
         if self.detector_embedding is not None:
            d = self.detector_embedding(detector_params)
            x = x + d.unsqueeze(1) # to give to all photons global infos on detector
        else:
           raise Exception("missing detector embedding")

        x = self.transformer(x, src_key_padding_mask=mask) # the transformer gets [batch, max_photons, embed_dim]
                                                                # and gives [batch, max_photons, embed_dim] 
                                                                # but now, each photon knows ifos about others

        valid = (~mask).unsqueeze(-1).float() #! mystic process to remove padding
        x = x * valid
        # final pooling
        event_embedding = x.sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        return event_embedding