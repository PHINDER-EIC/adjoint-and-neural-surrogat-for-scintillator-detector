from dataclasses import dataclass
import torch
from Photon_Simulator.utils.logger import get_logger
from Photon_Simulator.config.settings import settings

from .DifferentiablePhotonStep import DifferentiablePhotonStep



@dataclass
class PhotonState:
    pos: torch.Tensor
    direction: torch.Tensor
    weight: torch.Tensor
    time: torch.Tensor

class DifferentiableEngine:
    def __init__(self, scene, logger = None):
        self.scene = scene
        
        self.log = get_logger(settings.logger_name) if logger is None else logger
        
        #! Da spostare
        self.c_cm_s = 2.99792458e10
        
        def photon_step(self,state, hit):
            raise NotImplementedError
            
        def track_photon(self,state, n_steps):
            for _ in range(n_steps):
                state = self.photon_step(state)
        