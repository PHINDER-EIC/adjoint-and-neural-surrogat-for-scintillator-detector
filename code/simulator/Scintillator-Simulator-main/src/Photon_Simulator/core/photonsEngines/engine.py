from Photon_Simulator.utils.logger import get_logger
from abc import ABC, abstractmethod

class PhotonEngine(ABC):
    @abstractmethod
    def load_geometry(self, geometry) -> None:
        pass

    @abstractmethod
    def load_materials(self, materials) -> None:
        pass

    @abstractmethod
    def propagate(self, photons):
        pass