import torch
from torch.utils.data import Dataset

class PhotonEventDataset(Dataset):
    def __init__(self, events, targets, max_photons):
        self.events = events
        self.targets = targets
        self.max_photons = max_photons

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        photons = torch.tensor(self.events[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        n = min(len(photons), self.max_photons)
        photons = photons[:n]

        padded = torch.zeros(self.max_photons, photons.shape[1])
        mask = torch.ones(self.max_photons, dtype=torch.bool)

        padded[:n] = photons
        mask[:n] = False  # False = real photon, True = padding
        # print("Mask sum (number of padding photons):\n", mask.sum())
        return {
            "photons": padded,
            "mask": mask,
            "target": target,
        }
