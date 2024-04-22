from abc import abstractmethod

import torch
from torch.utils.data import Dataset


class ADTDataset(Dataset[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]]):
    @abstractmethod
    def __init__(self, **kwargs):
        self.time_shift = None

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def adjust_time_shift(self, time_shift: float):
        pass
