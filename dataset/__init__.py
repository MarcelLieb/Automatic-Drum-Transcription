import random
from typing import Generic, TypeVar

import torch
from torchaudio.transforms import Vol
from torch.utils.data import Dataset, Subset

from dataset.A2MD import A2MD, five_class_mapping
from torch.utils.data import DataLoader
from dataset.generics import ADTDataset


def audio_collate(batch: list[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]]):
    audio, annotation, gts = zip(*batch)
    audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=0.0)
    annotation = list(annotation)
    annotation = torch.nn.utils.rnn.pad_sequence(annotation, batch_first=True, padding_value=-1)
    audio = audio.permute(0, 2, 1)
    annotation = annotation.permute(0, 2, 1)
    return audio, annotation, gts


class Gain(torch.nn.Module):
    def __init__(self, min_gain: float = -20.0, max_gain: float = -1):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        gain = random.uniform(self.min_gain, self.max_gain)
        audio = Vol(gain, gain_type="db")(audio)
        return audio


def get_dataloader(dataset, batch_size, num_workers, is_train=False):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers,
                                             collate_fn=audio_collate, drop_last=False, pin_memory=True,
                                             prefetch_factor=1)
    return dataloader


T = TypeVar('T')


def get_splits(dataset: Dataset[Generic[T]], splits: list[float]) -> list[Subset[T]]:
    generator = torch.Generator().manual_seed(42)
    return torch.utils.data.random_split(dataset, splits, generator=generator)


def get_dataset(batch_size, num_workers, splits=None,
                version="L", time_shift=0.0, mapping=five_class_mapping,
                sample_rate=44100, hop_size=256, fft_size=2048, n_mels=82, center=False, pad_mode="constant"
                ) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]], DataLoader[
        tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]], DataLoader[
        tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]], ADTDataset]:
    if splits is None:
        splits = [0.8, 0.1, 0.1]
    A2md = A2MD(version, mapping=mapping, path="./data/a2md_public/", time_shift=time_shift,
                sample_rate=sample_rate, use_dataloader=True,
                hop_size=hop_size, fft_size=fft_size, n_mels=n_mels, center=center, pad_mode=pad_mode)
    train, val, test = get_splits(A2md, splits)
    dataloader_train = get_dataloader(train, batch_size, num_workers, is_train=True)
    dataloader_val = get_dataloader(val, batch_size, num_workers, is_train=False)
    dataloader_test = get_dataloader(test, batch_size, num_workers, is_train=False)
    return dataloader_train, dataloader_val, dataloader_test, A2md
