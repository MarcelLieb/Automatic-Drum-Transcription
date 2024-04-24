import random
from typing import Generic, TypeVar

import torch
from torchaudio.transforms import Vol
from torch.utils.data import Dataset, Subset

from dataset.A2MD import A2MD, five_class_mapping, get_tracks
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
                                             prefetch_factor=4)
    return dataloader


T = TypeVar('T')


def get_splits(version: str, splits: list[float], path: str) -> list[dict[str, list[str]]]:
    assert abs(sum(splits) - 1) < 1e-4
    generator = torch.Generator().manual_seed(42)
    cut_off = {
        "L": 0.7,
        "M": 0.4,
        "S": 0.2,
    }
    folders = [f"dist0p{x:02}" for x in range(0, int(cut_off[version] * 100), 10)]
    tracks = get_tracks(path)
    out = [{} for _ in range(len(splits))]
    for folder in folders:
        identifiers = tracks[folder]
        split: list[Subset] = torch.utils.data.random_split(range(len(identifiers)), splits, generator=generator)
        for i, s in enumerate(split):
            out[i][folder] = [identifiers[j] for j in s.indices]

    return out


def get_dataset(batch_size, num_workers, splits=None,
                version="L", time_shift=0.0, mapping=five_class_mapping, lead_in=0.25,
                sample_rate=44100, hop_size=256, fft_size=2048, n_mels=82, center=False, pad_mode="constant"
                ) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]], DataLoader[
        tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]], DataLoader[
        tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]]]:
    if splits is None:
        splits = [0.8, 0.1, 0.1]
    path = "./data/a2md_public/"
    train, val, test = get_splits(version, splits, path)
    train = A2MD(train, mapping=mapping, path=path, time_shift=time_shift,
                 sample_rate=sample_rate, use_dataloader=True, lead_in=lead_in, is_train=True,
                 hop_size=hop_size, fft_size=fft_size, n_mels=n_mels, center=center, pad_mode=pad_mode)
    val = A2MD(val, mapping=mapping, path=path, time_shift=0,
               sample_rate=sample_rate, use_dataloader=True, lead_in=lead_in, is_train=False,
               hop_size=hop_size, fft_size=fft_size, n_mels=n_mels, center=center, pad_mode=pad_mode)
    test = A2MD(test, mapping=mapping, path=path, time_shift=0,
                sample_rate=sample_rate, use_dataloader=True, lead_in=lead_in, is_train=False,
                hop_size=hop_size, fft_size=fft_size, n_mels=n_mels, center=center, pad_mode=pad_mode)
    dataloader_train = get_dataloader(train, batch_size, num_workers, is_train=True)
    dataloader_val = get_dataloader(val, 1, num_workers, is_train=False)
    dataloader_test = get_dataloader(test, 1, num_workers, is_train=False)
    return dataloader_train, dataloader_val, dataloader_test
