import random
from typing import TypeVar

import torch
from torchaudio.transforms import Vol
from torch.utils.data import Dataset, Subset

from dataset.A2MD import A2MD, get_tracks
from torch.utils.data import DataLoader

from dataset.RBMA13 import RBMA_13
from dataset.generics import ADTDataset
from settings import TrainingSettings, AudioProcessingSettings, AnnotationSettings


def audio_collate(batch: list[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]]):
    audio, annotation, gts = zip(*batch)
    audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=0.0)
    annotation = list(annotation)
    annotation = torch.nn.utils.rnn.pad_sequence(
        annotation, batch_first=True, padding_value=-1
    )
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
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        collate_fn=audio_collate,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=4,
    )
    return dataloader


T = TypeVar("T")


def get_splits(
    version: str, splits: list[float], path: str
) -> list[dict[str, list[str]]]:
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
        split: list[Subset] = torch.utils.data.random_split(
            range(len(identifiers)), splits, generator=generator
        )
        for i, s in enumerate(split):
            out[i][folder] = [identifiers[j] for j in s.indices]

    return out


def get_dataset(
    training_settings: TrainingSettings,
    audio_settings: AudioProcessingSettings,
    annotation_settings: AnnotationSettings,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]],
    DataLoader[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]],
    DataLoader[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]],
]:
    path = "./data/a2md_public/"
    train, val, test = get_splits(
        training_settings.dataset_version, training_settings.splits, path
    )
    train = A2MD(
        split=train,
        audio_settings=audio_settings,
        annotation_settings=annotation_settings,
        path=path,
        use_dataloader=True,
        is_train=True,
    )
    val = RBMA_13(
        root="./data/rbma_13",
        audio_settings=audio_settings,
        annotation_settings=annotation_settings,
        use_dataloader=True,
        is_train=False,
    )
    test = A2MD(
        split=test,
        audio_settings=audio_settings,
        annotation_settings=annotation_settings,
        path=path,
        use_dataloader=True,
        is_train=False,
    )
    dataloader_train = get_dataloader(
        train,
        training_settings.batch_size,
        training_settings.num_workers,
        is_train=True,
    )
    dataloader_val = get_dataloader(
        val, 1, training_settings.num_workers, is_train=False
    )
    dataloader_test = get_dataloader(
        test, 1, training_settings.num_workers, is_train=False
    )
    return dataloader_train, dataloader_val, dataloader_test
