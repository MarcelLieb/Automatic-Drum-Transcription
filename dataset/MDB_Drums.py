import os.path
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torchaudio

from dataset import load_audio, get_indices
from generics import ADTDataset
from dataset.mapping import DrumMapping, get_name_to_class_number
from settings import AudioProcessingSettings, AnnotationSettings

label_translator = {
    "KD": "BD",
    "SD": "SD",
    "SDB": "SD",
    "SDD": "SD",
    "SDF": "SD",
    "SDG": "SD",
    "SDNS": "SD",
    "CHH": "CHH",
    "OHH": "OHH",
    "PHH": "PHH",
    "LFT": "LT",
    "HFT": "LT",
    "MHT": "MT",
    "HIT": "HT",
    "RDC": "RD",
    "RDB": "RB",
    "CRC": "CRC",
    "CHC": "CHC",
    "SPC": "SPC",
    "SST": "SS",
    "TMB": "TB",
}


def get_annotations(root: str | Path, name: str, mapping: DrumMapping):
    labels = pl.read_csv(
        os.path.join(root, "annotations", "subclass", f"{name}_subclass.txt"),
        separator="\t",
        has_header=False,
        new_columns=["time", "class"],
    )
    labels = labels.select(pl.all().cast(pl.Utf8).str.strip_chars(" "))
    name_to_class = get_name_to_class_number(mapping)
    labels = labels.select(
        pl.col("time").cast(pl.Float32),
        pl.col("class")
        .replace(label_translator)
        .replace(name_to_class)
        .cast(pl.Float32),
    ).to_numpy()
    beats = np.loadtxt(
        os.path.join(root, "annotations", "beats", f"{name}_MIX.beats"), delimiter="\t"
    )
    beats = [beats[beats[:, 1] == 1][:, 0], beats[:, 0]]

    drums = [labels[labels[:, 1] == i][:, 0] for i in range(len(mapping))]

    return beats, drums


class MDBDrums(ADTDataset):
    def __init__(
        self,
        path: str | Path,
        audio_settings: AudioProcessingSettings,
        annotation_settings: AnnotationSettings,
        use_dataloader: bool = False,
        is_train: bool = True,
    ):
        super().__init__(
            audio_settings,
            annotation_settings,
            is_train=is_train,
            use_dataloader=use_dataloader,
        )
        self.path = path

        self.annotations = {}
        for root, dirs, files in os.walk(os.path.join(path, "annotations", "subclass")):
            for file in files:
                name = "_".join(file.split("_")[:2])
                self.annotations[name] = get_annotations(path, name, self.mapping)
        self.annotations = [(name, drums, beats) for name, (beats, drums) in self.annotations.items()]
        self.annotations.sort(key=lambda x: x[0])

    def __len__(self):
        return len(self.annotations)


    def get_full_path(self, identifier: str) -> Path:
        audio_path = os.path.join(self.path, "audio", "full_mix", f"{identifier}_MIX.wav")
        return Path(audio_path)
