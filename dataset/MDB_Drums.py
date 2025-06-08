import os.path
from pathlib import Path

import numpy as np
import polars as pl

from dataset import get_length, get_label_windows, get_segments
from dataset.generics import ADTDataset
from dataset.mapping import DrumMapping
from settings import DatasetSettings

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
    "MHT": "HT",
    "HIT": "HT",
    "RDC": "RD",
    "RDB": "RB",
    "CRC": "CRC",
    "CHC": "CHC",
    "SPC": "SPC",
    "SST": "SS",
    "TMB": "TB",
}

# from http://ifs.tuwien.ac.at/~vogl/dafx2018/mappings.py
mdb_drum_map = {
    'KD': 35,
    'SD': 38,
    'SDB': 38,
    'SDD': 38,
    'SDF': 38,
    'SDG': 38,
    'SDNS': 38,
    'CHH': 42,
    'OHH': 46,
    'PHH': 44,
    'HIT': 50,
    'MHT': 48,
    'HFT': 43,
    'LFT': 41,
    'RDC': 51,
    'RDB': 53,
    'CRC': 49,
    'CHC': 52,
    'SPC': 55,
    'SST': 37,
    'TMB': 54,
}

for key, value in label_translator.items():
    assert key in mdb_drum_map, f"{key}"
    assert value == \
           DrumMapping.EIGHTEEN_CLASS.value[0][DrumMapping.EIGHTEEN_CLASS.get_midi_to_class()[mdb_drum_map[key]]][
               0]


def get_annotations(root: str | Path, name: str, mapping: DrumMapping):
    labels = pl.scan_csv(
        os.path.join(
            root, "annotations", "subclass", f"MusicDelta_{name}_subclass.txt"
        ),
        separator="\t",
        has_header=False,
        new_columns=["time", "class"],
    )
    labels = labels.select(pl.all().cast(pl.Utf8).str.strip_chars(" "))

    midi_to_class = mapping.get_midi_to_class()
    midi_to_class = {i: value for i, value in enumerate(midi_to_class)}
    labels = (
        labels
        .select(
            pl.col("time").cast(pl.Float32),
            pl.col("class").replace_strict(mdb_drum_map, return_dtype=pl.Int8).replace_strict(midi_to_class),
        )
        .filter(pl.col("class") >= 0)
        .cast(pl.Float32)
        .collect()
        .to_numpy()
    )
    beats = np.loadtxt(
        os.path.join(root, "annotations", "beats", f"MusicDelta_{name}_MIX.beats"),
        delimiter="\t",
    )
    beats = [beats[beats[:, 1] == 1][:, 0], beats[:, 0]]

    drums = [labels[labels[:, 1] == i][:, 0] for i in range(len(mapping))]

    return beats, drums


def get_tracks(path: str | Path) -> list[str]:
    return [
        file.split("_")[1]
        for file in os.listdir(os.path.join(path, "annotations", "subclass"))
        if file.endswith(".txt")
    ]


class MDBDrums(ADTDataset):
    def __init__(
        self,
        path: str | Path,
        settings: DatasetSettings,
        segment: bool = True,
        split: list[str] | None = None,
        is_train: bool = False,
        use_dataloader: bool = False,
    ):
        super().__init__(
            settings,
            is_train=is_train,
            use_dataloader=use_dataloader,
            segment=segment,
        )
        self.path = path
        self.full = split is None
        self.split = split if split is not None else get_tracks(path)

        self.annotations = {}
        for name in self.split:
            self.annotations[name] = get_annotations(
                path, name, settings.annotation_settings.mapping
            )

        self.annotations = [
            (name, drums, beats) for name, (beats, drums) in self.annotations.items()
        ]
        self.annotations.sort(key=lambda x: x[0])

        if self.segment:
            lengths = [
                get_length(self.get_full_path(track[0])) for track in self.annotations
            ]
            if self.segment_type == "label":
                self.segments = get_label_windows(
                    lengths,
                    [drums for _, drums, *_ in self.annotations],
                    self.lead_in,
                    self.lead_out,
                    self.sample_rate,
                )
            elif self.segment_type == "frame":
                self.segments = get_segments(
                    lengths,
                    self.frame_length,
                    self.frame_overlap,
                    self.sample_rate,
                )

    def __len__(self):
        return len(self.segments) if self.segments is not None else len(self.annotations)

    def get_full_path(self, identifier: str) -> Path:
        audio_path = os.path.join(
            self.path, "audio", "full_mix", f"MusicDelta_{identifier}_MIX.wav"
        )
        return Path(audio_path)

    def get_identifier(self) -> str:
        if self.full:
            return "MDB_full"
        else:
            return "MDB_split"


if __name__ == "__main__":
    d = MDBDrums("../data/MDB Drums", DatasetSettings())
    _ = d[0]
