import os.path
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dataset import (
    DrumMapping,
    get_midi_to_class,
    get_length,
    get_label_windows,
    get_segments,
)
from generics import ADTDataset
from settings import DatasetSettings

drum_map = {
    "0": 36,  # bass drum
    "1": 38,  # snare drum
    "2": 37,  # side stick
    "3": 39,  # clap
    "4": 41,  # low tom
    "5": 45,  # mid tom
    "6": 50,  # high tom
    "7": 42,  # closed hi-hat
    "8": 44,  # pedal hi-hat
    "9": 46,  # open hi-hat
    "10": 54,  # tambourine
    "11": 51,  # ride cymbal
    "12": 53,  # ride bell
    "13": 49,  # crash cymbal
    "14": 55,  # splash cymbal
    "15": 52,  # chinese cymbal
    "16": 56,  # cowbell
    "17": 75,  # click
}

drum_map = {int(key): value for key, value in drum_map.items()}


def get_annotations(
    path: Path | str, identifier: str, mapping: DrumMapping
) -> tuple[str, list[np.array], list[np.array]]:
    file_path = os.path.join(path, "annotations", "drums_l", f"{identifier}.txt")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        drums = np.loadtxt(
            file_path,
            delimiter="\t",
        )
    if len(drums.shape) == 2:
        drums[:, 1] = np.vectorize(drum_map.get)(drums[:, 1].astype(int))

        midi_to_class = get_midi_to_class(mapping.value)
        drums[:, 1] = midi_to_class[drums[:, 1].astype(int)]

        drums = [drums[drums[:, 1] == i][:, 0] for i in range(len(mapping))]
    else:
        print(f"Empty drums for {identifier}")
        drums = [np.array([]) for _ in range(len(mapping))]

    beats = np.loadtxt(
        os.path.join(path, "annotations", "beats", f"{identifier}.txt"),
        delimiter="\t",
    )
    beats = [beats[beats[:, 1] == 1][:, 0], beats[:, 0]]

    return identifier, beats, drums


def get_tracks(path: str | Path) -> list[str]:
    return [
        ".".join(file.split(".")[:-1])
        for file in os.listdir(os.path.join(path, "mp3"))
        if file.endswith(".mp3")
    ]


class TMIDT(ADTDataset):
    def __init__(
        self,
        path: str | Path,
        settings: DatasetSettings,
        split: list[str] | None = None,
        is_train: bool = False,
        use_dataloader: bool = False,
    ):
        super().__init__(settings, is_train=is_train, use_dataloader=use_dataloader)

        self.path = path
        self.split = get_tracks(path) if split is None else split

        with torch.multiprocessing.Pool(torch.multiprocessing.cpu_count()) as pool:
            args = [(path, identifier, self.mapping) for identifier in self.split]
            self.annotations = pool.starmap(get_annotations, args)
            self.annotations.sort(key=lambda x: int(x[0].split("_")[0]))

            args = [(self.path, identifier) for identifier in self.split]
            paths = pool.starmap(self._get_full_path, args)
            if is_train:
                args = [(path,) for path in paths]
                lengths = pool.starmap(get_length, args)
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

    @staticmethod
    def _get_full_path(root: str, identification: str) -> Path:
        audio_path = os.path.join(root, "mp3", f"{identification}.mp3")
        return Path(audio_path)

    def get_full_path(self, identification: Any):
        return self._get_full_path(self.path, identification)


if __name__ == "__main__":
    a = get_annotations(
        Path("../data/midi"), "35_ABBA_-_SOS_accomp", DrumMapping.THREE_CLASS_STANDARD
    )
    d = TMIDT(Path("../data/midi"), DatasetSettings())
    _a = d[0]
