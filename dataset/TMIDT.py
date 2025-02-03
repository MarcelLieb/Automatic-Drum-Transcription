import os.path
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from dataset import (
    DrumMapping,
    get_midi_to_class,
    get_length,
    get_label_windows,
    get_segments, convert_to_wav,
)
from dataset.generics import ADTDataset
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
        drums = [np.array([]) for _ in range(len(mapping))]

    beats = np.loadtxt(
        os.path.join(path, "annotations", "beats", f"{identifier}.txt"),
        delimiter="\t",
    )
    beats = [beats[beats[:, 1] == 1][:, 0], beats[:, 0]]

    return identifier, drums, beats


def get_tracks(path: str | Path, instruments: Literal["all", "solo", "accomp"] = "all") -> list[str]:
    ids = [
        ".".join(file.split(".")[:-1])
        for file in os.listdir(os.path.join(path, "annotations", "drums_l"))
        if file.endswith(".txt")
    ]
    ids = [iden for iden in ids if os.stat(os.path.join(path, "annotations", "drums_l", iden + ".txt")).st_size > 0]
    if instruments == "all":
        pass
    elif instruments == "solo":
        ids = [iden for iden in ids if "accomp" not in iden]
    elif instruments == "accomp":
        ids = [iden for iden in ids if "accomp" in iden]
    out = []
    for iden in ids:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                get_length(os.path.join(path, "mp3", f"{iden}.mp3"))
            out.append(iden)
        except Exception as e:
            continue
    if not all(os.path.exists(os.path.join(path, "mp3", f"{iden}.wav")) for iden in out):
        with torch.multiprocessing.Pool(torch.multiprocessing.cpu_count()) as pool:
            pool.map(convert_to_wav, [os.path.join(path, "mp3", f"{iden}.mp3") for iden in out])
    return out


class TMIDT(ADTDataset):
    def __init__(
        self,
        path: str | Path,
        settings: DatasetSettings,
        segment: bool = True,
        split: list[str] | None = None,
        is_train: bool = False,
        use_dataloader: bool = False,
        instruments: Literal["all", "solo", "accomp"] = "all"
    ):
        super().__init__(settings, is_train=is_train, use_dataloader=use_dataloader, segment=segment)

        self.path = path
        self.instruments = instruments
        self.full = split is None
        self.split = get_tracks(path, instruments=instruments) if split is None else split

        with torch.multiprocessing.Pool(torch.multiprocessing.cpu_count()) as pool:
            args = [(path, identifier, self.mapping) for identifier in self.split]
            self.annotations = pool.starmap(get_annotations, args)
            self.annotations.sort(key=lambda x: int(x[0].split("_")[0]))

            self.annotations = [
                (iden, drums, beats) for iden, drums, beats in self.annotations if any(len(d) > 0 for d in drums)
            ]

            args = [(self.path, iden) for (iden, *_) in self.annotations]
            paths = pool.starmap(self._get_full_path, args)
            if self.segment:
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
        audio_path = os.path.join(root, "mp3", f"{identification}.wav")
        return Path(audio_path)

    def get_full_path(self, identification: Any):
        return self._get_full_path(self.path, identification)

    def get_identifier(self) -> str:
        return "TMIDT" + f"_{self.instruments}" if self.instruments != "all" else "" + "_split" if not self.full else ""


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    a = get_annotations(
        Path("../data/midi"), "35_ABBA_-_SOS_accomp", DrumMapping.THREE_CLASS_STANDARD
    )
    d = TMIDT(Path("../data/midi"), DatasetSettings())
    b = d[0]
    print(b)
