import os
from pathlib import Path

import numpy as np
import torch

from dataset import get_label_windows, get_length, get_segments, convert_to_wav, DrumMapping, get_splits
from dataset.generics import ADTDataset
from settings import DatasetSettings

rbma_13_path = "./data/rbma_13/"

# From http://ifs.tuwien.ac.at/~vogl/dafx2018/mappings.py
# RBMA annotations to gm midi drum note
rbma_drum_map = {
    0: 36,  # bass drum
    1: 38,  # snare drum
    2: 42,  # closed hi-hat
    3: 46,  # open hi-hat
    4: 44,  # pedal hi-hat
    5: 56,  # cowbell
    6: 53,  # ride bell
    7: 41,  # low floor tom
    9: 43,  # high floor tom
    10: 45,  # low tom
    11: 47,  # low-mid tom
    12: 48,  # high-mid tom
    13: 50,  # high tom
    14: 37,  # side stick
    15: 39,  # hand clap
    16: 51,  # ride cymbal
    17: 49,  # crash cymbal
    18: 55,  # splash cymbal
    19: 52,  # chinese cymbal
    20: 70,  # shaker, maracas
    21: 54,  # tambourine
    22: 75,  # claves, stick click
    23: 81,  # high bells / triangle
}


def load_rbma(rbma_path: str, mapping: DrumMapping) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    annotations = {}
    midi_to_class = mapping.get_midi_to_class()
    for root, dirs, files in os.walk(os.path.join(rbma_path, "annotations", "drums_full")):
        files = [file for file in files if file.endswith(".drums")]
        for file in files:
            track = file.split(".")[0]
            number = int(track.split("-")[-1])
            drums = np.loadtxt(os.path.join(root, file), delimiter="\t", )
            if len(drums) == 0:
                continue
            drums_midi = np.vectorize(rbma_drum_map.get)(drums[:, 1])
            drums[:, 1] = midi_to_class[drums_midi.astype(int)]
            drums = [drums[drums[:, 1] == i][:, 0] for i in range(len(mapping))]
            beats = np.loadtxt(
                os.path.join(rbma_path, "annotations", "beats_full", f"{track}.beats"),
                delimiter="\t",
            )
            beats = [beats[beats[:, 1] == 1][:, 0], beats[:, 0]]
            annotations[number] = (beats, drums)
    return annotations


def rename_rbma_audio_files():
    # load title list
    with open(os.path.join(rbma_13_path, "titles.txt")) as f:
        title_list = f.readlines()

    title_list = [x.strip() for x in title_list]

    positions = [[] for _ in range(30)]
    from fuzzywuzzy import process

    for root, dirs, files in os.walk(os.path.join(rbma_13_path, "audio")):
        for file in files:
            track = " ".join(file.split(".")[0].split(" ")[2:-1])
            title = process.extractOne(track, title_list)[0]
            # find position of track in title list
            position = title_list.index(title) + 1
            positions[position - 1].append(file)

    for i, files in enumerate(positions):
        if len(files) == 1:
            os.rename(
                os.path.join(rbma_13_path, "audio", files[0]),
                os.path.join(rbma_13_path, "audio", f"RBMA-13-Track-{i + 1:02}.mp3"),
            )

    return positions


def get_tracks(path: str) -> list[int]:
    return [
        int(file.split("-")[-1].split(".")[0])
        for file in os.listdir(os.path.join(path, "annotations", "drums_full"))
        if file.endswith(".drums")
    ]


def convert_to_wav_dataset(root: str):
    tracks = load_rbma(root, DrumMapping.THREE_CLASS)
    for track, _ in tracks.items():
        convert_to_wav(os.path.join(root, "audio", f"{track}.mp3"))


class RBMA13(ADTDataset):
    def __init__(
        self,
        path: str | Path,
        settings: DatasetSettings,
        segment: bool = True,
        splits: list[int] | None = None,
        is_train=False,
        use_dataloader=False,
    ):
        super().__init__(settings, is_train=is_train, use_dataloader=use_dataloader, segment=segment)
        self.path = path

        self.full = splits is None
        self.annotations = load_rbma(path, self.mapping)
        if splits is None:
            splits = list(self.annotations.keys())
        tracks = splits
        self.annotations = {track: self.annotations[track] for track in tracks}
        self.annotations = [
            (identifier, annotation[1], annotation[0])
            for identifier, annotation in self.annotations.items()
        ]
        self.annotations.sort(key=lambda x: int(x[0]))

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

    def adjust_time_shift(self, time_shift: float):
        self.time_shift = time_shift

    def get_full_path(self, number: int) -> Path:
        audio_path = os.path.join(self.path, "audio", f"RBMA-13-Track-{number:02}")
        if os.path.exists(audio_path + ".wav"):
            return Path(audio_path + ".wav")
        else:
            return Path(audio_path + ".mp3")

    def get_identifier(self) -> str:
        if self.full:
            return "RBMA_full"
        else:
            return "RBMA_split"


def main():
    convert_to_wav_dataset("../data/rbma_13")


if __name__ == "__main__":
    main()
