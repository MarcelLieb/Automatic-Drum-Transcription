import os
from pathlib import Path

import pretty_midi
import torch

from dataset import (
    # load_audio,
    get_label_windows,
    get_drums,
    get_length,
    get_segments,
    get_splits as get_splits_data, convert_to_wav,
)
from dataset.mapping import DrumMapping
from dataset.generics import ADTDataset
from settings import DatasetSettings

A2MD_PATH = "./data/a2md_public/"


def get_annotation(
    path: str,
    folder: str,
    identifier: str,
    mapping: DrumMapping = DrumMapping.THREE_CLASS,
):
    midi = pretty_midi.PrettyMIDI(
        midi_file=os.path.join(path, "align_mid", folder, f"align_mid_{identifier}.mid")
    )
    drums = get_drums(midi, mapping=mapping)
    if drums is None:
        return None
    beats = midi.get_beats()
    down_beats = midi.get_downbeats()
    return (folder, identifier), drums, [down_beats, beats]


def get_tracks(path: str) -> dict[str, list[str]]:
    folders = [f"dist0p{x:02}" for x in range(0, 70, 10)]
    out = {}
    for folder in folders:
        out[folder] = []
        for root, dirs, files in os.walk(os.path.join(path, "align_mid", folder)):
            for file in files:
                identifier = "_".join(file.split(".")[0].split("_")[2:4])
                out[folder].append(identifier)
    return out


def convert_to_wav_dataset(root: str):
    tracks = get_tracks(root)
    for folder, identifiers in tracks.items():
        print(f"Converting {folder}")
        for identifier in identifiers:
            convert_to_wav(A2MD._get_full_path(root, (folder, identifier)))


def get_splits(
    version: str, splits: list[float], path: str, seed: int = 42
) -> list[dict[str, list[str]]]:
    assert abs(sum(splits) - 1) < 1e-4
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
        split = get_splits_data(splits, identifiers, seed=seed)
        for i, s in enumerate(split):
            out[i][folder] = s
    # Check if the distribution is correct
    flat_tracks = [track for folder in folders for track in tracks[folder]]
    for i, s in enumerate(out):
        total = sum(len(arr) for _, arr in s.items())
        assert abs(total / len(flat_tracks) - splits[i]) < 2e-2, f"{total / len(flat_tracks)} != {splits[i]}"

    return out


class A2MD(ADTDataset):
    def __init__(
        self,
        path: Path | str,
        settings: DatasetSettings,
        segment: bool,
        split: dict[str, list[str]] | None = None,
        is_train: bool = False,
        use_dataloader: bool = False,
    ):
        super().__init__(settings, is_train=is_train, use_dataloader=use_dataloader, segment=segment)
        self.path = path
        self.split = get_tracks(path) if split is None else split
        self.full = split is None

        args = []
        for i, (folder, identifiers) in enumerate(self.split.items()):
            for identifier in identifiers:
                args.append((path, folder, identifier, self.mapping))
        with torch.multiprocessing.Pool(torch.multiprocessing.cpu_count()) as pool:
            self.annotations = pool.starmap(get_annotation, args)
            # filter tracks without drums
            self.annotations = [
                annotation for annotation in self.annotations if annotation is not None
            ]
            self.annotations.sort(key=lambda x: int(x[0][1].split("_")[-2]))
            args = [
                (self.path, identification) for identification, *_ in self.annotations
            ]
            # use static method to avoid passing self to pool
            paths = pool.starmap(A2MD._get_full_path, args)
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
            # args = [(path, self.sample_rate, self.normalize) for path in paths]
            # self.cache = pool.starmap(load_audio, args)

    def __len__(self):
        return len(self.segments) if self.segments is not None else len(self.annotations)

    @staticmethod
    def _get_full_path(root: str, identification: tuple[str, str]) -> Path:
        folder, identifier = identification
        if os.path.exists(os.path.join(root, "ytd_audio", folder, f"ytd_audio_{identifier}.wav")):
            audio_path = os.path.join(
                root, "ytd_audio", folder, f"ytd_audio_{identifier}.wav"
            )
        else:
            audio_path = os.path.join(
                root, "ytd_audio", folder, f"ytd_audio_{identifier}.mp3"
            )
        return Path(audio_path)

    def get_full_path(self, identification: tuple[str, str]) -> Path:
        return self._get_full_path(self.path, identification)

    def get_identifier(self) -> str:
        if self.full:
            return "A2MD_full"
        else:
            return "A2MD_split"


if __name__ == "__main__":
    convert_to_wav_dataset("../data/a2md_public/")
    print("Done")
