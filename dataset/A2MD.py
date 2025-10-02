import os
import subprocess
from pathlib import Path

import pretty_midi
import torch
import polars as pl
from sklearn.model_selection import KFold

from dataset import (
    # load_audio,
    get_label_windows,
    get_drums,
    get_length,
    get_segments,
    get_splits as get_splits_data,
    convert_to_wav,
    convert_to_flac,
)
from dataset.mapping import DrumMapping
from dataset.generics import ADTDataset
from settings import DatasetSettings

A2MD_PATH = "./data/a2md_public/"

PENALTY_CUTOFFS = {
    "S": 0.2,
    "M": 0.4,
    "L": 0.7,
}


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


def convert_to_flac_dataset(root: str):
    tracks = get_tracks(root)
    for folder, identifiers in tracks.items():
        print(f"Converting {folder}")
        for identifier in identifiers:
            convert_to_flac(A2MD._get_full_path(root, (folder, identifier)))
        # Add seek points to the flac files to improve random read performance
        cwd = str(Path(os.path.join(root, "ytd_audio", folder)).resolve())
        code = subprocess.run(
            "metaflac --add-seekpoint=0.1s *.flac", cwd=cwd, shell=True
        )
        if code.returncode != 0:
            print("Metaflac failed. Please ensure it is installed and in your PATH.")


def get_fold(
    cut_off: float, path: str, n_folds: int, fold: int, seed=42
) -> list[dict[str, list[str]]]:
    assert fold < n_folds, "Fold index out of range"
    assert 0.1 <= cut_off <= 0.7, "Cut-off must be between 0.1 and 0.7"
    folders = [f"dist0p{x:02}" for x in range(0, int(cut_off * 100), 10)]
    groups = pl.scan_csv(Path(path) / "groups.csv", has_header=True).filter(
        pl.col("folder").is_in(folders)
    )
    if n_folds in [3, 5, 10]:
        fold = (
            pl.scan_csv(Path(path) / f"splits_{n_folds}-folds_3.csv", has_header=True)
            .filter((pl.col("folder").is_in(folders)) & (pl.col("fold") == fold))
            .select("identifier")
            .collect()
            .to_series()
            .to_list()
        )
    else:
        k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        group_index = groups.select("group").unique().to_series().to_list()
        k_folds = k_fold.split(group_index)
        fold = list(k_folds)[fold][1]
        fold = (
            groups.filter(pl.col("group").is_in(fold))
            .select("identifier")
            .collect()
            .to_series()
            .to_list()
        )
    train_ids = (
        groups.filter(~(pl.col("identifier").is_in(fold)))
        .select("identifier")
        .collect()
        .to_series()
        .to_list()
    )
    val_ids = fold
    out = [{} for _ in range(3)]
    for folder in folders:
        identifiers = (
            groups.filter(pl.col("folder") == folder)
            .select("identifier")
            .collect()
            .to_series()
            .to_list()
        )
        out[0][folder] = sorted(set(identifiers) & set(train_ids))
        out[1][folder] = sorted(set(identifiers) & set(val_ids))
        out[2][folder] = sorted(set(identifiers) - set(train_ids) - set(val_ids))
    return out


def get_splits(
    cut_off: float,
    splits: list[float],
    path: str,
    seed: int = 42,
    return_seed: bool = False,
) -> list[dict[str, list[str]]] | tuple[list[dict[str, list[str]]], int]:
    assert 0.1 <= cut_off <= 0.7, "Cut-off must be between 0.1 and 0.7"
    assert abs(sum(splits) - 1) < 1e-4
    folders = [f"dist0p{x:02}" for x in range(0, int(cut_off * 100), 10)]

    group_index = pl.scan_csv(Path(path) / "groups.csv", has_header=True).filter(
        pl.col("folder").is_in(folders)
    )
    groups = sorted(
        group_index.select("group").unique().collect().to_series().to_list()
    )

    finished = False
    out = [{} for _ in range(len(splits))]

    while not finished:
        split = get_splits_data(splits, groups, seed=seed)

        out = [{} for _ in range(len(splits))]
        for folder in folders:
            identifiers = group_index.filter((pl.col("folder") == folder))
            for split_ixd, selected_groups in enumerate(split):
                out[split_ixd][folder] = sorted(
                    identifiers.filter(pl.col("group").is_in(selected_groups))
                    .select("identifier")
                    .collect()
                    .to_series()
                    .to_list()
                )

        # Check if the distribution per folder is acceptable
        finished = True
        for folder in folders:
            track_counts = torch.tensor(
                [len(out[i][folder]) for i in range(len(splits))]
            )
            total = sum(track_counts)
            fractions = track_counts.float() / total
            if not torch.allclose(fractions, torch.tensor(splits), atol=0.02):
                finished = False
                # Set the next seed
                seed += 1
                break

    if return_seed:
        return out, seed

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
        super().__init__(
            settings, is_train=is_train, use_dataloader=use_dataloader, segment=segment
        )
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
        return (
            len(self.segments) if self.segments is not None else len(self.annotations)
        )

    @staticmethod
    def get_splits(splits: list[float], data: dict, seed: int, **kwargs):
        cut_off = {
            "L": 0.7,
            "M": 0.4,
            "S": 0.2,
        }
        folders = [
            f"dist0p{x:02}"
            for x in range(0, int(cut_off[kwargs["dataset_version"]] * 100), 10)
        ]
        out = [{} for _ in range(len(splits))]
        for folder in folders:
            identifiers = data[folder]
            split = get_splits_data(splits, identifiers, seed=seed)
            for i, s in enumerate(split):
                out[i][folder] = s
        # Check if the distribution is correct
        flat_tracks = [track for folder in folders for track in data[folder]]
        for i, s in enumerate(out):
            total = sum(len(arr) for _, arr in s.items())
            assert abs(total / len(flat_tracks) - splits[i]) < 2e-2, (
                f"{total / len(flat_tracks)} != {splits[i]}"
            )

        return out

    @staticmethod
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

    @staticmethod
    def _get_full_path(root: str, identification: tuple[str, str]) -> Path:
        folder, identifier = identification
        if os.path.exists(
            os.path.join(root, "ytd_audio", folder, f"ytd_audio_{identifier}.flac")
        ):
            return Path(
                os.path.join(root, "ytd_audio", folder, f"ytd_audio_{identifier}.flac")
            )
        if os.path.exists(
            os.path.join(root, "ytd_audio", folder, f"ytd_audio_{identifier}.wav")
        ):
            return Path(
                os.path.join(root, "ytd_audio", folder, f"ytd_audio_{identifier}.wav")
            )
        return Path(
            os.path.join(root, "ytd_audio", folder, f"ytd_audio_{identifier}.mp3")
        )

    def get_full_path(self, identification: tuple[str, str]) -> Path:
        return self._get_full_path(self.path, identification)

    def get_identifier(self) -> str:
        if self.full:
            return "A2MD_full"
        else:
            return "A2MD_split"


if __name__ == "__main__":
    convert_to_flac_dataset("../data/a2md_public/")
    print("Done")
