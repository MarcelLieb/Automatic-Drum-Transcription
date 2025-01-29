import os
from pathlib import Path

import numpy as np
import torch

from dataset import get_label_windows, get_length, get_segments
from generics import ADTDataset
from settings import DatasetSettings

rbma_13_path = "./data/rbma_13/"


def load_rbma(rbma_path: str = rbma_13_path) -> dict[str, (np.array, np.array)]:
    annotations = {}
    for root, dirs, files in os.walk(os.path.join(rbma_path, "annotations", "drums")):
        for file in files:
            track = file.split(".")[0]
            drums = np.loadtxt(os.path.join(root, file), delimiter="\t")
            drums = [drums[drums[:, 1] == i][:, 0] for i in range(3)]
            beats = np.loadtxt(
                os.path.join(rbma_path, "annotations", "beats", f"{track}.txt"),
                delimiter="\t",
            )
            beats = [beats[beats[:, 1] == 1][:, 0], beats[:, 0]]
            annotations[track] = (beats, drums)
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
        for file in os.listdir(os.path.join(path, "annotations", "drums"))
        if file.endswith(".txt")
    ]


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
        self.annotations = load_rbma(path)
        if splits is None:
            splits = get_tracks(path)
        tracks = [f"RBMA-13-Track-{number:02}" for number in splits]
        self.annotations = {track: self.annotations[track] for track in tracks}
        self.annotations = [
            (identifier, annotation[1], annotation[0])
            for identifier, annotation in self.annotations.items()
        ]
        self.annotations.sort(key=lambda x: int(x[0].split("-")[-1]))

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

    def get_full_path(self, track: str) -> Path:
        audio_path = os.path.join(self.path, "audio", f"{track}.mp3")
        return Path(audio_path)

    def get_identifier(self) -> str:
        if self.full:
            return "RBMA_full"
        else:
            return "RBMA_split"


def main():
    settings = DatasetSettings()
    dataset = RBMA13(
        "../data/rbma_13/",
        settings,
    )
    averages = torch.zeros((4, 82))
    total_pos = torch.zeros(4)
    for i in range(len(dataset)):
        mel, labels, gt = dataset[i]
        spec_diff = mel[..., 1:] - mel[..., :-1]
        spec_diff = torch.clamp(spec_diff, min=0.0)
        spec_diff = torch.cat(
            (torch.zeros_like(spec_diff[..., -1:]), spec_diff), dim=-1
        )
        labels = labels.permute(1, 0)
        mask = labels == 1
        for j in range(4):
            positives = spec_diff[:, mask[j, :]].mean(dim=-1)
            positives = positives / torch.linalg.norm(
                positives, ord=1, dim=-1, keepdim=True
            )
            negatives = spec_diff[:, ~mask[j, :]].mean(dim=-1)
            negatives = negatives / torch.linalg.norm(
                negatives, ord=1, dim=-1, keepdim=True
            )
            total_pos[j] += torch.any(mask[j, :])
            if torch.any(mask[j, :]):
                total = positives - negatives
                averages[j, :] += total

    averages /= total_pos.unsqueeze(-1)
    print(averages)


if __name__ == "__main__":
    main()
