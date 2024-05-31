import os
from pathlib import Path

import numpy as np
import torch
import torchaudio

from dataset import get_segments
from generics import ADTDataset
from dataset.mapping import DrumMapping
from settings import AudioProcessingSettings, AnnotationSettings

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


def get_length(path: str, song: str):
    audio_path = os.path.join(path, "audio", f"{song}.mp3")
    meta_data = torchaudio.info(audio_path, backend="ffmpeg")
    return meta_data.num_frames / meta_data.sample_rate


class RBMA13(ADTDataset):
    def __init__(
        self,
        root: str | Path,
        audio_settings: AudioProcessingSettings,
        annotation_settings: AnnotationSettings,
        use_dataloader=False,
        splits: list[int] | None = None,
        is_train=False,
        **_kwargs,
    ):
        super().__init__(audio_settings, annotation_settings, is_train=is_train, use_dataloader=use_dataloader)
        self.path = root

        self.annotations = load_rbma(root)
        if splits is None:
            splits = [int(name.split("-")[-1]) for name in self.annotations.keys()]
        tracks = [f"RBMA-13-Track-{number:02}" for number in splits]
        self.annotations = {track: self.annotations[track] for track in tracks}
        self.annotations = [(identifier, annotation[1], annotation[0]) for identifier, annotation in self.annotations.items()]
        self.annotations.sort(key=lambda x: int(x[0].split("-")[-1]))

        lengths = [get_length(root, track[0]) for track in self.annotations]
        drum_labels = [track[2] for track in self.annotations]
        self.segments = (
            get_segments(
                lengths,
                drum_labels,
                self.lead_in,
                self.lead_out,
                self.sample_rate,
            )
            if self.is_train
            else None
        )

        self.label_spreader = torch.nn.MaxPool1d(3, stride=1, padding=1)

    def __len__(self):
        return len(self.annotations)

    def adjust_time_shift(self, time_shift: float):
        self.time_shift = time_shift


    def get_full_path(self, track: str) -> Path:
        audio_path = os.path.join(self.path, "audio", f"{track}.mp3")
        return Path(audio_path)



def main():
    a_settings = AudioProcessingSettings()
    an_settings = AnnotationSettings(mapping=DrumMapping.THREE_CLASS_STANDARD)
    dataset = RBMA13(
        "../data/rbma_13/",
        a_settings,
        an_settings,
        False,
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
