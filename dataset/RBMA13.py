import os
import random

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Vol, SpeedPerturbation

rbma_13_path = "./data/rbma_13/"


def load_rbma(rbma_path=rbma_13_path):
    annotations = {}
    for root, dirs, files in os.walk(os.path.join(rbma_path, "annotations", "drums")):
        for file in files:
            track = file.split(".")[0]
            data = np.loadtxt(os.path.join(root, file), delimiter="\t")
            annotations[track] = data
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
            os.rename(os.path.join(rbma_13_path, "audio", files[0]),
                      os.path.join(rbma_13_path, "audio", f"RBMA-13-Track-{i + 1:02}.mp3"))

    return positions


class RBMA_13(Dataset):
    def __init__(self, root, split, sample_rate=48000, hop_size=480, fft_size=2048, label_shift=-0.02,
                 pad_annotations=3, n_mels=82, center=False, pad_mode="constant"):
        self.root = root
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.fft_size = fft_size
        self.label_shift = label_shift
        self.train = "train" in split.lower()
        self.spectrum = torchaudio.transforms.Spectrogram(n_fft=self.fft_size, hop_length=self.hop_size,
                                                          win_length=self.fft_size//2, power=2, center=center,
                                                          pad_mode=pad_mode, normalized=False, onesided=True)
        self.filter_bank = torchaudio.transforms.MelScale(
            n_mels=n_mels, sample_rate=sample_rate, n_stft=self.fft_size // 2 + 1, f_min=0.0, f_max=20000, norm=None,
            mel_scale="htk",
        )

        annotations = load_rbma()
        self.annotations = {}
        self.label_spreader = torch.nn.MaxPool1d(pad_annotations, stride=1, padding=pad_annotations // 2)
        split = split.lower()
        assert split in ["train", "train_big", "validation", "test", "all"]
        if split == "all":
            self.annotations = annotations
            return
        if split == "train_big":
            with open(os.path.join(root, "splits", f"3-fold_cv_0.txt")) as f:
                tracks = f.readlines()
            with open(os.path.join(root, "splits", f"3-fold_cv_1.txt")) as f:
                tracks += f.readlines()
            tracks = [x.strip() for x in tracks]
            for track in tracks:
                self.annotations[track] = annotations[track]
            return
        index = ["train", "validation", "test"].index(split)
        with open(os.path.join(root, "splits", f"3-fold_cv_{index}.txt")) as f:
            tracks = f.readlines()
        tracks = [x.strip() for x in tracks]
        for track in tracks:
            self.annotations[track] = annotations[track]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        track = list(self.annotations.keys())[idx]

        annotation = self.annotations[track]
        audio_path = os.path.join(self.root, "audio", f"{track}.mp3")

        audio, sample_rate = torchaudio.load(audio_path, normalize=True, backend="ffmpeg")
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=48000)(audio)
        audio = torch.mean(audio, dim=0, keepdim=False, dtype=torch.float32)
        audio = audio / torch.max(torch.abs(audio))

        if self.train and False:
            transform = Gain(min_gain=-3, max_gain=0)
            audio = transform(audio)

            perturbation = round(random.uniform(0.9, 1.1), 1)
            audio = SpeedPerturbation(self.sample_rate, [perturbation])(audio)[0]
            annotation[:, 0] /= perturbation

        annotation[:, 0] += self.label_shift

        frames = (audio.shape[0] - self.fft_size) // self.hop_size + 1
        labels = torch.zeros((4, frames), dtype=torch.int64)
        indices = (annotation[:, 0] * self.sample_rate) // self.hop_size
        indices = torch.tensor(indices).long()
        for i in range(3):
            inst_indices = indices[annotation[:, 1] == i]
            labels[i, inst_indices] = 1
            labels[3, inst_indices] = 1
        labels = labels.float()
        labels = self.label_spreader(labels.unsqueeze(0)).squeeze(0)
        labels = labels.permute(1, 0)
        spectrum = self.spectrum(audio)
        spectrum = self.filter_bank(spectrum)
        spectrum = spectrum.permute(1, 0)

        return spectrum, labels


def get_dataloader(root, split, batch_size, num_workers, sample_rate=48000, hop_size=480, fft_size=2048,
                   label_shift=-0.02, **kwargs):
    dataset = RBMA_13(root, split, sample_rate, hop_size, fft_size, label_shift=label_shift, **kwargs)
    is_train = split.lower() == "train"
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers,
                                             collate_fn=collate_fn, drop_last=is_train)
    return dataloader


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    audio, annotation = zip(*batch)
    audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=0.0)
    annotation = list(annotation)
    annotation = torch.nn.utils.rnn.pad_sequence(annotation, batch_first=True, padding_value=-1)
    audio = audio.permute(0, 2, 1)
    annotation = annotation.permute(0, 2, 1)
    return audio, annotation


class Gain(torch.nn.Module):
    def __init__(self, min_gain: float = -20.0, max_gain: float = -1):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        gain = random.uniform(self.min_gain, self.max_gain)
        audio = Vol(gain, gain_type="db")(audio)
        return audio


if __name__ == '__main__':
    dataset = RBMA_13(rbma_13_path, "all", 48000, 480, 2048, label_shift=-0.01, pad_annotations=3)
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=480, win_length=1024, power=2, center=False,
                                                    pad_mode="constant", normalized=False, onesided=True)
    mel_scale = torchaudio.transforms.MelScale(n_mels=82, sample_rate=48000, n_stft=1025, f_min=0.0, f_max=20000,
                                               norm=None, mel_scale="htk")
    averages = torch.zeros((4, 82))
    averages_neg = torch.zeros((4, 82))
    total = torch.zeros(4)
    total_pos = torch.zeros(4)
    total_neg = torch.zeros(4)
    for i in range(len(dataset)):
        audio, labels = dataset[i]
        spec = spectrogram(audio)
        log = torch.log1p(spec * 0.1)
        mel = mel_scale(log)
        spec_diff = mel[..., 1:] - mel[..., :-1]
        spec_diff = torch.clamp(spec_diff, min=0.0)
        spec_diff = torch.cat((torch.zeros_like(spec_diff[..., -1:]), spec_diff), dim=-1)
        labels = labels.permute(1, 0)
        mask = labels == 1
        for j in range(4):
            positives = spec_diff[:, mask[j, :]].mean(dim=-1)
            positives = positives / torch.linalg.norm(positives, ord=1, dim=-1, keepdim=True)
            negatives = spec_diff[:, ~mask[j, :]].mean(dim=-1)
            negatives = negatives / torch.linalg.norm(negatives, ord=1, dim=-1, keepdim=True)
            total_pos[j] += torch.any(mask[j, :])
            if torch.any(mask[j, :]):
                total = positives - negatives
                averages[j, :] += total

    averages /= total_pos.unsqueeze(-1)
    print(a := averages)
