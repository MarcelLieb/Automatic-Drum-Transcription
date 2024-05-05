import os.path
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torchaudio

from dataset.generics import ADTDataset
from dataset.mapping import DrumMapping, get_name_to_class_number
from settings import AudioProcessingSettings, AnnotationSettings

label_translater = {
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
        .replace(label_translater)
        .replace(name_to_class)
        .cast(pl.Float32),
    ).to_numpy()
    beats = np.loadtxt(
        os.path.join(root, "annotations", "beats", f"{name}_MIX.beats"), delimiter="\t"
    )
    beats = [beats[beats[:, 1] == 1][:, 0], beats[:, 0]]

    drums = [labels[labels[:, 1] == i][:, 0] for i in range(len(mapping))]

    return beats, drums


def load_audio(path: str, identifier: str, sample_rate: int) -> torch.Tensor:
    audio_path = os.path.join(path, "audio", "full_mix", f"{identifier}_MIX.wav")
    audio, sr = torchaudio.load(audio_path, normalize=True, backend="ffmpeg")
    audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(audio)
    audio = torch.mean(audio, dim=0, keepdim=False, dtype=torch.float32)
    audio = audio / torch.max(torch.abs(audio))
    return audio


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
        )

        self.path = path
        self.use_dataloader = use_dataloader
        self.is_train = is_train

        self.pad = (
            torch.nn.MaxPool1d(3, stride=1, padding=1) if self.pad_annotations else None
        )

        self.annotations = {}

        for root, dirs, files in os.walk(os.path.join(path, "annotations", "subclass")):
            for file in files:
                name = "_".join(file.split("_")[:2])
                self.annotations[name] = get_annotations(path, name, self.mapping)

        self.spectrum = torchaudio.transforms.Spectrogram(
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.fft_size // 2,
            power=2,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=True,
            onesided=True,
        )

        self.filter_bank = torchaudio.transforms.MelScale(
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            f_min=self.mel_min,
            f_max=self.mel_max,
            n_stft=self.fft_size // 2 + 1,
        )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        name = list(self.annotations.keys())[idx]
        beats, drums = self.annotations[name]
        audio = load_audio(self.path, name, self.sample_rate)

        frames = (audio.shape[-1] - self.fft_size) // self.hop_size + 1
        labels = torch.zeros(((self.beats * 2) + 3, frames), dtype=torch.float32)

        if self.beats:
            down_beat_indices = (beats[0] * self.sample_rate) // self.hop_size
            down_beat_indices = torch.tensor(down_beat_indices, dtype=torch.long)
            down_beat_indices = down_beat_indices[down_beat_indices < frames]
            beat_indices = (beats[1] * self.sample_rate) // self.hop_size
            beat_indices = torch.tensor(beat_indices, dtype=torch.long)
            beat_indices = beat_indices[beat_indices < frames]
            labels[0, down_beat_indices] = 1
            labels[1, beat_indices] = 1

        hop_length = self.hop_size / self.sample_rate

        drum_indices = [(drum * self.sample_rate) // self.hop_size for drum in drums]
        drum_indices = [drum[drum < frames] for drum in drum_indices]
        for i, drum_class in enumerate(drum_indices):
            for j in range(round(self.time_shift // hop_length) + 1):
                shifted_drum_class = drum_class + j
                labels[
                    int(self.beats) * 2 + i,
                    shifted_drum_class[shifted_drum_class < frames],
                ] = 1
        if self.pad is not None:
            padded = self.pad(labels.unsqueeze(0)).squeeze(0) * self.pad_value
            labels = torch.maximum(labels, padded)

        gt_labels = [*beats, *drums]

        spectrum = self.spectrum(audio)
        spectrum = torch.log1p(spectrum)
        mel = self.filter_bank(spectrum)

        if self.use_dataloader:
            return mel.permute(1, 0), labels.permute(1, 0), gt_labels
        return spectrum, labels, gt_labels
