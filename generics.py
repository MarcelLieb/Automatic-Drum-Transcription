from abc import abstractmethod, ABC
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from dataset import get_labels, load_audio, segment_audio
from settings import DatasetSettings


class ADTDataset(Dataset[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]]):
    @abstractmethod
    def __init__(
        self,
        settings: DatasetSettings,
        is_train: bool,
        use_dataloader: bool = False,
    ):
        audio_settings = settings.audio_settings
        annotation_settings = settings.annotation_settings
        self.sample_rate = audio_settings.sample_rate
        self.hop_size = audio_settings.hop_size
        self.fft_size = audio_settings.fft_size
        self.n_mels = audio_settings.n_mels
        self.center = audio_settings.center
        self.pad_mode = audio_settings.pad_mode
        self.mel_min = audio_settings.mel_min
        self.mel_max = audio_settings.mel_max
        self.mapping = annotation_settings.mapping
        self.pad_annotations = annotation_settings.pad_annotations
        self.pad_value = annotation_settings.pad_value
        self.time_shift = annotation_settings.time_shift
        self.beats = annotation_settings.beats
        self.n_classes = annotation_settings.n_classes
        self.normalize = audio_settings.normalize

        self.segment_type = settings.segment_type

        self.lead_in = settings.label_lead_in
        self.lead_out = settings.label_lead_out

        self.segment_length = (
            self.lead_in + self.lead_out
            if settings.segment_type == "label"
            else settings.frame_length
        )
        self.segment_overlap = settings.frame_overlap

        self.is_train = is_train
        self.use_dataloader = use_dataloader

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
            f_min=audio_settings.mel_min,
            f_max=audio_settings.mel_max,
            n_stft=self.fft_size // 2 + 1,
        )

        self.annotations: list[tuple[Any, list[np.array], list[np.array]]] | None = None

        self.annotation_padder = (
            torch.nn.MaxPool1d(3, stride=1, padding=1) if self.pad_annotations else None
        )

        self.cache = None
        self.segments = None

    @abstractmethod
    def get_full_path(self, identification: Any):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def __getitem__(self, idx):
        if self.segments is not None:
            start, end, audio_idx = self.segments[idx]
            audio_idx = int(audio_idx)
        else:
            audio_idx = idx
            start, end = 0, -1

        assert self.annotations is not None, "Annotations are not loaded"

        identification, drums, beats = self.annotations[audio_idx]
        path = self.get_full_path(identification)

        if self.cache is None:
            audio = load_audio(path, self.sample_rate, self.normalize, start, end)
        else:
            full_audio = self.cache[audio_idx]
            start, end, segment_length = (
                np.array((start, end, self.segment_length)) * self.sample_rate
            ).astype(int)
            audio = segment_audio(full_audio, start, end, segment_length)

        if audio.shape[-1] < int(self.segment_length * self.sample_rate) and start == 0:
            audio = torch.cat(
                (
                    torch.zeros(
                        int(self.segment_length * self.sample_rate) - audio.shape[-1]
                    ),
                    audio,
                )
            )
        elif audio.shape[-1] < int(self.segment_length * self.sample_rate):
            audio = torch.cat(
                (
                    audio,
                    torch.zeros(
                        int(self.segment_length * self.sample_rate) - audio.shape[-1]
                    ),
                )
            )

        # segments get padded at the front if start is 0, therefore the true start may be negative
        start = (
            (end - start) - int(self.segment_length * self.sample_rate)
            if start == 0
            else start
        )
        if not self.center:
            # align frames with the end of the window
            audio = torch.nn.functional.pad(
                audio, (self.fft_size - self.hop_size, 0), mode=self.pad_mode
            )

        spectrum = self.spectrum(audio)
        spectrum = torch.log1p(spectrum)
        mel = self.filter_bank(spectrum)

        time_offset = start / self.sample_rate

        drums = [drum + self.time_shift for drum in drums]

        time_stamps = [*beats, *drums] if self.beats else [*drums]
        time_stamps = [cls[cls >= time_offset] - time_offset for cls in time_stamps]
        labels = get_labels(time_stamps, self.sample_rate, self.hop_size, mel.shape[-1])

        if self.pad_annotations:
            padded = (
                self.annotation_padder(labels.unsqueeze(0)).squeeze(0) * self.pad_value
            )
            labels = torch.maximum(labels, padded)

        gt_labels = [*beats, *drums]

        if self.use_dataloader:
            # allows use of torch.nn.utils.rnn.pad_sequence
            # it expects the variable length dimension to be the first dimension
            return mel.permute(1, 0), labels.permute(1, 0), gt_labels

        return mel, labels, gt_labels

    def adjust_time_shift(self, time_shift: float):
        self.time_shift = time_shift


class ConcatADTDataset(ADTDataset, ABC):
    def __init__(self, settings: DatasetSettings, datasets: list[ADTDataset]):
        super().__init__(settings, datasets[0].is_train, datasets[0].use_dataloader)
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.cumulative_lengths = np.cumsum(self.lengths)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > self.__len__():
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = self.__len__() + idx
        dataset_idx = np.searchsorted(self.cumulative_lengths, idx, side="right")
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_lengths[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def adjust_time_shift(self, time_shift: float):
        for dataset in self.datasets:
            dataset.adjust_time_shift(time_shift)

    def get_full_path(self, identification: Any):
        raise NotImplementedError(
            "get_full_path is not implemented for ConcatADTDataset"
        )
