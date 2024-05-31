from abc import abstractmethod
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from dataset import load_audio, segment_audio, get_labels
from settings import AudioProcessingSettings, AnnotationSettings


class ADTDataset(Dataset[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]]):
    @abstractmethod
    def __init__(
        self,
        audio_settings: AudioProcessingSettings,
        annotation_settings: AnnotationSettings,
        is_train: bool,
        use_dataloader: bool = False,
    ):
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
        self.lead_in = annotation_settings.lead_in
        self.lead_out = annotation_settings.lead_out
        self.time_shift = annotation_settings.time_shift
        self.beats = annotation_settings.beats
        self.n_classes = annotation_settings.n_classes
        self.normalize = audio_settings.normalize
        self.segment_length = self.lead_in + self.lead_out
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

        self.annotation_padder = torch.nn.MaxPool1d(3, stride=1, padding=1) if self.pad_annotations else None

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
        else:
            audio_idx = idx
            start, end = 0, -1

        assert self.annotations is not None, "Annotations are not loaded"

        identification, drums, beats = self.annotations[int(audio_idx)]
        path = self.get_full_path(identification)

        full_audio = (
            load_audio(path, self.sample_rate, self.normalize)
            if self.cache is None
            else self.cache[int(audio_idx)]
        )
        audio = segment_audio(full_audio, start, end, int(self.segment_length * self.sample_rate))
        if not self.center:
            # align frames with the end of the window
            audio = torch.nn.functional.pad(audio, (self.fft_size - self.hop_size, 0), mode=self.pad_mode)

        spectrum = self.spectrum(audio)
        spectrum = torch.log1p(spectrum)
        mel = self.filter_bank(spectrum)

        time_offset = start / self.sample_rate

        drums = [drum + self.time_shift for drum in drums]

        time_stamps = [*beats, *drums] if self.beats else [*drums]
        time_stamps = [cls[cls >= time_offset] - time_offset for cls in time_stamps]
        labels = get_labels(time_stamps, self.sample_rate, self.hop_size, mel.shape[-1])

        if self.pad_annotations:
            padded = self.annotation_padder(labels.unsqueeze(0)).squeeze(0) * self.pad_value
            labels = torch.maximum(labels, padded)

        gt_labels = [*beats, *drums]

        if self.use_dataloader:
            # allows use of torch.nn.utils.rnn.pad_sequence
            # it expects the variable length dimension to be the first dimension
            return mel.permute(1, 0), labels.permute(1, 0), gt_labels

        return mel, labels, gt_labels

    def adjust_time_shift(self, time_shift: float):
        self.time_shift = time_shift
