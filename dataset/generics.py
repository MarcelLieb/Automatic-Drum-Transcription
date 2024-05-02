from abc import abstractmethod
from dataclasses import asdict

import torch
from torch.utils.data import Dataset

from settings import AudioProcessingSettings, AnnotationSettings


class ADTDataset(Dataset[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]]):
    @abstractmethod
    def __init__(
        self,
        audio_settings: AudioProcessingSettings,
        annotation_settings: AnnotationSettings,
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
        self.time_shift = annotation_settings.time_shift
        self.beats = annotation_settings.beats
        self.n_classes = annotation_settings.n_classes

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def adjust_time_shift(self, time_shift: float):
        pass
