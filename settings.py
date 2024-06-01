from dataclasses import dataclass
from typing import Literal

from torch import nn

from dataset.mapping import DrumMapping


@dataclass
class AudioProcessingSettings:
    sample_rate: int = 44100
    hop_size: int = 441
    fft_size: int = 2048
    n_mels: int = 12 * 7
    center: bool = True
    pad_mode: Literal["constant", "reflect"] = "constant"
    mel_min: float = 20.0
    mel_max: float = 20000.0
    normalize: bool = False


@dataclass
class AnnotationSettings:
    mapping: DrumMapping = DrumMapping.THREE_CLASS_STANDARD
    pad_annotations: bool = False
    pad_value: float = 0.5
    time_shift: float = 0.0
    beats: bool = False

    @property
    def n_classes(self):
        return len(self.mapping) + 2 * int(self.beats)


@dataclass
class DatasetSettings:
    audio_settings: AudioProcessingSettings = AudioProcessingSettings()
    annotation_settings: AnnotationSettings = AnnotationSettings()
    segment_type: Literal["frame", "label"] | None = "frame"
    frame_length: float = 0.25
    frame_overlap: float = 0.10
    label_lead_in: float = 0.25
    label_lead_out: float = 0.10




@dataclass
class TrainingSettings:
    learning_rate: float = 1e-4
    epochs: int = 30
    batch_size: int = 512
    ema: bool = False
    scheduler: bool = True
    early_stopping: int | None = None
    dataset_version: Literal["S", "M", "L"] = "L"
    splits: list[float] = (0.85, 0.15, 0.0)
    num_workers: int = 64
    min_save_score: float = 0.62
    train_set: Literal["all", "a2md_train"] = "all"


@dataclass
class EvaluationSettings:
    peak_mean_range: int = 2
    peak_max_range: int = 2
    onset_cooldown: int = 0.02
    detect_tolerance: int = 0.025
    ignore_beats: bool = True
    min_test_score: float = 0.54


@dataclass
class CNNSettings:
    n_classes: int
    n_mels: int
    num_channels: int = 16
    num_residual_blocks: int = 9
    dropout: float = 0.1
    causal: bool = True
    flux: bool = True
    activation: nn.Module = nn.SELU()
    classifier_dim: int = 2**6
    down_sample_factor: 2 | 3 | 4 = 3


@dataclass
class CNNAttentionSettings:
    n_classes: int
    n_mels: int
    num_channels: int = 16
    dropout: float = 0.1
    causal: bool = True
    flux: bool = False
    activation: nn.Module = nn.SELU()
