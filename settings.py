from dataclasses import dataclass

from dataset.mapping import DrumMapping


@dataclass
class AudioProcessingSettings:
    sample_rate: int = 44100
    hop_size: int = 441
    fft_size: int = 2048
    n_mels: int = 82
    center: bool = False
    pad_mode: str = "constant"
    mel_min: float = 20.0
    mel_max: float = 44100 / 2


@dataclass
class AnnotationSettings:
    mapping: DrumMapping
    pad_annotations: bool = False
    pad_value: float = 0.5
    lead_in: float = 0.25
    time_shift: float = 0.0
    beats: bool = False

    @property
    def n_classes(self):
        return len(self.mapping) + 2 * int(self.beats)


@dataclass
class TrainingSettings:
    learning_rate: float = 1e-3
    epochs: int = 20
    batch_size: int = 2048
    ema: bool = False
    scheduler: bool = True
    n_mels: int = 12 * 7
    early_stopping: bool = False
    dataset_version: str = "L"
    splits: list[float] = (0.8, 0.1, 0.1)
    num_workers: int = 32
    ignore_beats: bool = True


@dataclass
class CNNSettings:
    n_classes: int
    n_mels: int
    num_channels: int = 32
    num_residual_blocks: int = 0
    dropout: float = 0.3
    assert 0.0 <= dropout < 1.0
    causal = True
    flux = True
