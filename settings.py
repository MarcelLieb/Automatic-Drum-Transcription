from abc import abstractmethod
from dataclasses import dataclass, asdict as dataclass_asdict, is_dataclass
from typing import Literal

from torch import nn
from torch.multiprocessing import cpu_count

from dataset.mapping import DrumMapping


def asdict(settings):
    dic = dataclass_asdict(settings)
    out = {}
    for key, value in dic.items():
        if is_dataclass(value) or isinstance(value, dict):
            if is_dataclass(value):
                value = asdict(value)
            for subkey, subvalue in value.items():
                out[subkey] = subvalue
        elif isinstance(value, (int, float, str, bool)):
            out[key] = value
        elif isinstance(value, nn.Module):
            out[key] = value.__class__.__name__
        else:
            out[key] = str(value)
    return out


@dataclass(frozen=True)
class AudioProcessingSettings:
    sample_rate: int = 44100
    hop_size: int = 441
    fft_size: int = 2048
    n_mels: int = 96
    center: bool = True
    pad_mode: Literal["constant", "reflect"] = "constant"
    mel_min: float = 20.0
    mel_max: float = 20000.0
    normalize: bool = False


@dataclass(frozen=True)
class AnnotationSettings:
    mapping: DrumMapping = DrumMapping.THREE_CLASS_STANDARD
    pad_annotations: bool = True
    pad_value: float = 0.5
    time_shift: float = 0.025
    beats: bool = False

    @property
    def n_classes(self):
        return len(self.mapping) + 2 * int(self.beats)


@dataclass(frozen=True)
class DatasetSettings:
    audio_settings: AudioProcessingSettings = AudioProcessingSettings()
    annotation_settings: AnnotationSettings = AnnotationSettings()
    segment_type: Literal["frame", "label"] | None = "frame"
    frame_length: float = 8.0
    frame_overlap: float = 0.1
    label_lead_in: float = 0.25
    label_lead_out: float = 0.10


@dataclass(frozen=True)
class TrainingSettings:
    learning_rate: float = 1e-4
    epochs: int = 20
    batch_size: int = 16
    weight_decay: float = 1e-5
    positive_weight: float = 1.0
    ema: bool = False
    scheduler: bool = False
    early_stopping: int | None = None
    dataset_version: Literal["S", "M", "L"] = "L"
    splits: list[float] = (0.85, 0.15, 0.0)
    num_workers: int = cpu_count()
    min_save_score: float = 0.62
    test_batch_size: int = 1
    train_set: Literal["all", "a2md_train"] = "a2md_train"
    model_settings: Literal["cnn", "cnn_attention", "mamba", "mamba_fast", "unet"] = "mamba_fast"


@dataclass(frozen=True)
class ModelSettingsBase:
    @abstractmethod
    def get_identifier(self):
        pass


@dataclass(frozen=True)
class EvaluationSettings:
    peak_mean_range: int = 2
    peak_max_range: int = 2
    onset_cooldown: int = 0.021
    detect_tolerance: int = 0.025
    ignore_beats: bool = True
    min_test_score: float = 0.48
    pr_points: int | None = 1000


@dataclass(frozen=True)
class CNNSettings(ModelSettingsBase):
    num_channels: int = 32
    num_residual_blocks: int = 7
    num_feature_layers: int = 2
    channel_multiplication: int = 2
    dropout: float = 0.1
    causal: bool = True
    flux: bool = True
    activation: nn.Module = nn.SELU()
    classifier_dim: int = 2 ** 6
    down_sample_factor: 2 | 3 | 4 = 2

    def get_identifier(self):
        return "cnn"


@dataclass(frozen=True)
class CNNAttentionSettings(ModelSettingsBase):
    num_channels: int = 24
    dropout: float = 0.1
    causal: bool = True
    flux: bool = True
    activation: nn.Module = nn.SELU()
    num_attention_blocks: int = 5
    num_heads: int = 8
    context_size: int = 200
    expansion_factor: int = 4
    use_relative_pos: bool = False

    def get_identifier(self):
        return "cnn_attention"


@dataclass(frozen=True)
class CNNMambaSettings(ModelSettingsBase):
    num_channels: int = 16
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1
    causal: bool = True
    flux: bool = False
    backbone: Literal["unet", "cnn"] = "cnn"
    activation: nn.Module = nn.SELU()
    n_layers: int = 5

    def get_identifier(self):
        return "mamba_fast"


@dataclass(frozen=True)
class CRNNSettings(ModelSettingsBase):
    num_channels: int = 32
    num_conv_layers: int = 2
    num_rnn_layers: int = 3
    dropout: float = 0.3
    causal: bool = True
    flux: bool = True
    activation: nn.Module = nn.SELU()

    def get_identifier(self):
        return "crnn"


@dataclass(frozen=True)
class UNetSettings(ModelSettingsBase):
    channels: int = 32

    def get_identifier(self):
        return "unet"
