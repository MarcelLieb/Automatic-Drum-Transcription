from abc import abstractmethod, ABC
from dataclasses import dataclass, asdict as dataclass_asdict, is_dataclass, field
from typing import Literal, Optional
from overrides import override
from ast import literal_eval as make_tuple

from torch import nn
from torch.multiprocessing import cpu_count

from dataset.mapping import DrumMapping


def asdict(settings):
    dic = settings
    if is_dataclass(settings):
        dic = dataclass_asdict(settings)
    out = {}
    for key, value in dic.items():
        if is_dataclass(value) or isinstance(value, dict):
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


@dataclass
class SettingsBase(ABC):
    @classmethod
    def from_flat_dict(cls, dic):
        dic = {key: item for key, item in dic.items()}  # make copy
        keys = dataclass_asdict(cls()).keys()
        class_attributes = [key for key in keys if key in dic]
        attributes = {
            key: dic[key] if dic[key] != "None" else None for key in class_attributes
        }
        return cls(**attributes)


@dataclass
class AudioProcessingSettings(SettingsBase):
    sample_rate: int = 44100
    hop_size: int = 441
    fft_size: int = 2048
    n_mels: int = 128
    pad_mode: Literal["constant", "reflect"] = "constant"
    mel_min: float = 20.0
    mel_max: float = 16000.0  # A2MD uses 128 kbps mp3 files, which have a maximum frequency of 16 kHz
    power: Literal[1, 2] = 1
    normalize: bool = False


@dataclass
class AnnotationSettings(SettingsBase):
    mapping: DrumMapping = DrumMapping.THREE_CLASS
    pad_annotations: bool = True
    pad_value: float = 0.5
    time_shift: float = 0.02
    beats: bool = False

    @property
    def n_classes(self):
        return len(self.mapping) + 2 * int(self.beats)

    @classmethod
    @override
    def from_flat_dict(cls, dic):
        dic = {key: item for key, item in dic.items()}  # make copy
        keys = dataclass_asdict(cls()).keys()
        class_attributes = [key for key in keys if key in dic]
        if "mapping" in class_attributes:
            dic["mapping"] = DrumMapping.from_str(str(dic["mapping"]))
        attributes = {
            key: dic[key] if dic[key] != "None" else None for key in class_attributes
        }
        return cls(**attributes)


@dataclass
class DatasetSettings(SettingsBase):
    audio_settings: AudioProcessingSettings = field(
        default_factory=AudioProcessingSettings
    )
    annotation_settings: AnnotationSettings = field(default_factory=AnnotationSettings)
    a2md_penalty_cutoff: float = 0.7
    splits: list[float] = (0.8, 0.2, 0.0)
    k_folds: Literal[None, 5, 10] = 5
    fold: int | None = 0
    full_length_test: bool = True
    per_song_sampling: bool = False
    num_workers: int = 8
    train_set: Literal["all", "a2md_train", "midi"] = "a2md_train"
    eval_set: str = "A2MD"
    test_sets: tuple[str] = ("RBMA", "MDB")
    segment_type: Literal["frame", "label"] | None = "frame"
    frame_length: float = 8.0
    frame_overlap: float = 0.0
    label_lead_in: float = 0.25
    label_lead_out: float = 0.10

    @classmethod
    @override
    def from_flat_dict(cls, dic):
        dic = {key: item for key, item in dic.items()}  # make copy
        keys = dataclass_asdict(cls()).keys()
        class_attributes = [key for key in keys if key in dic]
        if "splits" in class_attributes:
            dic["splits"] = make_tuple(str(dic["splits"]))
        if "test_sets" in class_attributes:
            dic["test_sets"] = make_tuple(str(dic["test_sets"]))
        class_attributes += ["audio_settings", "annotation_settings"]
        dic["audio_settings"] = AudioProcessingSettings.from_flat_dict(dic)
        dic["annotation_settings"] = AnnotationSettings.from_flat_dict(dic)
        attributes = {
            key: dic[key] if dic[key] != "None" else None for key in class_attributes
        }
        return cls(**attributes)


@dataclass
class TrainingSettings(SettingsBase):
    learning_rate: float = 0.004
    epochs: int = 30
    batch_size: int = 32
    weight_decay: float = 1e-12
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    gradient_clip_norm: Optional[float] = 1.0
    optimizer: Literal["adam", "radam"] = "radam"
    decoupled_weight_decay: bool = True
    use_pos_weight: bool = True
    ema: Optional[float] = None  # 0.998
    scheduler: bool = False
    early_stopping: int | None = None
    min_save_score: float = 0.75
    test_batch_size: int = 3
    model_settings: Literal["cnn", "cnn_attention", "mamba", "crnn", "vogl"] = (
        "cnn_attention"
    )

    def get_model_settings_class(self):
        match self.model_settings:
            case "cnn":
                return CNNSettings
            case "cnn_attention":
                return CNNAttentionSettings
            case "mamba" | "mamba_fast":
                return CNNMambaSettings
            case "crnn":
                return CRNNSettings
        return None

    @classmethod
    @override
    def from_flat_dict(cls, dic):
        dic = {key: item for key, item in dic.items()}  # make copy
        keys = dataclass_asdict(cls()).keys()
        class_attributes = [key for key in keys if key in dic]
        attributes = {
            key: dic[key] if dic[key] != "None" else None for key in class_attributes
        }
        return cls(**attributes)


@dataclass
class EvaluationSettings(SettingsBase):
    peak_mean_range: int = 2
    peak_max_range: int = 2
    onset_cooldown: int = 0.021
    detect_tolerance: float = 0.05
    ignore_beats: bool = True
    min_test_score: float = 0.79
    pr_points: int | None = 50


@dataclass
class ModelSettingsBase(SettingsBase):
    @abstractmethod
    def get_identifier(self):
        pass

    @classmethod
    @override
    def from_flat_dict(cls, dic):
        dic = {key: item for key, item in dic.items()}  # make copy
        keys = dataclass_asdict(cls()).keys()
        class_attributes = [key for key in keys if key in dic]
        if "activation" in class_attributes:
            name = (
                dic["activation"]
                if isinstance(dic["activation"], str)
                else dic["activation"].__class__.__name__
            )
            dic["activation"] = getattr(nn, name)()
        attributes = {
            key: dic[key] if dic[key] != "None" else None for key in class_attributes
        }
        return cls(**attributes)


@dataclass
class CNNSettings(ModelSettingsBase):
    num_channels: int = 32
    num_residual_blocks: int = 7
    num_feature_layers: int = 2
    channel_multiplication: int = 2
    cnn_dropout: float = 0.3
    dense_dropout: float = 0.5
    causal: bool = True
    flux: bool = True
    activation: nn.Module = nn.SELU()
    classifier_dim: int = 2**6
    down_sample_factor: Literal[2, 3, 4] = 2

    def get_identifier(self):
        return "cnn"


@dataclass
class CNNAttentionSettings(ModelSettingsBase):
    num_channels: int = 32
    cnn_dropout: float = 0.3
    attention_dropout: float = 0.5
    positional_encoding_dropout: float = 0.1
    causal: bool = True
    flux: bool = False
    activation: nn.Module = nn.SiLU()
    num_attention_blocks: int = 5
    num_heads: int = 4
    hidden_units: int = 64
    expansion_factor: int = 4
    use_relative_pos: bool = True
    down_sample_factor: int = 4
    num_conv_layers: int = 2
    channel_multiplication: int = 2

    def get_identifier(self):
        return "cnn_attention"


@dataclass
class CNNMambaSettings(ModelSettingsBase):
    num_channels: int = 32
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    cnn_dropout: float = 0.3
    mamba_dropout: float = 0.5
    dense_dropout: float = 0.5
    causal: bool = True
    flux: bool = False
    backbone: Literal["unet", "cnn"] = "cnn"
    activation: nn.Module = nn.SiLU()
    n_layers: int = 20
    down_sample_factor: int = 4
    num_conv_layers: int = 2
    channel_multiplication: int = 2
    classifier_dim: int = 128
    hidden_units: int = 128

    def get_identifier(self):
        return "mamba_fast"


@dataclass
class CRNNSettings(ModelSettingsBase):
    num_channels: int = 32
    num_conv_layers: int = 2
    channel_multiplication: int = 2
    down_sample_factor: int = 4
    num_rnn_layers: int = 3
    rnn_units: int = 256
    classifier_dim: int = 512
    cnn_dropout: float = 0.3
    rnn_dropout: float = 0.0
    dense_dropout: float = 0.5
    use_dense: bool = True
    causal: bool = True
    flux: bool = False
    activation: nn.Module = nn.SELU()

    def get_identifier(self):
        return "crnn"


@dataclass
class Config(SettingsBase):
    training: TrainingSettings = field(default_factory=TrainingSettings)
    dataset: DatasetSettings = field(default_factory=DatasetSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    model: ModelSettingsBase | None = None

    @classmethod
    @override
    def from_flat_dict(cls, dic):
        dic = {key: item for key, item in dic.items()}  # make copy
        keys = dataclass_asdict(cls()).keys()
        class_attributes = [key for key in keys if key in dic]
        class_attributes += ["dataset", "training", "model", "evaluation"]
        dic["dataset"] = DatasetSettings.from_flat_dict(dic)
        dic["training"] = TrainingSettings.from_flat_dict(dic)
        dic["evaluation"] = EvaluationSettings.from_flat_dict(dic)
        dic["model"] = (
            dic["training"].get_model_settings_class().from_flat_dict(dic)
            if dic["training"].get_model_settings_class() is not None
            else None
        )
        attributes = {
            key: dic[key] if dic[key] != "None" else None for key in class_attributes
        }
        return cls(**attributes)


def flatten_dict(_dict: dict) -> dict:
    out = {}
    inner_dicts = []
    for key, value in _dict.items():
        if isinstance(value, dict):
            inner_dicts.append(value)
        else:
            out[key] = value
    while len(inner_dicts) > 0:
        inner_dict = inner_dicts.pop()
        for key, value in inner_dict.items():
            if isinstance(value, dict):
                inner_dicts.append(value)
            else:
                out[key] = value
    return out