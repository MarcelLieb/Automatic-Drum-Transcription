import random
from pathlib import Path
from typing import Sequence, TypeVar

import librosa
import numpy as np
import pretty_midi
import sox
import torch
import torchaudio
from torch.utils.data import Subset
from torchaudio.transforms import Vol

from dataset.mapping import DrumMapping


class Gain(torch.nn.Module):
    def __init__(self, min_gain: float = -20.0, max_gain: float = -1):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        gain = random.uniform(self.min_gain, self.max_gain)
        audio = Vol(gain, gain_type="db")(audio)
        return audio


def load_audio(
    path: str | Path,
    sample_rate: int,
    normalize: bool,
    start: float = 0,
    end: float = -1,
    backend: str = "librosa",
) -> torch.Tensor:
    if (start, end) == (0, -1):
        audio, sr = torchaudio.load(path, normalize=True, backend="ffmpeg")
        audio = torch.mean(audio, dim=0, keepdim=False, dtype=torch.float32)
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(
            audio
        )
    else:
        if backend == "librosa":
            audio = librosa.load(path=path, sr=sample_rate, mono=True, offset=start, duration=end - start)[0]
            audio = torch.from_numpy(audio)
        elif backend == "sox":
            final_duration = end - start
            if start <= 0.28:
                offset = 2257
                end = end + (offset / sample_rate) + 0.1
            else:
                offset = 1105
                end = end + (offset / sample_rate) + 0.1
            tfm = sox.Transformer()
            tfm.trim(start, end)
            tfm.fade(fade_in_len=0, fade_out_len=0)
            tfm.set_output_format(rate=sample_rate, channels=1)
            audio = tfm.build_array(input_filepath=path)
            audio = torch.from_numpy(audio.copy())
            audio = audio / torch.iinfo(audio.dtype).max
            audio = audio[offset:offset + int(final_duration * sample_rate)]
        else:
            audio, og_sr = torchaudio.load(path, normalize=True, backend="ffmpeg",
                                           num_frames=int(sample_rate * (end - start)), offset=int(sample_rate * start))
            audio = torch.mean(audio, dim=0, keepdim=False, dtype=torch.float32)
            audio = torchaudio.transforms.Resample(orig_freq=og_sr, new_freq=sample_rate)(audio)
        assert audio.shape[-1] == int(
            sample_rate * (end - start)), f"{audio.shape[-1]} != {int(sample_rate * (end - start))}"
    if normalize:
        audio = audio / torch.max(torch.abs(audio))
    return audio


def get_length(path: str | Path) -> float:
    path = str(path)
    assert path.endswith(".wav") or path.endswith(".mp3") or path.endswith(".flac"), f"{path} is not a valid audio file"
    return librosa.get_duration(path=path)


def get_drums(midi: pretty_midi.PrettyMIDI, mapping: DrumMapping):
    drum_instruments = [
        instrument for instrument in midi.instruments if instrument.is_drum
    ]
    notes = np.array(
        [
            (note.pitch, note.start)
            for instrument in drum_instruments
            for note in instrument.notes
        ]
    )
    notes = np.array(sorted(notes, key=lambda x: x[1]))
    n_classes = len(mapping)
    midi_to_class = mapping.get_midi_to_class()
    if len(notes) == 0:
        return None
    if len(notes.shape) == 1:
        notes = notes[np.newaxis, ...]
    notes = notes[midi_to_class[notes[:, 0].astype(int)] != -1]

    drum_classes = []
    for i in range(n_classes):
        drum_classes.append(notes[midi_to_class[notes[:, 0].astype(int)] == i, 1])
        # two different instruments playing at the same time may get the same class
        drum_classes[i] = np.unique(drum_classes[i])

    return drum_classes


def get_label_windows(
    lengths: list[float],
    drum_labels: list[list[np.ndarray]],
    lead_in: float,
    lead_out: float,
    sample_rate: int,
    unique: bool = False,
) -> np.array:
    """
    :param lengths: List of lengths of the audio files in seconds
    :param drum_labels: List of drum labels
    :param lead_in: Length of the lead-in in seconds
    :param lead_out: Length of the lead-out in seconds
    :param sample_rate: Sample rate of the audio files
    :param unique: If True, only unique segments are returned
    :return: List of tuples containing the start and end indices of the segments and the index of the audio file
    """
    segments = []
    for i, (length, drum_label) in enumerate(zip(lengths, drum_labels)):
        labels = np.concatenate(drum_label)
        if unique:
            labels = np.unique(labels)
        start = (labels - lead_in)
        start = np.clip(start, 0, length)
        end = labels + lead_out - 1 / sample_rate
        end = np.clip(end, 0, length)
        out = np.stack((start, end, np.zeros_like(start) + i), axis=1)
        segments.append(out)
    return np.concatenate(segments, axis=0)


def get_segments(
    lengths: list[float], segment_length: float, overlap: float, sample_rate: int
) -> np.array:
    """
    Computes overlapping segments for a list of audio files
    :param lengths: List of lengths of the audio files in seconds
    :param segment_length: Length of the segments in seconds
    :param overlap: Overlap between segments in seconds
    :param sample_rate: Sample rate of the audio files
    :return: List of tuples containing the start and end indices of the segments and the index of the audio file
    """
    segments = []
    for i, length in enumerate(lengths):
        n_segments = (
            int(np.ceil((length - segment_length) / (segment_length - overlap))) + 1
        )
        starts = np.arange(0, n_segments) * (segment_length - overlap)
        ends = np.minimum(starts + segment_length, length)
        segment = np.stack((starts, ends, np.zeros_like(starts) + i), axis=1)
        segments.append(segment)
    segments = np.concatenate(segments, axis=0)
    return segments


def segment_audio(
    audio: torch.Tensor, start: int, end: int, length: int, pad: bool = True
) -> torch.Tensor:
    cut_audio = audio[..., start:end]
    if pad:
        if start == 0:
            cut_audio = pad_audio(cut_audio, length, front=True)
        else:
            cut_audio = pad_audio(cut_audio, length, front=False)
    return cut_audio


def pad_audio(audio: torch.Tensor, length: int, front: bool) -> torch.Tensor:
    if audio.shape[-1] >= length:
        return audio
    if front:
        cut_audio = torch.cat(
            (torch.zeros(int(length - audio.shape[-1])), audio)
        )
    else:
        cut_audio = torch.cat(
            (audio, torch.zeros(int(length - audio.shape[-1])))
        )
    assert cut_audio.shape[-1] == length, "Audio too short"
    return cut_audio


def get_labels(
    onset_times: Sequence[np.ndarray], sample_rate: float, hop_size: int, length: int
) -> torch.Tensor:
    labels = torch.zeros(len(onset_times), length)
    for i, cls in enumerate(onset_times):
        indices = get_indices(cls, sample_rate, hop_size)
        indices = indices[indices < length]
        labels[i, indices] = 1

    return labels


def get_indices(time_stamps: np.array, sample_rate: float, hop_size: int) -> np.array:
    return np.round((time_stamps * sample_rate) / hop_size).astype(int)


def get_time_index(length: int, sample_rate: float, hop_size: int) -> np.array:
    return (np.arange(length) * hop_size) / sample_rate


def audio_collate(batch: list[tuple[torch.Tensor, torch.Tensor, list[np.ndarray]]]):
    audio, annotation, gts = zip(*batch)
    audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=0.0)
    annotation = list(annotation)
    annotation = torch.nn.utils.rnn.pad_sequence(
        annotation, batch_first=True, padding_value=-1
    )
    audio = audio.permute(0, 2, 1)
    annotation = annotation.permute(0, 2, 1)
    return audio, annotation, gts


def get_dataloader(dataset, batch_size, num_workers, is_train=False):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        collate_fn=audio_collate,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=6 if num_workers > 0 else None,
    )
    return dataloader


def convert_to_wav(path: str | Path):
    path = Path(path)
    if path.suffix == ".wav":
        return
    audio, sr = torchaudio.load(path, normalize=True, backend="ffmpeg", channels_first=True)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(path.with_suffix(".wav"), audio, sr, channels_first=True, bits_per_sample=16, format="wav", encoding="PCM_S")


def convert_to_flac(path: str | Path):
    path = Path(path)
    if path.suffix == ".flac":
        return
    audio, sr = torchaudio.load(path, normalize=True, backend="ffmpeg", channels_first=True)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(path.with_suffix(".flac"), audio, sr, channels_first=True, bits_per_sample=16, format="flac", encoding="PCM_S")


T = TypeVar("T")


def get_splits(splits: list[float], data: list[T], seed=42) -> list[list[T]]:
    assert abs(sum(splits) - 1) < 1e-4
    generator = torch.Generator().manual_seed(seed)
    split: list[Subset] = torch.utils.data.random_split(
        range(len(data)), splits, generator=generator
    )
    out = [[]] * len(split)
    for i, s in enumerate(split):
        out[i] = [data[j] for j in s.indices]

    assert sum(len(s) for s in out) == len(data)
    for prob, s in zip(splits, out):
        assert abs(len(s) / len(data) - prob) < 0.02, f"{len(s) / len(data)} != {prob}"

    return out
