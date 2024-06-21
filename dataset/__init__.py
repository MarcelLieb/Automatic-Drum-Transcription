import random
from pathlib import Path
from typing import Sequence, TypeVar

import numpy as np
import pretty_midi
import torch
import torchaudio
from torch.utils.data import Subset
from torchaudio.transforms import Vol

from dataset.mapping import DrumMapping, get_midi_to_class


class Gain(torch.nn.Module):
    def __init__(self, min_gain: float = -20.0, max_gain: float = -1):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        gain = random.uniform(self.min_gain, self.max_gain)
        audio = Vol(gain, gain_type="db")(audio)
        return audio


def load_audio(path: str | Path, sample_rate: int, normalize: bool) -> torch.Tensor:
    audio, sr = torchaudio.load(path, normalize=True, backend="ffmpeg")
    audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(audio)
    audio = torch.mean(audio, dim=0, keepdim=False, dtype=torch.float32)
    if normalize:
        audio = audio / torch.max(torch.abs(audio))
    return audio


def get_length(path: str | Path) -> float:
    meta_data = torchaudio.info(path, backend="ffmpeg")
    return meta_data.num_frames / meta_data.sample_rate


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
    midi_to_class = get_midi_to_class(mapping.value)
    if len(notes) == 0:
        return None
    if len(notes.shape) == 1:
        notes = notes[np.newaxis, ...]
    notes = notes[midi_to_class[notes[:, 0].astype(int)] != -1]

    drum_classes = []
    for i in range(n_classes):
        drum_classes.append(notes[midi_to_class[notes[:, 0].astype(int)] == i, 1])

    return drum_classes


def get_label_windows(
    lengths: list[float],
    drum_labels: list[list[np.array]],
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
        start = ((labels * sample_rate) - (lead_in * sample_rate)).astype(int)
        start = np.clip(start, 0, length * sample_rate)
        end = (labels * sample_rate + lead_out * sample_rate).astype(int)
        end = np.clip(end, 0, length * sample_rate)
        out = np.stack((start, end, np.zeros_like(start) + i), axis=1).astype(int)
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
        for j in range(n_segments):
            start = int(j * (segment_length - overlap) * sample_rate)
            end = min(
                int(((j + 1) * (segment_length - overlap) + overlap) * sample_rate),
                int(length * sample_rate),
            )
            if end != int(length * sample_rate):
                assert (
                    abs((end - start) - int(segment_length * sample_rate)) <= 1
                ), f"Segment length not correct\n Expected:{int(segment_length * sample_rate)}, Got: {end - start}"
            if end - start > 0:
                segments.append((start, end, i))
    return np.array(segments) / sample_rate


def segment_audio(
    audio: torch.Tensor, start: int, end: int, length: int
) -> torch.Tensor:
    cut_audio = audio[..., start:end]
    if cut_audio.shape[-1] < length:
        if start == 0:
            cut_audio = torch.cat(
                (torch.zeros(int(length - cut_audio.shape[-1])), cut_audio)
            )
        else:
            cut_audio = torch.cat(
                (cut_audio, torch.zeros(int(length - cut_audio.shape[-1])))
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


def audio_collate(batch: list[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]]):
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
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return dataloader


T = TypeVar("T")


def get_splits(splits: list[float], data: list[T]) -> list[list[T]]:
    assert abs(sum(splits) - 1) < 1e-4
    generator = torch.Generator().manual_seed(42)
    split: list[Subset] = torch.utils.data.random_split(
        range(len(data)), splits, generator=generator
    )
    out = [[]] * len(split)
    for i, s in enumerate(split):
        out[i] = [data[j] for j in s.indices]

    return out
