from pathlib import Path

import numpy as np
import pretty_midi
import torch
import torchaudio
import os

from dataset.generics import ADTDataset
from dataset.mapping import get_midi_to_class, three_class_mapping, DrumMapping

A2MD_PATH = "./data/a2md_public/"


def get_drums(midi: pretty_midi.PrettyMIDI, mapping: tuple[tuple[str, ...], ...] = three_class_mapping):
    drum_instruments = [instrument for instrument in midi.instruments if instrument.is_drum]
    notes = np.array([(note.pitch, note.start) for instrument in drum_instruments for note in instrument.notes])
    notes = np.array(sorted(notes, key=lambda x: x[1]))
    n_classes = len(mapping)
    midi_to_class = get_midi_to_class(mapping)
    if len(notes) == 0:
        return None
    if len(notes.shape) == 1:
        notes = notes[np.newaxis, ...]
    notes = notes[midi_to_class[notes[:, 0].astype(int)] != -1]

    drum_classes = []
    for i in range(n_classes):
        drum_classes.append(notes[midi_to_class[notes[:, 0].astype(int)] == i, 1])

    return drum_classes


def get_annotation(path: str, folder: str, identifier: str, mapping: DrumMapping = DrumMapping.THREE_CLASS):
    midi = pretty_midi.PrettyMIDI(midi_file=os.path.join(path, "align_mid", folder, f"align_mid_{identifier}.mid"))
    drums = get_drums(midi, mapping=mapping.value)
    if drums is None:
        return None
    beats = midi.get_beats()
    down_beats = midi.get_downbeats()
    return folder, identifier, drums, beats, down_beats


def get_length(path: str, folder: str, identifier: str):
    audio_path = os.path.join(path, "ytd_audio", folder, f"ytd_audio_{identifier}.mp3")
    meta_data = torchaudio.info(audio_path, backend="ffmpeg")
    return meta_data.num_frames / meta_data.sample_rate


def calculate_segments(lengths: list[float], segment_length: float, sample_rate: int, fft_size: int) \
        -> list[tuple[int, int, int]]:
    """
    :param lengths: List of lengths of the audio files
    :param segment_length: Length of the segments in seconds
    :param sample_rate: Sample rate of the audio files
    :param fft_size: Size of the fft window
    :return: List of tuples containing the start and end indices of the segments and the index of the audio file
    """
    segments = []
    for i, length in enumerate(lengths):
        n_segments = int(np.ceil(length / segment_length))
        for j in range(n_segments):
            start = int(j * segment_length * sample_rate)
            end = min(int((j + 1) * segment_length * sample_rate), int(length * sample_rate))
            if end - start > fft_size:
                segments.append((start, end, i))
    return segments


def get_segments(lengths: list[float], drum_labels: list[list[np.array]], lead_in: float, sample_rate: int) -> np.array:
    """
    :param lengths: List of lengths of the audio files
    :param drum_labels: List of drum labels
    :param lead_in: Length of the lead-in in seconds
    :param sample_rate: Sample rate of the audio files
    :return: List of tuples containing the start and end indices of the segments and the index of the audio file
    """
    segments = []
    for i, (length, drum_label) in enumerate(zip(lengths, drum_labels)):
        labels = np.concatenate(drum_label)
        # labels = np.unique(labels)
        start = ((labels * sample_rate) - (lead_in * sample_rate)).astype(int)
        start = np.clip(start, 0, length * sample_rate)
        end = (labels * sample_rate + 0.125 * sample_rate).astype(int)
        end = np.clip(end, 0, length * sample_rate)
        out = np.stack((start, end, np.zeros_like(start) + i), axis=1).astype(int)
        segments.append(out)
    return np.concatenate(segments, axis=0)


def load_audio(path: str, folder: str, identifier: str, sample_rate: int) -> torch.Tensor:
    audio_path = os.path.join(path, "ytd_audio", folder, f"ytd_audio_{identifier}.mp3")
    audio, sr = torchaudio.load(audio_path, normalize=True, backend="ffmpeg")
    audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(audio)
    audio = torch.mean(audio, dim=0, keepdim=False, dtype=torch.float32)
    audio = audio / torch.max(torch.abs(audio))
    return audio


def get_tracks(path: str) -> dict[str, list[str]]:
    folders = [f"dist0p{x:02}" for x in range(0, 70, 10)]
    out = {}
    for folder in folders:
        out[folder] = []
        for root, dirs, files in os.walk(os.path.join(path, "align_mid", folder)):
            for file in files:
                identifier = "_".join(file.split(".")[0].split("_")[2:4])
                out[folder].append(identifier)
    return out


class A2MD(ADTDataset):
    def __init__(
        self,
        split: dict[str, list[str]],
        audio_settings: AudioProcessingSettings,
        annotation_settings: AnnotationSettings,
        path: Path | str = A2MD_PATH,
        is_train: bool = False,
        use_dataloader=False,
        **_kwargs,
    ):
        super().__init__(audio_settings, annotation_settings)
        self.path = path
        self.split = split
        self.use_dataloader = use_dataloader
        self.is_train = is_train
        self.pad = (
            torch.nn.MaxPool1d(3, stride=1, padding=1)
            if annotation_settings.pad_annotations
            else None
        )

        args = []
        for i, (folder, identifiers) in enumerate(split.items()):
            for identifier in identifiers:
                args.append((path, folder, identifier, self.mapping))
        torch.multiprocessing.set_sharing_strategy("file_system")
        with torch.multiprocessing.Pool() as pool:
            self.annotations = pool.starmap(get_annotation, args)
            # filter tracks without drums
            self.annotations = [
                annotation for annotation in self.annotations if annotation is not None
            ]
            args = [
                (path, folder, identifier)
                for folder, identifier, *_ in self.annotations
            ]
            self.lengths = pool.starmap(get_length, args) if is_train else None
            self.segments = (
                get_segments(
                    self.lengths,
                    [drums for _, _, drums, *_ in self.annotations],
                    annotation_settings.lead_in,
                    audio_settings.sample_rate,
                )
                if is_train
                else None
            )
            # self.segments = calculate_segments(self.lengths, 5.0, sample_rate, fft_size) if is_train else None
            args = [
                (path, folder, identifier, audio_settings.sample_rate)
                for folder, identifier, *_ in self.annotations
            ]
            self.cache = pool.starmap(load_audio, args) if is_train else None

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

    def __len__(self):
        return len(self.segments) if self.is_train else len(self.annotations)

    def __getitem__(self, idx):
        if self.is_train:
            start, end, audio_idx = self.segments[idx]
        else:
            audio_idx = idx
            start, end = 0, -1

        folder, identifier, drums, beats, down_beats = self.annotations[int(audio_idx)]

        audio = load_audio(self.path, folder, identifier, self.sample_rate) if self.cache is None else self.cache[
            int(audio_idx)]
        audio = audio[start:end]

        time_offset = start / self.sample_rate

        drums = [drum[drum >= time_offset] - time_offset for drum in drums]
        beats = beats[beats >= time_offset] - time_offset
        down_beats = down_beats[down_beats >= time_offset] - time_offset

        frames = (audio.shape[0] - self.fft_size) // self.hop_size + 1

        labels = torch.zeros((self.n_classes, frames), dtype=torch.float32)

        if self.beats:
            beat_indices = (beats * self.sample_rate) // self.hop_size
            beat_indices = torch.tensor(beat_indices, dtype=torch.long)
            beat_indices = beat_indices[beat_indices < frames]
            down_beat_indices = (down_beats * self.sample_rate) // self.hop_size
            down_beat_indices = torch.tensor(down_beat_indices, dtype=torch.long)
            down_beat_indices = down_beat_indices[down_beat_indices < frames]
            labels[0, down_beat_indices] = 1
            labels[1, beat_indices] = 1

        hop_length = self.hop_size / self.sample_rate

        drum_indices = [(drum * self.sample_rate) // self.hop_size for drum in drums]
        drum_indices = [drum[drum < frames] for drum in drum_indices]
        for i, drum_class in enumerate(drum_indices):
            for j in range(round(self.time_shift // hop_length) + 1):
                shifted_drum_class = drum_class + j
                labels[int(self.beats) * 2 + i, shifted_drum_class[shifted_drum_class < frames]] = 1

        if self.pad is not None:
            padded = self.pad(labels.unsqueeze(0)).squeeze(0) * self.pad_value
            labels = torch.maximum(labels, padded)
        spectrum = self.spectrum(audio)
        mel = self.filter_bank(spectrum)
        mel = torch.log1p(mel)

        gt_labels = [down_beats, beats, *drums]

        if self.use_dataloader:
            # allows use of torch.nn.utils.rnn.pad_sequence
            # it expects the variable length dimension to be the first dimension
            return mel.permute(1, 0), labels.permute(1, 0), gt_labels

        return mel, labels, gt_labels

    def adjust_time_shift(self, time_shift: float):
        self.time_shift = time_shift
