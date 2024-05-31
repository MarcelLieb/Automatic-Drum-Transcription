import os
from pathlib import Path

import numpy as np
import pretty_midi
import torch
import torchaudio

from dataset import load_audio, get_segments, get_drums
from dataset.mapping import DrumMapping
from generics import ADTDataset
from settings import AudioProcessingSettings, AnnotationSettings

A2MD_PATH = "./data/a2md_public/"


def get_annotation(
    path: str,
    folder: str,
    identifier: str,
    mapping: DrumMapping = DrumMapping.THREE_CLASS,
):
    midi = pretty_midi.PrettyMIDI(
        midi_file=os.path.join(path, "align_mid", folder, f"align_mid_{identifier}.mid")
    )
    drums = get_drums(midi, mapping=mapping)
    if drums is None:
        return None
    beats = midi.get_beats()
    down_beats = midi.get_downbeats()
    return (folder, identifier), drums, [down_beats, beats]


def get_length(path: str, folder: str, identifier: str):
    audio_path = os.path.join(path, "ytd_audio", folder, f"ytd_audio_{identifier}.mp3")
    meta_data = torchaudio.info(audio_path, backend="ffmpeg")
    return meta_data.num_frames / meta_data.sample_rate


def calculate_segments(
    lengths: list[float], segment_length: float, sample_rate: int, fft_size: int
) -> list[tuple[int, int, int]]:
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
            end = min(
                int((j + 1) * segment_length * sample_rate), int(length * sample_rate)
            )
            if end - start > fft_size:
                segments.append((start, end, i))
    return segments


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
        super().__init__(audio_settings, annotation_settings, is_train=is_train, use_dataloader=use_dataloader)
        self.path = path
        self.split = split

        args = []
        for i, (folder, identifiers) in enumerate(split.items()):
            for identifier in identifiers:
                args.append((path, folder, identifier, self.mapping))
        with torch.multiprocessing.Pool(torch.multiprocessing.cpu_count()) as pool:
            self.annotations = pool.starmap(get_annotation, args)
            # filter tracks without drums
            self.annotations = [
                annotation for annotation in self.annotations if annotation is not None
            ]
            self.annotations.sort(key=lambda x: int(x[0][1].split("_")[-2]))
            args = [
                (path, folder, identifier)
                for (folder, identifier), *_ in self.annotations
            ]
            lengths = pool.starmap(get_length, args) if is_train else None
            self.segments = (
                get_segments(
                    lengths,
                    [drums for _, drums, *_ in self.annotations],
                    annotation_settings.lead_in,
                    annotation_settings.lead_out,
                    audio_settings.sample_rate,
                )
                if is_train
                else None
            )
            args = [
                (self.path, identification)
                for identification, *_ in self.annotations
            ]
            # use static method to avoid passing self to pool
            paths = pool.starmap(A2MD._get_full_path, args)
            # self.segments = calculate_segments(self.lengths, 5.0, sample_rate, fft_size) if is_train else None
            args = [
                (path, audio_settings.sample_rate, self.normalize)
                for path in paths
            ]
            self.cache = pool.starmap(load_audio, args) if is_train else None

    def __len__(self):
        return len(self.segments) if self.is_train else len(self.annotations)


    @staticmethod
    def _get_full_path(root: str, identification: tuple[str, str]) -> Path:
        folder, identifier = identification
        audio_path = os.path.join(root, "ytd_audio", folder, f"ytd_audio_{identifier}.mp3")
        return Path(audio_path)

    def get_full_path(
            self, identification: tuple[str, str]
    ) -> Path:
        return self._get_full_path(self.path, identification)
