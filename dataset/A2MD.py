from pathlib import Path

import numpy as np
import pretty_midi
import torch
import torchaudio
from torch.utils.data import Dataset
import os

A2MD_PATH = "./data/a2md_public/"

# Mapping identical to the one used in "Towards multi-instrument drum transcription"
drum_midi_mapping = {
    "BD":  [35, 36],
    "SD":  [38, 40],
    "SS":  [37],
    "CLP": [39],
    "LT":  [41, 43],
    "MT":  [45, 47],
    "HT":  [48, 50],
    "CHH": [42],
    "PHH": [44],
    "OHH": [46],
    "TB":  [54],
    "RD":  [51, 59],
    "RB":  [53],
    "CB":  [56],
    "CRC": [49, 57],
    "SPC": [55],
    "CHC": [52],
    "CL":  [75],
}

drum_midi_mapping = {key: tuple(value) for key, value in drum_midi_mapping.items()}

three_class_mapping = (
    ("BD",),  # Bass Drum
    ("SD", "SS", "CLP"),  # Snare Drum + alike
    ("CHH", "PHH", "OHH")  # Hi-Hat
)

# Commonly used mapping
three_class_standard_mapping = (
    ("BD",),  # Bass Drum
    ("SD",),  # Snare Drum + alike
    ("CHH", "PHH", "OHH")  # Hi-Hat
)

four_class_mapping = (
    ("BD",),  # Bass Drum
    ("SD", "SS", "CLP"),  # Snare Drum + alike
    ("CHH", "PHH", "OHH"),  # Hi-Hat
    ("LT", "MT", "HT")  # Toms
)

# Mapping used in ADTOF
five_class_mapping = (
    ("BD",),  # Bass Drum
    ("SD", "SS", "CLP"),  # Snare Drum + alike
    ("CHH", "PHH", "OHH"),  # Hi-Hat
    ("LT", "MT", "HT"),  # Toms
    ("CRC", "SPC", "CHC", "RD")  # Cymbal + Ride
)

eight_class_mapping = (
    ("BD",),  # Bass Drum
    ("SD", "SS", "CLP"),  # Snare Drum + alike
    ("LT", "MT", "HT"),  # Toms
    ("CHH", "PHH", "OHH"),  # Hi-Hat
    ("CRC", "SPC", "CHC"),  # Cymbal
    ("RD",),  # Ride
    ("RB", "CB"),  # Bell
    ("CL",)  # Clave/Sticks
)

eighteen_class_mapping = tuple([(key, ) for key in drum_midi_mapping.keys()])


def get_midi_to_class(mapping: tuple[tuple[str, ...], ...]):
    reverse_map = np.zeros(128)
    reverse_map.fill(-1)
    for idx, drum_classes in enumerate(mapping):
        for drum_class in drum_classes:
            for pitch in drum_midi_mapping[drum_class]:
                reverse_map[pitch] = idx
    return reverse_map


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


def get_annotation(path: str, folder: str, identifier: str, mapping: tuple[tuple[str, ...], ...] = three_class_mapping):
    midi = pretty_midi.PrettyMIDI(midi_file=os.path.join(path, "align_mid", folder, f"align_mid_{identifier}.mid"))
    drums = get_drums(midi, mapping=mapping)
    if drums is None:
        return None
    beats = midi.get_beats()
    down_beats = midi.get_downbeats()
    return folder, identifier, drums, beats, down_beats


class A2MD(Dataset[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]]):
    def __init__(self, split: str, path: Path | str = A2MD_PATH,
                 pad_annotation: bool = True, pad_value: float = 0.5,
                 mapping: tuple[tuple[str, ...], ...] = three_class_mapping, time_shift=0.0,
                 sample_rate=44100, hop_size=512, fft_size=2048, n_mels=82, center=False,
                 pad_mode="constant", use_dataloader=False):
        self.path = path
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.fft_size = fft_size
        self.n_mels = n_mels
        self.center = center
        self.pad_mode = pad_mode
        self.split = split
        self.mapping = mapping
        self.use_dataloader = use_dataloader
        self.time_shift = time_shift
        self.n_classes = len(mapping) + 2
        self.pad = torch.nn.MaxPool1d(3, stride=1, padding=1) if pad_annotation else None
        self.pad_value = pad_value

        cut_off = {
            "L": 0.7,
            "M": 0.4,
            "S": 0.2,
        }

        folders = [f"dist0p{x:02}" for x in range(0, int(cut_off[split] * 100), 10)]

        args = []
        for folder in folders:
            for root, dirs, files in os.walk(os.path.join(path, "align_mid", folder)):
                for file in files:
                    identifier = "_".join(file.split(".")[0].split("_")[2:4])
                    args.append((path, folder, identifier, mapping))

        with torch.multiprocessing.Pool() as pool:
            self.annotations = pool.starmap(get_annotation, args)
            # filter tracks without drums
            self.annotations = [annotation for annotation in self.annotations if annotation is not None]

        self.spectrum = torchaudio.transforms.Spectrogram(n_fft=self.fft_size, hop_length=self.hop_size,
                                                          win_length=self.fft_size // 2, power=2, center=self.center,
                                                          pad_mode=self.pad_mode, normalized=False, onesided=True)
        self.filter_bank = torchaudio.transforms.MelScale(n_mels=self.n_mels, sample_rate=self.sample_rate,
                                                          f_min=0.0, f_max=self.sample_rate / 2,
                                                          n_stft=self.fft_size // 2 + 1)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        folder, identifier, drums, beats, down_beats = self.annotations[idx]
        audio_path = os.path.join(self.path, "ytd_audio", folder, f"ytd_audio_{identifier}.mp3")
        audio, sample_rate = torchaudio.load(audio_path, normalize=True, backend="ffmpeg")
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)(audio)
        audio = torch.mean(audio, dim=0, keepdim=False, dtype=torch.float32)
        audio = audio / torch.max(torch.abs(audio))

        frames = (audio.shape[0] - self.fft_size) // self.hop_size + 1

        labels = torch.zeros((self.n_classes, frames), dtype=torch.float32)

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
                labels[2 + i, shifted_drum_class[shifted_drum_class < frames]] = 1

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


if __name__ == '__main__':
    _dataset = A2MD("S", mapping=five_class_mapping, path="../data/a2md_public/")
    _test = _dataset[0]
    _audio, _labels = _test
