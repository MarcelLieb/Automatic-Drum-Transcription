import os

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

rbma_13_path = "./data/rbma_13/"


def load_rbma(rbma_path=rbma_13_path):
    annotations = {}
    for root, dirs, files in os.walk(os.path.join(rbma_path, "annotations", "drums")):
        for file in files:
            track = file.split(".")[0]
            data = np.loadtxt(os.path.join(root, file), delimiter="\t")
            annotations[track] = data
    return annotations


def rename_rbma_audio_files():
    # load title list
    with open(os.path.join(rbma_13_path, "titles.txt")) as f:
        title_list = f.readlines()

    title_list = [x.strip() for x in title_list]

    positions = [[] for _ in range(30)]
    from fuzzywuzzy import process
    for root, dirs, files in os.walk(os.path.join(rbma_13_path, "audio")):
        for file in files:
            track = " ".join(file.split(".")[0].split(" ")[2:-1])
            title = process.extractOne(track, title_list)[0]
            # find position of track in title list
            position = title_list.index(title) + 1
            positions[position - 1].append(file)

    for i, files in enumerate(positions):
        if len(files) == 1:
            os.rename(os.path.join(rbma_13_path, "audio", files[0]), os.path.join(rbma_13_path, "audio", f"RBMA-13-Track-{i+1:02}.mp3"))

    return positions


class RBMA_13(Dataset):
    def __init__(self, root, split, sample_rate=48000, hop_size=480, fft_size=2048):
        self.root = root

        annotations = load_rbma()
        self.annotations = {}
        split = split.lower()
        assert split in ["train", "validation", "test"]
        index = ["train", "validation", "test"].index(split)
        with open(os.path.join(root, "splits", f"3-fold_cv_{index}.txt")) as f:
            tracks = f.readlines()
        tracks = [x.strip() for x in tracks]
        for track in tracks:
            self.annotations[track] = annotations[track]

        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.fft_size = fft_size

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        track = list(self.annotations.keys())[idx]

        annotation = self.annotations[track]
        audio_path = os.path.join(self.root, "audio", f"{track}.mp3")

        audio, sample_rate = torchaudio.load(audio_path, normalize=True)
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=48000)(audio)
        audio = torch.mean(audio, dim=0, keepdim=False, dtype=torch.float32)

        frames = (audio.shape[0] - self.fft_size) // self.hop_size + 1
        labels = np.zeros(frames, dtype=np.int32)
        indices = (annotation[:, 0] * self.sample_rate) // self.hop_size
        indices = indices.astype(np.int32)
        labels[indices] = annotation[:, 1]
        labels = torch.tensor(labels, dtype=torch.int32)

        return audio, labels


def get_dataloader(root, split, batch_size, num_workers, sample_rate=48000, hop_size=480, fft_size=2048):
    dataset = RBMA_13(root, split, sample_rate, hop_size, fft_size)
    is_train = split.lower() == "train"
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader

def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    audio, annotation = zip(*batch)
    audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=-1)
    annotation = torch.nn.utils.rnn.pad_sequence(annotation, batch_first=True, padding_value=-1)
    return audio, annotation


if __name__ == '__main__':
    dataset = RBMA_13(rbma_13_path, "train")
    _audio, _annotation = dataset[0]
