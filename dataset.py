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
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.annotations = load_rbma()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        track_number = str(idx + 1)
        annotation = self.annotations[f"RBMA-13-Track-{track_number:02}"]
        audio_path = os.path.join(self.root, "audio", f"RBMA-13-Track-{track_number:02}.mp3")
        audio = torchaudio.load(audio_path)

        return audio, annotation


if __name__ == '__main__':
    dataset = RBMA_13(rbma_13_path)
    _audio, _annotation = dataset[0]
