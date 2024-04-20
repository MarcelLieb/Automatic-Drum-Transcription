import os
from pathlib import Path

import numpy as np
import torch

import rustimport.import_hook
rustimport.settings.compile_release_binaries = True
cargo_path = os.path.join(str(Path.home()), ".cargo", "bin")
if cargo_path not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + cargo_path
from speedup import calculate_pr as rust_calculate_pr


def peak_pick_max_mean(data: torch.tensor, sample_rate: int, hop_size: int, mean_range: int = 7, max_range: int = 3):
    mean_filter = torch.nn.AvgPool1d(kernel_size=mean_range + 1, stride=1, padding=0)
    max_filter = torch.nn.MaxPool1d(kernel_size=max_range + 1, stride=1, padding=0)
    padded = torch.nn.functional.pad(data, (mean_range, 0), mode="reflect")
    mean = mean_filter(padded)
    difference = data - mean
    padded = torch.nn.functional.pad(mean, (max_range, 0), mode="reflect")
    maximum = max_filter(padded)
    assert maximum.shape == mean.shape and maximum.shape == data.shape

    time = torch.tensor(range(data.shape[-1])) * hop_size / sample_rate

    out = []
    # iterate over batch
    for i in range(data.shape[0]):
        out.append([])
        # iterate over classes
        for j in range(data.shape[1]):
            # assume positive thresholds only
            out[i].append(torch.stack((time, difference[i, j]))[:, difference[i, j] >= 0.0])
    return out


def calculate_pr(peaks: list[list[torch.Tensor]], groundtruth: list[list[torch.Tensor]]):
    songs = []
    for i in range(len(peaks)):
        songs.extend([
            torch.stack((
                *peaks[i][j],
                torch.zeros(peaks[i][j].shape[1]) + i,
                torch.zeros(peaks[i][j].shape[1]) + j
            ), dim=0)
            for j in range(len(peaks[i]))
        ])
    all_detections = torch.cat(songs, dim=1)
    all_detections = all_detections.T
    precision, recall, f_score, threshold = rust_calculate_pr(np.array(all_detections), groundtruth, 0.03)

    return torch.tensor(precision), torch.tensor(recall), f_score, threshold
