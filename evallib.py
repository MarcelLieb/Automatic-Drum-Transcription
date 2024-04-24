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


def calculate_pr(peaks: list[list[torch.Tensor]], groundtruth: list[list[torch.Tensor]], ignore_beats: bool = False):
    classes = []
    gt = []

    for _ in range(len(peaks[0]) - 2 * int(ignore_beats)):
        classes.append([])
        gt.append([])
    for i in range(len(peaks)):
        for j in range(2 * int(ignore_beats), len(peaks[i])):
            classes[j - 2 * int(ignore_beats)].append(torch.stack((*peaks[i][j], torch.zeros(peaks[i][j].shape[1]) + i)))
    for i in range(len(classes)):
        classes[i] = np.array(torch.cat(classes[i], dim=1).T)
    if ignore_beats:
        groundtruth = [gt[2:] for gt in groundtruth]

    for i in range(len(groundtruth)):
        for j in range(len(groundtruth[i])):
            gt[j].append(groundtruth[i][j])
    prs, thresholds, f_score = rust_calculate_pr(classes, gt, 0.025)

    return torch.tensor(prs[0][0]), torch.tensor(prs[0][1]), f_score, torch.tensor(thresholds)
