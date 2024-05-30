import os
from pathlib import Path

import numpy as np
import torch

import rustimport.import_hook

from dataset import get_time_index

rustimport.settings.compile_release_binaries = True
cargo_path = os.path.join(str(Path.home()), ".cargo", "bin")
if cargo_path not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + cargo_path
from speedup import calculate_pr as rust_calculate_pr


def peak_pick_max_mean(
    data: torch.tensor,
    sample_rate: int,
    hop_size: int,
    fft_size: int,
    mean_range: int = 2,
    max_range: int = 2,
):
    mean_filter = torch.nn.AvgPool1d(kernel_size=mean_range + 1, stride=1, padding=0)
    max_filter = torch.nn.MaxPool1d(kernel_size=max_range + 1, stride=1, padding=0)
    padded = torch.nn.functional.pad(data, (mean_range, 0), mode="reflect")
    mean = mean_filter(padded)
    difference = data - mean
    padded = torch.nn.functional.pad(mean, (max_range, 0), mode="reflect")
    maximum = max_filter(padded)
    assert maximum.shape == mean.shape and maximum.shape == data.shape

    time = torch.tensor(get_time_index(data.shape[-1], sample_rate, hop_size))

    out = []
    # iterate over batch
    for i in range(data.shape[0]):
        out.append([])
        # iterate over classes
        for j in range(data.shape[1]):
            # assume positive thresholds only
            out[i].append(
                torch.stack((time, difference[i, j]))[:, difference[i, j] >= 0.0]
            )
    return out


def calculate_pr(
    peaks: list[list[torch.Tensor]],
    groundtruth: list[list[torch.Tensor]],
    ignore_beats: bool = False,
    detection_window: float = 0.05,
    onset_cooldown: float = 0.02,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    float,
    float,
    torch.Tensor,
]:
    classes = [[] for _ in peaks[0]]
    gt = [[] for _ in groundtruth[0]]

    for i in range(len(peaks)):
        for j in range(len(peaks[i])):
            classes[j].append(
                torch.stack((*peaks[i][j], torch.zeros(peaks[i][j].shape[1]) + i))
            )
    for i in range(len(classes)):
        classes[i] = np.array(torch.cat(classes[i], dim=1).T)

    for i in range(len(groundtruth)):
        for j in range(len(groundtruth[i])):
            gt[j].append(groundtruth[i][j])
    # beats get ignored automatically if peaks has fewer classes than groundtruth
    # this is due to the model not predicting beats
    if ignore_beats and len(gt) == len(classes):
        gt = gt[2:]
        classes = classes[2:]
    prts, best_thresholds, f_score, f_score_avg = rust_calculate_pr(
        classes, gt, detection_window, onset_cooldown, 1000
    )
    precisions = [torch.tensor(prts[i][0]) for i in range(len(prts))]
    recalls = [torch.tensor(prts[i][1]) for i in range(len(prts))]
    thresholds = [torch.tensor(prts[i][2]) for i in range(len(prts))]

    return (
        precisions,
        recalls,
        thresholds,
        f_score,
        f_score_avg,
        torch.tensor(best_thresholds),
    )


def calculate_f_score(precision: torch.Tensor, recall: torch.Tensor) -> torch.Tensor:
    return 2 * (precision * recall) / (precision + recall)
