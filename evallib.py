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


def peak_pick_max_mean(
    data: torch.tensor,
    sample_rate: int,
    hop_size: int,
    mean_range: int = 7,
    max_range: int = 3,
):
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
            out[i].append(
                torch.stack((time, difference[i, j]))[:, difference[i, j] >= 0.0]
            )
    return out


def calculate_pr(
    peaks: list[list[torch.Tensor]],
    groundtruth: list[list[torch.Tensor]],
    ignore_beats: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor], float, float, torch.Tensor]:
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
    prs, thresholds, f_score, f_score_avg = rust_calculate_pr(classes, gt, 0.025, 0.020)
    precisions = [torch.tensor(prs[i][0]) for i in range(len(prs))]
    recalls = [torch.tensor(prs[i][1]) for i in range(len(prs))]

    return (
        precisions,
        recalls,
        f_score,
        f_score_avg,
        torch.tensor(thresholds),
    )
