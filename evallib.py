import os
from pathlib import Path
from typing import Literal

import numpy as np
import rustimport.import_hook
import torch

from dataset import get_time_index

rustimport.settings.compile_release_binaries = True
cargo_path = os.path.join(str(Path.home()), ".cargo", "bin")
if cargo_path not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + cargo_path
from speedup import calculate_pr as rust_calculate_pr, combine_onsets as rust_combine_onsets, \
    evaluate_detections as rust_evaluate_onsets, evaluate_detection_stats as rust_evaluate_onset_stats


def peak_pick_max_mean(
    data: torch.Tensor,
    sample_rate: int,
    hop_size: int,
    fft_size: int,
    mean_range: int = 2,
    max_range: int = 2,
):
    mean_filter = torch.nn.AvgPool1d(kernel_size=mean_range + 1, stride=1, padding=0)
    padded = torch.nn.functional.pad(data, (mean_range, 0), mode="constant", value=0.0)
    mean: torch.Tensor = mean_filter(padded)
    difference: torch.Tensor = data - mean
    max_filter = torch.nn.MaxPool1d(kernel_size=max_range + 1, stride=1, padding=0)
    padded = torch.nn.functional.pad(data, (max_range, 0), mode="constant", value=0.0)
    maximum: torch.Tensor = max_filter(padded)
    assert maximum.shape == mean.shape and maximum.shape == data.shape

    time = torch.tensor(get_time_index(data.shape[-1], sample_rate, hop_size)).to(
        data.device
    )

    out = []
    # iterate over batch
    for i in range(data.shape[0]):
        out.append([])
        # iterate over classes
        for j in range(data.shape[1]):
            # assume positive thresholds only
            out[i].append(
                torch.stack((time, difference[i, j]))[
                :,
                torch.logical_and(
                    torch.logical_and(
                        difference[i, j] >= 0.0, data[i, j] >= maximum[i, j]
                    ),
                    data[i, j] >= 0.0,
                ),
                ]
            )
    return out


def calculate_pr(
    peaks: list[list[torch.Tensor]],
    groundtruth: list[list[torch.Tensor]],
    ignore_beats: bool = False,
    detection_window: float = 0.05,
    onset_cooldown: float = 0.02,
    pr_points: int | None = 1000,
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
        classes, gt, detection_window, onset_cooldown, pr_points
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


def combine_onsets(onsets: torch.Tensor, cool_down: float, strategy: Literal["min", "max", "avg"] = "min"):
    onsets = np.array(onsets)
    out = rust_combine_onsets(onsets, cool_down, strategy)
    return torch.tensor(out)


def evaluate_onsets(onsets: torch.Tensor, groundtruth: torch.Tensor, window: float):
    onsets = np.array(onsets)
    onsets.sort(kind="stable")
    groundtruth = np.array(groundtruth)
    groundtruth.sort(kind="stable")
    tp, fp, fn = rust_evaluate_onsets(onsets, groundtruth, window)
    return tp, fp, fn


def evaluate_onset_stats(onsets: torch.Tensor, groundtruth: torch.Tensor, window: float):
    onsets = np.array(onsets)
    onsets.sort(kind="stable")
    groundtruth = np.array(groundtruth)
    groundtruth.sort(kind="stable")
    tps, fps, fns = rust_evaluate_onset_stats(onsets, groundtruth, window)
    return tps, fps, fns

def calculate_f_score(precision: torch.Tensor, recall: torch.Tensor) -> torch.Tensor:
    f_score = 2 * (precision * recall) / (precision + recall)
    f_score[precision + recall == 0] = 0
    return f_score
