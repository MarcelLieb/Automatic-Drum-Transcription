import numpy as np
import torch
import os

from dataset import DrumMapping, get_time_index
from dataset.A2MD import get_annotation
from evallib import calculate_pr, combine_onsets, calculate_f_score, peak_pick_max_mean, evaluate_onsets
from madmom.evaluation.onsets import OnsetEvaluation


def peak_pick_max_mean_ref(raw_values: torch.Tensor, sample_rate, hop_size, mean_range=2, max_range=2):
    assert len(raw_values.shape) == 1
    time = torch.tensor(get_time_index(raw_values.shape[-1], sample_rate, hop_size))
    out = []
    for i in range(raw_values.shape[0]):
        mean = raw_values[max(i - mean_range, 0):i + 1].sum() / (mean_range + 1)
        difference = raw_values[i] - mean
        maximum = raw_values[max(i - max_range, 0):i + 1].max()
        if difference >= 0 and raw_values[i] >= maximum and raw_values[i] >= 0:
            out.append((time[i].item(), difference))
    return torch.tensor(out).movedim(0, 1)


def test_mean_max(mean_range=2, max_range=2):
    sample_rate = 44100
    hop_size = 441

    test_data = torch.load("../predictions/predictions_Validation_4.pt")
    raw_prediction = test_data["raw_predictions"]
    batched_pred = [tensor.unsqueeze(0) for tensor in raw_prediction]

    for i, (raw, batch_pred) in enumerate(zip(raw_prediction, batched_pred)):
        ref = [peak_pick_max_mean_ref(raw_val, sample_rate, hop_size, mean_range, max_range) for raw_val in raw]
        out = peak_pick_max_mean(batch_pred, sample_rate, hop_size, mean_range, max_range)[0]
        assert len(ref) == len(out), f"Length mismatch {len(ref)} != {len(out)}"
        for r, o in zip(ref, out):
            o = o.numpy()
            r = np.array(r)
            assert r.shape[-1] == o.shape[-1], f"Different number of onsets {r.shape[-1]} != {o.shape[-1]}"
            assert np.all(np.isclose(r, o,
                                     atol=1e-7)), f"Values do not match {r[~np.isclose(r, o, atol=1e-5)]} != {o[~np.isclose(r, o, atol=1e-5)]}"


def compare_evallib():
    test_data = torch.load("../predictions/predictions_Validation_4.pt")
    predictions = test_data["prediction"]
    audio_paths = test_data["paths"]
    audio_paths = [str(path) for path in audio_paths]
    root = os.path.join("..", *audio_paths[0].split("/")[:-3])
    folders = [path.split("/")[-2] for path in audio_paths]
    identifiers = ["_".join(path.split("/")[-1].split(".")[0].split("_")[-2:]) for path in audio_paths]
    annotations = [get_annotation(root, folder, identifier, DrumMapping.THREE_CLASS_STANDARD)[1:] for folder, identifier
                   in
                   zip(folders, identifiers)]
    annotations = [[*beats, *drums] for drums, beats in annotations]

    precisions, recalls, thresholds, f, f_avg, best_thresholds = calculate_pr(predictions, annotations,
                                                                              ignore_beats=True, detection_window=0.025,
                                                                              onset_cooldown=0.021, pr_points=200)
    print(f, f_avg)
    f_measures = [calculate_f_score(p, r) for p, r in zip(precisions, recalls)]
    for score in f_measures:
        assert not torch.isnan(score).any()

    per_class_onsets = [[song[i] for song in predictions] for i in range(len(predictions[0]))]
    per_class_gts = [[song[i] for song in annotations] for i in range(2, len(annotations[0]))]
    diff = []
    for pred, gts, thresh, f_score in zip(per_class_onsets, per_class_gts, thresholds, f_measures):
        for threshold, score in zip(thresh, f_score):
            count = np.zeros(4)
            other_count = np.zeros(4)
            total_unpicked = 0
            total_picked = 0
            for song, gt in zip(pred, gts):
                onsets = song[0, song[1] >= threshold]
                total_unpicked += len(onsets)
                onsets = combine_onsets(onsets, 0.021, "min")
                total_picked += len(onsets)
                ev = OnsetEvaluation(onsets, gt, window=0.025, combine=0)
                count += np.array([ev.num_tp, ev.num_fp, ev.num_fn, ev.num_tn])
                other_count[:3] += np.array(evaluate_onsets(onsets, gt, 0.025))
                if not np.isclose(count[:3], other_count[:3], atol=1e-5).all():
                    print("Problem in evaluate Onsets")
                    return
            # print(f"Threshold {threshold}: {count[:3]}")
            # print(f"Threshold {threshold}: {total_unpicked} {total_picked}")
            prec = count[0] / (count[0] + count[1]) if count[0] + count[1] > 0 else 0
            rec = count[0] / (count[0] + count[2]) if count[0] + count[2] > 0 else 0
            f = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
            prec = other_count[0] / (other_count[0] + other_count[1]) if other_count[0] + other_count[1] > 0 else 0
            rec = other_count[0] / (other_count[0] + other_count[2]) if other_count[0] + other_count[2] > 0 else 0
            other_f = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
            if not np.isclose(f, other_f, atol=1e-5, rtol=1e-5):
                print("Problem in evaluate Onsets")
            diff.append(abs(f - score.item()))
            if not np.isclose(f, score, atol=1e-5):
                print(f, score.item())
                print(f - score.item())
                print()
    print(np.mean(diff))


def main():
    # test_mean_max()
    compare_evallib()


if __name__ == '__main__':
    main()
