// rustimport:pyo3

// Add parallel processing library
//: [dependencies]
//: rayon = "1.8.0"

use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
fn calculate_pr(
    predictions: Vec<Vec<[f32; 3]>>,
    ground_truths: Vec<Vec<Vec<f32>>>,
    detect_window: f32,
    cool_down: f32,
    points: Option<usize>,
) -> (Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>, Vec<f32>, f32, f32) {
    let mut out = Vec::new();

    predictions
        .into_par_iter()
        .rev()
        .with_max_len(1)
        // Assumes beat annotations to be first in ground_truth
        // Filters out extra beat labels if length mismatches
        .zip(ground_truths.into_par_iter().rev())
        .map(|(values, labels)| {
            let mut tp: usize = 0;
            let mut fp: usize = 0;
            let mut r#fn: usize = labels.iter().map(|song| song.len()).sum();
            let mut max_f_score = 0.0;
            let (mut max_tp, mut max_fp, mut max_fn) = (tp, fp, r#fn);
            let mut threshold = f32::INFINITY;
            let mut pn = Vec::with_capacity(values.len());

            let mut precisions = Vec::with_capacity(values.len());
            let mut recalls = Vec::with_capacity(values.len());
            let mut thresholds = Vec::with_capacity(values.len());
            let mut labels = labels;
            let mut values = values;

            values.par_sort_by(|a, b| a[1].total_cmp(&b[1]));
            values.reverse();

            if points.is_none() {
                let mut peaks_by_song = split_songs(&values, labels.len());
                'calculation: for pred in values {
                    let [time, score, song] = pred;
                    let song = song as usize;

                    let peaks = &mut peaks_by_song[song];
                    // Search index of peak
                    let index = peaks.partition_point(|a| a[0] < time);
                    for i in (0..index).rev() {
                        // Check if peak is close enough to be affected by the onset cool down
                        if time - peaks[i][0] > cool_down {
                            // If not stop early
                            break;
                        }
                        // Ignore peak if a peak was detected closely before it
                        if peaks[i][1] > score {
                            peaks.remove(index);
                            continue 'calculation;
                        }
                    }

                    let annotations = &mut labels[song];
                    if annotations.is_empty() {
                        fp += 1;
                        continue;
                    }
                    let (index, dist) = find_closest_onset(time, &annotations).unwrap();
                    if dist < detect_window {
                        tp += 1;
                        r#fn -= 1;
                        annotations.remove(index - 1);
                    } else {
                        fp += 1;
                    }

                    let (p, r, f) = _calculate_prf(tp, fp, r#fn);
                    if f > max_f_score {
                        max_f_score = f;
                        threshold = score;
                        (max_tp, max_fp, max_fn) = (tp, fp, r#fn);
                    }
                    pn.push((threshold, tp, fp, r#fn));

                    precisions.push(p);
                    recalls.push(r);
                    thresholds.push(score);
                }
                (
                    pn, precisions, recalls, threshold, max_tp, max_fp, max_fn, thresholds,
                )
            } else {
                for chunk in values.chunks(values.len() / points.unwrap()) {
                    let score = chunk.last().unwrap()[1];
                    let peaks_by_songs = split_songs(chunk, labels.len());
                    let onsets_by_song: Vec<Vec<f32>> = peaks_by_songs
                        .into_par_iter()
                        .map(|onsets| {
                            onsets
                                .into_iter()
                                .map(|[time, _]| time)
                                .collect::<Vec<f32>>()
                        })
                        // Less onsets than with the direct approach get filtered here
                        .map(|onsets| _combine_onsets(&onsets, cool_down, "min"))
                        .collect();
                    let (n_tp, n_fp, _) = onsets_by_song
                        .into_iter()
                        .zip(labels.iter_mut())
                        .par_bridge()
                        .map(|(onsets, labels)| {
                            // Here the chunk wise differentiates from the direct approach slightly
                            //
                            _evaluate_detections(&onsets, labels, detect_window)
                        })
                        .reduce(|| (0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));

                    tp += n_tp;
                    fp += n_fp;
                    r#fn -= n_tp;
                    let (p, r, f) = _calculate_prf(tp, fp, r#fn);
                    if f > max_f_score {
                        max_f_score = f;
                        threshold = score;
                        (max_tp, max_fp, max_fn) = (tp, fp, r#fn);
                    }
                    pn.push((threshold, tp, fp, r#fn));

                    precisions.push(p);
                    recalls.push(r);
                    thresholds.push(score);
                }
                (
                    pn, precisions, recalls, threshold, max_tp, max_fp, max_fn, thresholds,
                )
            }
        })
        // Restore original order
        .rev()
        .collect_into_vec(&mut out);

    let (a, b, c) = out
        .par_iter()
        .map(|class| (class.4, class.5, class.6))
        .reduce(
            || (0, 0, 0),
            |acc, (a, b, c)| (acc.0 + a, acc.1 + b, acc.2 + c),
        );
    let (_, _, f_score) = _calculate_prf(a, b, c);
    let f_score_avg = out
        .par_iter()
        .map(|class| (class.4, class.5, class.6))
        .map(|(tp, fp, r#fn)| _calculate_prf(tp, fp, r#fn).2)
        .sum::<f32>()
        / out.len() as f32;

    (
        out.par_iter()
            .cloned()
            .map(|class| (class.1, class.2, class.7))
            .collect(),
        out.iter().cloned().map(|class| class.3).collect(),
        f_score,
        f_score_avg,
    )
}

fn _combine_onsets(onsets: &[f32], cool_down: f32, combine_strategy: &str) -> Vec<f32> {
    if onsets.is_empty() {
        return Vec::new();
    }
    let mut final_onsets = vec![*onsets.first().unwrap()];

    for onset_time in &onsets[1..] {
        let prev_time = *final_onsets.first().unwrap();
        if onset_time - prev_time > cool_down {
            final_onsets.push(*onset_time);
        } else {
            match combine_strategy {
                "min" | "left" => {}
                "max" | "right" => {
                    final_onsets.pop();
                    final_onsets.push(*onset_time);
                }
                "avg" | "mean" => {
                    final_onsets.pop();
                    final_onsets.push(0.5 * (prev_time + onset_time))
                }
                str => panic!(
                    "Unsupported combine strategy '{str}'\nSelect between 'min', 'max' or 'avg'"
                ),
            }
        }
    }

    final_onsets
}

fn _evaluate_detections(
    onsets: &[f32],
    labels: &mut Vec<f32>,
    detect_window: f32,
) -> (usize, usize, usize) {
    let n_labels = labels.len();
    let mut fp = 0;
    // Assume onsets are sorted ascending
    for onset_time in onsets.iter().rev() {
        if labels.is_empty() {
            break;
        }
        let (index, dist) = find_closest_onset(*onset_time, &labels).unwrap();

        if dist < detect_window + 10.0 * f32::EPSILON {
            labels.remove(index);
        } else {
            fp += 1;
        }
    }

    // Return tp, fp, fn
    (n_labels - labels.len(), fp, labels.len())
}

#[pyfunction]
fn combine_onsets(onsets: Vec<f32>, cool_down: f32, combine_strategy: &str) -> Vec<f32> {
    _combine_onsets(&onsets, cool_down, combine_strategy)
}

#[pyfunction]
fn evaluate_detections(
    onsets: Vec<f32>,
    labels: Vec<f32>,
    detect_window: f32,
) -> (usize, usize, usize) {
    let mut labels = labels;
    _evaluate_detections(&onsets, &mut labels, detect_window)
}

fn find_closest_onset(onset: f32, labels: &[f32]) -> Option<(usize, f32)> {
    if labels.is_empty() {
        return None;
    }
    // Assume labels are sorted ascending
    let index = labels.partition_point(|a| a < &onset);
    let mut dist: (f32, f32) = (f32::INFINITY, f32::INFINITY);
    if index > 0 {
        dist.0 = onset - labels[index - 1]
    }
    if let Some(next) = labels.get(index) {
        dist.1 = next - onset;
    }
    if dist.0 < dist.1 && dist.0.is_finite() {
        Some((index - 1, dist.0))
    } else if dist.1.is_finite() {
        Some((index, dist.1))
    } else {
        None
    }
}

#[inline(always)]
fn _calculate_prf(tp: usize, fp: usize, r#fn: usize) -> (f32, f32, f32) {
    let precision = tp as f32 / (tp + fp) as f32;
    let recall = tp as f32 / (tp + r#fn) as f32;
    let f_measure = (2 * tp) as f32 / (2 * tp + fp + r#fn) as f32;
    (precision, recall, f_measure)
}

fn split_songs(peaks: &[[f32; 3]], song_count: usize) -> Vec<Vec<[f32; 2]>> {
    (0..song_count)
        .map(|i| {
            let mut values: Vec<[f32; 2]> = peaks
                .iter()
                .cloned()
                .filter_map(|[t, v, s]| if s as usize == i { Some([t, v]) } else { None })
                .collect();
            values.sort_by(|a, b| a[0].total_cmp(&b[0]));
            values
        })
        .collect()
}