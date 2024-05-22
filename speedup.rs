// rustimport:pyo3

// Add parallel processing library
//: [dependencies]
//: rayon = "1.10.0"

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
            let mut labels = labels;
            let mut values = values;

            values.par_sort_by(|a, b| a[1].partial_cmp(&b[1]).unwrap());
            values.reverse();
            labels.par_iter_mut().for_each(|song| {
                song.par_sort_by(|a, b| a.partial_cmp(b).unwrap());
                song.dedup();
            });

            if points.is_none() {
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

                let mut peaks_by_song = split_songs(&values, labels.len());
                'calculation: for (i, pred) in values.into_iter().enumerate() {
                    let [time, score, song] = pred;
                    let song = song as usize;

                    let peaks = &mut peaks_by_song[song];
                    // Search index of peak
                    let index = peaks.partition_point(|a| a[0] < time);
                    debug_assert_eq!(time, peaks[index][0], "Iteration: {i}");
                    debug_assert!(
                        peaks
                            .binary_search_by(|a| a[0].partial_cmp(&time).unwrap())
                            .is_ok(),
                        "{time}, {:?}",
                        peaks.last()
                    );
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
                        annotations.remove(index);
                    } else {
                        fp += 1;
                    }

                    let (p, r, f) = _calculate_prf(tp, fp, r#fn);
                    if f > max_f_score {
                        max_f_score = f;
                        threshold = score;
                        (max_tp, max_fp, max_fn) = (tp, fp, r#fn);
                    }
                    pn.push((tp, fp, r#fn));

                    precisions.push(p);
                    recalls.push(r);
                    thresholds.push(score);
                }
                (
                    pn, precisions, recalls, threshold, max_tp, max_fp, max_fn, thresholds,
                )
            } else {
                let chunk_length = (values.len() as f32 / points.unwrap() as f32).round() as usize;
                let n_chunks = values.len().div_ceil(chunk_length);

                debug_assert!(n_chunks * chunk_length >= values.len());

                let iter: Vec<_> = (1..=n_chunks)
                    .into_par_iter()
                    .map(|i| {
                        let onsets = &values[..(i * chunk_length).min(values.len())];

                        let score = onsets.last().unwrap()[1];
                        let peaks_by_songs = split_songs(&onsets, labels.len());
                        let onsets_by_song: Vec<Vec<f32>> = peaks_by_songs
                            .into_par_iter()
                            .map(|onsets| {
                                onsets
                                    .into_iter()
                                    .map(|[time, _]| time)
                                    .collect::<Vec<f32>>()
                            })
                            .map(|onsets| _combine_onsets(&onsets, cool_down, "min"))
                            .collect();
                        let (n_tp, n_fp, n_fn) = onsets_by_song
                            .iter()
                            .zip(labels.iter())
                            .par_bridge()
                            .map(|(onsets, labels)| {
                                // Here the chunk wise differentiates from the direct approach slightly
                                _evaluate_detections(onsets, labels, detect_window)
                            })
                            .reduce(|| (0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));

                        debug_assert_ne!((n_tp, n_fp, n_fn), (0, 0, 0));

                        let (p, r, f) = _calculate_prf(n_tp, n_fp, n_fn);

                        #[cfg(not(debug_assertions))]
                        return (n_tp, n_fp, n_fn, p, r, f, score);

                        #[cfg(debug_assertions)]
                        {
                            let n_unfiltered = onsets.len();
                            let n_filtered: usize = onsets_by_song.iter().map(|v| v.len()).sum();
                            return (
                                n_tp,
                                n_fp,
                                n_fn,
                                p,
                                r,
                                f,
                                score,
                                n_unfiltered,
                                n_filtered,
                                (i * chunk_length).min(values.len()),
                            );
                        }
                    })
                    .collect();

                let pn = iter
                    .iter()
                    .cloned()
                    .map(|o| (o.0, o.1, o.2))
                    .collect::<Vec<_>>();

                let precisions = iter.iter().cloned().map(|o| o.3).collect::<Vec<_>>();
                let recalls = iter.iter().cloned().map(|o| o.4).collect::<Vec<_>>();

                debug_assert_eq!(iter.len(), n_chunks);
                #[cfg(debug_assertions)]
                recalls.windows(2).enumerate().for_each(|(i, w)| {
                    if w[0] > w[1] {
                        println!("{:?}\n{:?}\n", iter[i], iter[i + 1])
                    }
                });

                #[cfg(debug_assertions)]
                debug_assert!(
                    iter.windows(2).all(|w| w[0].7 < w[1].7),
                    "Chunk sizes are not monotonically increasing"
                );
                debug_assert!(
                    recalls.windows(2).all(|w| w[0] <= w[1]),
                    "Recall is not monotonically increasing"
                );

                let thresholds = iter.iter().cloned().map(|o| o.6).collect::<Vec<_>>();
                let (max_tp, max_fp, max_fn, threshold, _) = iter
                    .iter()
                    .cloned()
                    .map(|o| (o.0, o.1, o.2, o.6, o.5))
                    .max_by(|a, b| a.4.partial_cmp(&b.4).unwrap())
                    .unwrap();

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
        .iter()
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
    labels: &[f32],
    detect_window: f32,
) -> (usize, usize, usize) {
    let mut labels = Vec::from(labels);
    let mut tp = 0;
    let mut fp = 0;
    // Assume onsets are sorted ascending
    debug_assert!(
        onsets.windows(2).all(|w| w[0] < w[1]),
        "Onset times are not strictly monotonically increasing"
    );
    debug_assert!(
        labels.windows(2).all(|w| w[0] < w[1]),
        "Label times are not strictly monotonically increasing"
    );
    for (i, onset_time) in onsets.iter().enumerate().rev() {
        if labels.is_empty() {
            fp += i + 1;
            break;
        }

        let (index, dist) = find_closest_onset(*onset_time, &labels).unwrap();

        if dist < detect_window + 10.0 * f32::EPSILON {
            tp += 1;
            labels.remove(index);
        } else {
            fp += 1;
        }
    }

    // Return tp, fp, fn
    (tp, fp, labels.len())
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
    _evaluate_detections(&onsets, &labels, detect_window)
}

fn find_closest_onset(onset: f32, labels: &[f32]) -> Option<(usize, f32)> {
    if labels.is_empty() {
        return None;
    }
    // Assume labels are sorted ascending
    debug_assert!(
        labels.windows(2).all(|w| w[0] < w[1]),
        "Label times are not strictly monotonically increasing"
    );
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
    let out: Vec<Vec<[f32; 2]>> = (0..song_count)
        .into_par_iter()
        .map(|i| {
            let mut values: Vec<[f32; 2]> = peaks
                .iter()
                .cloned()
                .filter_map(|[t, v, s]| if s as usize == i { Some([t, v]) } else { None })
                .collect();
            values.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
            values
        })
        .collect();
    debug_assert_eq!(peaks.len(), out.iter().map(|v| v.len()).sum::<usize>());
    out
}