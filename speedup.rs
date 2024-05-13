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
            let mut pn = Vec::new();

            let mut precisions = Vec::new();
            let mut recalls = Vec::new();
            let mut thresholds = Vec::new();
            let mut labels = labels;
            let mut values = values;
            let mut peaks_by_song = split_songs(&values, labels.len());

            values.par_sort_by(|a, b| a[1].total_cmp(&b[1]));
            values.reverse();

            'calculation: for pred in values {
                let [time, score, song] = pred;
                let song = song as usize;

                let peaks = &mut peaks_by_song[song];
                // Search index of peak
                let index = peaks.partition_point(|a| a[0] < time);
                let mut i = index - 1;

                while let Some(peak) = peaks.get(i) {
                    // Check if peak is close enough to be affected by the onset cool down
                    if time - peak[0] > cool_down {
                        break;
                    }
                    // Ignore peak if a peak was detected closely before it
                    if peak[1] > score {
                        peaks.remove(index);
                        continue 'calculation;
                    }
                    i -= 1;
                }

                let annotations = &mut labels[song];
                if annotations.is_empty() {
                    fp += 1;
                    continue;
                }
                let index = annotations.partition_point(|a| a < &time);
                let prev = index.saturating_sub(1);
                let next = index.min(annotations.len() - 1);
                if (annotations[prev] + detect_window > time)
                    || annotations[next] - detect_window < time
                {
                    tp += 1;
                    r#fn -= 1;

                    if time - annotations[prev] < annotations[next] - time {
                        annotations.remove(prev);
                    } else {
                        annotations.remove(next);
                    }
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
            (pn, precisions, recalls, threshold, max_tp, max_fp, max_fn, thresholds)
        })
        // Restore original order
        .rev()
        .collect_into_vec(&mut out);

    let (a, b, c) = out
        .iter()
        .map(|class| (class.4, class.5, class.6))
        .fold((0, 0, 0), |acc, (a, b, c)| {
            (acc.0 + a, acc.1 + b, acc.2 + c)
        });
    let (_, _, f_score) = _calculate_prf(a, b, c);
    let f_score_avg = out
        .iter()
        .map(|class| (class.4, class.5, class.6))
        .map(|(tp, fp, r#fn)| _calculate_prf(tp, fp, r#fn).2)
        .sum::<f32>()
        / out.len() as f32;

    (
        out.iter()
            .cloned()
            .map(|class| (class.1, class.2, class.7))
            .collect(),
        out.iter().cloned().map(|class| class.3).collect(),
        f_score,
        f_score_avg,
    )
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