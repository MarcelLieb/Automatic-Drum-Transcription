// rustimport:pyo3

// Add parallel processing library
//: [dependencies]
//: rayon = "1.8.0"

use pyo3::prelude::*;

#[pyfunction]
fn calculate_pr(
    predictions: Vec<[f64; 4]>,
    ground_truths: Vec<Vec<Vec<f64>>>,
    detect_window: f64,
) -> (Vec<f64>, Vec<f64>, f64, f64) {
    let mut predictions = predictions;
    let mut ground_truths = ground_truths;
    let mut tp: usize = 0;
    let mut fp: usize = 0;
    let mut r#fn: usize = ground_truths
        .iter()
        .flat_map(|song| song.iter().map(|chan| chan.len()))
        .sum();
    let mut max_f_score = 0.0;
    let mut threshold = f64::INFINITY;

    let mut precisions = Vec::new();
    let mut recalls = Vec::new();

    predictions.sort_unstable_by(|a, b| a[1].partial_cmp(&b[1]).unwrap());
    predictions.reverse();

    for pred in predictions {
        let [time, score, song, class] = pred;
        let song = song as usize;
        let class = class as usize;
        let annotations = &mut ground_truths[song][class];
        if annotations.is_empty() {
            fp += 1;
            continue;
        }
        let index = annotations.partition_point(|a| a < &time);
        let prev = index.saturating_sub(1);
        let next = index.min(annotations.len() - 1);
        if (annotations[prev] + detect_window < time) || annotations[next] - detect_window < time {
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
        }

        precisions.push(p);
        recalls.push(r);
    }

    (precisions, recalls, max_f_score, threshold)
}

#[inline(always)]
fn _calculate_prf(tp: usize, fp: usize, r#fn: usize) -> (f64, f64, f64){
    let precision = tp as f64 / (tp + fp) as f64;
    let recall = tp as f64 / (tp + r#fn) as f64;
    let f_measure = (2 * tp) as f64 / (2 * tp + fp + r#fn) as f64;
    (precision, recall, f_measure)
}