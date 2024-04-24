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
) -> (Vec<(Vec<f32>, Vec<f32>)>, Vec<f32>, f32) {
    let mut out = Vec::new();
    predictions
        .into_par_iter()
        .with_max_len(1)
        .zip_eq(ground_truths)
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
            let mut labels = labels;
            let mut values = values;
            values.par_sort_by(|a, b| a[1].total_cmp(&b[1]));
            values.reverse();

            for pred in values {
                let [time, score, song] = pred;
                let song = song as usize;
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
            }
            (pn, precisions, recalls, threshold, max_tp, max_fp, max_fn)
        })
        .collect_into_vec(&mut out);

    let (a, b, c) = out
        .iter()
        .map(|class| (class.4, class.5, class.6))
        .fold((0, 0, 0), |acc, (a, b, c)| {
            (acc.0 + a, acc.1 + b, acc.2 + c)
        });
    let (_, _, f_score) = _calculate_prf(a, b, c);

    (
        out.iter()
            .map(|class| (class.1.clone(), class.2.clone()))
            .collect(),
        out.iter().cloned().map(|class| class.3).collect(),
        f_score,
    )
}

#[inline(always)]
fn _calculate_prf(tp: usize, fp: usize, r#fn: usize) -> (f32, f32, f32){
    let precision = tp as f32 / (tp + fp) as f32;
    let recall = tp as f32 / (tp + r#fn) as f32;
    let f_measure = (2 * tp) as f32 / (2 * tp + fp + r#fn) as f32;
    (precision, recall, f_measure)
}