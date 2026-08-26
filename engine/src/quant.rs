//! int8 weight-only quantisation.
//!
//! Decode reads every weight once per token and reuses nothing, so it is
//! bandwidth-bound. int8 does not help because integer arithmetic is faster --
//! it helps because a weight becomes one byte instead of four, and bytes moved
//! is the thing that sets the ceiling. On this hardware that raises the
//! theoretical limit from ~1,674 tok/s to ~6,700.
//!
//! Weight-only: activations stay f32. They are a negligible fraction of the
//! bytes moved during decode, so quantising them would cost accuracy and buy
//! almost nothing.

use crate::weights::Tensor;

/// A quantised 2-D weight matrix: int8 values plus one f32 scale per row.
pub struct QuantTensor {
    pub data: Vec<i8>,
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl QuantTensor {
    /// Symmetric per-row quantisation.
    ///
    /// Per-row rather than per-tensor: a single scale across a whole matrix is
    /// determined by its largest outlier, which crushes the resolution of every
    /// other row. Rows are the output channels, so each gets the full int8
    /// range for its own magnitude.
    ///
    /// Symmetric (no zero point) because trained weights are near
    /// zero-centred, and an asymmetric scheme would add an offset term to every
    /// dot product for no measurable accuracy gain.
    pub fn from_tensor(t: &Tensor) -> Self {
        let rows = t.rows();
        let cols = t.cols();
        let mut data = vec![0i8; rows * cols];
        let mut scales = vec![0.0f32; rows];

        for r in 0..rows {
            let row = &t.data[r * cols..(r + 1) * cols];
            let max_abs = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));

            // An all-zero row would give a zero scale and produce NaN on
            // dequantisation; leave it as zeros with a unit scale instead.
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            scales[r] = scale;

            let inv = 1.0 / scale;
            for (i, v) in row.iter().enumerate() {
                // round-half-away-from-zero, then clamp: 127 rather than 128 so
                // the range stays symmetric and -128 never appears.
                let q = (v * inv).round().clamp(-127.0, 127.0);
                data[r * cols + i] = q as i8;
            }
        }

        Self { data, scales, rows, cols }
    }

    /// Bytes held, including scales.
    pub fn bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4
    }

    /// Reconstruct f32 values, for measuring what quantisation cost.
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.rows * self.cols];
        for r in 0..self.rows {
            let scale = self.scales[r];
            for c in 0..self.cols {
                out[r * self.cols + c] = self.data[r * self.cols + c] as f32 * scale;
            }
        }
        out
    }

    /// Relative error introduced by quantising this tensor.
    ///
    /// Reported as a fraction of the tensor's RMS magnitude rather than
    /// elementwise, because near-zero weights have unbounded relative error
    /// while contributing almost nothing to any dot product.
    pub fn error_vs(&self, original: &Tensor) -> QuantError {
        let restored = self.dequantize();
        let mut sum_sq_err = 0.0f64;
        let mut sum_sq_val = 0.0f64;
        let mut max_abs_err = 0.0f32;

        for (o, r) in original.data.iter().zip(&restored) {
            let err = (o - r).abs();
            max_abs_err = max_abs_err.max(err);
            sum_sq_err += (err as f64) * (err as f64);
            sum_sq_val += (*o as f64) * (*o as f64);
        }

        QuantError {
            rms_relative: (sum_sq_err / sum_sq_val.max(1e-30)).sqrt(),
            max_absolute: max_abs_err,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QuantError {
    /// RMS of the error divided by RMS of the values.
    pub rms_relative: f64,
    pub max_absolute: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f32) -> Tensor {
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(f(r, c));
            }
        }
        Tensor { shape: vec![rows, cols], data }
    }

    #[test]
    fn round_trip_is_close() {
        let t = tensor(16, 64, |r, c| ((r * 7 + c * 3) % 100) as f32 / 50.0 - 1.0);
        let q = QuantTensor::from_tensor(&t);
        let err = q.error_vs(&t);
        // int8 over a symmetric range gives roughly 1/127 resolution; RMS
        // relative error well under 1% is the expected outcome.
        assert!(err.rms_relative < 0.01, "rms relative error {}", err.rms_relative);
    }

    #[test]
    fn per_row_scales_isolate_outliers() {
        // One row with a huge magnitude, the rest small. Per-row scaling must
        // keep the small rows accurate; a single tensor-wide scale would not.
        let t = tensor(4, 32, |r, c| {
            let base = ((c % 7) as f32 - 3.0) / 100.0;
            if r == 0 { base * 10_000.0 } else { base }
        });
        let q = QuantTensor::from_tensor(&t);
        assert!(q.scales[0] > q.scales[1] * 100.0, "outlier row should have its own scale");
        assert!(q.error_vs(&t).rms_relative < 0.01);
    }

    #[test]
    fn all_zero_row_does_not_produce_nan() {
        let t = tensor(3, 8, |r, _| if r == 1 { 0.0 } else { 0.5 });
        let q = QuantTensor::from_tensor(&t);
        assert!(q.dequantize().iter().all(|v| v.is_finite()));
        assert_eq!(q.scales[1], 1.0);
    }

    #[test]
    fn never_emits_negative_128() {
        // Clamping to -127 keeps the range symmetric; -128 has no positive twin
        // and would bias the reconstruction.
        let t = tensor(2, 16, |_, c| if c % 2 == 0 { -1.0 } else { 1.0 });
        let q = QuantTensor::from_tensor(&t);
        assert!(q.data.iter().all(|&v| v != i8::MIN));
    }

    #[test]
    fn four_times_smaller_than_f32() {
        let t = tensor(64, 256, |r, c| (r as f32 - c as f32) / 128.0);
        let q = QuantTensor::from_tensor(&t);
        let f32_bytes = t.data.len() * 4;
        // Scales add one f32 per row, so the ratio is slightly above 1/4.
        let ratio = q.bytes() as f64 / f32_bytes as f64;
        assert!(ratio > 0.24 && ratio < 0.26, "ratio {ratio}");
    }
}
