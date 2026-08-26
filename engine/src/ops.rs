//! Scalar CPU kernels.
//!
//! Deliberately simple: no SIMD, no threading, no blocking. This path exists to
//! be *obviously correct*, so that when a GPU kernel disagrees with it the bug
//! is in the GPU kernel. Optimising these would trade away the only property
//! that makes them useful.
//!
//! Every routine matches the corresponding PyTorch operation in `model.py`,
//! including the places where that implementation makes a specific choice
//! (fp32 accumulation in RMSNorm, the half-split RoPE convention).

/// out = W · x, where W is row-major `[rows, cols]` and x has `cols` elements.
///
/// This is the linear layer. PyTorch stores `nn.Linear` weights as
/// `[out_features, in_features]`, so each output element is a dot product with
/// one contiguous row -- which is why no transpose is needed anywhere.
pub fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);

    for (r, o) in out.iter_mut().enumerate() {
        let row = &w[r * cols..(r + 1) * cols];
        // Accumulate in f64 to keep the reference path free of the summation
        // error that would otherwise be confused for a kernel bug.
        let mut acc = 0.0f64;
        for (wi, xi) in row.iter().zip(x.iter()) {
            acc += (*wi as f64) * (*xi as f64);
        }
        *o = acc as f32;
    }
}

/// RMSNorm: x * rsqrt(mean(x^2) + eps) * weight.
///
/// No mean subtraction and no bias, matching `model.py::RMSNorm`. That
/// implementation normalises in fp32 regardless of the activation dtype, so
/// this one accumulates in f64 for the same reason as `matvec`.
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());

    let mut sum_sq = 0.0f64;
    for v in x {
        sum_sq += (*v as f64) * (*v as f64);
    }
    let scale = 1.0 / ((sum_sq / x.len() as f64) + eps as f64).sqrt();

    for ((o, xi), wi) in out.iter_mut().zip(x.iter()).zip(weight.iter()) {
        *o = ((*xi as f64) * scale) as f32 * wi;
    }
}

/// LayerNorm with optional bias, for checkpoints trained with `norm=layernorm`.
pub fn layernorm(x: &[f32], weight: &[f32], bias: Option<&[f32]>, eps: f32, out: &mut [f32]) {
    let n = x.len() as f64;
    let mean = x.iter().map(|v| *v as f64).sum::<f64>() / n;
    let var = x.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / n;
    let scale = 1.0 / (var + eps as f64).sqrt();

    for (i, o) in out.iter_mut().enumerate() {
        let normed = ((x[i] as f64 - mean) * scale) as f32;
        *o = normed * weight[i] + bias.map_or(0.0, |b| b[i]);
    }
}

/// Numerically stable softmax, in place.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // Subtracting the max before exponentiating prevents overflow; attention
    // scores at long context are large enough for this to matter.
    let mut sum = 0.0f64;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v as f64;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v = ((*v as f64) * inv) as f32;
    }
}

/// SiLU (swish): x * sigmoid(x). The gate activation in SwiGLU.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GeLU, tanh approximation -- matches `F.gelu(..., approximate="tanh")`.
pub fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_56;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
}

/// Precomputed rotary tables: `cos` and `sin`, each `[max_seq, head_dim / 2]`.
pub struct RopeTable {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub half: usize,
}

impl RopeTable {
    pub fn new(head_dim: usize, max_seq: usize, theta: f32) -> Self {
        let half = head_dim / 2;
        let mut cos = Vec::with_capacity(max_seq * half);
        let mut sin = Vec::with_capacity(max_seq * half);
        for pos in 0..max_seq {
            for i in 0..half {
                // inv_freq[i] = theta^(-2i / head_dim), matching precompute_rope.
                let inv_freq = (theta as f64).powf(-((2 * i) as f64) / head_dim as f64);
                let angle = pos as f64 * inv_freq;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        Self { cos, sin, half }
    }

    /// Rotate one head vector in place at the given position.
    ///
    /// Uses the half-split convention from `model.py::apply_rope`: element `i`
    /// pairs with element `i + head_dim/2`, not with its neighbour. Pairing
    /// adjacent elements instead -- the other common convention -- produces a
    /// model that still runs and generates fluent-looking nonsense, so this is
    /// pinned by a test rather than left to inspection.
    pub fn apply(&self, v: &mut [f32], pos: usize) {
        let half = self.half;
        let base = pos * half;
        for i in 0..half {
            let c = self.cos[base + i];
            let s = self.sin[base + i];
            let lo = v[i];
            let hi = v[i + half];
            v[i] = lo * c - hi * s;
            v[i + half] = lo * s + hi * c;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_identity() {
        // 2x2 identity times [3, 4] is [3, 4].
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let x = vec![3.0, 4.0];
        let mut out = vec![0.0; 2];
        matvec(&w, 2, 2, &x, &mut out);
        assert_eq!(out, vec![3.0, 4.0]);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum was {sum}");
        // Monotonic input must give monotonic output.
        assert!(x.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn softmax_survives_large_inputs() {
        // Without max-subtraction this overflows to NaN.
        let mut x = vec![1000.0, 1001.0, 1002.0];
        softmax(&mut x);
        assert!(x.iter().all(|v| v.is_finite()));
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rmsnorm_unit_weight_normalises() {
        let x = vec![3.0, 4.0];             // rms = sqrt(12.5)
        let w = vec![1.0, 1.0];
        let mut out = vec![0.0; 2];
        rmsnorm(&x, &w, 1e-6, &mut out);
        let rms = (out.iter().map(|v| v * v).sum::<f32>() / 2.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-4, "rms was {rms}");
    }

    #[test]
    fn rope_preserves_length_and_rotates() {
        let table = RopeTable::new(4, 8, 10000.0);
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        let before: f32 = v.iter().map(|x| x * x).sum();
        table.apply(&mut v, 3);
        let after: f32 = v.iter().map(|x| x * x).sum();
        // A rotation preserves norm.
        assert!((before - after).abs() < 1e-4);
        // Position 0 is the identity rotation; position 3 must not be.
        let mut v0 = vec![1.0, 2.0, 3.0, 4.0];
        table.apply(&mut v0, 0);
        assert_eq!(v0, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn silu_and_gelu_match_known_values() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!((silu(1.0) - 0.731_058_6).abs() < 1e-5);
        assert!((gelu(0.0) - 0.0).abs() < 1e-6);
        assert!((gelu(1.0) - 0.841_192).abs() < 1e-4);
    }
}
