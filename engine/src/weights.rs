//! Memory-mapped checkpoint access.
//!
//! Weights are never copied into owned buffers on load. The file is mapped and
//! `safetensors` hands back byte ranges into that mapping, so a multi-GB model
//! becomes available immediately and the OS pages it in on demand. Conversion
//! to f32 happens per tensor, only when a tensor is actually requested.

use anyhow::{anyhow, bail, Context, Result};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use std::fs::File;
use std::path::Path;

pub struct Weights {
    mmap: Mmap,
}

impl Weights {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: the mapping is read-only and lives as long as this struct.
        // A concurrent writer truncating the file would be undefined behaviour,
        // but checkpoints are immutable once exported.
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mapping {}", path.display()))?;
        Ok(Self { mmap })
    }

    fn tensors(&self) -> Result<SafeTensors<'_>> {
        SafeTensors::deserialize(&self.mmap).context("parsing safetensors header")
    }

    /// Every tensor name and shape in the checkpoint.
    pub fn inventory(&self) -> Result<Vec<(String, Vec<usize>, Dtype)>> {
        let st = self.tensors()?;
        let mut out = Vec::new();
        for name in st.names() {
            let view = st
                .tensor(name)
                .map_err(|e| anyhow!("reading tensor {name}: {e}"))?;
            out.push((name.to_string(), view.shape().to_vec(), view.dtype()));
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }

    /// Load one tensor as f32, converting from whatever it was stored as.
    ///
    /// f32 is the reference precision: the CPU path exists to validate numerics
    /// against PyTorch, and doing that in a reduced format would confound
    /// implementation bugs with rounding error.
    pub fn get(&self, name: &str) -> Result<Tensor> {
        let st = self.tensors()?;
        let view = st
            .tensor(name)
            .map_err(|_| anyhow!("no tensor named {name} in checkpoint"))?;
        let raw = view.data();

        let data: Vec<f32> = match view.dtype() {
            Dtype::F32 => raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            Dtype::BF16 => raw
                .chunks_exact(2)
                .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            Dtype::F16 => raw
                .chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            other => bail!("unsupported dtype {other:?} for tensor {name}"),
        };

        let shape = view.shape().to_vec();
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            bail!(
                "tensor {}: shape {:?} implies {} values, got {}",
                name,
                shape,
                expected,
                data.len()
            );
        }
        Ok(Tensor { shape, data })
    }

    pub fn total_bytes(&self) -> usize {
        self.mmap.len()
    }
}

/// A dense row-major f32 tensor.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; n],
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }

    pub fn cols(&self) -> usize {
        self.shape.get(1).copied().unwrap_or_else(|| self.len())
    }

    /// One row of a 2-D tensor, e.g. an embedding lookup.
    pub fn row(&self, i: usize) -> &[f32] {
        let c = self.cols();
        &self.data[i * c..(i + 1) * c]
    }
}
