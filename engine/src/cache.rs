//! Key/value cache for incremental decoding.
//!
//! Without a cache, generating token N re-runs attention over all N positions,
//! so producing a sequence costs O(N^2) work and the whole prompt is recomputed
//! on every single step. The cache keeps each position's projected keys and
//! values after they are computed once, making each new token O(N) instead.
//!
//! Layout is `[layer][position][kv_head * head_dim]`, contiguous in the last
//! dimension. Attention reads one position's keys for one head at a time, so
//! this puts exactly those values next to each other in memory.

use anyhow::{bail, Result};

use crate::config::Config;

pub struct KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    /// Positions currently held.
    len: usize,
    capacity: usize,
    /// Floats per position per layer: n_kv_head * head_dim.
    kv_dim: usize,
    n_layer: usize,
}

impl KvCache {
    pub fn new(cfg: &Config, capacity: usize) -> Self {
        let kv_dim = cfg.n_kv_head * cfg.head_dim();
        let total = cfg.n_layer * capacity * kv_dim;
        Self {
            k: vec![0.0; total],
            v: vec![0.0; total],
            len: 0,
            capacity,
            kv_dim,
            n_layer: cfg.n_layer,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes held by this cache, both tensors.
    pub fn bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * std::mem::size_of::<f32>()
    }

    /// Drop all cached positions without freeing the allocation, so a new
    /// sequence reuses the same memory.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    fn offset(&self, layer: usize, pos: usize) -> usize {
        (layer * self.capacity + pos) * self.kv_dim
    }

    /// Mutable slices for writing one position's keys and values.
    pub fn slot_mut(&mut self, layer: usize, pos: usize) -> (&mut [f32], &mut [f32]) {
        let start = self.offset(layer, pos);
        let end = start + self.kv_dim;
        (&mut self.k[start..end], &mut self.v[start..end])
    }

    /// One cached head's key vector at a position.
    pub fn key(&self, layer: usize, pos: usize, head: usize, head_dim: usize) -> &[f32] {
        let base = self.offset(layer, pos) + head * head_dim;
        &self.k[base..base + head_dim]
    }

    pub fn value(&self, layer: usize, pos: usize, head: usize, head_dim: usize) -> &[f32] {
        let base = self.offset(layer, pos) + head * head_dim;
        &self.v[base..base + head_dim]
    }

    /// Record that `n` more positions are now valid.
    ///
    /// Called once per token after every layer has written its slot, so a
    /// partially-filled position is never visible to attention.
    pub fn advance(&mut self, n: usize) -> Result<()> {
        if self.len + n > self.capacity {
            bail!(
                "kv cache full: {} + {n} exceeds capacity {}",
                self.len,
                self.capacity
            );
        }
        self.len += n;
        Ok(())
    }

    pub fn n_layer(&self) -> usize {
        self.n_layer
    }
}
