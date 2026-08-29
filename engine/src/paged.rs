//! Paged KV storage: fixed-size physical pages plus per-sequence page tables.
//!
//! A contiguous cache forces one decision at load time -- how many tokens one
//! sequence may ever hold -- and then charges every sequence that much whether
//! it uses it or not. Serving several requests at once from a contiguous cache
//! means either reserving the maximum context per request (mostly wasted) or
//! recopying the whole cache whenever a request enters or leaves. Paging
//! removes both: memory is handed out in small fixed blocks, sequences grow
//! lazily, and a finished request returns its pages without disturbing anyone
//! else's.
//!
//! # Layout
//!
//! One physical page holds `PAGE_TOKENS` positions for *every* layer:
//!
//! ```text
//! pool: [n_pages][n_layer][PAGE_TOKENS][kv_dim]
//! offset(page, layer, slot) =
//!     page * n_layer * PAGE_TOKENS * kv_dim      <- dynamic, from the page table
//!   + layer * PAGE_TOKENS * kv_dim               <- host constant per launch
//!   + slot * kv_dim
//! ```
//!
//! Putting all layers inside one page keeps the page table per *sequence*
//! rather than per (sequence, layer), which matters because that table is
//! copied to the device on every scheduling step.
//!
//! It also makes the split above fall out for free, and that split is what
//! makes paging cheap here: the layer term is exactly the `layer_base` argument
//! the projection kernels already take, and the page term is exactly the
//! per-step scalar they already read from device memory. Decode's K/V write and
//! RoPE therefore need no kernel changes, and CUDA graph capture stays valid,
//! because the only value that varies per step still arrives through the
//! parameter buffer rather than through a kernel argument.
//!
//! # Page size
//!
//! 16 tokens. Power of two so translation is a shift and a mask rather than a
//! division inside the attention inner loop. At `kv_dim = 192` floats that is
//! 12 KB of contiguous K per layer per page, well above the 128-byte
//! transaction granularity, so the coalesced reads the attention kernel was
//! tuned for are unaffected. Internal fragmentation is bounded by
//! `PAGE_TOKENS - 1` tokens per sequence regardless of context length.
//!
//! This module is deliberately free of CUDA types: it is the part of paging
//! that can be tested exhaustively on the CPU, and it is.

use anyhow::{bail, Result};

/// Tokens per physical page. See the module docs for why 16.
pub const PAGE_TOKENS: usize = 16;

/// Physical page index. `u32` because the table is copied to the device every
/// step and the kernels index it as `int`.
pub type PageId = u32;

/// Fixed-size physical pages plus a free list.
///
/// The allocator is intentionally boring: a stack of free ids and a used-flag
/// per page. Exhaustion is an error rather than a silent overwrite, and
/// double-free is rejected rather than corrupting the free list -- an aliased
/// page would show up as one sequence reading another's keys, which is the
/// hardest class of bug to notice from output that still looks fluent.
#[derive(Debug)]
pub struct PagePool {
    n_pages: usize,
    /// Floats per page: `n_layer * PAGE_TOKENS * kv_dim`.
    page_floats: usize,
    free: Vec<PageId>,
    in_use: Vec<bool>,
}

impl PagePool {
    pub fn new(n_pages: usize, n_layer: usize, kv_dim: usize) -> Self {
        // Reverse order so the first allocation is page 0, which makes test
        // expectations readable and dumps easier to follow.
        let free: Vec<PageId> = (0..n_pages as PageId).rev().collect();
        Self {
            n_pages,
            page_floats: n_layer * PAGE_TOKENS * kv_dim,
            free,
            in_use: vec![false; n_pages],
        }
    }

    pub fn n_pages(&self) -> usize {
        self.n_pages
    }

    pub fn free_pages(&self) -> usize {
        self.free.len()
    }

    pub fn used_pages(&self) -> usize {
        self.n_pages - self.free.len()
    }

    /// Floats in one page, across all layers and both of K/V separately.
    pub fn page_floats(&self) -> usize {
        self.page_floats
    }

    /// Bytes one page occupies in each of the K and V pools.
    pub fn page_bytes(&self) -> usize {
        self.page_floats * std::mem::size_of::<f32>()
    }

    /// Bytes for both pools together.
    pub fn total_bytes(&self) -> usize {
        2 * self.n_pages * self.page_bytes()
    }

    /// Positions the pool can hold in total, ignoring per-sequence rounding.
    pub fn capacity_tokens(&self) -> usize {
        self.n_pages * PAGE_TOKENS
    }

    pub fn alloc(&mut self) -> Result<PageId> {
        match self.free.pop() {
            Some(id) => {
                self.in_use[id as usize] = true;
                Ok(id)
            }
            None => bail!(
                "kv page pool exhausted: all {} pages ({} tokens) are in use",
                self.n_pages,
                self.capacity_tokens()
            ),
        }
    }

    pub fn release(&mut self, id: PageId) -> Result<()> {
        let idx = id as usize;
        if idx >= self.n_pages {
            bail!("page {id} out of range (pool has {} pages)", self.n_pages);
        }
        if !self.in_use[idx] {
            bail!("double free of kv page {id}");
        }
        self.in_use[idx] = false;
        self.free.push(id);
        Ok(())
    }
}

/// One sequence's logical-to-physical mapping.
///
/// `len` is the number of committed positions. Pages are allocated only as the
/// sequence actually grows, so a 7-token request holds one page, not the 64 a
/// full-context reservation would take.
#[derive(Debug, Default, Clone)]
pub struct SequencePages {
    pages: Vec<PageId>,
    len: usize,
}

impl SequencePages {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn pages(&self) -> &[PageId] {
        &self.pages
    }

    pub fn n_pages(&self) -> usize {
        self.pages.len()
    }

    /// Slots allocated but not yet occupied. This is the internal
    /// fragmentation the page size buys, and it is worth reporting rather than
    /// leaving implicit.
    pub fn wasted_slots(&self) -> usize {
        self.pages.len() * PAGE_TOKENS - self.len
    }

    /// Logical position to (physical page, slot within page).
    pub fn translate(&self, pos: usize) -> Result<(PageId, usize)> {
        if pos >= self.len {
            bail!("position {pos} outside sequence of length {}", self.len);
        }
        Ok((self.pages[pos / PAGE_TOKENS], pos % PAGE_TOKENS))
    }

    /// Make room for `extra` more positions and commit them.
    ///
    /// If the pool runs out partway, every page taken by *this* call is
    /// returned before reporting the error, so a failed grow leaves the pool
    /// exactly as it was. Leaking pages on the error path would turn a clean
    /// "pool exhausted" into a slow leak that only shows up much later.
    pub fn grow(&mut self, pool: &mut PagePool, extra: usize) -> Result<()> {
        let target = self.len + extra;
        let needed = target.div_ceil(PAGE_TOKENS);
        let mut taken = Vec::new();
        while self.pages.len() + taken.len() < needed {
            match pool.alloc() {
                Ok(id) => taken.push(id),
                Err(e) => {
                    for id in taken {
                        pool.release(id).expect("page just allocated must be live");
                    }
                    return Err(e);
                }
            }
        }
        self.pages.extend(taken);
        self.len = target;
        Ok(())
    }

    /// Return every page to the pool and reset to empty, so the same struct can
    /// serve the next request.
    pub fn release(&mut self, pool: &mut PagePool) -> Result<()> {
        for id in self.pages.drain(..) {
            pool.release(id)?;
        }
        self.len = 0;
        Ok(())
    }

    /// Page table padded to `stride` entries, for upload to the device.
    ///
    /// Unused entries are filled with 0 rather than left uninitialised. A
    /// kernel must never read them -- `seq_len` bounds the loop -- but a
    /// deterministic filler makes a bug that does read them reproducible
    /// instead of dependent on whatever the buffer last held.
    pub fn table_padded(&self, stride: usize) -> Vec<i32> {
        let mut out = vec![0i32; stride];
        for (i, p) in self.pages.iter().enumerate() {
            out[i] = *p as i32;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shape of the trained 120M model, so the tests exercise real strides.
    const N_LAYER: usize = 12;
    const KV_DIM: usize = 192;

    fn pool(n_pages: usize) -> PagePool {
        PagePool::new(n_pages, N_LAYER, KV_DIM)
    }

    #[test]
    fn allocates_in_order_and_tracks_counts() {
        let mut p = pool(4);
        assert_eq!(p.free_pages(), 4);
        assert_eq!(p.used_pages(), 0);
        assert_eq!(p.alloc().unwrap(), 0);
        assert_eq!(p.alloc().unwrap(), 1);
        assert_eq!(p.free_pages(), 2);
        assert_eq!(p.used_pages(), 2);
    }

    #[test]
    fn page_geometry_matches_layout() {
        let p = pool(64);
        assert_eq!(p.page_floats(), N_LAYER * PAGE_TOKENS * KV_DIM);
        assert_eq!(p.capacity_tokens(), 64 * PAGE_TOKENS);
        // Both pools together must equal the contiguous cache they replace.
        assert_eq!(p.total_bytes(), 2 * N_LAYER * 1024 * KV_DIM * 4);
    }

    #[test]
    fn grows_lazily_one_page_at_a_time() {
        let mut pl = pool(8);
        let mut s = SequencePages::new();
        s.grow(&mut pl, 1).unwrap();
        assert_eq!(s.n_pages(), 1);
        assert_eq!(pl.used_pages(), 1);

        // Filling the rest of the first page must not take a second.
        s.grow(&mut pl, PAGE_TOKENS - 1).unwrap();
        assert_eq!(s.len(), PAGE_TOKENS);
        assert_eq!(s.n_pages(), 1);
        assert_eq!(s.wasted_slots(), 0);

        // One more token crosses the boundary.
        s.grow(&mut pl, 1).unwrap();
        assert_eq!(s.n_pages(), 2);
        assert_eq!(s.wasted_slots(), PAGE_TOKENS - 1);
    }

    #[test]
    fn translation_crosses_page_boundaries() {
        let mut pl = pool(8);
        let mut s = SequencePages::new();
        s.grow(&mut pl, PAGE_TOKENS * 2 + 3).unwrap();
        assert_eq!(s.n_pages(), 3);

        assert_eq!(s.translate(0).unwrap(), (s.pages()[0], 0));
        assert_eq!(s.translate(PAGE_TOKENS - 1).unwrap(), (s.pages()[0], PAGE_TOKENS - 1));
        assert_eq!(s.translate(PAGE_TOKENS).unwrap(), (s.pages()[1], 0));
        assert_eq!(s.translate(PAGE_TOKENS + 1).unwrap(), (s.pages()[1], 1));
        assert_eq!(s.translate(2 * PAGE_TOKENS).unwrap(), (s.pages()[2], 0));
    }

    #[test]
    fn translation_rejects_positions_past_the_end() {
        let mut pl = pool(8);
        let mut s = SequencePages::new();
        s.grow(&mut pl, 5).unwrap();
        assert!(s.translate(5).is_err());
        assert!(s.translate(usize::MAX).is_err());
    }

    #[test]
    fn sequences_never_share_a_physical_page() {
        let mut pl = pool(16);
        let mut a = SequencePages::new();
        let mut b = SequencePages::new();
        let mut c = SequencePages::new();
        // Interleave growth so the allocation order is not trivially blocked.
        for _ in 0..3 {
            a.grow(&mut pl, PAGE_TOKENS).unwrap();
            b.grow(&mut pl, PAGE_TOKENS).unwrap();
            c.grow(&mut pl, PAGE_TOKENS).unwrap();
        }
        let mut all: Vec<PageId> = Vec::new();
        all.extend(a.pages());
        all.extend(b.pages());
        all.extend(c.pages());
        let unique: std::collections::HashSet<PageId> = all.iter().copied().collect();
        assert_eq!(all.len(), 9);
        assert_eq!(unique.len(), 9, "a physical page was handed to two sequences");
    }

    #[test]
    fn release_returns_pages_and_they_are_reused() {
        let mut pl = pool(4);
        let mut s = SequencePages::new();
        s.grow(&mut pl, PAGE_TOKENS * 3).unwrap();
        assert_eq!(pl.free_pages(), 1);

        s.release(&mut pl).unwrap();
        assert_eq!(pl.free_pages(), 4);
        assert_eq!(s.len(), 0);
        assert_eq!(s.n_pages(), 0);

        // The pool must hand the same pages out again rather than growing.
        let mut t = SequencePages::new();
        t.grow(&mut pl, PAGE_TOKENS * 4).unwrap();
        assert_eq!(t.n_pages(), 4);
        assert_eq!(pl.free_pages(), 0);
    }

    #[test]
    fn exhaustion_is_an_explicit_error() {
        let mut pl = pool(2);
        let mut s = SequencePages::new();
        let err = s.grow(&mut pl, PAGE_TOKENS * 3).unwrap_err();
        assert!(err.to_string().contains("exhausted"), "got: {err}");
    }

    #[test]
    fn a_failed_grow_leaves_the_pool_untouched() {
        let mut pl = pool(3);
        let mut s = SequencePages::new();
        s.grow(&mut pl, PAGE_TOKENS).unwrap();
        assert_eq!(pl.used_pages(), 1);

        // Asks for four more pages when only two remain.
        assert!(s.grow(&mut pl, PAGE_TOKENS * 4).is_err());
        assert_eq!(pl.used_pages(), 1, "failed grow leaked pages");
        assert_eq!(pl.free_pages(), 2);
        assert_eq!(s.len(), PAGE_TOKENS, "failed grow changed the sequence");
        assert_eq!(s.n_pages(), 1);
    }

    #[test]
    fn double_free_is_rejected() {
        let mut pl = pool(2);
        let id = pl.alloc().unwrap();
        pl.release(id).unwrap();
        let err = pl.release(id).unwrap_err();
        assert!(err.to_string().contains("double free"), "got: {err}");
    }

    #[test]
    fn releasing_an_out_of_range_page_is_rejected() {
        let mut pl = pool(2);
        assert!(pl.release(7).is_err());
    }

    #[test]
    fn reset_then_reuse_gives_a_clean_sequence() {
        let mut pl = pool(8);
        let mut s = SequencePages::new();
        s.grow(&mut pl, 40).unwrap();
        s.release(&mut pl).unwrap();
        assert!(s.translate(0).is_err(), "released sequence still translates");
        s.grow(&mut pl, 3).unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(s.n_pages(), 1);
    }

    #[test]
    fn padded_table_is_deterministic() {
        let mut pl = pool(8);
        let mut s = SequencePages::new();
        s.grow(&mut pl, PAGE_TOKENS + 1).unwrap();
        let t = s.table_padded(6);
        assert_eq!(t.len(), 6);
        assert_eq!(t[0], s.pages()[0] as i32);
        assert_eq!(t[1], s.pages()[1] as i32);
        assert_eq!(&t[2..], &[0, 0, 0, 0]);
    }

    #[test]
    fn admission_after_reclaim_reuses_the_freed_pages() {
        // The scheduler's core loop: the pool is full, a request retires, and
        // the next one is admitted into exactly what it gave back.
        let mut pl = pool(4);
        let mut a = SequencePages::new();
        let mut b = SequencePages::new();
        a.grow(&mut pl, PAGE_TOKENS * 2).unwrap();
        b.grow(&mut pl, PAGE_TOKENS * 2).unwrap();
        assert_eq!(pl.free_pages(), 0);

        let mut waiting = SequencePages::new();
        assert!(waiting.grow(&mut pl, PAGE_TOKENS).is_err(), "admitted with no pages");

        a.release(&mut pl).unwrap();
        waiting.grow(&mut pl, PAGE_TOKENS * 2).unwrap();
        assert_eq!(pl.free_pages(), 0);
        assert_eq!(waiting.n_pages(), 2);
        assert_eq!(b.len(), PAGE_TOKENS * 2, "retirement disturbed a live sequence");
    }

    #[test]
    fn capacity_accounting_adds_up() {
        let mut pl = pool(10);
        let mut seqs: Vec<SequencePages> = Vec::new();
        for len in [1usize, 17, 33] {
            let mut s = SequencePages::new();
            s.grow(&mut pl, len).unwrap();
            seqs.push(s);
        }
        let held: usize = seqs.iter().map(|s| s.n_pages()).sum();
        let wasted: usize = seqs.iter().map(|s| s.wasted_slots()).sum();
        let live: usize = seqs.iter().map(|s| s.len()).sum();

        assert_eq!(held, pl.used_pages());
        assert_eq!(held + pl.free_pages(), pl.n_pages());
        // Every allocated slot is either occupied or counted as waste.
        assert_eq!(held * PAGE_TOKENS, live + wasted);
    }

    #[test]
    fn interleaved_free_and_realloc_stays_consistent() {
        let mut pl = pool(6);
        let mut a = SequencePages::new();
        let mut b = SequencePages::new();
        a.grow(&mut pl, PAGE_TOKENS * 2).unwrap();
        b.grow(&mut pl, PAGE_TOKENS * 2).unwrap();
        a.release(&mut pl).unwrap();

        let mut c = SequencePages::new();
        c.grow(&mut pl, PAGE_TOKENS * 4).unwrap();
        assert_eq!(pl.free_pages(), 0);

        // b must be untouched by a's release and c's allocation.
        assert_eq!(b.len(), PAGE_TOKENS * 2);
        for p in b.pages() {
            assert!(!c.pages().contains(p), "c took a page b still holds");
        }
    }
}
