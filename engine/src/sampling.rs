//! Token selection: greedy, temperature and top-k, with a deterministic RNG.
//!
//! One implementation, shared by the CLI `generate` command and the
//! continuous-batching runtime. Two samplers with subtly different semantics is
//! the failure this module exists to prevent -- it would show up as the same
//! prompt and seed producing different text depending on which entry point ran
//! it, which is the kind of bug nobody notices until they are trying to
//! reproduce a result.
//!
//! No CUDA types and no feature gate: this is the reference implementation, and
//! `cargo test` exercises it without a GPU.
//!
//! # Two semantics were unified here, deliberately
//!
//! The CLI's greedy branch used `max_by`, which returns the *last* maximum on a
//! tie. The service's device argmax returns the *first*. Both are defensible
//! and they disagreed, so this module standardises on **lowest index**, which
//! is what the GPU kernel does and what the existing service tests pin. The
//! practical effect is limited to exact ties between f32 logits, but having one
//! rule is worth more than preserving an accident.
//!
//! The CLI also compared with `partial_cmp().unwrap()`, which panics on a NaN
//! logit. In a CLI that is a crash; in a shared inference thread it would take
//! down every other request in the batch. Comparisons here are NaN-safe and
//! NaN never wins, matching the device argmax. That is a deliberate hardening,
//! not an accident, and it is tested.
//!
//! # Candidate order is canonical, and that changed generated text once
//!
//! The candidate set used to come out of `select_nth_unstable_by` and be walked
//! in whatever order that left behind: correct as a set, arbitrary as a
//! sequence. Which token a given random threshold lands on depends on that
//! sequence, so the arbitrary order was load-bearing without being defined
//! anywhere.
//!
//! That became untenable once top-k extraction moved onto the GPU, where thread
//! scheduling would otherwise decide the order. Candidates are therefore sorted
//! into one **canonical order** -- higher logit first, ties to the lower token
//! id -- which is a strict total order (token ids are unique, so nothing is
//! ever tied) and so reproducible anywhere, host or device.
//!
//! This changed which token a given seed produces. The candidate *set* is
//! identical and each candidate's probability is identical, so the distribution
//! being sampled from did not change at all -- only which member of it a
//! particular threshold selects. Seeds recorded before this change do not
//! reproduce their old text; seeds recorded after it reproduce everywhere,
//! which is the property actually worth having.

/// xorshift64*, so sampling is reproducible without pulling in a rand crate.
///
/// Each request owns one of these. State must travel with request identity, not
/// with a scheduler slot: a request's sampled sequence cannot be allowed to
/// change because another request joined the batch or finished.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// `seed | 1` because zero is a fixed point of xorshift: seeding with it
    /// would emit zeros forever.
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    /// Uniform in `[0, 1)`, from the top 24 bits.
    pub fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / (1u32 << 24) as f32
    }

    /// Raw state, for asserting that an RNG did or did not advance.
    pub fn state(&self) -> u64 {
        self.0
    }
}

/// Per-request generation settings, fixed for the request's lifetime.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    /// `<= 0` means greedy. This is the switch, not a separate mode flag, so a
    /// request cannot be "sampling with temperature 0".
    pub temperature: f32,
    pub top_k: usize,
    pub seed: u64,
}

/// Matches the CLI's historical defaults so a documented seed reproduces.
pub const DEFAULT_TEMPERATURE: f32 = 0.8;
pub const DEFAULT_TOP_K: usize = 40;
pub const DEFAULT_SEED: u64 = 1234;

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 64,
            // Greedy by default: the service has always been greedy and a
            // caller that omits sampling parameters must keep getting what it
            // got before.
            temperature: 0.0,
            top_k: DEFAULT_TOP_K,
            seed: DEFAULT_SEED,
        }
    }
}

impl GenerationConfig {
    pub fn is_greedy(&self) -> bool {
        !(self.temperature > 0.0)
    }

    pub fn greedy(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            ..Default::default()
        }
    }
}

/// Total order on logits with NaN treated as smallest.
///
/// `partial_cmp().unwrap()` panics on NaN. Sorting NaN to the bottom instead
/// means a NaN logit is never selected, which is exactly what the device argmax
/// does, and one malformed row cannot take down a shared inference thread.
fn cmp_desc(a: f32, b: f32) -> std::cmp::Ordering {
    b.partial_cmp(&a).unwrap_or_else(|| {
        // At least one is NaN. NaN is "smallest", so it sorts last descending.
        match (a.is_nan(), b.is_nan()) {
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            _ => std::cmp::Ordering::Equal,
        }
    })
}

/// Highest logit, ties resolved to the lowest index.
///
/// The same rule as the `argmax_rows_f32` kernel: the winner is the lowest
/// index `i` such that no `j` has `v[j] > v[i]`. NaN never displaces anything.
pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in logits.iter().enumerate() {
        if *x > logits[best] {
            best = i;
        }
    }
    best
}

/// The canonical order over `(token id, logit)` candidates.
///
/// Higher logit first; equal logits resolved to the lower token id. Because
/// token ids are unique this is a *strict total order* -- no two candidates
/// ever compare `Equal` -- which is the property that matters: a correct top-k
/// set has exactly one canonical arrangement, so the host and the GPU can
/// produce it independently and agree by construction rather than by luck.
///
/// Consistent with `argmax`: the first candidate under this order is the
/// highest logit with the lowest index, which is exactly what `argmax` returns.
pub fn cmp_candidates(a: (usize, f32), b: (usize, f32)) -> std::cmp::Ordering {
    cmp_desc(a.1, b.1).then_with(|| a.0.cmp(&b.0))
}

/// Order-preserving map from a logit to an unsigned integer.
///
/// Bitwise throughout, with two cases that exist to match `cmp_desc`: NaN maps
/// below every real value including -inf, and -0.0 maps to +0.0's key because
/// IEEE says they are equal and the order must then fall through to the token
/// id. Key 0 is reserved for NaN and nothing else can produce it -- that would
/// need `0xFFFFFFFF`, itself a NaN.
///
/// The `topk_value_key` device function is this, line for line. Having one
/// definition of the order in two languages is the price of computing it in two
/// places; having two definitions would be the bug.
/// Written with masks rather than branches: the sign of a logit is a coin
/// flip, so a branch here is 50304 unpredictable jumps per row.
fn value_key(v: f32) -> u32 {
    let u = v.to_bits();
    // -0.0 keys as +0.0.
    let u = u & !((u == 0x8000_0000) as u32).wrapping_neg();
    // Flip the sign bit for positives, invert everything for negatives.
    let key = u ^ ((((u as i32) >> 31) as u32) | 0x8000_0000);
    // NaN below everything, including -inf.
    key & (((u & 0x7fff_ffff) <= 0x7f80_0000) as u32).wrapping_neg()
}

/// The canonical order as a single sortable integer, descending.
///
/// High half orders by value, low half by the complement of the token id, so
/// one `u64` comparison decides what `cmp_candidates` decides. Selecting on
/// these rather than on `(usize, f32)` pairs halves the bytes moved and turns
/// each comparison into one instruction instead of a call through a closure --
/// worth doing on a 50304-entry vocabulary, where selection is the dominant
/// cost of the full-logit path.
fn sort_key(v: f32, i: usize) -> u64 {
    debug_assert!(i <= u32::MAX as usize, "token id must fit the key's low half");
    ((value_key(v) as u64) << 32) | (u32::MAX - i as u32) as u64
}

/// The `k` best candidates of `logits`, in canonical order.
///
/// The reference implementation of what the device top-k kernel computes, and
/// the fallback used when a request asks for more candidates than the kernel
/// can hold.
pub fn top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    if logits.is_empty() {
        return Vec::new();
    }
    let mut keys: Vec<u64> = logits
        .iter()
        .enumerate()
        .map(|(i, v)| sort_key(*v, i))
        .collect();
    // top_k = 0 means "no limit" in most APIs; here it clamps to 1, matching
    // the CLI. Values above the vocabulary clamp down to it.
    let k = k.clamp(1, keys.len());
    // Partial selection: the full vocabulary is 50304 entries and only k of
    // them can be chosen, so a full sort would be wasted work. Keys are unique,
    // so the split is exact -- there is no boundary tie left to resolve
    // arbitrarily.
    keys.select_nth_unstable_by(k - 1, |a, b| b.cmp(a));
    keys.truncate(k);
    // ...and only then sort, which is k log k rather than n log n.
    keys.sort_unstable_by(|a, b| b.cmp(a));
    keys.into_iter()
        .map(|key| {
            let i = (u32::MAX - (key as u32)) as usize;
            (i, logits[i])
        })
        .collect()
}

/// Select one token from a candidate set that is already in canonical order.
///
/// The only place temperature, the probability sum and the threshold walk are
/// implemented. Whether the candidates were found by `top_k` on the host or by
/// the top-k kernel on the device, the arithmetic that turns them into a token
/// is this function and nothing else.
pub fn sample_candidates(cands: &[(usize, f32)], cfg: &GenerationConfig, rng: &mut Rng) -> usize {
    if cands.is_empty() {
        return 0;
    }
    // Canonical order puts the best candidate first, so this is `argmax` over
    // the whole row without needing the whole row.
    let best = cands[0].0;

    let max = cands
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, |m, v| if v > m { v } else { m });

    // Accumulate in f64: with a low temperature the exponentials span many
    // orders of magnitude and an f32 sum loses the tail entirely.
    let mut weights: Vec<f32> = Vec::with_capacity(cands.len());
    let mut sum = 0.0f64;
    for (_, v) in cands {
        let scaled = ((*v - max) / cfg.temperature).exp();
        let w = if scaled.is_finite() { scaled } else { 0.0 };
        weights.push(w);
        sum += w as f64;
    }

    if !(sum > 0.0) {
        // Every candidate underflowed or was NaN. Fall back to the best token
        // rather than returning an arbitrary one.
        return best;
    }

    let threshold = rng.next_f32() as f64 * sum;
    let mut acc = 0.0f64;
    for (i, (id, _)) in cands.iter().enumerate() {
        acc += weights[i] as f64;
        if acc >= threshold {
            return *id;
        }
    }
    // Reachable only through floating-point accumulation error.
    cands.last().map(|(i, _)| *i).unwrap_or(0)
}

/// Select one token.
///
/// `temperature <= 0` is greedy and does not touch `rng`, so a greedy request
/// consumes no randomness -- which is what lets a batch mix greedy and sampled
/// requests without one perturbing the other.
pub fn sample(logits: &[f32], cfg: &GenerationConfig, rng: &mut Rng) -> usize {
    if cfg.is_greedy() {
        return argmax(logits);
    }
    if logits.is_empty() {
        return 0;
    }
    sample_candidates(&top_k(logits, cfg.top_k), cfg, rng)
}

/// Validate a configuration, returning a message a caller can act on.
pub fn validate(cfg: &GenerationConfig, vocab: usize) -> Result<(), String> {
    if cfg.temperature.is_nan() {
        return Err("temperature must be a number".into());
    }
    if cfg.temperature > 100.0 {
        return Err(format!(
            "temperature {} is out of range (max 100)",
            cfg.temperature
        ));
    }
    if cfg.top_k > vocab {
        // Clamping is the documented behaviour, so this is informational
        // rather than an error; reject only nonsense that suggests confusion.
        return Ok(());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(temperature: f32, top_k: usize, seed: u64) -> GenerationConfig {
        GenerationConfig {
            max_tokens: 16,
            temperature,
            top_k,
            seed,
        }
    }

    #[test]
    fn rng_is_deterministic_and_in_range() {
        let mut a = Rng::new(1234);
        let mut b = Rng::new(1234);
        for _ in 0..1000 {
            let x = a.next_f32();
            assert_eq!(x, b.next_f32());
            assert!((0.0..1.0).contains(&x), "{x} outside [0,1)");
        }
    }

    #[test]
    fn seed_zero_does_not_stick_at_zero() {
        // xorshift has zero as a fixed point; `seed | 1` is what avoids it.
        let mut r = Rng::new(0);
        let first = r.next_f32();
        let second = r.next_f32();
        assert!(first != 0.0 || second != 0.0, "rng stuck at zero");
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let xs: Vec<f32> = (0..8).map(|_| a.next_f32()).collect();
        let ys: Vec<f32> = (0..8).map(|_| b.next_f32()).collect();
        assert_ne!(xs, ys);
    }

    // --- greedy ---

    #[test]
    fn greedy_picks_the_maximum() {
        let logits = vec![0.1, 5.0, -2.0, 4.9];
        let mut rng = Rng::new(1);
        assert_eq!(sample(&logits, &cfg(0.0, 40, 1), &mut rng), 1);
    }

    #[test]
    fn greedy_breaks_ties_to_the_lowest_index() {
        // Matches argmax_rows_f32 on the GPU. The CLI's old max_by returned the
        // last maximum; this is the deliberate unification.
        let logits = vec![1.0, 9.0, 3.0, 9.0];
        let mut rng = Rng::new(1);
        assert_eq!(sample(&logits, &cfg(0.0, 40, 1), &mut rng), 1);
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn greedy_consumes_no_randomness() {
        // This is what lets greedy and sampled requests share a batch without
        // one perturbing the other.
        let logits = vec![0.1, 5.0, -2.0];
        let mut rng = Rng::new(99);
        let before = rng.state();
        sample(&logits, &cfg(0.0, 40, 99), &mut rng);
        assert_eq!(rng.state(), before, "greedy advanced the RNG");
    }

    #[test]
    fn negative_temperature_is_greedy() {
        let logits = vec![0.1, 5.0, -2.0];
        let mut rng = Rng::new(1);
        assert_eq!(sample(&logits, &cfg(-1.0, 40, 1), &mut rng), 1);
    }

    #[test]
    fn greedy_handles_all_negative_logits() {
        let logits = vec![-9.0, -1.5, -3.0];
        let mut rng = Rng::new(1);
        assert_eq!(sample(&logits, &cfg(0.0, 40, 1), &mut rng), 1);
    }

    #[test]
    fn nan_never_wins_and_does_not_panic() {
        // The old CLI comparator panicked here. In a shared inference thread
        // that would kill every other request in the batch.
        let logits = vec![1.0, f32::NAN, 3.0, f32::NAN];
        assert_eq!(argmax(&logits), 2);
        let mut rng = Rng::new(7);
        assert_eq!(sample(&logits, &cfg(0.0, 40, 7), &mut rng), 2);
        // And in the sampled path.
        let picked = sample(&logits, &cfg(0.8, 4, 7), &mut rng);
        assert!(picked == 0 || picked == 2, "picked a NaN slot: {picked}");
    }

    #[test]
    fn nan_at_index_zero_matches_the_kernel() {
        // Nothing is > NaN, so index 0 is never displaced -- the documented
        // device-argmax behaviour.
        let logits = vec![f32::NAN, 1.0, 2.0];
        assert_eq!(argmax(&logits), 0);
    }

    #[test]
    fn infinities_are_handled() {
        let logits = vec![1.0, f32::INFINITY, 2.0];
        assert_eq!(argmax(&logits), 1);
        let mut rng = Rng::new(3);
        assert_eq!(sample(&logits, &cfg(0.7, 3, 3), &mut rng), 1);
    }

    // --- top-k ---

    #[test]
    fn top_k_one_is_deterministic_and_equals_greedy() {
        let logits = vec![0.5, 2.0, 1.0, 1.9];
        for seed in [1u64, 42, 9999] {
            let mut rng = Rng::new(seed);
            assert_eq!(sample(&logits, &cfg(1.0, 1, seed), &mut rng), 1);
        }
    }

    #[test]
    fn top_k_zero_clamps_to_one() {
        let logits = vec![0.5, 2.0, 1.0];
        let mut rng = Rng::new(5);
        assert_eq!(sample(&logits, &cfg(1.0, 0, 5), &mut rng), 1);
    }

    #[test]
    fn top_k_larger_than_the_vocabulary_clamps_down() {
        let logits = vec![0.5, 2.0, 1.0];
        let mut rng = Rng::new(5);
        let picked = sample(&logits, &cfg(1.0, 10_000, 5), &mut rng);
        assert!(picked < 3);
    }

    #[test]
    fn top_k_restricts_the_candidate_set() {
        // Index 3 is far below the top two and must never be chosen at k=2.
        let logits = vec![5.0, 4.9, -20.0, -30.0];
        for seed in 0..200u64 {
            let mut rng = Rng::new(seed);
            let picked = sample(&logits, &cfg(1.0, 2, seed), &mut rng);
            assert!(picked < 2, "seed {seed} picked {picked} outside the top 2");
        }
    }

    // --- temperature ---

    #[test]
    fn low_temperature_concentrates_on_the_best_token() {
        let logits = vec![3.0, 1.0, 0.5];
        let mut hits = 0;
        for seed in 0..300u64 {
            let mut rng = Rng::new(seed);
            if sample(&logits, &cfg(0.05, 40, seed), &mut rng) == 0 {
                hits += 1;
            }
        }
        assert!(hits > 290, "low temperature was not concentrated: {hits}/300");
    }

    #[test]
    fn high_temperature_spreads_the_distribution() {
        let logits = vec![3.0, 1.0, 0.5];
        let mut seen = std::collections::HashSet::new();
        for seed in 0..300u64 {
            let mut rng = Rng::new(seed);
            seen.insert(sample(&logits, &cfg(5.0, 40, seed), &mut rng));
        }
        assert!(seen.len() >= 2, "high temperature never left the top token");
    }

    #[test]
    fn sampling_is_reproducible_for_a_seed() {
        let logits: Vec<f32> = (0..500).map(|i| ((i as f32) * 0.7).sin() * 3.0).collect();
        let run = || {
            let mut rng = Rng::new(4242);
            let c = cfg(0.8, 40, 4242);
            (0..64).map(|_| sample(&logits, &c, &mut rng)).collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn a_different_seed_gives_a_different_sequence() {
        let logits: Vec<f32> = (0..500).map(|i| ((i as f32) * 0.7).sin() * 3.0).collect();
        let run = |seed: u64| {
            let mut rng = Rng::new(seed);
            let c = cfg(1.0, 50, seed);
            (0..64).map(|_| sample(&logits, &c, &mut rng)).collect::<Vec<_>>()
        };
        assert_ne!(run(1), run(2));
    }

    #[test]
    fn identical_values_across_the_vocabulary_still_select() {
        // Uniform logits: every candidate is equally likely and nothing may
        // fall through the accumulation loop.
        let logits = vec![1.0f32; 128];
        for seed in 0..100u64 {
            let mut rng = Rng::new(seed);
            let picked = sample(&logits, &cfg(1.0, 128, seed), &mut rng);
            assert!(picked < 128, "seed {seed} produced {picked}");
        }
    }

    #[test]
    fn extreme_negative_logits_do_not_produce_an_invalid_token() {
        // Every exponential underflows to zero; the fallback must still return
        // a valid index rather than an arbitrary one.
        let logits = vec![-1e30, -1e30, -1e30];
        let mut rng = Rng::new(11);
        let picked = sample(&logits, &cfg(0.001, 3, 11), &mut rng);
        assert!(picked < 3);
    }

    #[test]
    fn config_reports_greedy_correctly() {
        assert!(GenerationConfig::greedy(8).is_greedy());
        assert!(cfg(0.0, 40, 1).is_greedy());
        assert!(cfg(-0.5, 40, 1).is_greedy());
        assert!(!cfg(0.0001, 40, 1).is_greedy());
        assert!(cfg(f32::NAN, 40, 1).is_greedy(), "NaN temperature must not sample");
    }

    #[test]
    fn validation_rejects_nonsense_temperature() {
        let vocab = 50304;
        assert!(validate(&cfg(0.8, 40, 1), vocab).is_ok());
        assert!(validate(&cfg(0.0, 40, 1), vocab).is_ok());
        assert!(validate(&cfg(f32::NAN, 40, 1), vocab).is_err());
        assert!(validate(&cfg(1000.0, 40, 1), vocab).is_err());
        // Clamping is documented, so an oversized top_k is accepted.
        assert!(validate(&cfg(0.8, 999_999, 1), vocab).is_ok());
    }

    // --- canonical candidate order ---
    //
    // These pin the contract the top-k kernel has to reproduce. If one of them
    // starts failing, the GPU and the host have stopped agreeing about what
    // "the top k candidates" means, which is a silent-divergence bug rather
    // than a crash.

    /// Brute-force top-k: sort everything, take k. Slow and obviously correct.
    fn reference_top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut all: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        all.sort_by(|a, b| cmp_candidates(*a, *b));
        all.truncate(k.clamp(1, logits.len()));
        all
    }

    fn same(a: &[(usize, f32)], b: &[(usize, f32)]) -> bool {
        a.len() == b.len()
            && a.iter().zip(b).all(|(x, y)| {
                x.0 == y.0 && (x.1 == y.1 || (x.1.is_nan() && y.1.is_nan()))
            })
    }

    #[test]
    fn the_packed_key_orders_exactly_as_the_comparator_does() {
        // top_k selects on packed keys for speed; cmp_candidates is the
        // definition. If they ever disagree, the fast path silently returns a
        // different candidate set from the one the docs describe -- and from
        // the one the GPU kernel, which uses the same key, returns.
        let values = [
            0.0f32, -0.0, 1.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN,
            f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 3.4e38, -3.4e38, 1e-30, -1e-30,
        ];
        for (ia, a) in values.iter().enumerate() {
            for (ib, b) in values.iter().enumerate() {
                if ia == ib {
                    continue;
                }
                let by_key = sort_key(*b, ib).cmp(&sort_key(*a, ia));
                let by_cmp = cmp_candidates((ia, *a), (ib, *b));
                assert_eq!(by_key, by_cmp, "{a} at {ia} vs {b} at {ib}");
            }
        }
    }

    #[test]
    fn top_k_matches_brute_force_on_random_rows() {
        for seed in 0..40u64 {
            let mut r = Rng::new(seed);
            let logits: Vec<f32> = (0..1000).map(|_| r.next_f32() * 20.0 - 10.0).collect();
            for k in [1usize, 2, 5, 40, 128, 999, 1000] {
                let got = top_k(&logits, k);
                let want = reference_top_k(&logits, k);
                assert!(same(&got, &want), "seed {seed} k {k}");
            }
        }
    }

    #[test]
    fn canonical_order_is_descending_by_value_then_ascending_by_id() {
        let logits = vec![1.0, 3.0, 3.0, 2.0, 3.0];
        let got = top_k(&logits, 4);
        // Three-way tie at 3.0 resolves by id: 1, 2, 4. Then 2.0 at index 3.
        assert_eq!(got.iter().map(|c| c.0).collect::<Vec<_>>(), vec![1, 2, 4, 3]);
    }

    #[test]
    fn ties_at_the_k_boundary_select_the_lower_id() {
        // Four values tied at 5.0; k=2 must take ids 0 and 1, never 2 or 3.
        let logits = vec![5.0, 5.0, 5.0, 5.0];
        let got = top_k(&logits, 2);
        assert_eq!(got.iter().map(|c| c.0).collect::<Vec<_>>(), vec![0, 1]);
    }

    #[test]
    fn negative_zero_ties_with_positive_zero() {
        // IEEE says -0.0 == 0.0, so these are tied and resolve by id. The
        // device key mapping has to agree, which is why this is pinned here.
        let logits = vec![-1.0, 0.0, -0.0, -2.0];
        let got = top_k(&logits, 2);
        assert_eq!(got.iter().map(|c| c.0).collect::<Vec<_>>(), vec![1, 2]);
        let flipped = vec![-1.0, -0.0, 0.0, -2.0];
        let got = top_k(&flipped, 2);
        assert_eq!(got.iter().map(|c| c.0).collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn nan_sorts_below_negative_infinity() {
        let logits = vec![f32::NAN, f32::NEG_INFINITY, f32::NAN, -1.0e38];
        let got = top_k(&logits, 4);
        assert_eq!(got.iter().map(|c| c.0).collect::<Vec<_>>(), vec![3, 1, 0, 2]);
    }

    #[test]
    fn infinity_sorts_above_everything() {
        let logits = vec![1.0e38, f32::INFINITY, 0.0, f32::INFINITY];
        let got = top_k(&logits, 3);
        assert_eq!(got.iter().map(|c| c.0).collect::<Vec<_>>(), vec![1, 3, 0]);
    }

    #[test]
    fn maxima_at_the_first_and_last_index_are_both_found() {
        let mut logits = vec![0.0f32; 500];
        logits[0] = 9.0;
        logits[499] = 8.0;
        let got = top_k(&logits, 2);
        assert_eq!(got.iter().map(|c| c.0).collect::<Vec<_>>(), vec![0, 499]);
    }

    #[test]
    fn sample_candidates_agrees_with_sample_on_full_logits() {
        // The device path calls sample_candidates with candidates the kernel
        // found; the host path calls sample. Identical candidates must give
        // identical tokens, or the two paths generate different text.
        let logits: Vec<f32> = (0..2000).map(|i| ((i as f32) * 0.37).sin() * 6.0).collect();
        for (temp, k) in [(0.8f32, 40usize), (1.0, 5), (0.3, 128), (2.0, 1)] {
            let c = cfg(temp, k, 77);
            let mut a = Rng::new(77);
            let mut b = Rng::new(77);
            let cands = top_k(&logits, k);
            for step in 0..64 {
                let via_full = sample(&logits, &c, &mut a);
                let via_cands = sample_candidates(&cands, &c, &mut b);
                assert_eq!(via_full, via_cands, "temp {temp} k {k} step {step}");
            }
        }
    }

    #[test]
    fn sample_candidates_never_leaves_the_candidate_set() {
        let logits: Vec<f32> = (0..300).map(|i| ((i as f32) * 1.1).cos() * 4.0).collect();
        let cands = top_k(&logits, 7);
        let ids: std::collections::HashSet<usize> = cands.iter().map(|c| c.0).collect();
        for seed in 0..300u64 {
            let mut rng = Rng::new(seed);
            let picked = sample_candidates(&cands, &cfg(1.5, 7, seed), &mut rng);
            assert!(ids.contains(&picked), "seed {seed} escaped the candidate set");
        }
    }

    #[test]
    fn an_all_nan_candidate_set_still_returns_a_valid_token() {
        let cands = vec![(0usize, f32::NAN), (1, f32::NAN)];
        let mut rng = Rng::new(3);
        assert_eq!(sample_candidates(&cands, &cfg(0.8, 2, 3), &mut rng), 0);
    }
}
