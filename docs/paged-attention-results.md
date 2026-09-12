# Paged GQA prefill attention: milestone results

The selected implementation is `exact-q4-k64`: four adjacent query rows and
their four GQA heads share 64-position paged K/V tiles while preserving the
reference f32 reduction order. Three alternating service A/B rounds against a
fresh build of the starting commit improve every long workload in every pair.
Short bursts retain the established packed-prefill performance. The online
softmax candidate passed tensor tolerances but changed seeded generation and
is disabled by default.

This is category **A: an attention service win**. Final numerical, model,
protocol, sustained-service and policy checks have completed. Budget 1024
and chunk cap off remain the defaults; packed serving graphs remain off.
It is custom tiled GQA attention with bounded full-history score vectors in
shared memory. It is not FlashAttention or an online-softmax production path.
The implementation and checks are described in
[the design](paged-attention-design.md); complete round statistics, resources,
and measurement provenance are in [the evidence](paged-attention-evidence.json).
Raw machine-specific logs and orchestration remain outside Git.

All table brackets mean minimum–maximum across the stated trials or outer
rounds. A median of per-round p95 values is not a pooled percentile. Tiny
bursts do not support p99 claims. CUDA-event kernel timing, instrumented
transformer stages, synchronized replay, host wall, and HTTP TTFT are kept
separate.

## 1. Starting HEAD

`135b0953e51801d9748d52672713f19cb48b7ffa`,
`feat(runtime): add packed prefill with bounded token scheduling`.
The prior packed-prefill implementation, scheduler, validation harnesses and
[packed design](packed-prefill-design.md),
[results](packed-prefill-results.md), and
[evidence](packed-prefill-evidence.json) were inspected before changing attention.
The source remained authoritative when the proposed online architecture
conflicted with deterministic generation.

## 2. Clean-tree and baseline identity

The starting working tree was clean. The baseline was freshly built from a
separate archive of the actual starting commit, not an older executable.
All 77 extracted regular files matched a fresh in-memory Git archive by SHA-256:
zero missing files, mismatches, or extras.

| Artifact | SHA-256 |
|---|---|
| Starting-commit archive | `f97dfd30f931ca918b3443aa33aaa65d91fbebe51c1ef187164a768db5e3b012` |
| Baseline release executable | `749af057e62d939605d258a6f0dc362997ed82a6eb2900ac51b1679f3f8f7903` |
| Baseline `kernels.cu` | `3d049614d2e8572f47f2b49560467fb22f5030ae50bfe725e33f968b0244595e` |

Immediately before cleanup, all 77 source files were reverified against the
starting Git archive with zero missing files, mismatches or extras. The
disposable extracted source was then removed with a guarded path check;
binaries and raw evidence remain outside the repository.
No commit has been made.

## 3. Exact old attention algorithm

The unchanged `attention_prefill_paged_impl` serves both old paged and packed
entry points. One 256-thread CTA, eight warps, computes one query head of one
query row; grid `(12,T,1)`. It allocates 4,096 dynamic shared bytes for a
1024-token context and materializes one full causal score vector. The source
declares another 264 bytes of reduction storage; the CUDA driver reports
272 static bytes.

Warp `w` walks history `j=w,w+8,...`. Lane `d` accumulates Q/K dimensions
`d` then `d+32`, followed by the shuffle-add tree 16/8/4/2/1 and scale
`rsqrtf(64)=0.125`. Thread `t` finds maxima over `t,t+256,...`; each warp
reduces, then thread 0 combines eight maxima in ascending warp order.
Each thread evaluates `__expf(score-max)` and accumulates its same strided
positions; a warp reduction followed by a warp-0 reduction of eight
partials yields the denominator. Thread 0 publishes its reciprocal.
Threads 0–63 each accumulate one output component over V in strictly
ascending history order and multiply by that reciprocal. The other 192
threads are idle during this PV loop. There are five CTA barriers.

## 4. Read, write, and reduction structure

K projection writes reusable current-slice scratch, RoPE rotates it, and K
scatters to the paged pool. V projection overwrites that scratch and scatters
to the V pool. Q projection/RoPE precede attention, which rereads both old and
current K/V through the pool. All writes precede attention on the same stream.

Each old CTA reads `L*64` K floats and `L*64` V floats for causal length
`L`: `512*L` logical payload bytes. Shared scores receive the QK writes,
the maximum pass, exponential overwrite/denominator pass, and PV reads.
There is no old global dense score matrix either. Source page-table
references are warp-uniform in the K loop and duplicated across the two V
warps: 96*L scalar lane references before compiler optimization, but only
`ceil(L/16)` distinct page indices. These are source accounting counts,
not DRAM transactions.

## 5. Measured old baseline and power discipline

The first fresh baseline profile reproduced long attention as the largest
individual stage: 4 × 256 attention **6.4846 ms**, outer-job median range
6.302–6.523 ms; packed wall **16.9843 ms**. At 16 × 16, attention was
0.3064 ms and packed wall 8.1459 ms. Those first jobs used an enforced 135 W
envelope and must not be divided by candidate measurements made at another
power envelope. Sections 31–32 record the subsequent paired attribution.

The hardware is an RTX PRO 4000 Blackwell Laptop, sm_120, 60 SMs, 16 GB,
maximum SM clock 3090 MHz. AC power and Windows Best Performance were
independently confirmed, including the September 12 continuation. Dell
Ultra Performance could not be independently verified without administrative
access; no such verification is claimed and no system settings were changed.

The major service comparison used three alternating pairs, A/B, B/A, A/B,
with identical model, quantization, server arguments and warmup. Whole-process
loaded-SM medians baseline→candidate were 2763.5→2790, 2775→2782 and
2771→2775 MHz: changes below 1%. Enforced-power medians were
145.02→146.285, 145→145 and 145→145 W. Each run had ten loaded samples;
maximum SM clock stayed 3090 MHz. Samples cover startup/warmup as well as
workloads and are not aligned to each request or timed region.
They support the repeatable long-service result, not sub-percent claims.

Later micro/profile runs had changing power/clocks. Their complete samples
are retained and those runs are explicitly qualified below. GPU utilization
is device activity, never achieved occupancy. WSL did not provide reliable
Nsight GPU activity/hardware counters.

## 6. GQA redundancy analysis

For `Hq=12,Hkv=3,Dh=64`, each four-head GQA group independently rereads the
same K/V history in the reference. Adjacent query rows also reload their
overlapping histories. Let
`S=sum_i(t_i*p_i+t_i*(t_i+1)/2)` for slice length `t_i` and prefix `p_i`.
Reference logical K+V reads per layer are `6144*S` bytes. At zero prefix,
16 × 16 has S=2,176; 4 × 256 has S=131,584.

The selected CTA stages history once for up to 16 independent distributions.
Its logical K+V payload is `512*Hkv*sum_query_tiles(max_causal_length)`,
charging each tile for its longest valid query and its ragged tail.
This proves transport reuse in the CUDA source. Cache hits already serve
some reference reads, so neither a 4× nor 16× DRAM-bandwidth claim follows.

## 7. Candidate architectures and ablations

The benchmark and tensor harness cover grouped one-row CTAs and multi-row
tiles with query sizes 1/2/4 and history sizes 16/32/64. The online family
uses tile-sized scores and running softmax state. The exact family retains
the reference reduction order and full causal scores. Hybrid current-slice
scratch and page-table caching are independently selectable.

The initial online matrix had 12 variants. It demonstrated substantial
kernel reuse wins but `q2-k32` failed the 16-step generation gate.
The first exact prototype used fixed 1024-score capacity and q1/q2;
q1 lost all six measured shapes, while q2 improved long shapes.
Bounding score capacity and adding q4 were separate measured revisions,
not retroactively assigned to earlier results.

A compact q4 selection ablation used five alternating rounds of 50 launches.
At 4 × 256, replay medians in µs were:

| Variant | Replay µs |
|---|---:|
| reference | 506.233 |
| exact-q2-k32 | 362.452 |
| exact-q4-k16 | 322.807 |
| exact-q4-k32 | 306.845 |
| exact-q4-k64 | 297.928 |
| exact-q4-k32-hybrid | 302.074 |
| exact-q4-k32-cache | 302.550 |

At 16 × 16, q4/k16, q4/k32 and q4/k64 were 11.121, 11.528 and
12.068 µs against reference 20.846 µs. Tile selection balances those shapes
with the stronger long-shape result and real-service acceptance.
The earlier broad bounded-capacity matrix contains large power-related
outliers; it remains in the evidence as exploratory data, not a headline.

## 8. Selected kernel architecture

Production selects **`exact-q4-k64`** for supported paged f32 attention.
One CTA shares K/V across four adjacent query rows in one request segment and
the four query heads belonging to one KV head. The first history traversal
stages K and produces full causal score vectors. Each query warp emulates
the old eight virtual softmax warps, including both reduction trees and
identity values for empty virtual warps. The second traversal reuses the
tile buffer for V and preserves ascending-history accumulation.

Only K/V transport and work assignment change. Per-query score, normalization
and output state remain independent. There is no reduced precision, new
durable KV layout, gathered history, host dense mask, third-party attention
dependency, or Blackwell-specific copy dependency.

## 9. CTA and grid mapping

Descriptors are `int4[packed_start,chunk_len,absolute_start,table_index]`,
derived directly from validated packed slices. Grid is
`(Hkv,ceil(max_chunk/4),request_count)`. A CTA owns one segment, one KV head,
and one four-row query tile. A warp owns one query-row/head distribution.
Fully empty query tiles return uniformly; inactive warps in partial tiles
still take every CTA barrier. Extra grid CTAs for short ragged segments are
launch padding, not semantic padding.

## 10. Threads per block

Selected blocks contain **512 threads, 16 warps**. The q1/q2 ablations use
128/256 threads. The reference uses 256 threads. Each selected warp lane
holds two Q components and computes two output components; only transport
is shared across the 16 distributions.

## 11. Query tile

**Four adjacent query rows** belonging to the same request slice. No tile
crosses request ownership. The causal end is separately computed for each
row; the tile's largest causal end only bounds shared transport.

## 12. History tile

**64 positions** per K or V tile, corresponding to four logical 16-token
pages. All four page indices are translated separately. Physically adjacent
pages are never assumed. Cooperative loads read consecutive dimensions
within a KV head. The same shared allocation holds K, then V, in two passes.

## 13. Shared-memory resources

For history capacity C, selected dynamic shared bytes are
`4*(64*64+4*4*C)`. C is the maximum validated absolute slice end rounded
up to 256, bounded by model page capacity.

| C | Dynamic shared bytes | Calculated blocks/SM | Resident-thread ceiling |
|---:|---:|---:|---:|
| 256 | 32,768 | 2 | 66.7% |
| 512 | 49,152 | 2 | 66.7% |
| 768 | 65,536 | 1 | 33.3% |
| 1024 | 81,920 | 1 | 33.3% |

The page-table-cache experiment adds 256 bytes at the current 64-entry table
stride. The reference uses 4,096 dynamic plus 272 driver-reported static
bytes and permits six blocks/SM. Capacity is allocation topology, not causal
visibility: unused reserved score slots are never read as history.
K/V traversal has two CTA barriers per tile in each pass, plus one initial
barrier only when optional table caching is enabled.

## 14. Registers, local memory, and occupancy limits

CUDA function attributes report selected **56 registers/thread, zero static
shared bytes, zero local bytes/thread**. Reference reports 40 registers and
zero local bytes. Zero local bytes is not an instruction-level PTXAS spill
audit; no achieved occupancy is inferred from the occupancy calculator.
Long-history q4 can win despite a one-block ceiling because staged K/V serves
16 distributions. Resource data for all recorded ablations remains in JSON.

## 15. Online-softmax implementation and rejection

The online experiment maintains independent running `m,l,O` per query head:

```text
m_next = max(m, max(tile_scores))
alpha  = exp(m - m_next)
p_j    = exp(score_j - m_next)
l_next = alpha*l + sum(p_j)
O_next = alpha*O + sum(p_j*V_j)
out    = O/l                       # only after the final visible tile
```

It uses f32 and tile-sized score scratch, rescales both normalization and
output when the maximum changes, and ignores empty later tiles. This is
mathematically stable but reorders f32 operations relative to the old global
softmax/PV reduction. It passed the unchanged tensor tolerance and failed
seeded generation. Production therefore uses exact reduction order;
`q2-k32` and other online names remain explicit diagnostic controls.

## 16. Paged traversal

K/V retain `pool[physical_page][layer][16][kv_dim]`,
`physical_page=request_table[position>>4]`, `slot=position&15`.
Each descriptor identifies its own table, and every logical page crossed by
a tile is independently translated. No full history gather, alternate cache,
or contiguous compatibility copy is used in production.

## 17. Causal masks

A packed row at absolute P consumes only its own request's positions 0..P.
Cooperative transport may stage a later row needed by another query in the
same CTA, but each distribution masks its score and PV reads with its own
causal end. Future poisoning verifies this distinction. A descriptor cannot
select a neighbor's table because ownership is established before upload.

## 18. GQA mapping

`kv_head=query_head/(n_head/n_kv_head)`, unchanged. Heads 0–3, 4–7 and
8–11 use KV heads 0, 1 and 2. Tests assign distinct vectors to every group
and include adversarial running maxima, zeros and shuffled physical pages.

## 19. Hybrid current-K/V experiment

The optional hybrid path keeps current rotated K in additional persistent
scratch and V in existing scratch. History before the slice start reads the
paged pool; current visible positions read `packed_start+position-start`.
K and V still scatter to their canonical pools before attention on the same
stream, so decode and later chunks observe complete durable state.

The q4 hybrid/cache effects were small and shape dependent; neither becomes
a default. Tensor tests require exact bits for every exact hybrid/cache
combination. The selected-hybrid int8 model extension also passes 1,084 exact
sequences, 878 packed calls and 24 fuzz sets with zero final-logit difference. The hybrid avoids a page-pool address path, not the mandatory
pool write or all global-memory reads.

## 20. Additional memory and metadata cost

| Added payload | Bytes |
|---|---:|
| Persistent device segment descriptors, max batch 16 | 256 |
| Host segment staging, max batch 16 | 256 |
| Optional current K, 1024×192 f32 | 786,432 = 0.75 MiB |
| Additional current V | 0; existing scratch |
| Global score/mask/history matrix | 0 |
| Per-call CUDA allocations | 0 |

The optional K buffer is shared across layers. It is allocated only for an
explicit hybrid variant before inference and may remain at the diagnostic
high-water mark until teardown. Container headers, scalar state and existing
graph implementation bookkeeping are not included in payload totals.

One bulk descriptor upload adds `16*active_requests` bytes per packed call;
a singleton uploads 16 bytes. There is no per-row transfer or CPU mask
construction. Descriptor construction is a linear append alongside existing
validated slice preparation. The CPU scheduling planner is unchanged.
The existing prefill traces retain host prepare measurements; those include
validation and uploads and do not isolate a descriptor-only cost.
No new synchronized attention timings or adapter-specific metrics were added.

## 21. Files changed

Six implementation/validation files and four documentation files form the
reviewable change:

| File | Purpose |
|---|---|
| `engine/kernels/kernels.cu` | Online and exact GQA tiled kernels; reference retained |
| `engine/src/gpu.rs` | Variant parsing, launch/resource wrappers, shared-memory setup |
| `engine/src/gpu_model.rs` | Segment ownership, optional scratch, selected dispatch, graph topology |
| `engine/src/gpu_attention_validation.rs` | Tensor oracle, adversarial fixtures, attention benchmark |
| `engine/src/gpu_packed_validation.rs` | Explicit reference model and controlled divergence diagnostic |
| `engine/src/main.rs` | CLI commands, paged CE option, reference storage-only parity control |
| `README.md` | Engineering decision and reproduction commands |
| `docs/paged-attention-design.md` | Arithmetic, ownership, memory and graph contract |
| `docs/paged-attention-results.md` | This 54-item report |
| `docs/paged-attention-evidence.json` | Compact portable numerical/performance provenance |

No runtime scheduler, protocol adapter, vocabulary projection or sampling
algorithm was changed.

## 22. Numerical tensor comparison

The final tensor run uses master seed **20260908**, **53 fixed + 128 fuzz
cases**, 12 online plus 36 exact variants, and **8,688 reference tensor
comparisons**. Each variant compares **86,522,880 f32 output elements**.
All 36 exact variants have **zero max absolute error, zero max relative
error, and identical f32 bits** versus the unchanged reference. The recorded
half-rounding diagnostic likewise finds zero changed rounded values for
exact outputs.

Online variants retain the original element gate
`abs_error <= 2e-5 + 2e-5*abs(reference)`. Their maximum absolute errors
are about 1.10e-6, while relative errors reach about 0.1013 on values close
to zero using a 1e-6 denominator floor. Relative and absolute maxima need
not identify the same element. A sampled f64 CPU oracle independently checks
the mathematical result; GPU-reference equality alone is not its substitute.
No existing tolerance was loosened.

The four-step online model run passed 847 sequences but had maximum final
logit difference 0.005815029. Extending to 16 steps exposed a failure at
request 114, prompt 256, chunk 32, temperature 0.8, top-k 500, seed 141025:

```text
reference: 430 315 43 272 73 21841 7258 43 1921 1168 84 43 42 23 44 17
q2-k32:    430 315 43 272 73 21841 7258 43 8158 8979 25 8979 25 8979 25 8979
```

Teacher-forcing the canonical preceding token at every step isolates the
first divergence. Packed-reference logits are bit-identical to monolithic
reference through all 16 steps. RNG states also match exactly. At token nine,
the same draw 0.9933525323867798 falls in rank 429, but attention drift
reverses near-tied candidates:

| Distribution | Rank 428 | Rank 429 |
|---|---|---|
| Reference | token 8158, logit 7.364331722 | token 1921, logit 7.364254475 |
| Online q2-k32 | token 1921, logit 7.364475727 | token 8158, logit 7.364439011 |

The top-500 set is unchanged, and maximum logit difference at that step is
0.0007739067078. The selected interval changes label even though its CDF
bounds differ only slightly. This is attention-dependent numerical drift,
not packing ownership, changed RNG advancement, or a new sampling defect.
The exact kernel avoids it without changing sampling.

## 23. Exact token sequences

The final int8 selected-path model check compares **1,084 complete generated
sequences**, **878 packed GPU calls**, **24 model fuzz sets**, up to 16
generated steps. Tokens and seeded RNG state agree exactly with explicit
reference attention; maximum final-logit absolute difference is **zero**.

The separate f32-weight run covers **847 sequences**, **620 packed calls**,
four generated steps and zero random model fuzz sets; it also has zero final
logit difference. This is a shorter f32 corpus, not a 16-step f32 claim.
Both retain greedy and mixed top-k 5/40/128/500 paths and canonical independent
vocabulary arithmetic.

A freshly recorded starting-commit HTTP oracle contains 232 independent
reference sequences. The candidate HTTP comparison passes **384 exact
comparisons across 42 workload runs**, including reuse of oracle entries,
24 deterministic fuzz sets and 80 packed batches; every page is reclaimed.

## 24. Fuzz count and seeds

The synthetic tensor seed is 20260908, with 128 randomized compositions/
layouts in the final run. The model corpus has 24 deterministic fuzz sets
in addition to fixed cases; their varying request counts total part of the
1,084 sequences, not 24 sequences. HTTP reference comparison uses the
existing deterministic master seed 20260905 and 24 fuzz sets.
All reported counts are from completed logs, not requested workloads that
stopped at an earlier failure.

## 25. Permutation invariance

The final tensor run performs **6,000 exact composition/permutation/page-remap
checks** across all exact variants. Model boundary groups run three packed
orders for chunk sizes 32/64/128/256 and irregular 37/73/131.
Mixed sampling reverses packing order and verifies exact tokens/RNG.
The online failure is separately preserved and never relabeled an
invariance pass for complete generation.

## 26. Composition invariance

Persistent tensor/model fixtures change active request composition, including
16→1→8→2 and 1/2/4/8/16/1/8/2 reuse patterns. Final rows finish together in
counts 1/2/3/4/8/16 with mixed sampling. The 16-request mixed decode check
shrinks to eight and compares eager/graph and reversed-order execution:
64 exact sequences. Unused descriptor capacity and stale previous metadata
have no semantic effect.

## 27. Page-boundary torture

Fixed fixtures cover causal positions and prompt lengths around 1, 15/16/17,
31/32/33, 63/64/65, 127/128/129, 255/256/257, 511/512 and context-edge
941/1023/1024 where valid. Non-zero and non-page-aligned prefixes,
independently shuffled physical maps, adjacent pages owned by different
requests and heterogeneous slice sizes are included.
The final tensor run additionally passes **960 exact future-poison checks**.

## 28. Cancellation and page reuse

Model cancellation at boundaries 0/1/3/6 followed by immediate page reuse
passes 20 exact survivor sequences. The max_tokens=1 and constrained 32-page
admission/decode-growth paths also pass. HTTP cancellation at zero/one token,
native early disconnect, SDK overload, TUI cancellation and sustained
intentional disconnects all return their pages.

Host validation rejects malformed shapes, absolute/packed overflow,
bad tables/pages, duplicate ownership and invalid final/token indices before
device access; 20 malformed-metadata cases remain in the model harness.
The selected kernel retains no page indices across calls. Submitted stream
work completes before reclamation, and unrecoverable CUDA execution failures
retain the runtime's existing fail-closed behavior.

## 29. Held-out cross entropy

A fresh starting-commit build and the final candidate both report decode
CE **3.720334**, perplexity **41.2782**, over 1,024 scored positions.
Legacy contiguous prefill context 32 reports CE **3.319376**, perplexity
**27.6431**, over **31 scored final positions**, not 1,024 positions.
The latter by itself does not exercise optimized paged attention.

The new `gpu-eval --paged` control explicitly compares reference and selected
int8 paged prefill. All final pairs match at the printed precision:

| Context | Scored final positions | Reference CE | Exact q4/k64 CE | Perplexity, both |
|---:|---:|---:|---:|---:|
| 32 | 31 | 3.319376 | 3.319376 | 27.6431 |
| 256 | 63 | 3.424866 | 3.424866 | 30.7185 |
| 941 | 17 | 4.048477 | 4.048477 | 57.3101 |

Context 32 uses 1,024 input tokens; contexts 256 and 941 use 16,384.
These prefill evaluations score one final target per disjoint context
window. Printed-metric equality supports the larger exact-token corpus;
it is not a full dense teacher-forced evaluation of every input token or
a bit-identity assertion about the unprinted CE accumulator.

## 30. Attention microbenchmark matrix

The dedicated `gpu-prefill-attention-bench` measures synthetic f32 Q/K/V
for layer 1 of a three-layer fixture with the production 12/3/64 head geometry.
It excludes allocations, H2D and D2H. Each of five trials warms eager launches
and a temporary graph of 50 repeated attention kernels; variant order
alternates by trial. Eager-event and eager-wall timing include launch
submission gaps. The table reports temporary-graph CUDA-event µs per kernel
and paired reference/exact ratios, with all five values retained in JSON.

**Envelope limitation:** the final full matrix had 28 loaded samples with
enforced power 115–132.14 W and loaded SM clock 315–2662 MHz
(median 2171 MHz). Values are descriptive measurements with alternating
references, not a flat-clock hardware ranking. The much larger table is
not used alone to enable the default; the separate closely matched service
A/B and q4 ablation support that decision. Same-shape measurements repeated
later in the matrix need not match when clocks move.

| Shape (see offsets below) | Query rows | Σ causal positions | Reference µs | Exact q4/k64 µs | Paired reference/exact |
|---|---|---|---|---|---|
| single-1 | 1 | 1 | 2.157 [2.113–2.268] | 3.286 [3.142–3.361] | 0.68 [0.63–0.69] |
| single-16 | 16 | 136 | 3.414 [3.404–3.820] | 5.613 [5.551–5.670] | 0.61 [0.60–0.69] |
| single-32 | 32 | 528 | 6.397 [6.367–6.447] | 8.317 [8.278–8.379] | 0.77 [0.76–0.77] |
| single-64 | 64 | 2080 | 13.978 [13.946–13.990] | 14.164 [14.122–14.298] | 0.98 [0.98–0.99] |
| single-128 | 128 | 8256 | 37.499 [37.407–38.222] | 36.445 [36.300–37.128] | 1.03 [1.03–1.04] |
| single-256 | 256 | 32896 | 128.763 [127.850–136.381] | 108.381 [100.957–108.592] | 1.26 [1.18–1.28] |
| single-512 | 512 | 131328 | 505.928 [497.219–611.576] | 349.855 [312.800–393.269] | 1.56 [1.44–1.59] |
| single-768 | 768 | 295296 | 1158.227 [1104.963–1458.991] | 818.486 [783.189–854.479] | 1.45 [1.33–1.78] |
| single-941 | 941 | 443211 | 1924.155 [1837.116–1960.075] | 1223.516 [1154.478–1308.145] | 1.56 [1.50–1.67] |
| single-1024 | 1024 | 524800 | 2201.128 [2092.338–2279.274] | 1394.830 [1350.791–1432.618] | 1.56 [1.54–1.63] |
| 16x16 | 256 | 2176 | 21.387 [21.276–21.542] | 12.436 [12.295–13.061] | 1.73 [1.64–1.74] |
| 8x32 | 256 | 4224 | 28.642 [27.834–28.783] | 18.888 [18.244–19.131] | 1.52 [1.50–1.53] |
| 4x64 | 256 | 8320 | 43.765 [43.701–46.138] | 31.203 [29.874–31.693] | 1.46 [1.40–1.46] |
| 4x128 | 512 | 33024 | 140.772 [136.149–148.592] | 109.352 [107.163–112.573] | 1.28 [1.25–1.36] |
| 4x256 | 1024 | 131584 | 576.836 [514.553–639.773] | 360.977 [328.791–394.246] | 1.62 [1.43–1.74] |
| 8x128 | 1024 | 66048 | 281.013 [268.701–307.409] | 186.015 [182.937–197.184] | 1.51 [1.47–1.56] |
| 2x512 | 1024 | 262656 | 1115.538 [1047.991–1204.231] | 645.834 [586.879–665.259] | 1.75 [1.62–2.05] |
| history-64-at-0 | 256 | 8320 | 42.824 [42.474–43.690] | 29.649 [29.202–30.032] | 1.45 [1.43–1.46] |
| history-64-at-64 | 256 | 24704 | 101.524 [100.053–119.992] | 75.078 [73.853–90.679] | 1.35 [1.32–1.35] |
| history-64-at-256 | 256 | 73856 | 290.392 [262.272–335.456] | 221.832 [200.923–252.945] | 1.31 [1.27–1.33] |
| history-64-at-512 | 256 | 139392 | 574.552 [539.932–612.863] | 467.839 [419.103–510.167] | 1.23 [1.06–1.37] |
| history-64-at-960 | 256 | 254080 | 1158.167 [1067.503–1182.150] | 777.009 [753.381–816.936] | 1.45 [1.42–1.49] |
| scheduled-4x512-first | 1024 | 131584 | 599.821 [546.641–663.099] | 401.427 [325.234–411.727] | 1.65 [1.39–1.72] |
| scheduled-4x512-second | 1024 | 393728 | 1746.705 [1735.187–1832.270] | 927.771 [900.934–961.471] | 1.91 [1.85–1.96] |
| heterogeneous | 467 | 212282 | 980.237 [935.359–1015.951] | 665.290 [603.621–681.171] | 1.49 [1.41–1.62] |

The `history-64-at-X` rows are four requests, each 64 current query rows at
absolute offset X. The scheduled 4×512 cases split four 512-token prompts
into 4×256 at offset 0 and 4×256 at offset 256 to respect aggregate capacity
1024. Heterogeneous starts are [0,63,255,512], lengths [17,64,129,257].
Other shapes start at zero.

The tiny singleton kernel loses by about 1–3 µs; single-64 is essentially
the crossover, and packed short shapes already win because CTA/data reuse
differs. There is no simple history-only threshold that maps all shapes.
Since short real-service bursts show no material regression, production
uses the selected path for all supported paged shapes instead of adding a
threshold solely from kernel timing.

## 31. Stage profile: 16 × 16

The extended paired series contains three alternating baseline/candidate
outer jobs per shape. Each job contains three instrumented rounds of
**100 full transformer passes**; each stage is summed over 12 layers per
pass and reduced to the outer-job median. Tables report median/range across
the three outer jobs. `event_total` is independently aggregated, not the
sum of the table's independently chosen medians. These are not HTTP TTFT.

| Stage, summed over 12 layers (ms) | Starting commit | Exact q4/k64 |
|---|---|---|
| embed | 0.0217 [0.0216–0.0219] | 0.0219 [0.0217–0.0241] |
| norm | 0.1992 [0.1953–0.1993] | 0.1971 [0.1931–0.1985] |
| qkv_gemm | 2.4385 [2.4237–2.4522] | 2.4719 [2.4605–2.4975] |
| rope | 0.1565 [0.1539–0.1577] | 0.1540 [0.1532–0.1589] |
| kv_store | 0.1508 [0.1454–0.1531] | 0.1475 [0.1463–0.1498] |
| attention | 0.3601 [0.3537–0.3686] | 0.2399 [0.2347–0.2467] |
| o_gemm | 0.8431 [0.8395–0.8488] | 0.8544 [0.8489–0.8648] |
| residual | 0.1701 [0.1697–0.1720] | 0.1820 [0.1788–0.1906] |
| ffn_gate_up_gemm | 2.0737 [2.0607–2.0858] | 2.0978 [2.0837–2.1264] |
| swiglu | 0.1346 [0.1320–0.1402] | 0.1453 [0.1394–0.1501] |
| down_gemm | 2.1014 [2.0888–2.1128] | 2.1268 [2.1121–2.1644] |
| final_gather | 0.0059 [0.0057–0.0061] | 0.0058 [0.0057–0.0059] |
| final_norm | 0.0080 [0.0079–0.0081] | 0.0082 [0.0081–0.0082] |
| final_head | 0.2397 [0.2384–0.2406] | 0.2419 [0.2404–0.2442] |
| argmax | 0.0412 [0.0410–0.0412] | 0.0416 [0.0416–0.0428] |
| event_total | 8.9366 [8.8939–8.9869] | 8.9297 [8.8833–9.0631] |
| Uninstrumented packed host wall | 9.9448 [9.9285–9.9801] | 9.7936 [9.7885–9.8241] |
| Temporary complete packed replay, synchronized host wall | 8.5576 [8.5465–8.5664] | 8.4354 [8.3883–8.4607] |

**The profile A/B is descriptive only and excluded from causal speedup
acceptance.** Longer runs increased loaded samples to 9–10 per process but
did not match clocks: short-shape candidate loaded-SM medians were 7.52%,
4.33% and 2.03% below their paired baseline. Enforced-power medians differed
by -0.88%, 0% and 0%; maximum SM clock remained 3090 MHz. Event insertion
also perturbs execution and leaves submission gaps inside stage intervals.
The earlier 20-pass series and every raw outer/internal round remain in
the evidence. The closely matched service A/B is the default-acceptance
measurement; this series records remaining costs on the candidate itself.

The replay field historically named `packed_gpu_ms` is measured using
host `Instant` around repeated graph launches and synchronization. Its
temporary complete greedy packed graph includes final-row selection; it
is neither a CUDA-event measurement nor an isolated transformer-body graph.

## 32. Stage profile: 4 × 256

| Stage, summed over 12 layers (ms) | Starting commit | Exact q4/k64 |
|---|---|---|
| embed | 0.0336 [0.0307–0.0337] | 0.0336 [0.0322–0.0343] |
| norm | 0.2858 [0.2750–0.2876] | 0.2502 [0.2479–0.2527] |
| qkv_gemm | 2.7424 [2.7278–2.7509] | 2.6611 [2.6516–2.6675] |
| rope | 0.2313 [0.2249–0.2321] | 0.2119 [0.2064–0.2122] |
| kv_store | 0.2031 [0.2007–0.2038] | 0.1965 [0.1955–0.1974] |
| attention | 5.9924 [5.9294–6.0337] | 3.9534 [3.9401–3.9693] |
| o_gemm | 1.1096 [1.1022–1.1136] | 1.0601 [1.0566–1.0631] |
| residual | 0.3071 [0.3057–0.3081] | 0.2920 [0.2844–0.2954] |
| ffn_gate_up_gemm | 4.0132 [3.9755–4.0313] | 3.8006 [3.7870–3.8181] |
| swiglu | 0.3452 [0.3393–0.3485] | 0.3228 [0.3220–0.3238] |
| down_gemm | 2.7157 [2.6996–2.7285] | 2.5856 [2.5778–2.5923] |
| final_gather | 0.0062 [0.0061–0.0062] | 0.0059 [0.0059–0.0060] |
| final_norm | 0.0082 [0.0082–0.0082] | 0.0077 [0.0074–0.0078] |
| final_head | 0.0838 [0.0831–0.0839] | 0.0810 [0.0808–0.0811] |
| argmax | 0.0412 [0.0410–0.0412] | 0.0389 [0.0387–0.0390] |
| event_total | 18.1238 [17.9461–18.2040] | 15.5039 [15.4320–15.5682] |
| Uninstrumented packed host wall | 19.8164 [19.6558–19.9303] | 18.2750 [18.1029–18.3447] |
| Temporary complete packed replay, synchronized host wall | 17.1017 [16.8936–17.3600] | 15.0218 [14.9494–15.0364] |

This extended series also fails the clock-match requirement for causal
profile A/B attribution: candidate loaded-SM medians were **+3.65%, +5.28%
and +6.85%** relative to the baseline, with 16–18 loaded samples per process.
Enforced-power medians differed by -1.94%, -1.64% and 0%; maximum SM clock
remained 3090 MHz. Unchanged candidate GEMM stages also measure faster.
Therefore the observed reference/candidate stage differences are retained
as descriptive evidence and are not advertised as a precise attention-only
or total-model speedup. No clock-normalization formula is substituted for
a matched experiment.

Within the candidate measurements, attention remains the largest individual
stage at **3.9534 ms**, close to gate/up at **3.8006 ms**. QKV is 2.6611 ms
and down is 2.5856 ms. Attention's cost has not disappeared; the measured
ranking supports further investigation without starting another optimization
inside this milestone. The separate repeated service gains establish the
practical benefit under the better-matched service envelopes.

## 33. Long-service A/B

All requests generate 64 tokens. The baseline is the fresh frozen starting
commit. The candidate explicitly selects `exact-q4-k64`, identical to the
final default behavior; packing/budget/chunk/graphs and all other arguments
match. Three warmups precede timed workloads in each outer run.
Each latency statistic is computed per burst and then summarized across
three alternating pairs. Ranges retain all rounds.

| Workload | Baseline TTFT p50 ms | Exact TTFT p50 ms | Baseline TTFT p95 ms | Exact TTFT p95 ms | Baseline tok/s | Exact tok/s |
|---|---|---|---|---|---|---|
| long-4x256 | 23.66 [23.57–23.99] | 22.18 [21.90–22.18] | 23.83 [23.81–24.11] | 22.21 [22.15–22.31] | 2849 [2839–2867] | 2913 [2904–2913] |
| long-4x512 | 35.59 [34.90–35.93] | 30.04 [29.83–30.24] | 48.12 [47.17–48.13] | 40.42 [40.22–40.55] | 1903 [1902–1926] | 2059 [2058–2066] |
| long-4x941 | 75.19 [73.45–75.24] | 63.80 [63.70–63.92] | 117.37 [115.65–117.74] | 98.26 [98.06–99.24] | 1088 [1081–1099] | 1207 [1206–1209] |
| long-8x512 | 61.09 [59.66–62.26] | 49.06 [48.58–49.47] | 98.40 [95.87–99.39] | 77.06 [76.56–79.46] | 2440 [2418–2485] | 2814 [2728–2820] |

Within-round changes avoid treating separately chosen medians as a pair:

| Workload | Paired TTFT p50 change % | Paired TTFT p95 change % | Paired throughput change % |
|---|---|---|---|
| long-4x256 | -7.43 [-7.57–-5.88] | -7.07 [-7.47–-6.73] | 2.24 [1.58–2.30] |
| long-4x512 | -15.05 [-16.41–-14.54] | -15.74 [-16.01–-14.72] | 8.20 [6.86–8.61] |
| long-4x941 | -15.04 [-15.16–-13.28] | -15.45 [-16.72–-15.04] | 10.87 [9.81–11.89] |
| long-8x512 | -20.48 [-20.53–-17.77] | -20.14 [-21.69–-20.05] | 13.49 [12.85–15.33] |

All four long shapes improve TTFT and throughput in every pair. The 4×256
throughput gain is modest; 4×512, 4×941 and 8×512 provide the clearest
service acceptance evidence. Prefill/attention wall attribution is measured
separately in sections 30–32; HTTP does not insert per-request GPU timers.
No service attention-wall number is fabricated by subtracting TTFT components.

## 34. Short-service regression and mixed shape

| Workload | Baseline TTFT p50 ms | Exact TTFT p50 ms | Baseline TTFT p95 ms | Exact TTFT p95 ms | Baseline tok/s | Exact tok/s |
|---|---|---|---|---|---|---|
| short-2 | 10.12 [9.92–10.12] | 10.11 [10.11–10.34] | 13.01 [12.94–13.06] | 13.21 [13.02–13.23] | 2271 [2266–2272] | 2280 [2234–2298] |
| short-4 | 13.87 [13.86–13.98] | 13.75 [13.70–14.03] | 14.00 [13.97–14.12] | 14.01 [13.79–14.15] | 3709 [3660–3722] | 3718 [3667–3718] |
| short-8 | 15.13 [14.54–15.30] | 15.07 [14.63–15.13] | 15.49 [14.75–15.62] | 15.27 [14.89–15.33] | 5961 [5908–5964] | 5910 [5828–5942] |
| short-16 | 16.54 [15.77–16.56] | 16.06 [15.96–16.29] | 17.05 [16.43–17.14] | 16.71 [16.53–16.84] | 8746 [8439–8958] | 8917 [8901–8925] |
| mixed-8 | 26.52 [25.12–48.51] | 23.39 [22.76–40.39] | 48.72 [45.67–57.08] | 49.31 [40.64–50.51] | 2693 [2670–2817] | 2820 [2812–2846] |

Short burst 8 throughput is consistently about 0.3–1.35% lower, paired
median -0.91%. Short burst 16 throughput has a paired median +1.95% but
overlapping ranges. These are small relative to the unchanged short-prompt
service envelope, not evidence for a new short-workload speed claim.
Mixed-8 p95 is slightly worse at the median (48.72→49.31 ms), and its TTFT
p50 has substantial arrival/composition variation. That tail regression is
retained. The acceptance case is repeatable long service with preserved
short service, not universal improvement.

## 35. Head-of-line and existing-stream gaps

The existing decode stream stays active while arrivals enter. Single-arrival
tests cover 33, 535 and 941 prompt tokens. The mixed `hol-4` arrival set is
33/127/535/941, so 127 is exercised within that mixed arrival workload rather
than mislabeled as an independently repeated single-arrival test.
Gap statistics below cover the disturbance interval.

| Arrival(s) | Attention | Arriving TTFT p50 ms | Arriving TTFT p95 ms | Existing gap p50 ms | Existing gap p95 ms | Existing worst gap ms |
|---|---|---|---|---|---|---|
| stall-33 | baseline | 7.16 [7.16–7.17] | 7.16 [7.16–7.17] | 0.682 [0.622–0.690] | 4.484 [4.342–4.516] | 6.512 [6.512–6.707] |
| stall-33 | candidate | 7.22 [7.04–7.27] | 7.22 [7.04–7.27] | 0.666 [0.646–0.713] | 4.311 [4.241–4.550] | 6.602 [6.592–6.637] |
| stall-535 | baseline | 14.76 [14.58–14.80] | 14.76 [14.58–14.80] | 1.212 [1.210–1.219] | 8.644 [8.152–8.740] | 13.680 [13.534–13.740] |
| stall-535 | candidate | 13.82 [13.78–13.83] | 13.82 [13.78–13.83] | 1.220 [1.218–1.232] | 8.119 [7.952–8.124] | 12.634 [12.584–12.674] |
| stall-941 | baseline | 28.78 [28.49–28.80] | 28.78 [28.49–28.80] | 1.729 [1.701–1.823] | 16.124 [15.819–17.254] | 27.490 [27.274–27.808] |
| stall-941 | candidate | 25.51 [25.44–25.64] | 25.51 [25.44–25.64] | 1.671 [1.666–1.711] | 14.366 [14.251–15.347] | 24.403 [24.375–24.538] |
| hol-4 | baseline | 41.20 [19.57–43.66] | 51.70 [50.54–53.12] | 2.111 [2.050–2.125] | 2.402 [2.375–2.520] | 26.864 [23.541–28.026] |
| hol-4 | candidate | 21.55 [16.29–28.22] | 41.32 [37.95–46.16] | 2.052 [2.037–2.053] | 2.307 [2.301–2.564] | 25.211 [25.177–25.817] |

The 941-token arrival reduces the median worst gap from 27.490 to 24.403 ms.
The 33-token worst gap moves slightly upward (6.512→6.602 ms).
The mixed HOL TTFT p50 range is broad because admission timing changes which
arrivals share a plan; its median reduction must not be presented as an
isolated deterministic 48% kernel gain. Existing p50/p95 gaps stay similar
and the long single-arrival worst gaps improve.

## 36. Chunk-policy re-evaluation

All **three alternating-order rounds of all six configurations completed**:
budget 1024 with cap off/256/128/64, and budgets 512/256 with cap off.
Every configuration runs the same short/mixed/long/stall/HOL workloads.
Rounds 1–2 were recorded September 10; round 3 resumed September 12.
The resumed round's loaded-SM process medians were about 1860–1867 MHz,
compared with about 2760–2775 MHz in rounds 1–2. Configuration medians were
close within each round, but enforced power still varied. Absolute pooled
medians across these sessions do not represent one stable-clock experiment.

The tables therefore compare each configuration with the **same-round
1024/no-cap control**, then report the median and range of the three
percentage changes. Positive latency is worse; positive throughput is better.
Every absolute per-round statistic and envelope is retained in the evidence.

| Budget / cap | Short-16 p95 TTFT Δ% | Mixed-8 p95 TTFT Δ% | 8×512 median TTFT Δ% | 8×512 tok/s Δ% |
|---|---|---|---|---|
| 1024 / off | 0 (control) | 0 (control) | 0 (control) | 0 (control) |
| 1024 / 256 | -1.7 [-4.7–3.3] | 4.7 [-0.4–7.6] | 25.8 [21.0–27.1] | 1.3 [-1.2–2.6] |
| 1024 / 128 | 0.8 [-2.5–1.7] | 56.2 [46.6–56.6] | 55.8 [53.0–56.4] | 5.0 [1.2–5.1] |
| 1024 / 64 | -0.7 [-1.0–-0.1] | 138.9 [135.2–143.6] | 95.2 [88.2–99.7] | -8.1 [-8.3–-3.1] |

| Budget / cap | 941 arrival TTFT Δ% | 941 arrival worst gap Δ% | HOL p95 TTFT Δ% |
|---|---|---|---|
| 1024 / off | 0 (control) | 0 (control) | 0 (control) |
| 1024 / 256 | 82.8 [77.3–85.4] | -44.3 [-44.9–-43.0] | 21.9 [-15.3–29.2] |
| 1024 / 128 | 184.0 [165.0–186.9] | -55.6 [-57.3–-54.7] | 89.9 [78.3–91.7] |
| 1024 / 64 | 376.0 [235.8–376.8] | -62.7 [-73.3–-62.3] | 181.8 [133.2–190.2] |

**Keep chunk cap off.** Caps reduce the longest disturbance of an existing
decode request, but cap 256 raises the 941 arrival's TTFT by 77–85% and
8×512 median TTFT by 21–27%. Cap 128 slightly improves 8×512 aggregate
throughput while materially worsening mixed and long arrival latency.
Cap 64 reduces worst stalls further but has the largest arrival cost.
These are useful explicit latency tradeoffs, not a service-wide improvement
that justifies changing the default. Short-16 alone would miss the loss.

## 37. Aggregate token-budget re-evaluation

The same completed sweep tests budgets **256, 512 and 1024** with chunk
cap off. The following same-round percentage changes use the method and
clock qualifications in section 36:

| Budget / cap | Short-16 p95 TTFT Δ% | Mixed-8 p95 TTFT Δ% | 8×512 median TTFT Δ% | 8×512 tok/s Δ% |
|---|---|---|---|---|
| 1024 / off | 0 (control) | 0 (control) | 0 (control) | 0 (control) |
| 512 / off | 3.4 [-4.2–8.7] | 7.4 [7.3–12.1] | 10.0 [5.9–18.4] | -8.3 [-12.2–-5.3] |
| 256 / off | 51.7 [48.2–55.0] | 64.1 [38.9–65.2] | 131.0 [120.0–135.3] | -25.2 [-29.0–-23.2] |

| Budget / cap | 941 arrival TTFT Δ% | 941 arrival worst gap Δ% | HOL p95 TTFT Δ% |
|---|---|---|---|
| 1024 / off | 0 (control) | 0 (control) | 0 (control) |
| 512 / off | 21.2 [16.0–21.9] | -24.8 [-26.6–-24.2] | 15.4 [11.8–20.5] |
| 256 / off | 83.4 [73.5–87.8] | -44.0 [-46.4–-43.8] | 74.9 [54.0–77.7] |

**Keep aggregate token budget 1024 and packing on.** Budget 512 improves
the 941 arrival's worst disturbance by about 24–27%, but raises that
arrival's TTFT by 16–22%, reduces 8×512 throughput in every round, and
worsens mixed/HOL p95 TTFT. Budget 256 trades still smaller stalls for
larger throughput and latency losses, including roughly 48–55% worse
short-16 p95 TTFT. The default balances short packed throughput, long
arrival latency and bounded work; optional lower budgets remain available
when a deployment deliberately prioritizes its worst stall. No adaptive
scheduler was implemented.

## 38. Optional packed-graph experiment

After selecting attention, the existing temporary packed replay benchmark
was repeated in the extended 100-pass profile series. On the candidate:

| Shape | Eager packed host wall ms | Temporary graph host wall ms | Same-job wall reduction |
|---|---|---|---|
| 16×16 | 9.7936 [9.7885–9.8241] | 8.4354 [8.3883–8.4607] | 14.14% [13.61–14.30] |
| 4×256 | 18.2750 [18.1029–18.3447] | 15.0218 [14.9494–15.0364] | 17.72% [17.42–18.11] |

These are **host `Instant` timings around synchronized replay of a complete
greedy packed graph**, including final-row selection. The legacy output
label `packed_gpu_ms` does not make them CUDA-event or pure GPU times.
Capture, allocation and dynamic service workload amortization are outside
replay timing. Eager wall versus replay mixes launch/submission,
synchronization and timing effects; their difference is not pure launch
cost. The prepared-input replay is not a production cache-hit test, and the
profile power/clock limitations still apply.

The repeatable gap is a useful opportunity for a separate experiment. It
does not quantify the benefit of a clean transformer-body-only cache,
whose topology distribution and capture amortization have not been measured
under real serving traffic.

## 39. Packed-graph implementation decision

Packed serving remains **eager**. A graph cache was investigated at the design
level because temporary replay has a visible benefit, but that benefit does
not establish a clean production cache hit rate or capture amortization.
A transformer-body-only graph could leave metadata upload and final-row
selection/sampling outside. Its actual topology includes aggregate rows,
descriptor count, maximum query-tile grid extent and score capacity, plus a
fixed attention variant/cache lifetime—not only Tpacked. A bounded cache
would need pointer-lifetime invalidation, full-cache eager fallback and
measured capture cost/hit distribution.

Request IDs, page IDs, positions, prompt contents, RNG and scheduler slots
are data and must not become cache keys. That implementation and cache
experiment are deferred; the attention milestone does not require category D.
Existing singleton prefill graphs remain enabled with the corrected
`(length,want_logits,score_capacity)` key, a 64-entry limit and eager
fallback. A variant change synchronizes and invalidates captures/prepared
shape before changing scratch or launch topology.

## 40. Sustained service and cross-protocol proof

The below-capacity run offers 35 requests/s for 60 seconds:
**2,100 submitted, 1,976 completed, 124 requested disconnects, zero errors**,
queue peak 1, no client backpressure, zero runtime failures, all 1024 pages
returned. Aggregate generation is 2,105.75 tok/s including the stated
workload/drain method. Its naturally sparse arrivals produced zero packed
batches during that traffic interval; it is a stability test, not a packed
utilization claim.

The overload run offers 100 requests/s for 30 seconds:
**3,000 submitted, 1,689 completed, 104 requested disconnects, 1,207 expected
HTTP overload responses**, 404 packed batches, zero runtime failures,
queue peak exactly the configured bound 64, all 1024 pages returned.
Its throughput is 3,458.77 tok/s. The harness field named `errors` counts
the expected overload responses; none is silently described as a successful
request or a runtime failure.

Both runs separately prove all five surfaces share one packed batch:
a 941-token blocker is followed by native, OpenAI completion, OpenAI chat,
Anthropic and TUI prompts of 16/16/23/23/16 tokens. Exactly 1,035 prompt rows
are consumed in two prefill batches, the second containing **94 rows across
all five surfaces**. Six final rows cause 24 D2H bytes, peak decode batch is
five, five requests complete and one TUI cancellation is reclaimed.
This is runtime counter/ordering proof of one scheduler, one optimized
attention path and one decode runtime, not simply five simultaneous clients.

## 41. Pressure and backpressure

With **32 KV pages and queue limit 4**, the 24-request pressure burst yields
four accepted responses and 20 HTTP 429 responses. Peak queue is four,
peak page use 24, active/queued return to zero and all 32 pages return.
The complete test includes its setup/warmup activity separately in lifetime
counters; the four accepted burst responses must not be replaced by total
server completions. No malformed metadata reaches the GPU, and failed
request count remains zero.

The sustained overload result above verifies the larger queue bound during
a meaningful arrival interval; model pressure validation separately exercises
admission and decode growth with exact sequences.

## 42. Native regression

All **42 native HTTP checks** pass: request validation, mixed concurrency,
solo/batched output equality, actual shared decode, early disconnect,
cancellation accounting, page return and health after bad inputs.

## 43. OpenAI regression and official SDK

All **125 OpenAI compatibility checks** pass, including official
`openai-python 3.8.0` SDK coverage for model listing, completion/chat,
stream/non-stream equality, role delta, usage, finish reasons and mapped
errors. Overload stays 200/429 with prompt rejection rather than blocking;
all pages return.

## 44. Anthropic regression and official SDK

All **125 Anthropic checks** pass, including the installed official SDK:
message/non-stream/stream consistency, request IDs, usage, stop reasons,
system/multi-turn input, count_tokens, model endpoints and mapped errors.
No Anthropic-specific attention branch was added.

## 45. TUI regression

All **19 TUI smoke checks** pass: actual interactive generation,
cancellation, reuse after cancellation, shared decode with an independent
client, resize, clean shutdown and all pages returned.
The separate 94-row mixed-protocol proof establishes TUI packed participation,
which TUI smoke alone would not prove.

## 46. Complete build and GPU regressions

| Required command/suite | Completed result |
|---|---|
| `cargo test --manifest-path engine/Cargo.toml --locked` | 144 tests passed |
| `cargo test --manifest-path engine/Cargo.toml --features cuda --locked` | 190 tests passed |
| `cargo check --manifest-path engine/Cargo.toml --no-default-features --features tui --locked` | Passed |
| `cargo build --manifest-path engine/Cargo.toml --release --features cuda --locked` | Passed |
| `gpu-validate` | Passed |
| `gpu-graph-check` | Passed |
| `gpu-batch` | Passed, plus the five-slot boundary/admission check |
| `gpu-paged --graph` | Passed, exact paged/contiguous storage parity |
| `gpu-sampling` | Passed |
| `gpu-prefill-check` | Passed |
| `gpu-prefill-graph-check` | Passed across lengths, offsets, sampling and reuse |
| `gpu-packed-prefill-check`, int8 | 1,084 exact sequences, 878 packed calls, 24 fuzz sets |
| `gpu-packed-prefill-check`, f32 | 847 exact sequences, 620 calls, shorter four-step corpus |
| `gpu-prefill-attention-check` | 53 fixed + 128 fuzz cases; 8,688 comparisons |
| Packed HTTP reference comparison | 384 exact comparisons, all pages returned |
| Prefill trace/default and constrained budget | Passed |

`gpu-paged` explicitly selects reference attention because it isolates
storage layout equivalence; the optimized tensor and model gates separately
establish attention correctness. Passing a reference-only storage suite is
not relabeled as optimized-kernel coverage. Existing two unused-variable
warnings were left unchanged. No tolerance was weakened.

## 47. Sanitizer and profiler status

No Compute Sanitizer coverage is claimed for this milestone in WSL.
The existing WDDM environment cannot initialize its CUDA debugger; no risky
system/admin modification was attempted. Nsight GPU activity/counters are
also unavailable/restricted with this WSL/driver/toolchain combination.
The occupancy figures are CUDA resource estimates.

On native Linux with working CUDA debugger support, the portable follow-up is:

```sh
compute-sanitizer --tool memcheck --error-exitcode 99 \
  engine/target/release/llm-engine gpu-prefill-attention-check \
  --fuzz 128 --seed 20260908
CRUCIBLE_PREFILL_ATTN=exact-q4-k64 compute-sanitizer --tool memcheck \
  --error-exitcode 99 engine/target/release/llm-engine \
  gpu-packed-prefill-check "$MODEL" --quant int8 --steps 16 --fuzz 24
```

The selected build uses native Crucible CUDA, without new precision,
dependency or platform-specific runtime requirements.

## 48. Production defaults

| Control | Selected behavior |
|---|---|
| Canonical reference attention | Retained unchanged; `CRUCIBLE_PREFILL_ATTN=reference` |
| Optimized paged prefill | On: `exact-q4-k64`; `exact` is an alias |
| Geometry | head_dim=64 and four query heads per KV head |
| Unsupported geometry / legacy contiguous path | Reference |
| History/row dispatch threshold | None |
| Hybrid current-slice K/V | Off |
| Shared page-table cache | Off |
| Packing | On |
| Aggregate prefill budget | 1024, retained after the completed three-round sweep |
| Chunk cap | Off, retained after the completed three-round sweep |
| Singleton prefill graphs | On, bounded to 64 exact topology entries |
| Packed serving graphs | Off |
| Adaptive scheduler | None |

Reference attention, packing off/on, optional chunking and singleton graphs
remain diagnostic controls. Variant names preserve serious A/B reproducibility
without presenting rejected online paths as recommendations.

## 49. Rejected or deferred approaches

Online softmax is rejected as a default because tiny attention drift changes
seeded output. It is retained only for explicit diagnosis. Fixed-1024 score
capacity and q1 exact tiles lose too much on small workloads; bounded
capacity and multi-row reuse replace them. q4/k16 wins a small packed kernel
but q4/k64 has the stronger long-shape tradeoff. Hybrid K scratch and
page-table caching offer small, shape-dependent effects and remain off.
A tiny-history dispatch threshold lacks a demonstrated service advantage.

Reduced-precision tensor-core attention, asynchronous Blackwell copy paths,
third-party attention dependencies, new durable KV layouts and an adaptive
scheduler were not added. Temporary packed graphs show an opportunity but
production cache complexity/amortization is not yet established.
No next milestone was started.

## 50. Next measured bottleneck

The candidate's extended profile measures long-shape attention at
**3.9534 ms**, gate/up GEMMs at **3.8006 ms**, QKV at **2.6611 ms** and
down at **2.5856 ms**. Attention remains the largest individual stage and
is close to gate/up; claiming it has been eliminated as a bottleneck would
overstate the result. At 16×16, QKV **2.4719 ms**, down **2.1268 ms** and
gate/up **2.0978 ms** dominate attention **0.2399 ms**.

These rankings describe the candidate's own instrumented execution. The
clock-mismatched reference/candidate profile pairs are excluded from causal
speedup acceptance. The temporary complete-graph host timing also records
a submission/replay opportunity, without establishing a production cache
benefit. No GEMM work, graph-cache implementation, or next milestone follows
inside this task.

## 51. README, design, evidence and reproduction

The README engineering section explains the baseline redundancy, exact and
online designs, rejected drift, selected tile/resources, hybrid costs,
crossover, service comparison, retained scheduling defaults and remaining
measured costs.
The design document specifies page ownership, arithmetic order,
synchronization, shape validation, scratch lifetime and graph topology.
This report maps all 54 requested results; portable JSON retains every
round, resource limit and envelope qualification. Raw data, model exports,
profiler traces and machine-specific orchestration are outside Git.

```sh
cargo build --manifest-path engine/Cargo.toml --release --features cuda --locked
engine/target/release/llm-engine gpu-prefill-attention-check --fuzz 128 --seed 20260908
engine/target/release/llm-engine gpu-prefill-attention-bench \
  --variants exact-q4-k64 --iters 50 --trials 5 --seed 20260908
CRUCIBLE_PREFILL_ATTN=exact-q4-k64 engine/target/release/llm-engine \
  gpu-packed-prefill-check "$MODEL" --quant int8 --steps 16 --fuzz 24
CRUCIBLE_PREFILL_ATTN=reference engine/target/release/llm-engine \
  gpu-eval "$MODEL" "$HELDOUT" --quant int8 --graph --paged \
  --tokens 16384 --prefill-ctx 941
```

The benchmark automatically includes reference. For the rejected numerical
case, explicitly select `q2-k32` and add `--diagnose-request 114` to the
packed model check. That teacher-forced diagnostic explains divergence; it
is not an acceptance pass.

## 52. Exact working-tree status and closure

All serialized final phases completed with exit 0, including the extended
hybrid model check and paired paged CE. The longer profile follow-up and
all three rounds of the policy sweep are incorporated above with their
clock limitations. No profile or policy result remains awaiting incorporation.

The disposable baseline source was reverified against the starting Git
archive immediately before guarded removal: **77 identical files, zero
missing files, extras or mismatches**. Generated `__pycache__` directories
were reversibly archived outside the repository. Raw benchmark logs,
profiler output, model files, binaries and orchestration remain outside Git.

The verified `git status --short --untracked-files=all` is:

```text
 M README.md
 M engine/kernels/kernels.cu
 M engine/src/gpu.rs
 M engine/src/gpu_model.rs
 M engine/src/gpu_packed_validation.rs
 M engine/src/main.rs
?? docs/paged-attention-design.md
?? docs/paged-attention-evidence.json
?? docs/paged-attention-results.md
?? engine/src/gpu_attention_validation.rs
```

`git diff --check` passes. HEAD remains
`135b0953e51801d9748d52672713f19cb48b7ffa`; there are exactly the ten
intentional files listed above. **No commit has been made.** The proposed
single commit below completes this milestone; no next milestone was started.

## 53. Proposed single commit title

```text
perf(cuda): add exact tiled GQA prefill attention
```

## 54. Complete proposed commit body

```text
Share paged K/V tiles across four query rows and their four GQA heads
during prefill. Preserve the reference f32 QK, softmax and ascending V
reduction order so packing, page placement and decode neighbors retain
identical logits, generated tokens and seeded RNG behavior.

Select exact-q4-k64 for supported paged attention after paired service
tests improve long-prompt TTFT and throughput without a material short
burst regression. Keep reference attention and explicit online,
hybrid-current-K/V and page-table-cache diagnostic variants. Reject
online softmax as a default after reproducing and isolating a seeded
token change caused by near-tied logit ordering.

Build compact request-segment descriptors from validated packed slices,
bound score shared memory by required history capacity, and include that
capacity in singleton graph topology. Keep one canonical paged KV pool,
bulk metadata uploads and persistent scratch; leave scheduler, protocol,
vocabulary and sampling semantics unchanged. Retain budget 1024, chunk
cap off and eager packed serving after the recorded policy/graph review.

Add deterministic tensor/page/causal/GQA validation, a 25-shape attention
microbenchmark with resource reporting, an explicit paged held-out eval
control and a teacher-forced numerical diagnostic. Document architecture,
ablations, service and stage measurements, power-envelope limits, exact
memory costs, rejected alternatives and the remaining measured stages.

Validation:
- cargo test: 144 passed; CUDA-feature tests: 190 passed
- tui-only check and CUDA release build passed
- existing GPU validation, graph, batching, paged, sampling and prefill
  suites passed
- all 36 exact variants passed bit identity across 53 fixed + 128 fuzz
  tensor cases, including future poisoning and page/layout invariance
- int8 packed model: 1,084 exact sequences, 878 calls, 24 fuzz sets
- f32 packed model: 847 exact sequences in the four-step corpus
- hybrid int8 model: 1,084 exact sequences and zero final-logit error
- paged held-out CE matches printed reference metrics at contexts
  32, 256 and 941; all three rounds of six policy configurations completed
- 384 exact HTTP comparisons; native 42, OpenAI 125, Anthropic 125 and
  TUI 19 checks passed, including official SDKs
- five protocol surfaces shared a 94-row packed batch
- stable traffic, bounded overload, cancellation and 32-page pressure
  checks completed with zero runtime failures and all pages returned

No Compute Sanitizer or achieved-occupancy coverage is claimed under
WSL; portable native-Linux memcheck commands are documented.
```

