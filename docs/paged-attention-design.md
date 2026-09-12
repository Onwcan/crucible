# Ragged paged prefill attention: design and reference audit

This is the design record for the attention milestone starting at
`135b0953e51801d9748d52672713f19cb48b7ffa`,
`feat(runtime): add packed prefill with bounded token scheduling`.
The reference audit below describes that committed source. The selected
production path is **`exact-q4-k64`**: four adjacent query rows and their four
GQA heads share 64-position K and V tiles, with f32 arithmetic preserving the
reference reduction order. It retains full causal score vectors in bounded
CTA shared memory. This is tiled GQA attention, not online attention or
FlashAttention. The online `q2-k32` experiment passed tensor tolerances but
changed the ninth generated token in an adversarial model run and was rejected.
All 36 exact variants passed tensor bit-identity checks; the selected model
passed 1,084 complete sequences and three paired service rounds show repeatable
long-prompt gains. The final numerical, protocol, sustained-service and
three-round scheduler checks have completed. Measurements and their
limitations are recorded in
[the results report](paged-attention-results.md) and
[portable evidence](paged-attention-evidence.json).

The preceding [packed-prefill contract](packed-prefill-design.md),
[measurement report](packed-prefill-results.md), and
[evidence](packed-prefill-evidence.json) establish the existing scheduler and
motivation. Their 4 × 256 attention attribution of 6.297 ms was an
instrumented result under that experiment's clocks. A fresh starting-commit
build measured 6.485 ms, with outer-job medians from 6.302 to 6.523 ms.
Those jobs used a 135 W envelope; the initial online microbenchmark used
145–151.56 W, so those separate runs do not form a matched A/B. The prior 16 × 16
profile was predominantly transformer GEMMs; improving long history must not
erase that workload's established packing gain.

## Baseline identity

The separately extracted baseline source was compared, read-only, against an
in-memory `git archive` of the full starting commit. All **77 regular files**
matched SHA-256 content hashes, with zero missing files, mismatches, or extra
files. A fresh release build from that extracted tree completed, and its
stage profiles and held-out evaluations are recorded in the results report.
Immediately before guarded cleanup, all 77 files were reverified against
the starting archive with zero mismatches. The disposable extracted source
was removed; its freshly built reference executable and raw evidence remain
outside Git.

| Item | SHA-256 |
|---|---|
| Starting-commit tar archive | `f97dfd30f931ca918b3443aa33aaa65d91fbebe51c1ef187164a768db5e3b012` |
| `engine/kernels/kernels.cu` | `3d049614d2e8572f47f2b49560467fb22f5030ae50bfe725e33f968b0244595e` |
| `engine/src/gpu.rs` | `42a46067f3430a00f5eb4e2077d51b7c1e56baa8e034be1ce2cfe51816aa8a87` |

The source anchors are `attention_prefill_paged_impl`, its two public prefill
entry points, and the reduction helpers in
[kernels.cu](../engine/kernels/kernels.cu); launch wrappers and
`PrefillAttentionVariant` in [gpu.rs](../engine/src/gpu.rs); and
`queue_prefill_transformer_impl`, descriptor upload, and scratch ownership in
[gpu_model.rs](../engine/src/gpu_model.rs). Use the starting commit when
reproducing the old algorithm; these relative links follow the reviewed tree.

## Geometry and durable state

For the 120M model, `D=768`, query heads `Hq=12`, KV heads `Hkv=3`, head width
`Dh=64`, `KD=192`, layers `N=12`, and context capacity `C=1024`. The group size
is `G=Hq/Hkv=4`. Heads 0–3 use KV head 0, heads 4–7 use KV head 1, and heads
8–11 use KV head 2. The mapping is division, not interleaving.

K and V are separate f32 pools with the same canonical layout:

```text
pool[physical_page][layer][16][KD]
physical_page = request_page_table[absolute_position >> 4]
slot          = absolute_position & 15
element_offset = ((physical_page * N + layer) * 16 + slot) * KD
                 + kv_head * Dh + dimension
```

The default table stride is `ceil(C/16)=64` int32 entries. Logical adjacent
pages need not be physically adjacent. Decode, later chunks, and all candidate
prefill paths retain this one durable representation. Neither a contiguous
history gather nor a prefill-only cache is part of the production design.

Q and attention output have shape `[Tpacked,Hq,Dh]`. For descriptor `i`, local
slice row `j` maps to packed row `packed_start_i+j` and absolute position
`absolute_start_i+j`. Only positions `0..=absolute_position` of descriptor
`i` are visible. Tensor adjacency cannot extend that interval or select
another request's page table.

## Exact old algorithm

Both `attention_prefill_paged_f32` and `attention_prefill_packed_f32` call the
same `attention_prefill_paged_impl`. The singleton entry supplies causal
length `params[PARAM_PREFILL_POS]+blockIdx.y+1`. The packed entry supplies
`positions[row]+1` and table `page_tables+row_request[row]*table_stride`.
The arithmetic is otherwise shared.

| Property | Starting-commit behavior |
|---|---|
| Grid | `(Hq, query_rows, 1)`; one CTA per query row and query head |
| Block | 256 threads, eight warps |
| Dynamic shared memory | `4*C` bytes, 4,096 bytes at context 1024, even for short causal lengths |
| Additional declared shared storage | `warp_max[32]`, reduction `partial[32]`, `smax`, `ssum`: 264 bytes nominal |
| Score storage | One shared f32 score per visible history position; overwritten by unnormalized exponentials |
| K/V reuse | No explicit sharing across query heads or query rows |
| Output | 64 f32 components per CTA; threads 0–63 compute them |
| CTA synchronization | Five barriers: after QK, two for maximum, one inside denominator reduction, one after reciprocal publication |

The nominal shared total is 4,360 bytes. Declared storage and compiler-reported
static shared size need not be identical because the compiler can reuse
lifetimes; compiled resource attributes must be reported separately.

For causal length `L`, the floating-point order is:

1. Warp `w` handles history positions `j=w,w+8,w+16,...`. Within a position,
   lane `d` first accumulates dimension `d`, then dimension `d+32`, into one
   f32 dot partial. `warp_reduce_sum` uses shuffle-down additions at offsets
   16, 8, 4, 2, 1. Lane 0 writes `dot*rsqrtf(64)` to `scores[j]`.
2. Thread `t` takes maxima over `j=t,t+256,...`. Each warp reduces with the
   same descending shuffle offsets using `fmaxf`; lane 0 publishes its warp
   maximum. Thread 0 then takes the maximum of the eight warp results in
   ascending warp order and publishes `smax`.
3. Each thread revisits its `t+256k` positions, evaluates
   `__expf(scores[j]-smax)`, overwrites the score, and sums those exponentials
   in ascending position order. A warp sum followed by a warp-0 sum of the
   eight shared partials (remaining lanes zero) produces the denominator.
   Thread 0 publishes its reciprocal as `ssum`.
4. For each output dimension `d`, one thread accumulates
   `scores[j]*V[j,d]` in strictly increasing `j=0..L-1`, then multiplies by
   `ssum`. There is no split-history output reduction in this prefill kernel.

These describe source-level f32 operations. The existing NVRTC compilation
options can fuse multiply-adds; instruction-level identities require the
compiled kernel and numerical comparison, not algebraic reassociation.

### Read, write, and lookup accounting

Per `(query row, query head)` CTA, the source performs `L*Dh` K-element reads
and `L*Dh` V-element reads: `2*L*64*4 = 512*L` logical payload bytes. Each of
the four sibling query-head CTAs repeats those reads for the same KV head.
Different query rows repeat their overlapping histories independently.
Cache hits may already serve some of that reuse; these counts are **not DRAM
bytes, transaction counts, or achieved bandwidth**.

The Q expression is evaluated for `L*Dh` source-level operands too, although
its fixed per-thread addresses permit compiler/register/cache reuse. For the
page table, the K loop names one lookup per participating lane and position,
and the V loop names one per output dimension and position. At `Dh=64`, that
is `32*L+64*L=96*L` scalar lane-level references, or three warp-uniform
lookup instructions per position before compiler optimization. Only
`ceil(L/16)` distinct table entries are useful. A broadcast or cache hit must
not be counted as another memory transaction.

Scores incur `L` writes during QK, `L` reads for maximum, `L` reads and writes
for exponentiation, and `Dh*L` reads during PV. They stay in shared memory;
the baseline does not allocate a global `[T,T]` attention matrix.

For request slice length `t_i` with prefix `p_i`, define
`S=sum_i(t_i*p_i+t_i*(t_i+1)/2)`. Baseline logical K+V bytes per layer are
`2*Dh*4*Hq*S = 6144*S` for this model. Thus 16 × 16 at zero prefix has
`S=2176`, while 4 × 256 has `S=131584`: equal-cost assumptions based only on
aggregate rows are invalid. These are accounting formulas, not measured
bandwidth predictions.

### Current-slice cache round trip

The old transformer loop projects K into reusable `[Tpacked,KD]` scratch,
applies RoPE, and scatters it into the paged K pool. V projection then
overwrites that same scratch and scatters into the V pool. Q projection and
RoPE follow. Attention reads all visible K/V, including the current slice,
from those pools. The same stream orders every scatter before attention;
future rows may already have been written but remain causally masked.

This establishes the current-slice reread opportunity. It does not establish
that avoiding a pool address calculation will reduce device-memory traffic:
the alternative also reads f32 global scratch and must still write the pool.

## Shared descriptor and tiled launch architecture

Both tiled families explicitly specialize `Dh=64` and `Hq=4*Hkv` with
f32 Q/K/V and f32 accumulation. It has no attention WMMA, fp16/bf16 conversion,
third-party attention dependency, or Blackwell-only asynchronous staging.
Unsupported geometry retains the reference path at model dispatch.

A persistent descriptor contains four int32 values:

```text
[packed_start, chunk_len, absolute_start, table_index]
```

The host builds these directly from validated request slices; the kernel does
not discover segments by scanning rows. Owners/positions remain available to
RoPE and cache stores. Active descriptors are bulk-uploaded before execution
and remain device data under graph replay.

For `Q_TILE` in 1/2/4 and `KV_TILE` in 16/32/64:

```text
grid = (Hkv, ceil(max_chunk / Q_TILE), request_count)
CTA  = one descriptor × one KV head × Q_TILE adjacent query rows
warp = one query row × one of that KV head's four query heads
threads_per_CTA = 32 * 4 * Q_TILE = 128 / 256 / 512
```

Every warp owns one query-head distribution and two output components per
lane. Q's two components per lane are loaded once before history traversal.
Softmax state and output accumulation are never shared between heads or
query rows; only K/V transport is shared. Production uses `Q_TILE=4`,
`KV_TILE=64`, 512 threads, and grid `(Hkv,ceil(max_chunk/4),request_count)`.

A CTA whose query tile starts past the descriptor's end returns uniformly.
In a partial query tile, inactive warps skip Q/output accesses but still
participate in every CTA barrier. The grid may contain such empty CTAs for
short descriptors because it uses `max_chunk`; this is launch padding, not
semantic token padding or cross-request attention.

### Rejected online candidate: traversal and normalization

The CTA traverses logical positions from zero to the largest valid query's
causal end in `KV_TILE` steps. Cooperative lanes stage K and V arrays
`[KV_TILE,64]`, translating every constituent logical page independently.
Consecutive lanes load consecutive dimensions within a head. A tile spanning
32 or 64 positions never assumes adjacent physical pages.

Within each warp, valid scores use the same two per-lane dot components and
shuffle sum as the reference. The candidate stores only the current tile's
scores/probabilities in shared memory, `[Q_TILE*4,KV_TILE]`. It removes the
full-history score vector, not all score materialization. Each query's
`visible=min(KV_TILE,causal_length-tile_start)` bounds every score and PV read;
the later rows staged for another query in that CTA are not visible to it.

For one independent distribution, initialize `m=-infinity`, `l=0`, `O=0`.
For a nonempty visible tile with scores `s_j`, compute:

```text
m_next = max(m, max_j(s_j))
alpha  = exp(m - m_next)
p_j    = exp(s_j - m_next)
l_next = alpha*l + sum_j(p_j)
O_next = alpha*O + sum_j(p_j*V_j)
```

For an active query, the first tile contains at least one key, so its finite
maximum makes the initial `alpha` zero. Empty later tiles leave its state
unchanged. Division occurs once at the end, `out=O/l`. The running output is
rescaled whenever the running maximum changes; normalization is not dropped
between tiles.

Tile maxima and sums use warp reductions. Each lane rescales its two output
components, then accumulates visible V positions in ascending order within
the tile. This order differs from the reference's global softmax and final
ascending-history sum, even when the QK dots match. Bit identity is therefore
an evidence question. The exact old kernel remains the numerical oracle.

There are two CTA barriers per history tile: after cooperative loading and
before overwriting shared storage for the next tile. Warp synchronization
separates score writes, probability writes, and their consumers. Optional
page-table staging adds one initial CTA barrier.

### On-chip storage and reuse accounting

The candidate declares no static shared arrays. Dynamic shared payload is
`4*(2*KV_TILE*64 + Q_TILE*4*KV_TILE)` bytes, plus `4*table_stride` if table
caching is enabled.

| Query rows/CTA | Threads/CTA | KV tile 16, bytes | KV tile 32, bytes | KV tile 64, bytes |
|---:|---:|---:|---:|---:|
| 1 | 128 | 8,448 | 16,896 | 33,792 |
| 2 | 256 | 8,704 | 17,408 | 34,816 |
| 4 | 512 | 9,216 | 18,432 | 36,864 |

At context 1024, table caching adds 256 bytes/CTA. It stages only entries up
to that CTA's history end while reserving the full stride. The cache option
must be compared with direct table lookup, including its load/barrier cost.
Shared-memory broadcasts and bank behavior remain compiled/measured questions.

For one query row, staged K/V source reads are shared across four sibling
heads: logical payload is four times smaller than four reference CTAs, before
caches and staging overhead. With several query rows, let `L_max,k` be the
largest valid causal length in query tile `k`. Candidate logical K+V bytes
per layer are `2*Dh*4*Hkv*sum_k(L_max,k)`, compared with the baseline expression
above. This explicitly charges the common tile for the largest query's
history and handles ragged tails. It is not a claim of 4× or
`4*Q_TILE` measured DRAM bandwidth reduction or wall-clock speedup.

The initial online run did not record compiled registers/thread, local
bytes/thread, static shared bytes, or active-block limits. The resource helper uses CUDA function
attributes and the occupancy API. Local memory can include stack as well as
spills; occupancy calculated from launch resources is a ceiling, not achieved
occupancy. Increasing reuse can still lose if 512-thread CTAs, shared storage,
or register pressure reduce useful concurrency.

## Selected exact arithmetic and bounded score storage

The `exact-q{1,2,4}-k{16,32,64}` family shares paged K/V transport while
retaining complete causal score vectors in CTA shared memory. The selected
`exact-q4-k64` uses 16 independent distributions per CTA: four query rows
times four query heads. Its 512 threads form 16 warps. This path does **not**
use online softmax; preserving the reference's arithmetic proved necessary
for deterministic seeded generation.

There are two tiled history traversals. The first cooperatively stages K,
shares each tile across the four GQA heads and four query rows, and writes
each distribution's full score vector. Each query warp then emulates the
reference's eight virtual softmax warps: virtual thread `t` visits
`t,t+256,...`; maxima are combined in ascending virtual-warp order; both
levels of the denominator shuffle tree retain offsets 16/8/4/2/1. Empty
virtual warps contribute the same identities. The second traversal reuses
the same shared tile buffer for V. Each output component accumulates visible
positions in ascending order across tile boundaries, then multiplies by the
reciprocal once. K dots retain the reference's two per-lane terms and shuffle
tree. The change is shared transport and work assignment, not reassociation
of the independent distributions' sums.

For a validated launch, let `H=max_i(absolute_start_i+chunk_len_i)`. The score
capacity is `C=ceil(H/256)*256`, capped at the allocated page-table capacity.
At the current 1024-token context, the possible bounds are 256/512/768/1024.
The host validates every descriptor's absolute end before uploading it.
Capacity controls scratch stride and launch shared memory; per-query causal
length controls semantic visibility. Reserving 256 scores for a one-token
query does not make the remaining 255 positions visible.

Dynamic shared payload is `4*(KV_TILE*64 + Q_TILE*4*C)` bytes, with
`4*table_stride` additional bytes only for the table-cache experiment.
One tile buffer serves K and V at different times. There are two CTA barriers
per tile in each traversal; optional table staging adds one. No global
`[T,T]` matrix, padded batch matrix, host causal mask, or gathered history
buffer is allocated.

| Selected q4/k64 score capacity | Threads/CTA | Dynamic shared bytes | Blocks/SM ceiling | Resident-thread ceiling |
|---:|---:|---:|---:|---:|
| 256 | 512 | 32,768 | 2 | 66.7% |
| 512 | 512 | 49,152 | Resource measurement recorded with final matrix | — |
| 768 | 512 | 65,536 | 1 | 33.3% |
| 1024 | 512 | 81,920 | 1 | 33.3% |

The q4 resource run reports 56 registers/thread, zero static shared bytes,
and zero local bytes/thread. The reference reports 40 registers, 272 static
plus 4,096 dynamic shared bytes, zero local bytes, and six blocks/SM (100%
resident-thread ceiling). These are CUDA function attributes and occupancy
calculator limits, not achieved occupancy. No instruction-level PTXAS spill
audit is inferred from the local-byte field. q4 can win despite a one-block
ceiling at long histories because each staged tile serves 16 distributions.

The earlier exact experiment reserved 1024 scores even for short histories
and tested only q1/q2. Its timing and resource record remains separate in
the evidence. Bounded capacity reduces shared pressure on short launches;
adding q4 provides more row reuse. All 36 exact tile/hybrid/cache combinations
pass the later bit-identity tensor gate. The selected complete-model corpus
also reports zero final-logit difference and 1,084 identical sequences.

## Hybrid current-slice experiment

The hybrid variant preserves rotated current K in separate persistent scratch
and keeps V in the existing K/V scratch. It retains the simple ordering:

```text
K projection -> K RoPE -> paged K scatter
V projection -> paged V scatter
Q projection -> Q RoPE
attention: paged prefix + contiguous current-slice K/V
```

For a descriptor with absolute start `a`, a visible history position `j<a`
uses its page table and pool. Position `j>=a` instead reads scratch row
`packed_start+j-a`. The query's causal limit still applies. Both sources hold
the same f32 values; the pool remains authoritative after this layer and is
ready for subsequent chunks and decode. This variant does not delay cache
writes or maintain two durable caches.

| Added payload beyond the packed-prefill baseline | Bytes |
|---|---:|
| Device segment descriptors, capacity `B` | `16*B` = 256 at B=16 |
| Host segment staging, capacity `B` | `16*B` = 256 at B=16 |
| Optional retained current K | `Tcapacity*KD*4` = 786,432 at 1024 × 192 |
| Additional current V | 0; reuses existing scratch |
| New global attention score/mask/history matrix | 0 |

The extra K buffer is **0.75 MiB**, shared across layers, not 0.75 MiB per
layer. It is allocated only for the explicit hybrid experiment before timed
inference; a diagnostic switch may retain it afterward until model teardown.
Descriptor payloads exclude container headers and the two host shape scalars.
Tiled packed calls add one bulk descriptor upload of `16*request_count`
bytes; singleton tiled calls upload one 16-byte descriptor. There are no
per-query H2D copies or per-call CUDA allocations in the intended path.

## Dispatch, graphs, and lifecycle invariants

The model default is `CRUCIBLE_PREFILL_ATTN=exact`, an alias for
`exact-q4-k64`. It applies to paged prefill with `Dh=64` and `Hq=4*Hkv`;
unsupported geometry and the legacy contiguous path retain reference
attention. `CRUCIBLE_PREFILL_ATTN=reference` explicitly selects the unchanged
oracle. There is no history threshold: the measured tiny-kernel penalty is
roughly 1–3 microseconds, while the short service rounds show no material
regression. A threshold would add graph/dispatch complexity without a
measured service advantage.

Diagnostic names encode query and history tiles, with independent `-hybrid`
and `-cache` suffixes. Names without `exact-` select the online experiment;
`grouped`, `tiled`, and `hybrid` remain its q1-k16, q2-k16, and q2-k16-hybrid
aliases. They are reproducibility controls, not production recommendations.
Hybrid current-K/V and page-table staging are off by default.

The singleton graph key is `(slice_length,want_logits,score_capacity)`.
The allocation bound is true launch topology: replaying a graph captured
with 256 score slots for a later 512-slot history would be unsafe. Positions
and physical pages remain dynamic descriptor/table contents and do not enter
the key. The graph cache is bounded to 64 entries. A full cache runs eager;
capture failure falls back to eager and is remembered for that key. Capture
must end even if queuing fails so the stream cannot remain half-captured.

Changing the variant synchronizes outstanding work, invalidates singleton
captures and packed prepared-shape state, and provisions optional scratch
before subsequent inference. Consequently the variant is fixed for each
cache lifetime and need not be duplicated in every key. Scratch resizing
likewise invalidates captures that contain old addresses. Metadata H2D writes
remain outside capture, in bulk, before transformer execution.

Packed serving remains eager. The optional temporary packed replay experiment
measures an already-prepared packed call; it does not implement a production
cache. A future transformer-body graph would leave metadata upload and final
row selection/sampling outside. Its key must encode actual launch topology:
aggregate rows, descriptor count, maximum query-tile extent, attention
variant/score capacity, and any projection topology that truly changes.
Request IDs, page IDs, positions, tokens, RNG states and scheduler slots are
data, never keys. A cache would need a fixed entry/memory budget, eager
fallback, pointer-lifetime invalidation and measured capture amortization.
The present milestone does not add that cache. The temporary complete
greedy packed graph includes final-row selection and is timed with host
`Instant` around repeated launches and synchronization, despite the legacy
`packed_gpu_ms` output label. Its same-job host-wall gap is 14.14% for
16×16 and 17.72% for 4×256. This is neither a CUDA-event measurement nor
an isolated transformer-body experiment, and it does not establish a
production cache hit rate or capture amortization.

Host validation establishes positive lengths, checked packed/absolute bounds,
exact table stride, in-range page IDs, no duplicate physical ownership, and
valid token/final-row selections before upload. Descriptors derive from those
validated slices. Every consumed descriptor is overwritten; unused capacity
from a previous call is not semantic input.

The scheduler still admits bounded residents, decodes established requests,
runs one bounded prefill plan, and retires/reclaims. Cancellation is observed
at those boundaries. Submitted work completes before pages can be reused;
attention retains no page index across calls. Unrecoverable execution errors
retain the existing fail-closed behavior. No attention-specific GPU timing
synchronization is added to metrics.

Final-row gather, RMSNorm, vocabulary arithmetic, top-k/full-logit selection,
and request RNG remain outside this optimization. All native, OpenAI,
Anthropic, and TUI requests enter the same scheduler and GPU stream. Packing
remains on, aggregate prefill budget 1024, chunk cap off, singleton graphs on,
and packed serving graphs off. The completed three-round sweep retains
these defaults: smaller budgets/caps reduce worst decode stalls but impose
material arrival-latency or throughput costs, as recorded in the results. There is no adaptive scheduler or protocol-specific attention path.

## Selection evidence and limits

The online candidate passed the unchanged tensor tolerance, but its maximum
absolute error near 1.10e-6 did not establish generation equivalence. In the
controlled request-114 diagnosis, packed reference logits equal the
independent monolithic control bit-for-bit at all 16 steps, and candidate
RNG states equal the reference states. At token nine, the same uniform draw
selects rank 429 after attention-dependent drift reverses tokens 1921 and
8158 at ranks 428/429. The online candidate is therefore disabled by default.
No tolerance or sampling rule was changed to hide that failure.

The exact family preserves all tested output bits, including adversarial
page layouts and future poisoning. The selected int8 model run passes 1,084
complete sequences over 878 packed GPU calls and 24 seeded model fuzz sets,
with zero final-logit difference versus monolithic inference. The synthetic
corpus independently checks all three KV groups, per-query causal ends,
noncontiguous physical pages, changed packing composition/order, page remaps,
and positions around page boundaries. These finite tests support the audited
arithmetic; they are not a proof for arbitrary future compiler/hardware changes.

Three paired real-service rounds use a fresh build of the actual starting
commit and the selected implementation under the new roughly 145 W envelope.
The long 4×256, 4×512, 4×941 and 8×512 workloads improve in every pair.
Old 135 W stage profiles are retained as motivation, not divided into new
candidate times. Kernel-only timing, instrumented transformer stages,
uninstrumented wall time, and HTTP latency have separate meanings in the
results report. The longer 100-pass stage profiles still have mismatched
reference/candidate clocks and are excluded from causal speedup acceptance.
They describe remaining candidate costs: long attention is 3.9534 ms and
gate/up is 3.8006 ms, so attention remains the largest individual stage.
Paged held-out CE matches the printed reference metrics at contexts 32,
256 and 941; complete model and protocol checks pass.

The hybrid scratch path and page-table staging remain diagnostic ablations:
their small, shape-dependent effects do not justify changing the selected
path or its extra memory. Reduced-precision tensor-core attention and fragile
Blackwell-specific copy mechanisms were not added. A rounded-fp16 diagnostic
quantifies input-precision drift but is not an attention implementation or a
production option. The actual attention stays f32 and custom CUDA.

Compute Sanitizer could not initialize WDDM instrumentation in the prior WSL
attempts; no sanitizer or Nsight hardware-counter coverage is claimed here.
The results provide native-Linux memcheck commands. No system/admin changes,
third-party attention dependency, prefix caching, speculation, new KV layout,
or next-stage GEMM optimization belongs to this milestone.
