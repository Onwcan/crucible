# Packed prefill: measurement and validation record

Cross-request packed prefill is integrated into the real serving scheduler.
The implementation combines prompt rows before the existing transformer GEMMs,
preserves separate positions and paged histories, and selects only completed
requests' final rows. Correctness validation passes for both int8 and f32,
including adversarial request composition and cancellation. The measured
outcome is a **partial win**: short bursts benefit substantially; small chunks
still impose an expensive GPU floor, and long-request tail latency determines
whether a chunk policy is useful.

The selected configuration enables packing with a 1,024-token aggregate budget,
keeps the additional per-request chunk cap disabled, uses the existing graph
path for one contributor, and uses eager packed execution for two or more.
The [design contract](packed-prefill-design.md) contains the complete stage
shape table, indexing equations, source links, bounds, and lifecycle rules.
This document records the evidence and its limits.

> Final serving, chunk-policy, and decode-only comparisons use the corrected
> candidate. Earlier policy sweeps explicitly precede that correction and are
> not final-binary performance. The final one-token bursts, sustained load,
> five-client packed-batch proof, and closing regression checks are complete.

## Baseline identity and measurement envelope

The starting commit was `8f5c745c6309f25f4dae0466a2275598bf333042`.
`git status --short` was empty before implementation. The comparison binary
was rebuilt from an archive of that exact commit into a separate build
directory; it was not a previously compiled executable from the modified tree.
The milestone is left uncommitted for review.

Measurements use the exported 120M model and int8 weights unless a check
explicitly says f32. The GPU is the NVIDIA RTX PRO 4000 Blackwell Generation
Laptop GPU, 16 GB, `sm_120`, 60 SMs. The model context is 1,024 tokens, resident
capacity is 16 requests, and the normal server pool is 1,024 pages of 16 tokens.
The host was on AC, battery status 2 at 100%, with Windows' Best Performance
overlay. Dell Ultra Performance was **not independently verified**: the Dell
CLI required administrator access even for its help query, so it did not
establish the active thermal mode.

The GPU's enforced limit changes with laptop power sharing. It is therefore
reported as an observed range, alongside loaded clocks, rather than as one
nominal TGP. All experiments warm the model before timing. The following
summaries retain samples at at least 50% reported GPU utilization for the
loaded-clock statistic; startup transitions remain visible in the minima.

| Experiment | Enforced limit, W | Maximum SM clock, MHz | Loaded SM clock min / median / max, MHz |
|---|---:|---:|---:|
| Baseline microbenchmark monitor | 145.00–159.38 | 3090 | 1875 / 2812 / 2827 |
| Baseline HTTP, three trials | 148.26–154.26 | 3090 | 2302 / 2782 / 2805 |
| Final 30-second sustained run | 145.00–158.13 | 3090 | 2587 / 2730 / 2752 |

The sustained monitor recorded 59 samples, 58 above the utilization threshold;
loaded utilization ranged from 79% to 90%, median 85%. These are coarse device
activity samples, not achieved occupancy or tensor-core utilization counters.
Per-policy envelopes are included with the sweep tables below.

GPU microbenchmark wall time, pure replay time, and HTTP TTFT measure different
boundaries. HTTP TTFT begins when the client submits a request and ends at its
first received token. It includes queueing and scheduling. Pure replay times
the captured kernel sequence, including device argmax, while excluding
metadata upload, readback, host sampling, and HTTP. Replay is timed with a
host clock over synchronized graph launches, so graph-launch and final-sync
overhead are amortized into the result; it is not a hardware-counter measure
of kernel duration. Instrumented CUDA-event stage intervals are diagnostic
and may include host submission gaps.

## What inspection established

There are seven projection GEMMs per layer: K, V, Q, O, gate, up, and down.
For 12 layers this is **84 transformer GEMMs per prefill call**, with the final
vocabulary projection counted separately. The earlier six-per-layer estimate
omitted one projection. The shared transformer loop now passes the aggregate
real-row count as GEMM `M`; it does not loop over requests to run independent
transformers.

For example, four 16-token slices change four sets of GEMMs with `M=16` into
one set with `M=64`; sixteen such requests yield `M=256`. Normalization,
residuals, SwiGLU, and embeddings also operate over those packed rows. Only
the operations that depend on sequence state use ownership metadata.

The [stage analysis](packed-prefill-design.md#shapes-and-what-can-be-packed)
documents each input/output shape. The following contracts are the critical
review points:

| Area | Implemented contract |
|---|---|
| Representation | Consecutive real rows; int32 owner and absolute-position arrays; stable request IDs in the host plan; temporary descriptor indices on the GPU. |
| RoPE | Uses the row's absolute prompt position, including nonzero chunk offsets. |
| KV routing | Uses that owner's page table, with `PAGE_TOKENS=16` and the existing decode-compatible KV pool layout. |
| Attention | Reads only positions `0..=absolute_position` through that request's table; the packed and reference paths share the paged attention implementation. |
| Scheduling | Admit bounded residents, decode existing streams first, execute one bounded prefill plan, then retire and reclaim. |
| Fairness | Visit the round-robin queue once per plan; one positive slice per contributor; unfinished contributors move to the back. A surviving resident receives work within at most the resident count of nonempty plans. |
| Final rows | Gather only finishing prompt rows, normalize and project those rows, then select each request's first token. Non-final work has no logits or D2H result. |
| Sampling | Request-owned seed/RNG; greedy tie semantics unchanged; device top-k through 128; full-logit fallback only for selected rows above that bound. |
| Cancellation | Observe at scheduler boundaries. Complete an already submitted bounded GPU call before reclaiming its pages. Remove cancelled descriptors before later plans. |
| Admission | Prefilling plus decoding is bounded by `max_batch`; the semaphore bounds waiting work across both the channel and runtime pending queue. |
| Reservations | Reserve lifetime growth as `ceil((prompt + max_tokens - 1)/16)` pages. Allocate prompt pages on admission and later decode pages lazily. |
| Graphs | Singleton exact-length graphs retain their 64-entry cache; packed serving remains eager. Benchmark-only replay graphs do not imply packed serving graph support. |
| Ownership | Exactly one inference thread owns the GPU and one execution stream. All protocol handlers submit through the shared queue. |

The fairness bound applies after admission and is a bound in plans, not in
milliseconds. A token budget bounds rows per GPU call, not a universal latency
deadline: longer cached histories make attention more expensive. Without the
optional chunk cap, the budget may still split a prompt at the end of a plan.

## Baseline measurements

Each microbenchmark cell is the median of three round medians, each round
using 30 iterations. Parentheses show the minimum and maximum round medians
for graph wall time. These are final chunks that produce logits.

| Prompt tokens | Eager wall, ms | Graph wall, ms (range) | Synchronized replay, ms | Eager-minus-graph, ms |
|---:|---:|---:|---:|---:|
| 1 | 6.924 | 5.919 (5.879–5.933) | 5.746 | 1.008 |
| 8 | 6.412 | 5.911 (5.864–5.949) | 5.745 | 0.501 |
| 16 | 6.824 | 6.297 (6.264–6.308) | 6.129 | 0.527 |
| 32 | 6.725 | 6.211 (6.154–6.223) | 6.036 | 0.531 |
| 64 | 6.779 | 6.256 (6.225–6.267) | 6.091 | 0.523 |
| 128 | 7.128 | 6.608 (6.568–6.617) | 6.441 | 0.511 |
| 256 | 8.767 | 8.256 (8.229–8.275) | 8.071 | 0.511 |

The 1-token GPU floor is reproduced: replay is 97% of graph wall time. More
launch-overhead reduction alone cannot remove it. The eager 256-token round
medians varied from 8.688 to 9.468 ms while graph medians varied by only
0.046 ms; this is why graph and paired measurements matter.

The initial HTTP baseline uses three trials, 64 generated tokens per request,
verified prompt-token counts, and graph prefill. Short bursts cycle 8/16/32/64
tokens; the mixed burst is 8/17/33/64/127/256/512/941. TTFT percentiles pool the
requests from the three trials; aggregate generation throughput is the median
of the three trial throughputs. With so few requests, these p95 values describe
the tested bursts and are not population tail estimates; p99 is not inferred.

| Workload | TTFT p50 / p95, ms | Aggregate generation, tok/s |
|---|---:|---:|
| Isolated 1 token | 7.08 / 7.11 | 1407 |
| Isolated 8 | 7.04 / 7.10 | 1364 |
| Isolated 16 | 7.34 / 7.39 | 1347 |
| Isolated 32 | 7.23 / 7.32 | 1336 |
| Isolated 64 | 7.37 / 7.42 | 1250 |
| Isolated 128 | 7.97 / 8.01 | 1102 |
| Isolated 256 | 9.64 / 9.96 | 943 |
| Short burst 2 | 10.79 / 14.59 | 2222 |
| Short burst 4 | 26.85 / 28.08 | 3192 |
| Short burst 8 | 52.60 / 53.17 | 4207 |
| Short burst 16 | 103.15 / 106.16 | 5132 |
| Mixed burst 8 | 83.39 / 84.46 | 2386 |
| 4 × 256 | 35.54 / 35.79 | 2539 |
| 4 × 512 | 53.76 / 54.57 | 1866 |
| 4 × 941 | 116.39 / 117.43 | 1121 |

For one established stream plus one arriving prompt, baseline medians of the
three trials' worst gaps were 6.81, 14.27, and 28.45 ms for arriving lengths
33, 535, and 941. Their observed maximum gaps were 7.11, 14.49, and 28.53 ms.
The arriving requests' median TTFTs were 7.28, 15.42, and 29.86 ms.

## Token-budget and chunk-cap experiments

**These policy sweeps preceded the final decode vocabulary-projection
correction described below.** Every candidate has its own exact-HEAD baseline
measurement, with three trials on each side. The tables inform the scheduling
tradeoff; final-binary gains must come from the later paired measurements.
Throughputs are generated tokens per second over the whole request workload,
not prompt rows per second.

First, keep the requested per-request cap at 128 and vary the aggregate budget.
The effective cap cannot exceed the remaining aggregate budget. The last column
is the median of each trial's largest established-stream gap when a single
941-token prompt arrives.

| Aggregate budget | Short-16 TTFT p50 / p95, ms | Short-16 tok/s | Mixed TTFT p50 / p95, ms | Mixed tok/s | 4 × 941 TTFT p95, ms | 941 arrival worst gap, ms |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 32.89 / 71.57 | 6142 | 75.45 / 283.21 | 1255 | 523.06 | 11.27 |
| 128 | 23.99 / 42.41 | 7292 | 36.33 / 161.80 | 1757 | 307.06 | 13.15 |
| 256 | 17.17 / 27.29 | 8216 | 25.15 / 114.95 | 2087 | 213.54 | 13.16 |
| 512 | 16.89 / 17.62 | 8720 | 17.88 / 107.79 | 2118 | 162.59 | 13.11 |
| 1024 | 17.26 / 18.03 | 8603 | 19.19 / 110.35 | 2109 | 163.17 | 13.34 |

Across the five paired baselines, short-16 throughput ranged 5,032–5,133 tok/s,
mixed p95 83.53–90.52 ms, 4 × 941 p95 120.00–121.98 ms, and the 941-arrival
worst-gap median 28.39–29.19 ms. The 512/1024 difference at cap 128 is small
beside the common long-prompt penalty. A 64-token budget can protect an
established decoder while making the mixed-burst tail more than three times
the baseline; it is unsuitable as a universal default.

Second, hold the aggregate budget at 1024 and vary the additional per-request
cap. “None” disables that cap while retaining the aggregate budget.

| Per-request cap | Short-16 tok/s | Mixed TTFT p50 / p95, ms | Mixed tok/s | 4 × 512 TTFT p95, ms | 4 × 941 TTFT p95, ms | 941 arrival TTFT / worst gap, ms |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 8418 | 26.14 / 293.06 | 1230 | 152.81 | 339.07 | 246.49 / 9.69 |
| 64 | 8877 | 15.62 / 164.83 | 1726 | 98.48 | 226.48 | 133.38 / 10.87 |
| 128 | 8744 | 18.46 / 107.33 | 2139 | 64.66 | 159.74 | 81.05 / 12.71 |
| 256 | 8652 | 22.57 / 72.70 | 2471 | 52.61 | 133.84 | 51.37 / 14.99 |
| None | 8683 | 45.20 / 56.66 | 2636 | 51.24 | 125.13 | 29.34 / 27.74 |

The paired no-cap baseline was 5,082 tok/s for short-16; mixed TTFT
83.69/84.38 ms at 2,336 tok/s; 4 × 512 p95 55.10 ms; 4 × 941 p95 120.76 ms;
and 941-arrival TTFT/worst gap 29.92/28.80 ms. Thus the no-cap candidate
substantially improves short and mixed bursts, but its 4 × 941 tail and
throughput (1,053 versus 1,100 tok/s) remain a limitation. Cap 256 improves the
mixed distribution but still charges long requests additional work. Cap 32
or 64 improves the mixed median while imposing a much worse tail.

The data supports leaving per-request chunking opt-in. It does not establish
a simple active-decode-aware switch that dominates the static default: lowering
the budget trades arriving-request TTFT and aggregate throughput for smaller
decode stalls. No adaptive policy was added.

| Sweep candidate | Enforced limit, W | Loaded SM clock min / median / max, MHz |
|---|---:|---:|
| Budget 64, cap 128 | 145.00–149.20 | 2640 / 2775 / 2782 |
| Budget 128, cap 128 | 145.00–149.17 | 2610 / 2775 / 2782 |
| Budget 256, cap 128 | 145.00–149.24 | 2580 / 2775 / 2782 |
| Budget 512, cap 128 | 145.00–150.12 | 2677 / 2775 / 2782 |
| Budget 1024, cap 128 | 145.00–148.08 | 2377 / 2767 / 2782 |
| Budget 1024, cap 32 | 145.00–152.69 | 2632 / 2782 / 2797 |
| Budget 1024, cap 64 | 145.00–150.00 | 2685 / 2775 / 2782 |
| Budget 1024, cap 128 repeat | 145.00–149.52 | 2572 / 2775 / 2782 |
| Budget 1024, cap 256 | 145.00–146.75 | 2497 / 2775 / 2782 |
| Budget 1024, no cap | 145.00–147.63 | 2497 / 2775 / 2782 |

The maximum SM clock was 3090 MHz for every sweep sample. Paired baselines
had enforced limits 145.00–152.42 W; their loaded-clock medians ranged
2763.5–2790 MHz. The substantial cap-driven differences therefore cannot be
explained by a low-power baseline.

## Final performance measurements

The final service comparison uses the rebuilt starting-commit baseline and the
candidate containing the decode determinism correction. There are three
independent, freshly warmed rounds per side, in **A/B, B/A, A/B** order, with
one trial of each workload per round. Cells below are medians of the three
round statistics. In particular, p95 is the median of per-round p95 values,
not a pooled percentile; an isolated trial contains only one request, so its
p50 and p95 are identical. All generated tokens count toward workload
throughput, including prefill's first token. The
[portable evidence summary](packed-prefill-evidence.json) retains each round's
values and envelopes. No best-run selection is used.

### Real-service latency and throughput

These requests generate 64 tokens each. Short bursts cycle 8/16/32/64 prompt
tokens; mixed bursts use 8/17/33/64/127/256/512/941. The separate one-token
HTTP burst table and GPU crossover matrix follow below.

| Workload | Baseline TTFT p50 / p95, ms | Packed TTFT p50 / p95, ms | Baseline / packed, tok/s | Throughput change |
|---|---:|---:|---:|---:|
| Isolated 1 | 6.95 / 6.95 | 6.35 / 6.35 | 1426 / 1413 | -0.9% |
| Isolated 8 | 6.84 / 6.84 | 6.33 / 6.33 | 1397 / 1407 | +0.7% |
| Isolated 16 | 7.30 / 7.30 | 6.78 / 6.78 | 1381 / 1372 | -0.7% |
| Isolated 32 | 7.23 / 7.23 | 6.79 / 6.79 | 1353 / 1331 | -1.6% |
| Isolated 64 | 7.30 / 7.30 | 6.55 / 6.55 | 1279 / 1289 | +0.8% |
| Isolated 128 | 7.75 / 7.75 | 7.03 / 7.03 | 1173 / 1160 | -1.1% |
| Isolated 256 | 9.60 / 9.60 | 8.69 / 8.69 | 974 / 969 | -0.5% |
| Short burst 2 | 10.96 / 14.26 | 9.99 / 12.93 | 2246 / 2271 | +1.1% |
| Short burst 4 | 27.49 / 27.75 | 14.03 / 14.17 | 3121 / 3709 | +18.8% |
| Short burst 8 | 52.79 / 53.49 | 14.89 / 15.32 | 4162 / 5912 | +42.0% |
| Short burst 16 | 105.90 / 107.01 | 16.15 / 16.99 | 5031 / 8739 | +73.7% |
| Mixed burst 8 | 83.26 / 83.38 | 44.77 / 54.77 | 2385 / 2734 | +14.6% |
| 4 × 256 | 35.87 / 36.01 | 23.85 / 24.10 | 2553 / 2846 | +11.5% |
| 4 × 512 | 54.51 / 54.71 | 35.17 / 47.28 | 1868 / 1928 | +3.2% |
| 4 × 941 | 116.47 / 116.63 | 73.92 / 115.08 | 1127 / 1107 | -1.7% |
| 8 × 512 | 109.72 / 109.94 | 58.31 / 93.59 | 2389 / 2547 | +6.6% |

Short-16 p95 falls **84.1%**, from 107.01 to 16.99 ms, while generation
throughput rises **73.7%**. Mixed p95 falls **34.3%**, with throughput up
**14.6%**. The isolated improvements use the singleton graph path and are not
evidence of cross-request packing. At two short requests, throughput changes
only 1.1%; scheduler arrival timing can leave the first request as a singleton.
The GPU-only crossover therefore should not be read as an HTTP crossover.

The long-workload result is mixed. For 4 × 941, median TTFT improves but p95
changes only from 116.63 to 115.08 ms and generation throughput falls 1.7%.
This does not establish a long-prompt throughput win. The 4 × 512 and 8 × 512
workloads gain 3.2% and 6.6% throughput respectively. Selected round ranges
make the variability visible:

| Workload | Baseline p95 range, ms | Packed p95 range, ms | Baseline throughput range, tok/s | Packed throughput range, tok/s |
|---|---:|---:|---:|---:|
| Short burst 2 | 14.22–14.30 | 12.79–13.07 | 2216–2298 | 2248–2273 |
| Short burst 16 | 106.40–107.60 | 16.70–17.00 | 5025–5081 | 8696–8945 |
| Mixed burst 8 | 82.54–83.89 | 44.30–56.55 | 2355–2389 | 2672–2737 |
| 4 × 512 | 53.84–55.04 | 47.06–47.57 | 1844–1883 | 1898–1928 |
| 4 × 941 | 115.73–118.18 | 114.79–116.00 | 1117–1132 | 1076–1115 |
| 8 × 512 | 106.86–110.94 | 93.15–96.14 | 2381–2496 | 2453–2568 |

### Arrival order and established-stream stalls

Fairness trials request 1 ms arrival spacing in sorted, reverse, and fixed
shuffled order. Actual candidate starts preserve those orders in all three
rounds, with 6.91–7.11 ms from first to last for the shuffled trials. The
shuffle is 127/8/941/33/512/17/256/64. This tests the same prompt multiset and
content variants with different queue order; it does not turn the plan-count
fairness bound into a time guarantee.

| Arrival order | Baseline TTFT p50 / p95, ms | Packed TTFT p50 / p95, ms | Baseline / packed, tok/s |
|---|---:|---:|---:|
| Sorted | 40.24 / 77.88 | 14.29 / 49.03 | 2356 / 2595 |
| Reverse | 82.68 / 85.86 | 43.21 / 46.40 | 2369 / 2807 |
| Shuffled | 79.61 / 82.70 | 41.02 / 50.39 | 2409 / 2752 |

Per-request medians retain the identity of the long request instead of hiding
it in an aggregate percentile. Each cell is baseline / packed TTFT in ms,
measured from that request's own submission:

| Prompt tokens | Sorted arrival | Reverse arrival | Shuffled arrival |
|---:|---:|---:|---:|
| 8 | 6.95 / 6.31 | 80.23 / 40.94 | 82.88 / 34.21 |
| 17 | 41.32 / 15.83 | 80.99 / 41.77 | 78.96 / 49.62 |
| 33 | 40.80 / 14.79 | 82.24 / 42.70 | 80.91 / 32.12 |
| 64 | 39.68 / 13.78 | 83.12 / 43.72 | 77.28 / 47.83 |
| 127 | 38.52 / 12.91 | 83.84 / 44.78 | 7.78 / 7.20 |
| 256 | 37.84 / 11.86 | 84.95 / 45.85 | 78.20 / 48.64 |
| 512 | 78.07 / 34.26 | 86.34 / 46.65 | 80.24 / 50.80 |
| 941 | 77.49 / 57.53 | 28.97 / 27.56 | 82.36 / 33.42 |

In reverse order, the first 941-token request still blocks later arrivals
until its bounded call completes: the 8-token request's median TTFT is
40.94 ms. In sorted order, the 941-token request waits 57.53 ms. All measured
requests finish, but packed scheduling does not promise first-come-first-token
ordering; requests finishing in one call can also be observed in different
SSE delivery order.

The single-arrival stall test starts an established decoder before submitting
one new prompt. The four-arrival HOL test adds 33/127/535/941-token requests
while one established stream is active. “Worst gap” is the median of the
three trials' largest established-stream inter-token gaps.

| Arriving workload | Baseline / packed TTFT p50, ms | Baseline / packed worst gap, ms |
|---|---:|---:|
| One 33-token prompt | 7.15 / 7.18 | 6.71 / 6.60 |
| One 535-token prompt | 15.24 / 14.64 | 14.23 / 13.50 |
| One 941-token prompt | 29.95 / 28.61 | 28.71 / 27.37 |
| Four mixed prompts | 35.94 / 19.31 | 43.03 / 26.64 |

During the four-arrival disturbance, established-stream gap p50/p95 is
2.096/2.391 ms for baseline and 2.109/2.588 ms for packed. The worst-gap round
range is 29.49–48.93 ms versus 22.86–28.69 ms. Packed reduces the large stalls
in these trials, while the gap p95 is slightly higher. Arriving-request p95
ranges 53.78–57.77 ms for baseline and 40.87–52.89 ms for packed. These small
samples do not support population p99 or a universal HOL deadline.

### Final chunk-policy comparison

Three rounds also compare the frozen starting-commit binary with singleton
chunking enabled against the corrected packed candidate at caps 128 and 64,
at aggregate budget 1024. The singleton chunked reference therefore retains
the baseline vocabulary arithmetic; it is not a candidate ablation with only
packing disabled. For context, the earlier final A/B monolithic-baseline and
packed-default medians are repeated in this table; those two configurations
were not rerun inside the three-policy experiment.

| Configuration | Short-16 p95, ms / tok/s | Mixed p50 / p95, ms / tok/s | 4 × 941 p95, ms / tok/s | 941 arrival TTFT / worst gap, ms |
|---|---:|---:|---:|---:|
| Exact starting commit | 107.01 / 5031 | 83.26 / 83.38 / 2385 | 116.63 / 1127 | 29.95 / 28.71 |
| Starting commit, chunk 128 | 122.45 / 4616 | 39.90 / 139.76 / 1738 | 308.00 / 591 | 76.14 / 20.42 |
| Packed default | 16.99 / 8739 | 44.77 / 54.77 / 2734 | 115.08 / 1107 | 28.61 / 27.37 |
| Packed chunk 128 | 16.92 / 8739 | 17.76 / 90.47 / 2154 | 153.54 / 967 | 80.93 / 12.56 |
| Packed chunk 64 | 16.72 / 8842 | 20.08 / 141.64 / 1729 | 214.14 / 787 | 133.46 / 10.73 |

Packing recovers much of singleton chunking's throughput loss, but the
additional cap still trades tail latency and throughput for smaller stalls.
Compared with packed default, cap 128 cuts the 941-arrival worst gap from
27.37 to 12.56 ms while increasing that request's TTFT from 28.61 to
80.93 ms. Cap 64 lowers the gap to 10.73 ms but raises TTFT to 133.46 ms.
Mixed p95 rises from 54.77 to 90.47 or 141.64 ms. These final measurements
confirm the decision to leave the additional chunk cap opt-in.

| Configuration | 4 × 256 p95 / tok/s | 4 × 512 p95 / tok/s | 8 × 512 p95 / tok/s | HOL arrivals p95 / established worst gap, ms |
|---|---:|---:|---:|---:|
| Exact starting commit | 36.01 / 2553 | 54.71 / 1868 | 109.94 / 2389 | 57.59 / 43.03 |
| Starting commit, chunk 128 | 59.04 / 2024 | 131.93 / 1151 | 269.98 / 1352 | 124.69 / 21.09 |
| Packed default | 24.10 / 2846 | 47.28 / 1928 | 93.59 / 2547 | 49.71 / 26.64 |
| Packed chunk 128 | 29.13 / 2691 | 62.68 / 1759 | 99.11 / 2520 | 95.62 / 13.45 |
| Packed chunk 64 | 42.92 / 2310 | 94.16 / 1449 | 127.03 / 2222 | 148.82 / 11.31 |

The capped runs are repeatable at the scale of the policy difference. Mixed
p95 ranges 139.32–140.34 ms for singleton cap 128, 89.26–90.85 ms for packed
cap 128, and 138.31–150.09 ms for packed cap 64. Their 4 × 941 p95 ranges
are 307.21–309.11, 150.84–156.85, and 214.09–215.52 ms respectively. Loaded
clock medians range 2771–2797 MHz for singleton cap 128, 2775–2797 MHz for
packed cap 128, and 2775–2797 MHz for packed cap 64; enforced limits across
those groups are 145.00–150.44, 145.00–152.31, and 145.00–153.64 W. Maximum
SM clock is 3090 MHz. The evidence file retains each of the nine envelopes.

| Final service round | Enforced limit, W | Loaded SM clock min / median / max, MHz |
|---|---:|---:|
| Baseline 1 | 145.00–152.07 | 2565 / 2756 / 2812 |
| Baseline 2 | 145.00–150.10 | 2617 / 2797 / 2797 |
| Baseline 3 | 145.00–151.12 | 2587 / 2767 / 2797 |
| Candidate 1 | 145.00–150.54 | 2632 / 2786 / 2805 |
| Candidate 2 | 145.00–152.02 | 2617 / 2782 / 2797 |
| Candidate 3 | 145.00–145.00 | 2580 / 2778.5 / 2797 |

Maximum SM clock was 3090 MHz throughout. Loaded candidate median clocks are
2778.5–2786 MHz and baseline medians 2756–2797 MHz; the large short-burst gain
is not a comparison against an idle or low-power baseline.

### One-token HTTP bursts

Three additional alternating rounds submit 2/4/8/16 one-token prompts, with 64 generated tokens each. These are actual HTTP arrivals; a burst is not guaranteed to become one GPU batch. All rounds and their envelopes are retained. The two-client case does not show a service win, despite the low-level two-contributor crossover. Packing adds no wait-to-fill window.

| Concurrent one-token prompts | Baseline / packed TTFT p50, ms | Baseline / packed p95, ms | Baseline / packed generation tok/s |
|---:|---:|---:|---:|
| 2 | 11.32 / 15.66 | 14.83 / 19.42 | 1875 / 1162 |
| 4 | 24.91 / 20.04 | 29.27 / 20.45 | 2626 / 2048 |
| 8 | 58.54 / 14.89 | 60.94 / 16.21 | 3642 / 3841 |
| 16 | 102.15 / 21.59 | 104.20 / 25.69 | 4126 / 5154 |

Small-burst arrival and clock variation is material; the evidence file retains ranges rather than selecting a best run. The production threshold counts actual contributors in a constructed plan, not simultaneous HTTP clients.

### Packed GPU crossover and equivalent-row scaling

The final microbenchmark covers every requested count 1/2/3/4/8/16 and chunk
1/8/16/32/64/128/256 that fits the 1,024-row scratch: **39 shapes**, three
rounds of 15 iterations. Each iteration alternates the serial-eager,
serial-graph, and packed-eager ordering; each path is warmed first. Serial
comparators use the candidate binary, so this isolates combining requests
from the decode correctness repair. All prompts begin at position zero and
all requests finish; histories from earlier chunks are validated elsewhere.

Wall values include the model call, metadata, synchronization and final token
readback. Replay is a separate benchmark-only captured compute sequence with
no upload or readback. Serving never uses that packed replay graph. “Rows/s”
is real prompt rows divided by packed wall time; speedup is the median of
within-round serial-graph/packed-wall ratios. Ranges are the three wall round
medians. No padding tokens count as useful work.

| Requests × chunk | Rows | Serial eager / graph, ms | Packed wall (range), ms | Synchronized replay, ms | Packed rows/s | Graph / packed |
|---|---:|---:|---:|---:|---:|---:|
| 1 × 1 | 1 | 6.748 / 6.325 | 6.543 (6.490–6.892) | 5.779 | 153 | 0.96× |
| 1 × 8 | 8 | 6.753 / 6.333 | 6.528 (6.510–6.545) | 5.764 | 1,226 | 0.97× |
| 1 × 16 | 16 | 7.139 / 6.684 | 6.900 (6.885–6.909) | 6.133 | 2,319 | 0.97× |
| 1 × 32 | 32 | 7.001 / 6.635 | 6.787 (6.754–6.814) | 6.047 | 4,715 | 0.98× |
| 1 × 64 | 64 | 7.127 / 6.682 | 6.844 (6.808–6.853) | 6.107 | 9,351 | 0.97× |
| 1 × 128 | 128 | 7.451 / 7.092 | 7.226 (7.178–7.253) | 6.477 | 17,715 | 0.98× |
| 1 × 256 | 256 | 9.084 / 8.727 | 8.838 (8.835–8.887) | 8.155 | 28,964 | 0.99× |
| 2 × 1 | 2 | 13.174 / 12.279 | 6.543 (6.523–6.543) | 5.791 | 306 | 1.88× |
| 2 × 8 | 16 | 13.176 / 12.198 | 6.870 (6.840–6.933) | 6.125 | 2,329 | 1.78× |
| 2 × 16 | 32 | 13.961 / 12.985 | 6.779 (6.772–6.821) | 6.034 | 4,720 | 1.91× |
| 2 × 32 | 64 | 13.674 / 12.777 | 6.798 (6.752–6.802) | 6.075 | 9,415 | 1.88× |
| 2 × 64 | 128 | 13.847 / 12.979 | 7.025 (6.949–7.038) | 6.300 | 18,222 | 1.84× |
| 2 × 128 | 256 | 14.553 / 13.656 | 8.204 (8.152–8.228) | 7.501 | 31,205 | 1.66× |
| 2 × 256 | 512 | 18.283 / 17.214 | 10.674 (10.561–10.776) | 11.008 | 47,967 | 1.61× |
| 3 × 1 | 3 | 19.669 / 18.129 | 6.582 (6.516–6.608) | 5.832 | 456 | 2.75× |
| 3 × 8 | 24 | 19.509 / 18.130 | 6.433 (6.413–6.434) | 5.690 | 3,731 | 2.82× |
| 3 × 16 | 48 | 20.818 / 19.150 | 6.811 (6.720–6.813) | 6.057 | 7,048 | 2.82× |
| 3 × 32 | 96 | 20.364 / 18.957 | 6.841 (6.826–6.896) | 6.128 | 14,033 | 2.76× |
| 3 × 64 | 192 | 20.666 / 19.179 | 7.312 (7.279–7.319) | 6.600 | 26,256 | 2.62× |
| 3 × 128 | 384 | 21.611 / 20.298 | 8.784 (8.672–8.813) | 8.140 | 43,717 | 2.33× |
| 3 × 256 | 768 | 27.193 / 25.941 | 14.531 (14.197–14.556) | 14.553 | 52,851 | 1.78× |
| 4 × 1 | 4 | 26.035 / 24.213 | 6.473 (6.440–6.474) | 5.709 | 618 | 3.74× |
| 4 × 8 | 32 | 25.875 / 24.038 | 6.791 (6.725–6.817) | 6.041 | 4,712 | 3.55× |
| 4 × 16 | 64 | 27.689 / 25.593 | 6.816 (6.756–6.816) | 6.050 | 9,390 | 3.78× |
| 4 × 32 | 128 | 27.106 / 25.321 | 6.958 (6.912–6.993) | 6.231 | 18,397 | 3.63× |
| 4 × 64 | 256 | 27.373 / 25.513 | 7.894 (7.817–7.909) | 7.170 | 32,431 | 3.23× |
| 4 × 128 | 512 | 28.851 / 27.023 | 9.469 (9.396–9.494) | 9.183 | 54,074 | 2.85× |
| 4 × 256 | 1024 | 36.325 / 34.556 | 16.328 (16.109–16.354) | 17.600 | 62,713 | 2.12× |
| 8 × 1 | 8 | 51.746 / 47.651 | 6.569 (6.565–6.612) | 5.847 | 1,218 | 7.23× |
| 8 × 8 | 64 | 51.584 / 47.726 | 6.817 (6.796–6.842) | 6.082 | 9,388 | 6.98× |
| 8 × 16 | 128 | 54.693 / 50.632 | 6.957 (6.928–6.983) | 6.226 | 18,399 | 7.28× |
| 8 × 32 | 256 | 54.146 / 49.984 | 7.796 (7.792–7.871) | 7.033 | 32,836 | 6.38× |
| 8 × 64 | 512 | 54.660 / 50.702 | 8.901 (8.837–9.141) | 8.345 | 57,523 | 5.70× |
| 8 × 128 | 1024 | 57.424 / 53.995 | 13.737 (13.719–14.288) | 13.820 | 74,544 | 3.92× |
| 16 × 1 | 16 | 104.158 / 95.995 | 7.176 (7.057–7.328) | 6.301 | 2,229 | 13.38× |
| 16 × 8 | 128 | 104.842 / 96.394 | 7.240 (7.084–7.284) | 6.303 | 17,679 | 13.38× |
| 16 × 16 | 256 | 112.423 / 103.788 | 8.017 (7.786–8.160) | 7.059 | 31,933 | 12.92× |
| 16 × 32 | 512 | 110.962 / 102.550 | 8.886 (8.642–9.050) | 8.135 | 57,617 | 11.49× |
| 16 × 64 | 1024 | 112.223 / 103.876 | 12.903 (12.602–13.212) | 12.482 | 79,360 | 7.99× |

One-contributor packed execution loses to the singleton graph at every tested
length (0.96–0.99×), supporting singleton dispatch. Two contributors already
win 1.61–1.91× over serial graphs. At 16 × 16, packed wall is 8.017 ms versus
103.788 ms for serial graphs, a 12.92× within-round speedup; at 16 × 64 it is
12.903 versus 103.876 ms, or 7.99×. The one-token floor remains: packed replay
is about 5.7–6.3 ms across the tested contributor counts, even though real row
throughput rises as more useful requests share that floor.

Equivalent aggregate rows do not imply equivalent work. At 256 rows,
1 × 256 takes 8.838 ms, 2 × 128 takes 8.204 ms, 4 × 64 takes 7.894 ms,
8 × 32 takes 7.796 ms, and 16 × 16 takes 8.017 ms. Shorter independent
histories reduce attention work, while more final rows add vocabulary
projection and selection work. At 1,024 rows, 4 × 256 takes 16.328 ms versus
12.903 ms for 16 × 64. This is a reason to report sequence shape as well as
GEMM M.

Individual iterations are noisy: the largest within-round min/max spread is
85.11% at 3 × 256, although its round wall medians are 14.197–14.556 ms.
Replay and wall are measured at different times and have different launch
behavior. Some long-shape replay values exceed eager wall (4 × 256 is
17.600 versus 16.328 ms); subtracting them would not be a valid host-overhead
estimate. The stable short-shape replay floor and repeated paired wall ratios
support the packing result without that subtraction.

| Micro round | Enforced limit, W | Loaded SM clock min / median / max, MHz |
|---|---:|---:|
| 1 | 145.00–155.36 | 2542 / 2797 / 2827 |
| 2 | 145.00–151.94 | 1852 / 2790 / 2805 |
| 3 | 145.00–151.87 | 2430 / 2790 / 2805 |

Loaded utilization medians are 93%, 94%, and 95%; maximum SM clock remains
3090 MHz. These are device activity observations, not tensor-core or occupancy
measurements.

### Per-stage diagnostic attribution

CUDA events instrument the real shared prefill loop and final-row selection.
For each shape there are three rounds of 20 passes, using greedy final rows.
Each cell is the median of the three round means, with its min–max range;
layer stages sum their intervals across all 12 layers. Event insertion can
perturb execution, intervals can include host submission gaps, and metadata
H2D/readback are excluded. Independently taking each stage's median need not
sum to the median total.

| Stage | Operations/pass | 16 × 16, ms (range) | 4 × 256, ms (range) |
|---|---:|---:|---:|
| Embedding | 1 | 0.024 (0.024–0.026) | 0.035 (0.030–0.036) |
| Transformer normalization | 24 | 0.192 (0.190–0.200) | 0.289 (0.266–0.292) |
| Q/K/V GEMMs | 36 | 2.298 (2.208–2.582) | 2.955 (2.766–2.960) |
| Q/K RoPE | 24 | 0.163 (0.154–0.186) | 0.316 (0.242–0.333) |
| K/V paged stores | 24 | 0.148 (0.145–0.220) | 0.206 (0.200–0.209) |
| Paged attention | 12 | 0.300 (0.299–0.300) | 6.297 (5.807–6.554) |
| O GEMM | 12 | 0.760 (0.760–0.761) | 1.150 (1.079–1.183) |
| Residual adds | 24 | 0.200 (0.159–0.206) | 0.311 (0.301–0.320) |
| FFN gate/up GEMMs | 24 | 1.939 (1.870–1.941) | 4.184 (3.922–4.332) |
| SwiGLU | 12 | 0.126 (0.125–0.163) | 0.371 (0.370–0.470) |
| Down GEMM | 12 | 1.888 (1.883–1.893) | 2.816 (2.693–2.896) |
| Final-row gather | 1 | 0.006 (0.006–0.006) | 0.006 (0.006–0.007) |
| Final-row normalization | 1 | 0.006 (0.006–0.006) | 0.008 (0.008–0.009) |
| Final vocabulary projection | 1 | 0.206 (0.206–0.206) | 0.087 (0.082–0.088) |
| Argmax | 1 | 0.037 (0.037–0.037) | 0.043 (0.040–0.044) |
| Total recorded intervals | 209 | 8.303 (8.123–8.670) | 19.182 (17.905–19.532) |

For 16 × 16, the 84 transformer GEMMs occupy a median **82.8%** of summed
instrumented intervals; attention is **3.6%**. For 4 × 256, the corresponding
shares are **58.2%** and **32.8%**. Attention is then the largest individual
stage at 6.297 ms, ahead of gate/up at 4.184 ms. This supports investigating
long-history attention next for the long-prompt tail, while GEMM execution
remains the main aggregate cost for short packed prompts. It does not justify
claiming achieved occupancy, tensor-core saturation, or DRAM bandwidth, and
no GEMM retuning or attention rewrite was added.

The profile's power limit is 145 W at loaded samples. Short-profile loaded
clocks are 2520–2775 MHz; long-profile clocks are 2017–2775 MHz with median
2437 MHz. This differs from the uninstrumented microbenchmark envelope and
helps explain why absolute profile totals cannot replace the performance
measurements. Nsight Systems captured CUDA driver APIs but **no GPU activity
records** in this WSL environment, so no kernel-level GPU counters or device
copy durations are inferred from that trace. The trace also reports a CUDA
13.2 driver newer than the profiler's supported 13.1 libraries, and contains
no NVTX phase markers; warm-call boundaries were reconstructed from source
control flow and driver-call order.

### CPU planning, memory, and transfers

The final runtime trace of the mixed 8/17/33/64/127/256/512/941 burst uses two
prefill calls per trial. The first packs 1,024 rows from eight requests and
finishes seven; the second is a graph-eligible singleton for the last 934 rows
of the 941-token request. Across three trials, six plans consumed 34.694 us
of planning time, 5.782 us per plan, versus 135.209 ms of prefill-call wall time.
One singleton planning sample was 31.159 us; the other five were 0.415–1.092 us.
The aggregate planning fraction was 0.0257% of prefill wall time and 0.0064%
of runtime wall time. A tighter 256-token trace used 27 calls, averaging
0.754 us of planning per call. These are trace-mode CPU observations; they do
not include all host metadata packing or establish a p99 bound at burst 16.

The first mixed trial decomposes runtime TTFT as follows. This starts at
runtime submission, so it excludes HTTP/tokenization/network latency; the
HTTP tables above retain those user-visible costs. Shared-call wall sums the
complete host wall duration of every prefill call containing that request,
not its exclusive device work, and must not be summed across requests. The
941-token request participates in both calls, so its 45.0233 ms is cumulative.

| Requests | Queue wait, ms | Prefill wait, ms | Shared-call wall, ms | Post-call, ms | Runtime TTFT, ms |
|---|---:|---:|---:|---:|---:|
| First seven, 8–512 tokens | 0.0047 | 0.0010 | 17.6223 | 0.0086 | 17.6367 |
| 941-token request | 0.0047 | 1.6979 | 45.0233 | 0.0028 | 46.7287 |

At token capacity `T` and resident capacity `B`, added device metadata is
`8*T + 4*B` bytes: **8,256 bytes at T=1024, B=16**. Added model-side host
payload is `12*T + 5*B + 4*pool_pages` bytes, **16,464 bytes** at 1,024 pool
pages. The runtime also holds a capacity-sized plan, a `B * table_stride`
int32 staging table (4,096 bytes here), and reservation bookkeeping. These
payload counts exclude allocation/container headers. Packed activations reuse
the existing prefill scratch; finishing hidden rows, logits and selection
buffers reuse decode scratch. There is no new `[packed_tokens,vocabulary]`
buffer and no per-call GPU allocation in the serving path. Small host
descriptor/readback allocations remain; “no host allocations” is not claimed.

The packed upload uses five bulk writes, plus one when final rows exist:
tokens, owners, positions, the capacity-sized page table, per-row top-k policy,
and final-row indices. The total payload is
`12*Tpacked + 4*B*table_stride + 4*F + 4*B` bytes. This is **7,296 bytes** for
16 × 16 final tokens and **16,512 bytes** for 16 × 64, at `B=16` and stride 64.
It includes reused token/page-table buffers, so it is not all additional
traffic versus the reference. In the Nsight 16 × 16 trace, the 20 warm timed
calls contain 120 H2D driver calls, matching six per pass. Summed H2D API
duration per pass is median **42.09 us**, mean **82.51 us**, range
21.89–686.49 us. These are host API durations, not device transfer latency;
the trace has no GPU copy activities. The same measured interval contains
4,180 kernel-launch APIs (209 per pass), 20 D2H APIs, and no allocation/free
APIs. This corroborates warm-buffer reuse for this shape; it does not prove
all code paths are allocation-free.

Greedy first-token selection copies `4*F` bytes. Device top-k adds
`8*F*128` bytes when any final row uses it. Each selected full-logit fallback
row adds `4*50304` bytes. Non-final packed chunks return zero result bytes.
The pressure test's seven greedy final rows copied exactly **28 bytes**;
the three cross-protocol probes copied 4,112, 204,304 and 205,328 bytes for
their different mixes of bounded top-k and full-logit rows. Packing did not
restore full-vocabulary readback for every request.

## Request isolation and regression validation

The model oracle is an independently executed monolithic request, compared
by generated token IDs. Tests exercise the original reduction semantics;
no tolerance was relaxed to accept different tokens. The int8 and f32 packed
checks both reported maximum final-logit absolute difference **0.000000e0**.

| Validation | Final result |
|---|---|
| CPU/default Rust suite | 130 unit tests and 14 mock-server integration tests passed. |
| CUDA-feature Rust suite | 176 unit tests and 14 mock-server integration tests passed. |
| TUI without default features | `cargo check --no-default-features --features tui --locked` passed. |
| Int8 packed model check | 1,084 exact generated sequences; 878 packed GPU calls; 24 fixed-seed fuzz sets; passed again after the profiling refactor. |
| F32 packed model check | 847 exact generated sequences; 620 packed GPU calls; deterministic matrix, zero additional fuzz sets. |
| HTTP independent-reference check | 384 exact comparisons across 42 workload runs; 24 deterministic fuzz sets; 78 packed batches; every page reclaimed. |
| Native HTTP | All 42 checks passed. |
| OpenAI compatibility and official SDK | All 125 checks passed. |
| Anthropic compatibility and official SDK | All 125 checks passed. |
| TUI smoke | All 19 checks passed, including shutdown, healthy server and returned pages. |
| `gpu-validate` | Passed kernels, sampling edges and 104 random sampling rows. |
| `gpu-graph-check` | Bit-identical eager/graph decode across graph-cache transitions and slot permutations. |
| `gpu-batch` | Exact generated output alone and batched, with simultaneous and staggered admission. |
| `gpu-sampling` | Exact seeded output for device-top-k and full-logit paths; cancellation/reuse passed. |
| `gpu-paged` | Paged versus contiguous prefill and decode bit-identical through length 1023; every page returned. |
| `gpu-prefill-check` | Chunked versus monolithic exact across lengths, chunks, offsets and mid-prompt cancellation. |
| `gpu-prefill-graph-check` | Exact eager/graph results across lengths, chunks, sampling policies, offsets and unrelated graph reuse. |
| Compute Sanitizer | Unavailable: WDDM debugger initialization failed; no instrumented kernel coverage. |

The sanitizer attempt reported six initialization/device-support errors,
including a disabled WDDM debugger interface requiring an administrator-run
enablement script. Those are tool initialization failures, **not six detected
kernel memory errors**. The functional batch check still ran and matched its
references, but it cannot substitute for sanitizer coverage. No sanitizer pass
is claimed.

The adversarial packed matrix covers mixed page-boundary lengths
15/16/17, 31/32/33, 63/64/65, 127/128/129, 255/256/257 and 511/512;
aligned caps 32/64/128/256 and misaligned caps 37/73/131; nonzero prefixes;
three permutations; changed neighbors between chunks; and radically different
histories through independent pages. Composition/stale-buffer sequences
1/2/4/8/16/1/8/2 and simultaneous final-row counts 1/2/3/4/8/16 pass.

Sixteen concurrent requests mix greedy and top-k 5/40/128/500, then shrink
from sixteen to eight; eager/graph and reversed-order variants yield 64 exact
sequences. Cancellation at boundaries 0/1/3/6 with immediate page reuse yields
20 exact surviving sequences. `max_tokens=1`, near-context validation,
constrained 32-page admission/growth, malformed descriptors and invalid HTTP
requests are covered. Rejection occurs before invalid metadata reaches a
kernel; one malformed request does not fail valid neighbors.

### A pre-existing decode determinism defect

The expanded HTTP corpus exposed a seeded-output mismatch on the unchanged
baseline. The int8 decode vocabulary projection switched from GEMV to a
half-activation WMMA path above eight requests. A request could therefore
receive different logits when unrelated neighbors changed its decode batch
size. Packed prefill made this old batch-composition weakness easier to hit.

The final implementation uses the independent-request GEMV arithmetic for the
vocabulary projection at every supported decode batch size, 1–16. The same
final-row rule is used for packed prefill. This restores the required seeded
independence, while preserving the existing transformer GEMM kernels and
weight format. It is a correctness repair whose end-to-end cost is measured below.

The repeated decode-only comparison uses three alternating A/B pairs, each
with three inner trials, batches 1/4/8/16, and 256 generated tokens per
request. Cells are medians of the three round medians, with round ranges.
Both binaries time 254 decode intervals: untimed admission already generates
the first decode token after prefill's first token. The frozen baseline's
displayed throughput counted 255, so it is corrected by **254/255**; the
candidate corrects this accounting in source. The token workload is identical.

| Decode batch | Baseline tok/s (range) | Candidate tok/s (range) | Median change |
|---:|---:|---:|---:|
| 1 | 1325.8 (1308.8–1345.7) | 1317.0 (1285.0–1340.0) | -0.66% |
| 4 | 2775.1 (2762.1–2781.1) | 2753.0 (2729.0–2791.0) | -0.80% |
| 8 | 4623.8 (4586.9–4653.7) | 4601.0 (4573.0–4662.0) | -0.49% |
| 16 | 7019.4 (7015.4–7065.2) | 7084.0 (6998.0–7086.0) | +0.92% |

Every baseline/candidate range overlaps. This experiment detects no material
end-to-end decode-throughput regression at the measured resolution, including
batch 16; the +0.92% median there is not a demonstrated speedup. It also does
not isolate the vocabulary projection's own kernel cost. Scheduler planning
cannot explain these decode-only results because there are no arriving
prefills in the timed interval. The correctness repair is retained for exact
request independence, not because this noisy measurement proves it free.

Loaded median SM clocks are 2730–2756 MHz for baseline and 2733.5–2756 MHz
for candidate. Enforced limits range 145.00–152.09 W and 145.00–151.96 W,
respectively, with maximum SM clock 3090 MHz. The earlier three 128-token,
one-inner-trial pairs had material variance: baseline batch-1 results ranged
1203.4–1477.3 tok/s. Those samples remain in the evidence file for transparency
and are not selected as the final cost estimate. Increasing trial length and
repetition, rather than choosing a favorable sample, produced the table above.

The held-out int8 decode check evaluated 1,024 positions and reported
cross-entropy **3.720334**, perplexity **41.2782**, exactly matching the
rebuilt baseline's printed values. The prefill check reported cross-entropy
**3.319376**, perplexity **27.6431**, over 31 evaluated positions, also matching
the baseline. These are the repository's targeted checks, not a new full
held-out evaluation or a claim that decode and prefill use identical reductions.

## Sustained arrivals, protocols, and page pressure

The final 30-second sustained experiment offers **100 requests/s**, mixing
native, OpenAI completions, OpenAI chat, and Anthropic requests; short, medium
and long prompts; greedy and top-k 5/40/128/500; and scheduled disconnects.
It is an **overload/backpressure experiment**, not proof that this service can
complete 100 requests/s at low latency. All counters below are deltas for the
measured interval, except where an end-state gauge is explicitly stated.

| Metric | Observed result |
|---|---:|
| Offered / submitted | 3000 / 3000 |
| Completed | 1735 |
| Requested and observed cancellations | 116 |
| Expected queue-full responses | 1149: 881 HTTP 429, 268 Anthropic HTTP 529 |
| Runtime failed requests / monitor errors | 0 / 0 |
| Wall including drain | 31.294 s |
| Completed requests/s including drain | 55.44 |
| Generated tokens/s including drain | 3550.15 |
| Native TTFT p50 / p95 / p99 / max | 1097.40 / 1159.01 / 1170.10 / 1177.72 ms |
| Native inter-token gap p50 / p95 / p99 / max | 2.637 / 15.329 / 19.225 / 24.956 ms |
| Peak / median sampled queue depth | 64 / 63, configured limit 64 |
| Prefill calls / request slices / packed calls | 1342 / 1797 / 384 |
| Prompt rows / rows in packed calls | 394381 / 185881 |
| Mean request slices / real rows per prefill call | 1.34 / 293.88 |
| Sampled prefill request count min / median / max | 1 / 1 / 4 |
| Sampled real rows per prefill call min / median / max | 14 / 263 / 1024 |
| Sampled decode batch min / median / max | 1 / 16 / 16 |
| Peak KV pages | 336 / 1024 |
| Final active / prefilling / queued / used KV pages | 0 / 0 / 0 / 0 |

There were 1,529 metrics snapshots. The “last prefill” gauges can represent a
call completed before the snapshot and are not batch-size histograms. Polling
can miss short-lived prefilling states; the peak observed prefilling count of
one does not override the cumulative proof of 384 packed calls. Sampled-policy
counts are 600 offers of each policy, not 600 successful completions each.
Native SSE supplies TTFT and token-gap distributions; the other protocol
clients here measure complete-response latency, so they do not supply TTFT.

Three dedicated four-protocol probes each completed one request per adapter
and each recorded a packed call, demonstrating cross-protocol sharing.
A subsequent isolated proof included **all five client surfaces together**:
native, OpenAI completions, OpenAI chat, Anthropic, and a real pty TUI.
It passed on its first attempt. A 941-token singleton prefill held the owner
while the five short prompts arrived. Exactly six prompt slices and final
rows executed in two calls: one singleton and one packed call of **94 rows**
(16 native + 16 completion + 23 chat + 23 Anthropic + 16 TUI).
Because 94 is smaller than the blocker and each prompt was consumed once,
the packed call necessarily contained all five clients. Decode reached batch
five; five requests completed, the TUI cancellation was observed, all pages
returned, no request failed, and first-token D2H totaled 24 bytes for six rows.
Adapter usage counted each formatted prompt separately. The earlier TUI smoke
suite also passed all 19 checks. The probe retains its exact counter evidence
and submission/acceptance timing, rather than inferring participation merely
from concurrent clients.

### Below-capacity mixed arrivals

A second run offered 35 requests/s for **60 seconds**, with the same four
HTTP adapters, greedy and top-k 5/40/128/500, and scheduled disconnects.
All 2100 requests were submitted without client backpressure:
1976 completed and 124 disconnected, with **zero errors**.
The peak waiting queue was **2** out of 64. Including drain, completed
rate was 32.88 requests/s and generation rate
2105.25 tokens/s. Native TTFT p50/p95/p99 was
9.735/10.741/11.169 ms; native gap p50/p95/p99 was
1.854/8.393/14.615 ms. The final queue, resident counts
and used pages were zero. At these spaced arrivals there were no multi-request
prefill calls during the measured steady interval: singleton fallback served
the work. The separate burst and five-client probes establish actual packing;
this run establishes bounded, stable service below the measured saturation
point. It is a one-minute stress observation, not a long-term availability claim.

The 32-page, queue-limit-four pressure test submitted a 24-request burst:
four accepted requests produced exact 128-token survivor outputs, twenty
received HTTP 429, resident cancellation reclaimed capacity, and later work
reused pages. Peak queue depth was exactly four, peak physical usage 24 pages,
and two packed batches executed. It ended with zero active, prefilling or
queued requests, all 32 pages free, and zero failed requests. This exercises
reservation and queue bounds together rather than relying on a large pool.

## Defaults, rejected alternatives, and remaining limits

| Control | Production default |
|---|---|
| Cross-request packing | On; `CRUCIBLE_BATCHED_PREFILL=0` restores reference scheduling. |
| Aggregate prefill token budget | 1024, clamped to model scratch capacity. |
| Maximum contributing requests | Model resident capacity, normally 16. |
| Additional per-request chunk cap | Off. Opt-in cap defaults to 128 when chunking is enabled. |
| Singleton prefill graph | On, exact `(length, want_logits)` key, bounded cache. |
| Packed serving graph | Not implemented; packed serving executes eagerly. |
| Dispatch | One contributor uses the singleton path; two or more use packed work. |
| Vocabulary projection | Independent GEMV arithmetic at every supported batch size. |

Packed rows improve workload shape, but the policy sweep rejects blanket
small-chunk scheduling. It also gives no reason to retune the existing WMMA
tiles, pad semantic tokens, introduce a dense cross-request attention mask,
or add an adaptive policy. Packed serving graphs remain deferred pending a
measured benefit and exact-topology capture design. The singleton graph remains
valuable because a one-request call has no cross-request utilization gain.

Known limits are the small 120M model and one laptop envelope; small numbers
of independent burst trials; non-streamed completion-only timing for three
protocol clients in sustained load; polling rather than complete batch
histograms; and the long-prompt tail/throughput tradeoff. The 30-second
saturation test and 60-second below-capacity test exercise different operating
points; neither is a service-level or long-term availability guarantee.
Event attribution identifies GEMMs as the main short-shape cost and attention
as the largest individual long-shape stage. GPU hardware counters and sanitizer
coverage remain unavailable in this environment. Repeated paired decode
throughput ranges overlap, so the vocabulary repair has no separately resolved
cost in this experiment; no attention rewrite is included in this milestone.

Prefix caching, shared KV across requests, FlashAttention, speculation,
multi-stream overlap, distributed serving and new quantization formats remain
outside scope. The README/roadmap marks the implemented, validated,
benchmarked server integration complete while retaining chunking as opt-in.
Those separate features remain incomplete.

## Portable reproduction

Run from the repository root. Commands assume a CUDA-capable Linux or WSL
environment with the existing CUDA toolchain configured. Substitute your model
export, tokenizer and held-out token file; no benchmark depends on a particular
user directory. Environment variables belong on the **server process**, not
on the Python client.

```sh
git rev-parse HEAD
git status --short
cargo test --manifest-path engine/Cargo.toml --locked
cargo test --manifest-path engine/Cargo.toml --features cuda --locked
cargo check --manifest-path engine/Cargo.toml --no-default-features --features tui --locked
cargo build --manifest-path engine/Cargo.toml --release --features cuda --locked

engine/target/release/llm-engine gpu-packed-prefill-check export/120m --quant int8 --steps 16 --fuzz 24
engine/target/release/llm-engine gpu-packed-prefill-check export/120m --quant f32 --steps 4 --fuzz 0
engine/target/release/llm-engine gpu-prefill-bench export/120m --lengths 1,8,16,32,64,128,256 --iters 30
engine/target/release/llm-engine gpu-packed-prefill-bench export/120m --batches 1,2,3,4,8,16 --chunks 1,8,16,32,64,128,256 --iters 15
engine/target/release/llm-engine gpu-packed-prefill-bench export/120m --batches 16 --chunks 16 --iters 20 --packed-only
engine/target/release/llm-engine gpu-packed-prefill-bench export/120m --batches 4 --chunks 256 --iters 20 --packed-only
engine/target/release/llm-engine gpu-prefill-trace export/120m --budget 1024 --trials 3
```

Record a reference against a dedicated server with packing disabled. Restart
with packing enabled before running the comparison, preserving all other
model, queue, context, sampling and graph settings. Run one GPU-owning server
at a time. The token limit below permits the long arrival and interruption
workloads as well as the near-context tests.

```sh
CRUCIBLE_BATCHED_PREFILL=0 engine/target/release/llm-engine serve export/120m --tokenizer export/gpt2.tok --max-prompt-tokens 941 --max-new-tokens 900
python scripts/test_packed_prefill.py --record-reference packed-reference.json --fuzz-rounds 24

# Restart the server with the default packing configuration.
engine/target/release/llm-engine serve export/120m --tokenizer export/gpt2.tok --max-prompt-tokens 941 --max-new-tokens 900
python scripts/test_packed_prefill.py --reference packed-reference.json --fuzz-rounds 24
python scripts/bench_packed_prefill.py --trials 3 --label packed-default --require-packed --output packed-default.json
python scripts/bench_packed_prefill.py --scenarios sustained --trials 1 --duration 30 --arrival-rate 100 --max-inflight 96 --queue-limit 64 --require-packed --label packed-sustained --output packed-sustained.json
python scripts/test_serve.py
python scripts/test_openai.py
python scripts/test_anthropic.py
python scripts/smoke_tui.py --binary engine/target/release/llm-engine
```

For the policy sweep, restart the candidate for every combination using
`CRUCIBLE_PREFILL_TOKEN_BUDGET` and `CRUCIBLE_PREFILL_CHUNK` with
`CRUCIBLE_CHUNKED_PREFILL=1`; unset chunking for the no-cap case. Compare each
candidate with a freshly warmed exact-commit baseline, alternate A/B ordering,
and retain per-round JSON. Use `--scenarios short,mixed,long,stall,hol,fairness`
to focus service measurements; add `tiny` for one-token HTTP bursts. Repeat the microbenchmark in separate rounds.
Record `nvidia-smi` enforced power limit, maximum/current SM clocks, utilization
and draw throughout each experiment, alongside before/after snapshots.

For pressure, restart with `--kv-pages 32 --max-queue 4` and a generation
limit of at least 512, then run:

```sh
python scripts/test_packed_pressure.py --pages 32 --queue-limit 4 --output packed-pressure.json
```

Retain experiment JSON and profiler traces outside the source tree. The
portable harnesses belong under root `scripts/`; temporary baseline builds
and machine-specific orchestration do not. The final review should enumerate
the intentional changed files and `git status --short`, and propose one
commit without creating it automatically.


## Completed working tree and proposed commit

HEAD remains `8f5c745c6309f25f4dae0466a2275598bf333042`; no commit was created.
The temporary baseline source directory was compared byte-for-byte with all
68 Git blobs at that commit before removal. The frozen executable and raw
experiment logs are retained outside the repository. No machine-specific
benchmark harness, model export, temporary build tree, or generated Python
cache is included in the changes.

Intentional files in `git status --short --untracked-files=all`:

```text
 M README.md
 M engine/kernels/kernels.cu
 M engine/src/gpu.rs
 M engine/src/gpu_model.rs
 M engine/src/lib.rs
 M engine/src/main.rs
 M engine/src/protocol.rs
 M engine/src/runtime.rs
 M engine/src/server.rs
?? docs/packed-prefill-design.md
?? docs/packed-prefill-evidence.json
?? docs/packed-prefill-results.md
?? engine/src/gpu_packed_validation.rs
?? engine/src/gpu_prefill_trace.rs
?? engine/src/prefill.rs
?? scripts/bench_packed_prefill.py
?? scripts/test_packed_prefill.py
?? scripts/test_packed_pressure.py
```

The three closing checks passed after the profiler refactor: 130 CPU unit and
14 TUI tests; 176 CUDA-feature CPU/mock-server unit and 14 TUI tests; and the
TUI-only build. The release build, final 1,084-sequence packed GPU regression,
functional batch limits three and five, Python syntax checks, and whitespace
checks passed. No test tolerance was weakened. Sanitizer instrumentation was
unavailable as described above.

Proposed single commit, **not executed**:

```text
feat(runtime): add packed prefill with bounded token scheduling

Pack independent prompt rows into shared transformer GEMMs while routing
RoPE, paged KV writes and causal attention by request and absolute position.
Gather only completing rows and preserve compact per-request token selection.

Schedule one decode-first, round-robin prefill plan per step with explicit
token/request limits. Reserve lifetime KV growth and bound the combined HTTP
and runtime waiting queue. Reclaim cancelled work at completed GPU boundaries
and fail closed on runtime execution errors.

Keep singleton prefill graphs and the monolithic/chunked reference controls.
Enable packing at two contributors with a 1024-token default budget; leave
the additional chunk cap disabled after measuring its tail/throughput cost.
Remove the batch-size-dependent int8 vocabulary projection crossover to
preserve seeded outputs, and bound non-power-of-two GEMV scratch accesses.

Add planner, isolation, malformed-input, cancellation/reuse and pressure
validation; portable HTTP benchmarks; runtime timing and CUDA-event profiling.
Document paired measurements, power envelopes, defaults and limitations.

Validated: CPU/CUDA-feature/TUI checks; all existing GPU suites; 1084 int8
and 847 f32 exact sequences; native and official OpenAI/Anthropic SDK suites;
384 HTTP sequence comparisons; five-client packed execution; sustained load,
page pressure, and unchanged held-out CE. No sanitizer pass is claimed.
```
