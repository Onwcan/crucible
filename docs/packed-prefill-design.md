# Cross-request packed prefill

Packed prefill concatenates real prompt rows from several resident requests and
runs one transformer over their combined row count. Its purpose is to give the
existing prefill GEMMs more work per launch. The scheduler ships with packing
enabled, a 1,024-token aggregate budget (clamped to model capacity), and optional
per-request chunking disabled. The measurements and default decision belong in
the [README](../README.md); this document records the implementation contract.

The implementation is in [prefill.rs](../engine/src/prefill.rs),
[runtime.rs](../engine/src/runtime.rs),
[gpu_model.rs](../engine/src/gpu_model.rs),
[gpu.rs](../engine/src/gpu.rs), and
[kernels.cu](../engine/kernels/kernels.cu). All HTTP adapters enter the same
[server](../engine/src/server.rs) and expose the same
[protocol metrics](../engine/src/protocol.rs).

## Shapes and what can be packed

For the 120M architecture, `D = 768`, `KD = 192` (three KV heads of width 64),
`H = 2048`, `V = 50304`, and there are 12 layers and 12 query heads. Let `R` be
the requests contributing one slice each, `t_i` the slice lengths,
`Tpacked = sum(t_i)`, and `F` the number of slices that finish their prompts.
`0 <= F <= R <= max_batch <= 16`, and `Tpacked` is bounded by the configured token
budget and the existing prefill scratch capacity. Rows are consecutive, with
no semantic padding to the longest request or to a graph bucket.

The GEMM interface computes `C[M,N] = A[M,K] * W[N,K]^T`. Every projection below
uses `M = Tpacked`, including when the requests have different prompt lengths
or offsets. Weight layout and quantization are unchanged.

| Stage | Input/output shape or GEMM `(M,N,K)` | Packing rule |
|---|---|---|
| Embedding | token IDs `[Tpacked]` to `[Tpacked,D]` | Independent token lookup. |
| Attention RMSNorm | `[Tpacked,D]` to `[Tpacked,D]` | Independent row reduction. |
| K projection | `(Tpacked,KD,D)` | One shared-weight GEMM; output uses the reusable `[Tpacked,KD]` scratch. |
| K RoPE and cache write | `[Tpacked,KD]` | Rotation uses each row's absolute position; the write uses its request's page table. |
| V projection and cache write | `(Tpacked,KD,D)` | One GEMM reuses the K/V scratch after K has been stored; V is not rotated. |
| Q projection and RoPE | `(Tpacked,D,D)` | One GEMM; rotation uses per-row absolute positions. |
| Causal GQA attention | Q/output `[Tpacked,D]`; each row reads its own prefix | One launch over `(query head, packed row)`, with request-specific page lookup and causal length. |
| Attention output projection | `(Tpacked,D,D)` | One GEMM, followed by a `[Tpacked,D]` residual add. |
| MLP RMSNorm | `[Tpacked,D]` | Independent row reduction. |
| Gate projection | `(Tpacked,H,D)` | One GEMM. |
| Up projection | `(Tpacked,H,D)` | One separate GEMM. |
| SwiGLU | two `[Tpacked,H]` inputs to `[Tpacked,H]` | Elementwise `silu(gate) * up`; gate/up projections remain separate launches. |
| Down projection | `(Tpacked,D,H)` | One GEMM, followed by a `[Tpacked,D]` residual add. |
| Final-row gather | `[Tpacked,D]` to `[F,D]` | Gather only the final prompt position of each finishing request, in descriptor order. |
| Final RMSNorm | `[F,D]` | Omitted when `F = 0`. |
| Vocabulary projection | logically `(F,V,D)`, output `[F,V]` | Uses final-row GEMV arithmetic, not a new WMMA GEMM; omitted when `F = 0`. |
| First-token selection | `F` independent distributions | Greedy IDs, bounded top-k candidates, or explicitly selected full-logit rows. |

There are **seven transformer projection GEMMs per layer**: K, V, Q, O, gate,
up, and down, or **84 per packed call for 12 layers**. The final vocabulary
projection is separate. It is incorrect to count six projections per layer or
to describe this as a host loop running independent prefills.

`queue_prefill_transformer` is shared by reference and packed execution. The
existing GEMM dispatcher chooses 16x64 or 64x64 WMMA tiles per launch using
`M = Tpacked` and each projection's `N`. Tile boundaries may include masked
lanes internally, but extra rows never become tokens, KV writes, attention
queries, or final rows. The final vocabulary projection preserves the
standalone GEMV reduction order when the number of finishing requests changes;
for applicable int8 widths it uses the existing batched int8 GEMV.

The same rule now applies to int8 decode at all supported batch sizes. The
previous decode dispatcher switched the vocabulary projection to WMMA above
eight active requests, rounding its activations to half precision. The new
adversarial HTTP corpus reproduced a seeded-output mismatch on the unchanged
baseline. Removing that crossover keeps independent-request arithmetic when
batch membership changes. This is a correctness repair with a separately
reported batch-16 performance cost, not a GEMM retuning experiment.

## Sequence ownership and causal attention

The planner records stable request IDs, packed row starts, prompt offsets,
slice lengths, and whether each slice is final. GPU descriptors borrow the
corresponding token and page-table slices. Descriptor indices are temporary
owners for one call; they are not stable request IDs or decode slots.

For packed row `r` belonging to descriptor `i` at local offset `j`:

```text
owners[r]    = i
positions[r] = prompt_start_i + j
table        = page_tables + i * table_stride
causal length = positions[r] + 1
```

RoPE reads `positions[r]`. Both K and V cache stores use the same owner and
position. Packed attention passes that owner's table and causal length into
`attention_prefill_paged_impl`, which is also used by single-request paged
prefill. Dot products, softmax reductions, value accumulation, and the mapping
`kv_head = query_head / (n_head / n_kv_head)` are shared. A row therefore reads
positions `0..=positions[r]` of its own request, including previously cached
chunks. Neighbors in the packed tensor cannot extend that history.

K and V each retain the decode-compatible layout:

```text
pool[n_pages][n_layer][16][KD]
offset(page, layer, position, component)
  = ((page * n_layer + layer) * 16 + (position & 15)) * KD + component
page = request_page_table[position >> 4]
table_stride = ceil(model_context_capacity / 16)
```

The pool is allocated once. No packed KV pool, per-row replicated page table,
cache conversion, or temporary padded attention history is introduced. The K
and V kernels complete before attention on the same CUDA stream. Although a
chunk's later K/V positions have been written, a query's causal bound prevents
it from reading them.

Before the first device write, `prepare_packed_prefill` validates nonempty
chunks, request/token capacities, checked offsets, token IDs, exact table
stride, every used page index, and duplicate physical pages within or across
the supplied request histories. It also validates final-row selection indices,
top-k bounds, and duplicate/conflicting selection rows. A persistent page-owner
array makes alias checking bounded by the configured pool. Every consumed
metadata element is overwritten before its next use; unused capacity is not
semantic input. The single-request dispatch uses this validation too.

## Scheduling and fairness

The packed scheduler iteration is:

```text
admit   pending -> prefilling, while resident slots and lifetime pages fit
decode  one batched step for requests that already have a first token
prefill one plan, bounded by aggregate tokens and contributing requests
retire  completed requests and reclaim their pages and reservation
```

Admission is FCFS. A pending request that cannot obtain its full reservation
stops admission for that iteration; younger requests do not pass it. Prefilling
and active requests share the resident bound:
`active.len() + prefilling.len() <= max_batch`.

`PrefillBatchPlan::build` visits the round-robin queue once, in order, and takes
at most one slice from each visited request:

```text
slice length = min(prompt remaining, per-request cap, aggregate budget remaining)
```

The request limit also stops planning. With chunking disabled, the per-request
cap is the scratch capacity, so the aggregate budget still bounds work and can
split a request at the end of a plan. Enabling chunking adds the configured
smaller per-request cap; it does not replace the aggregate limit. After the
completed GPU call, each unfinished contributor moves to the back of the
queue, finished contributors become active, and new admissions join the back.
Progress and output are applied by request identity, independent of later
decode-slot compaction or cancellation.

Fairness follows directly from the queue order. Consider a surviving resident
prefill request with at most `R - 1` peers ahead of it. Every nonempty plan
that has not reached it consumes positive work from at least the front peer;
that peer either leaves or moves behind the survivor. Arrivals join behind it,
and cancellation only removes peers. Thus the survivor receives work within
at most `R` nonempty plans when at most `R` requests are resident. Repeating
the argument gives progress until completion. This is a bound in plans, not
a wall-clock latency guarantee, and applies after admission. Pending requests
can wait for earlier residents to release slots or reservations.

Each packed iteration services existing decoders before prompt work. A finished
prefill produces the first token directly and enters decode for the following
iteration. Retirement in the same iteration handles `max_tokens = 1`, avoiding
an extra token and an unnecessary cache write.

## Memory ownership and admission bounds

For prompt length `P` and generation budget `G`, a request reserves
`ceil((P + G - 1) / 16)` logical pages through completion. The first output
comes from prefill logits; only the remaining `G - 1` outputs require decode
writes. Zero lengths, integer overflow, context overflow, and a requirement
larger than the entire pool are rejected before admission. Direct runtime
submission also validates sampling, vocabulary bounds, and live ID uniqueness
across pending, prefilling, and active requests.

Logical reservations and physical page assignment are separate. All prompt
pages are physically assigned at admission, even if prompt execution will use
several slices. Subsequent generation pages are assigned lazily as decode
crosses page boundaries. Reservations ensure that a new prompt cannot consume
physically free pages already promised to a resident decoder's growth. A
failed physical admission releases its tentative logical reservation. Normal
completion and resident cancellation release both the assigned pages and the
full logical reservation.

The HTTP adapters retain their existing conservative `P + G <= context`
validation. The internal reservation counts actual KV writes, `P + G - 1`;
packing does not relax the public context limit.

The HTTP waiting bound is enforced with one semaphore of `max_queue` permits,
shared across the job channel and `Runtime.pending`. A handler acquires a
permit before submitting. Draining the channel transfers that permit into
the inference thread's `Live` entry; it does not make another waiting slot
available. The permit is dropped when a successful scheduler step reports
admission, or when the waiting job is rejected, cancelled, or dropped.
Consequently the count remains conservative during the admitting GPU step.
The channel is also bounded by `max_queue`, and submission uses `try_send`.
`queued_requests` is `max_queue - available_permits`, including jobs still in
the channel. Resident capacity is separately bounded by `max_batch`. Native
and OpenAI overload responses are 429; the Anthropic adapter maps the same
queue-full condition to 529.

Packed tensor work reuses the existing prefill scratch. Final-row hidden
states, logits, argmax IDs, and top-k buffers reuse batched decode scratch after
that iteration's decode has completed. There is no new `[Tpacked,V]` allocation.
For token capacity `T`, resident capacity `B`, and pool size `pages`, added
persistent metadata payloads are:

| Allocation | Bytes |
|---|---:|
| Device owners and absolute positions, both int32 `[T]` | `8*T` |
| Device final-row indices, int32 `[B]` | `4*B` |
| **Added device total** | **`8*T + 4*B`** |
| Host token IDs, owners, positions, each int32 `[T]` | `12*T` |
| Host final-row indices int32 `[B]`, selection kind uint8 `[B]` | `5*B` |
| Host physical-page owner check int32 `[pages]` | `4*pages` |
| **Packed model host payload** | **`12*T + 5*B + 4*pages`** |

At `T = 1024`, `B = 16`, the added device payload is **8,256 bytes**. These
formulas exclude allocator/container headers. The runtime additionally retains
a plan with capacity `B` (`B * size_of::<PrefillSlice>()` payload), its
`B * table_stride` int32 staging table, and the reservation ledger. The model's
device/host page tables and batched selection buffers already exist for decode.
Transient work includes small descriptor and selection-row vectors and output
readback vectors; prompt token storage remains owned by resident requests.
The design makes persistent GPU allocations outside the inference step, not
an assertion that the host performs no allocations during a step.

At the default 1024-page pool, packed model host payload is **16,464 bytes**.
For a non-power-of-two batch limit, the pre-existing GEMV source buffers now
round to their kernel's 1/2/4/8/16 instantiation capacity to keep inactive
loads in bounds. This adds `4 * (next_power_of_two(B) - B) * (3*D + H)` bytes:
17,408 bytes at `B=3`, 52,224 at `B=5`, and zero at the default `B=16`.

## Dispatch, selection, and failure behavior

A plan with one contributor uses `prefill_single_mixed`: the existing
exact-length prefill graph (or its eager fallback), followed by device-to-device
logit placement into the selection scratch. A plan with two or more contributors
uses eager `prefill_packed`. The single-request graph key remains
`(slice length, want_logits)`, with dynamic tokens, pages, and offsets uploaded
outside capture. Its cache is bounded at 64 entries. Capture failures are
remembered and fall back to eager execution. Packed serving does not create a
graph cache over request compositions; `time_packed_replay` creates only a
temporary benchmark graph for pure GPU timing.

Only finishing rows reach the vocabulary projection. Each has its own
generation policy and RNG initialized from that request's seed, then carried
into decode. Greedy rows read one int32 ID. If any final row uses device top-k,
readback includes fixed-capacity candidate values and IDs for all `F` final
rows, with capacity `Kmax = 128`; full-logit fallback copies only the selected
rows. For `U` full-logit rows the measured transfer accounting is:

```text
prefill D2H bytes = 4*F + (any device top-k ? 8*F*Kmax : 0) + 4*U*V
```

A non-final plan omits all selection work, explicitly synchronizes the stream,
and transfers zero result bytes. Final selection readback also waits for the
GPU work before the scheduler can recycle pages. The single GPU-owning thread
executes a plan as one bounded unit; cancellation is checked between scheduler
iterations, never while a submitted kernel sequence is using those pages.
A disconnected prefilling request produces no subsequent first token and its
pages/reservation are reclaimed at that boundary. Already submitted work can
finish before the disconnect is observed.

Malformed submissions fail only that request before it owns pages. An execution
or cancellation error that escapes the runtime is fatal to the inference
thread: it closes admission, records the fatal state, attempts to notify all
live and channel-queued jobs, drains their permits, and drops the runtime/CUDA
state. `/health` then reports service unavailable. Notification uses bounded
nonblocking sends, so a closed or full client channel cannot delay cleanup.

Metrics distinguish one prefill execution (`prefill_batches`) from its request
slices (`prefill_requests` and the compatible `prefill_chunks` counter).
`packed_prefill_batches` and `packed_prefill_tokens` count only calls with at
least two contributors. `prefill_final_rows` counts requests reaching selection;
`prefill_tokens` counts real prompt rows. Last/max batch dimensions, averages,
and `prefill_d2h_bytes` allow the service benchmark to verify which execution
path actually ran. These are scheduler counters, not timings inferred from
kernel launches.

## Controls and reproducible validation

| Control | Meaning |
|---|---|
| `CRUCIBLE_BATCHED_PREFILL=0` | Disable packed scheduling; preserve the reference single-request path. Packing is enabled when unset. |
| `CRUCIBLE_PREFILL_TOKEN_BUDGET` or `--prefill-token-budget` | Aggregate real-row limit; default 1024 clamped to capacity. Explicit values must be within capacity. |
| `CRUCIBLE_MAX_PREFILL_REQUESTS` or `--max-prefill-requests` | Contributors per plan; default model request capacity (`max_batch`). |
| `CRUCIBLE_CHUNKED_PREFILL=1` | Enable the additional per-request slice cap; disabled when unset. |
| `CRUCIBLE_PREFILL_CHUNK` | Per-request cap when chunking is enabled; default 128, clamped to capacity. |
| `--prefill-chunk-tokens` | Set the per-request cap and enable chunking; the CLI flag implies opting in. |
| `CRUCIBLE_PREFILL_GRAPH=0` | Disable single-request prefill graph reuse; packed serving is already eager. |
| `CRUCIBLE_DEVICE_TOPK=0` | Send sampled final rows through full-logit selection. |
| `CRUCIBLE_GEMM=tiled`, `wmma-small`, or `wmma-big` | Scalar/fixed-tile diagnostic controls; unset uses existing automatic WMMA dispatch. |

With packing disabled, chunking disabled runs the preserved monolithic
reference before decode; enabling chunking runs one oldest-request slice after
decode. This control changes scheduling as well as the cross-request GEMMs,
so wall-clock service results must record the full configuration.

Run portable planner and reservation tests from the repository root; they do
not need CUDA:

```text
cargo test --manifest-path engine/Cargo.toml --no-default-features
```

On a CUDA-capable Linux/WSL setup, build and run the kernel/model checks. Replace
`export/120m` with a local export containing its configuration and weights:

```text
cargo test --manifest-path engine/Cargo.toml --features cuda
cargo build --manifest-path engine/Cargo.toml --release --features cuda
engine/target/release/llm-engine gpu-packed-prefill-check export/120m --quant int8 --steps 16 --fuzz 12
engine/target/release/llm-engine gpu-packed-prefill-check export/120m --quant f32 --steps 16 --fuzz 12
engine/target/release/llm-engine gpu-packed-prefill-bench export/120m --batches 1,2,3,4,8,16 --chunks 1,8,16,32,64,128,256 --iters 15
```

The check uses independent monolithic requests as the oracle. It exercises
page boundaries, nonzero and misaligned offsets, request permutations,
heterogeneous sampling, cancellation and immediate reuse, fixed-seed fuzzing,
and malformed metadata rejection. Compare token IDs, not merely fluent text.

For HTTP validation, run one dedicated server at a time with a prompt limit
of at least 941 and generation limit of at least 16. First set
`CRUCIBLE_BATCHED_PREFILL=0`, start the server, and record independent results;
then restart with packing enabled and the same model and test arguments:

```text
engine/target/release/llm-engine serve export/120m --tokenizer export/gpt2.tok --max-prompt-tokens 941 --max-new-tokens 512
python scripts/test_packed_prefill.py --record-reference packed-reference.json
```

After restarting the server with packing enabled:

```text
python scripts/test_packed_prefill.py --reference packed-reference.json
python scripts/bench_packed_prefill.py --trials 3 --label packed-1024 --require-packed --output packed-service.json
```

Environment settings apply to the server process, not the Python client. Set
them with the host shell's usual environment syntax. The HTTP harnesses use
only Python's standard library and can run from a different host with
`--host` and `--port`. They check prompt token counts through the service.
For page pressure, restart with `--kv-pages 32 --max-queue 4` and
`--max-new-tokens 512`, then run:

```text
python scripts/test_packed_pressure.py --pages 32 --queue-limit 4 --output packed-pressure.json
```

Keep per-round benchmark JSON, warm each configuration, alternate reference and
candidate ordering, and record power/clock state. Compare isolated latency,
bursts, heterogeneous/long prompts, established-stream gaps, and sustained
traffic separately. A temporary pure-GPU replay timing establishes kernel cost;
it includes the device argmax stage but excludes queueing, metadata transfer,
result readback, host sampling, and HTTP delivery. It is not a service
throughput result. `--packed-only` also prints opt-in CUDA-event stage intervals;
these include instrumentation and possible submission gaps, so compare their
attribution with the uninstrumented replay measurement.
