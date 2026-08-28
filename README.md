# crucible

Training small language models from scratch and building a custom inference
engine, targeting **NVIDIA Blackwell (`sm_120`)** hardware.

> **Status:** 120M model trained (1.44B tokens, val loss 3.28). Rust engine
> decodes at **~1460 tok/s** (1.7x llama.cpp at batch 1, token-identical greedy
> output) and prefills at **~39,700 tok/s** at seq 512 on a hand-written
> tensor-core GEMM — 2.4x the scalar kernel it replaced. Remaining gap to
> llama.cpp on prefill is 2.8x, down from 7.2x.

## Why Blackwell specifically

`sm_120` is recent silicon, and the surrounding software stack is still
maturing — llama.cpp, vLLM, and Triton all have less-optimised Blackwell paths
than they do for Ampere or Ada. That makes it a useful target: FP8 tensor-core
work here is measurable against baselines that have not yet been tuned to death.

## Hardware

| | |
|---|---|
| GPU | NVIDIA RTX PRO 4000 Blackwell Laptop, 16 GB GDDR7 ECC |
| Compute capability | `sm_120` (12.0) |
| Driver | 596.86 (CUDA 13.2 capable) |
| CPU | Intel Core Ultra 9 285HX |
| RAM | 64 GB DDR5 |
| Host OS | Windows 11 + WSL2 (Ubuntu 26.04 LTS) |
| Power limit | 134 W (requires Dell Optimizer "Ultra Performance") |

## Setup

```bash
bash scripts/setup_wsl.sh
```

All GPU work happens inside WSL2. This is a deliberate choice, not convenience:

- **Triton** and the CUDA toolchain are first-class here. (An earlier version
  of this list also cited vLLM's lack of Windows support; that reason was
  withdrawn once vLLM turned out not to run under WSL2 either — see the
  comparison section.)
- **Triton** — which `torch.compile` uses to generate fused kernels — is
  first-class on Linux and unreliable on Windows.
- CUDA C++ on Windows requires the **MSVC** toolchain; on Linux `gcc` suffices.

The one place native Windows would be better is profiling: Nsight Compute can
hit restrictions collecting some hardware counters under WSL2 virtualisation.

### Environment traps this handles

1. **Ubuntu 26.04 ships only Python 3.14**, which PyTorch does not yet publish
   wheels for. The setup script installs a standalone Python 3.12 via `uv`
   rather than fighting the system interpreter.
2. **Never install an NVIDIA driver inside WSL.** The Windows driver already
   exposes the GPU through `/dev/dxg`. The script installs the CUDA *toolkit*
   only; installing a driver in the guest breaks passthrough.
3. **`sm_120` requires CUDA ≥ 12.8.** Earlier toolkits compile but fall back to
   PTX JIT, which is slower and easy to miss. `verify_gpu.py` checks the arch
   list explicitly rather than trusting that CUDA "works".

## Measured hardware baseline

PyTorch 2.13.0+cu130 reports arch list `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`,
so `sm_120` kernels are native rather than PTX-JIT'd.

Median of 15 trials × 50 iterations, 4096³ matmul, via `scripts/bench.py`:

| Precision | TFLOP/s | spread |
|---|---:|---:|
| FP32 | 17.6 | 21.7% |
| TF32 | 50.2 | 8.6% |
| FP16 | 78.8 | 22.2% |
| BF16 | 76.4 | 23.2% |
| FP8 e4m3 | 179.7 | 13.6% |
| FP8 e4m3 (fast accum) | **186.0** | 4.6% |

### What these numbers support

- **FP8 is worth targeting** — roughly 2.4× BF16, the headline path for the
  inference engine.
- **BF16 and FP16 are equivalent in speed.** An early single-shot run suggested
  BF16 was ~20% faster; repeated measurement showed that was an artifact. BF16
  remains the training default for numerical range, not throughput.
- **FP8 requires `torch._scaled_mm`.** Plain `a @ b` raises
  `NotImplementedError` for FP8 dtypes regardless of hardware support, and the
  right-hand operand must be column-major.
- **`fast_accum` is not established as faster.** It led in two consecutive runs
  but by less than the noise floor. Recorded as suggestive, not as a result.

### Measurement caveats

The GPU idles at ~700 MHz and ramps to ~1700 MHz under load (49 °C → 67 °C,
24 W → 134 W within a single run). Workloads execute in fixed order, so the
first one is measured coldest, which inflates every ratio computed against it.
Reported FP32 speedups reached 11.1× but are realistically closer to 10×.

`bench.py` now warms the GPU before the first workload; the table above predates
that fix and is kept for the caveat it documents. Absolute per-precision medians
are unaffected by ordering — only cross-workload ratios are.

## Training

```bash
# tokenise a corpus (streams from the Hub, writes uint16 shards)
python data/prepare.py --tokens 1e8 --out data/tiny

# train
python train.py --preset 30m --data data/tiny --steps 2000
```

| Preset | Total params | Non-embedding | Embedding share |
|---|---:|---:|---:|
| 30m | 28.5 M | 9.1 M | 68% |
| 120m | 113.0 M | 74.4 M | 34% |
| 350m | 317.4 M | 265.9 M | 16% |

Parameter counts are reported **non-embedding** by default, following scaling-law
convention. This matters at small scale: with a 50k vocabulary the embedding
table is 19 M parameters, so two-thirds of the "30M" preset is embeddings.
Comparing presets on total parameters would mostly compare embedding tables.

### Choosing a model size for a fixed GPU budget

Training the largest model that fits is the wrong instinct on one GPU. Measured
throughput per preset (`scripts/probe_batch.py`) against an 8-hour budget:

| preset | non-embedding | tok/s | 8h tokens | tokens/param | vs Chinchilla |
|---|---:|---:|---:|---:|---:|
| 30m | 9.4M | 250,000 | 2.00B | 211.8 | 1059% |
| 120m | 74.3M | 50,827 | 1.46B | **19.7** | **98%** |
| 350m | 265.9M | 17,995 | 0.52B | 1.9 | 10% |

At 8 hours the 350M preset reaches under 2 tokens per parameter against the
~20 that compute-optimal training calls for. It is so far undertrained that the
120M, trained properly on the same GPU-hours, should reach a lower loss.
`scripts/compute_budget.py` reproduces this table from measured throughput.

### Trained model

The 120M preset, trained on 1.44B tokens (22,000 steps × 65,536):

| | 30M (ablation control) | 120M |
|---|---:|---:|
| non-embedding params | 9.1 M | 74.3 M |
| tokens seen | 0.13 B | 1.44 B |
| tokens / param | 14 | 19.4 |
| **best val loss** | 4.3733 | **3.2839** |

A 1.09 improvement in validation loss, and the first run in this repo where GQA
is genuinely GQA (`n_kv_head=3`) rather than the degenerate MQA the 30M sweep
silently used.

### VRAM: this platform does not raise OutOfMemoryError

Probing batch sizes at the 350M preset:

```
 batch   ms/step   peak GB      tok/s
     4     227.6     12.61     17,995
     6     419.4     16.57     14,651
     8    2778.5     20.61      2,948
    12   44883.2     28.57        274
```

Peak allocation runs past the card's 16.3 GB to 28 GB without failing, because
the WSL2 driver spills to host RAM over PCIe rather than aborting. Batch 12 is
**200× slower** than batch 4 and still reports success. Any batch-size search
relying on `torch.cuda.OutOfMemoryError` will therefore recommend a
configuration that technically runs and is unusable. `probe_batch.py` detects
the throughput cliff and the VRAM overshoot directly instead.

### Throughput

Measured against the 76.4 TFLOP/s BF16 figure above rather than a datasheet
number:

| preset | MFU | tokens/s | ms/step |
|---|---:|---:|---:|
| 30m | 27% | 250,000 | 282 |
| 120m | 19% → **52%** | 26,000 → 70,600 | 2,520 → 929 |

The 120M figure is two numbers because the run had two regimes. Steps 0–16,000
held ~2,520 ms/step; from step ~16,000 to the end it held ~929 ms/step. A clean
2.7× shift, stable on both sides, with nothing in the training code changing at
that point.

The cause is almost certainly the **laptop power profile being switched
mid-run** — this machine toggles between a ~55 W eco cap and a ~135 W
performance cap, and 2.7× is the right magnitude for that change. It cannot be
confirmed from the log, because `train.py` recorded step time and MFU but never
recorded the power limit. That omission is what made a one-line explanation
look like a mystery, and it is why every GPU benchmark in this repo now prints
`enforced.power.limit` alongside its result.

The consequence for the schedule was large: **12.5 hours instead of the 6.1
projected**, entirely from the slow regime. Two lessons are now baked into the
repo: throughput must be monitored *during* long runs rather than sampled at the
start, and a projection built from the first 20 steps is worthless.

At the fast regime, 52% against 27% at 30M confirms the low figure at 30M is a
small-model artifact — kernel launch overhead dominating a model too small to
saturate the GPU — not a defect in the training loop.

Replacing the naive data loader with a vectorised gather plus background
prefetch thread changed 30M MFU by **0.3 points — nothing**, which is what ruled
the input pipeline out as the cause. The prefetch loader is kept because it is
not worse and matters at larger batch sizes, but it was not the fix.

## Architecture ablations

```bash
python sweep.py --steps 2000 --seeds 3      # run the sweep
python sweep.py --report                    # aggregate results
```

Five axes, each varied independently against a fixed control
(`gqa / rope / swiglu / pre-rmsnorm`):

| Axis | Variants |
|---|---|
| `attention` | mha, **gqa**, mqa |
| `pos_encoding` | **rope**, alibi, learned, none |
| `activation` | **swiglu**, gelu |
| `norm` | **rmsnorm**, layernorm |
| `norm_placement` | **pre**, post |

SwiGLU's hidden dimension is scaled to `2/3 × 4d` so its parameter count stays
comparable to a GeLU MLP — otherwise the comparison would measure extra
parameters rather than the activation function.

### Methodology

Every configuration runs across multiple seeds, and results are classified into
three outcomes rather than two:

1. **Diverged.** A seed whose best validation loss exceeds the control by more
   than 1.0 has failed to train, not merely done worse. It is excluded from the
   mean and SD, and its variant is flagged `UNSTABLE` regardless of how the
   surviving seeds did.
2. **A difference**, if |Δ| exceeds **2× the larger seed-to-seed standard
   deviation**.
3. **Within noise** otherwise. A single-seed result is `unknown`, never a
   finding.

Step 1 is not hypothetical — it was added *because* the first version got
`norm_placement=post` wrong. Averaging its diverged seed in produced an SD of
1.78 against the control's 0.016, and that inflated SD then swallowed the very
difference it should have exposed, reporting a training failure as "within
noise".

The two-sigma rule has likewise already overturned a result here: a single-shot
benchmark showed BF16 outperforming FP16 by 20%, which repeated measurement
revealed to be pure noise. Architecture differences at this scale are small
enough that the same trap applies throughout.

Each run executes in a separate process so CUDA state, compilation caches, and
RNG do not leak between configurations. Completed runs are skipped, so a sweep
is resumable.

### Results

27 runs (9 configs × 3 seeds), 2000 steps each, 5.6 h on one GPU.
Full table in [RESULTS.md](RESULTS.md); figures in `figures/`.

Control `gqa/rope/swiglu/pre-rmsnorm`: **4.3733 ± 0.0157** best validation loss.

| Axis | Variant | Δ vs control | Verdict |
|---|---|---:|---|
| `pos_encoding` | `none` | +0.2579 | worse |
| `pos_encoding` | `learned` | +0.1843 | worse |
| `pos_encoding` | `alibi` | −0.0188 | within noise |
| `attention` | `mha` | −0.0031 | within noise |
| `attention` | `mqa` | −0.0070 | within noise |
| `activation` | `gelu` | −0.0005 | within noise |
| `norm` | `layernorm` | −0.0049 | within noise |
| `norm_placement` | `post` | +0.1821 | **UNSTABLE — 1/3 seeds diverged** |

**At this scale, positional encoding is the only architecture choice that
measurably affects loss.** Activation and norm type sit inside seed noise, so
they can be chosen on efficiency grounds rather than quality — `swiglu` vs
`gelu` differs by 0.0005 with parameter count held constant.

Post-norm diverged on one seed of three. A single-seed experiment would more
likely than not have reported it as merely "somewhat worse" and missed the
instability entirely.

> **Correction — the `attention` axis in this sweep is partly void.**
> `GPTConfig` resolved `gqa` via `max(1, n_head // 4)`, which collapses to
> `n_kv_head = 1` for any `n_head <= 7`. The 30M preset has 6 heads, so the
> `gqa` control and the `mqa` variant were **the same architecture**, and their
> null result is vacuous rather than informative. The row labelled *control*
> throughout this sweep is therefore MQA, not GQA.
>
> What survives: `mha` (6 KV heads) and `mqa` (1 KV head) *were* genuinely
> different configurations, and the difference between them was inside seed
> noise. So the real finding is that **cutting from 6 KV heads to 1 costs
> nothing measurable in loss at this scale** — a 6× KV-cache reduction, not the
> 12× originally claimed here, which came from the default 12-head config
> rather than the preset actually trained.
>
> Every other axis is unaffected: all variants shared the same attention
> baseline, so those comparisons remain valid. GQA proper is untested at 30M.
> Fixed in `GPTConfig.__post_init__` (floor of 2, plus a hard failure when GQA
> would degenerate) and pinned by `test_gqa_never_degenerates`.

### Throughput observations, and why the MFU column proved unusable

Three effects make MFU unsafe to compare across configurations:

1. **`torch.compile` is bistable.** Two `layernorm` seeds ran at 759 ms/step and
   the third at 296 ms — identical code and config, 2.6× apart, because
   compilation landed on a different kernel schedule. Loss was unaffected
   (SD 0.0022, the tightest of any variant).
2. **Sustained thermal drift.** Runs slowed from 9.9 to ~20 min over 5.6 hours.
3. **Within-run consistency is not evidence.** The sweep showed MHA at
   270–282 ms against GQA at 287–288 ms, consistent across all three seeds,
   which looked like a real ~5% effect and an obvious culprit:
   `repeat_interleave` materialising the KV-head expansion.

Point 3 was wrong, and `scripts/bench_attention.py` was written to check it:

```
variant                    ms/step   spread   peak GB
MHA                           72.6     3.9%     6.78
GQA repeat_interleave         70.3     2.3%     6.76
GQA fused                     72.6     3.4%     6.64
MQA repeat_interleave         71.3     7.5%     6.76
MQA fused                     72.0     9.8%     6.64
```

Total range across variants is 3.2% against a 9.8% noise floor. **No attention
implementation is faster than another at this size.** The apparent MHA advantage
was a compiled-schedule artifact that happened to be stable within one sweep —
the same failure mode as the LayerNorm bistability, and precisely why
seed-consistency alone cannot establish a throughput result.

The fused-GQA path (`enable_gqa`) is kept anyway, on the one measurable
difference: **6.64 GB vs 6.76 GB** peak memory, in the expected direction for
both GQA and MQA. That gap grows with KV-cache length, which matters for
inference rather than training. `tests/test_attention.py` pins it to produce
outputs identical to the materialised path.

### Known limitation

At 2000 steps the 30M preset consumes 131 M tokens against a 90 M-token corpus,
so data repeats. Conclusions transfer to larger scale only loosely. Expanding
the corpus (`prepare.py --tokens 5e8`) removes this.

## Inference engine (Rust)

```bash
cd engine && cargo build --release
./target/release/llm-engine inspect ../export/120m
./target/release/llm-engine logits ../export/120m --tokens 464,2159,318,1719
```

Checkpoints cross the Python/Rust boundary as safetensors (`export.py`), which
is flat, zero-copy and memory-mappable — unlike `.pt`, which is a Python pickle.
`config.json` carries the architecture, because head counts, norm type and
positional encoding all change the compute graph and the engine refuses to infer
them from tensor shapes.

### Validating the forward pass

The CPU path is written to be obviously correct rather than fast: scalar
kernels, f64 accumulation, no SIMD or threading. It exists so that when a GPU
kernel disagrees with it, the bug is known to be in the GPU kernel.

Correctness is established by comparing logits against PyTorch on identical
input, not by inspecting output text:

| | Rust | PyTorch |
|---|---|---|
| sum over vocabulary | −334002.5076 | −334002.4851 |
| min | −12.244563 | −12.244562 |
| max | 7.800590 | 7.800590 |

Top-10 token ids match in order, values to ~6 decimal places. The remaining
difference is f32 summation order across 50,304 logits — 6.7e-8 relative.

This matters more than it might look. A transposed projection, the other RoPE
pairing convention, or query heads mapped to the wrong KV heads under GQA all
produce a model that loads, runs, and generates fluent-looking text. None of
them announce themselves, and none would be caught by reading samples.
`scripts/reference_logits.py` regenerates the PyTorch side.

Current CPU throughput: 445 ms to load, 261 ms for a 4-token forward pass —
single-threaded scalar code, and the baseline every optimisation is measured
against.

### Prefill and decode are different problems

The engine has two paths, and the reason is arithmetic rather than tidiness.

**Decode** has one token in flight. Every matmul is a matrix-vector product, no
weight is reused, and the work is bandwidth-bound — which is why int8 and CUDA
graphs are what move it.

**Prefill** has the whole prompt at once with no sequential dependency between
its tokens, so one weight matrix serves every row. That is matrix-matrix:
compute-bound, and it wants tiling and tensor cores instead.

Using the decode path for prefill, as this engine did at first, applies
bandwidth-bound tooling to a compute-bound problem and pays ~150 kernel launches
per prompt token. `GpuModel::forward` now routes any multi-token input to
`prefill`; `CRUCIBLE_PREFILL=serial` forces the old path, which is how the two
were compared.

Both paths write into one KV cache layout, so a prompt can be prefilled and then
extended token by token with no conversion between them.

### Tokenizer

GPT-2 BPE, reimplemented in Rust. The vocabulary is exported from the same
tiktoken encoding the training data was built with
(`scripts/export_tokenizer.py`), in a trivial binary format so the engine needs
no JSON or base64 dependency.

```bash
python scripts/export_tokenizer.py --out export/gpt2.tok
./target/release/llm-engine tokenize export/gpt2.tok "The World is a stage"
```

The pre-tokenizer pattern needs `fancy-regex` rather than the standard `regex`
crate, because `\s+(?!\S)` is a negative lookahead and `regex` has no lookaround
support.

**A bug worth recording.** The first merge loop removed the merged element
before recomputing neighbouring ranks, so lookahead indexed past the wrong
boundary and merging stopped early: `"The"` encoded as `[817, 68]` (`"Th"`,
`"e"`) instead of `[464]`, and the probe string produced 29 tokens instead of
14. **Decoding round-tripped perfectly the whole time** — the text came back
byte-identical, so nothing looked wrong. The only symptom would have been a
model fed ids it was never trained on, which presents as degraded output and
reads like a bad model rather than a bad tokenizer.

It was caught by comparing ids against tiktoken, and `matches_tiktoken_ids`
now pins them.

### Generation

```bash
./target/release/llm-engine generate export/120m \
    --tokenizer export/gpt2.tok \
    --prompt "The capital of France is" --max-tokens 30 --temperature 0.7
```

Output from the 120M model (val loss 3.28):

> **The capital of France is** in the possession of the royal treasury. The
> French government is in the possession of the French and French governments
> of the entire territory of the world.

> **Photosynthesis is the process by which** plants convert light into energy.
> Plants can use photosynthesis to generate heat. The resulting heat is used to
> power the plant's turbines

> **The three branches of government are** the three branches of government, the
> executive branch and the judiciary, and the judiciary.

Fluent and locally coherent, with the failure modes expected at this scale:
factual drift, repetition, and word-sense confusion (*plant* the organism versus
*plant* the factory).

Sampling is top-k with temperature, seeded by a small xorshift64\* so runs
reproduce exactly without a `rand` dependency.

### KV cache

Without a cache, generating token *N* re-runs attention over all *N* positions,
so a sequence costs O(N²) and the entire prompt is recomputed every step. The
cache retains each position's projected keys and values, making each new token
O(N).

Measured on the same prompt and seed, 30 tokens:

| | no cache | with cache |
|---|---:|---:|
| decode time | 35.22 s | **3.10 s** |
| throughput | 0.85 tok/s | **9.66 tok/s** |
| per token | 1,174 ms | **103 ms** |

**11.4× faster, byte-identical output.** Identical output is the check that
matters: a broken cache still generates fluent text, just different text, so
matching the uncached path token-for-token under a fixed seed is what proves it
correct.

The advantage grows with length, since only the uncached path is quadratic. At
150 tokens throughput holds at 9.42 tok/s — essentially flat — while the
uncached path would need roughly 25× its 30-token time. That is **~55× at 150
tokens**, widening further from there.

This required restructuring the forward pass from layer-major to **token-major**:
each token now flows through every layer before the next token begins. By the
time position *p* reaches layer *L*, every earlier position has already written
its layer-*L* keys and values, so attention reads them instead of recomputing.
The logits still match PyTorch exactly (`sum -334002.5076`), which is how the
restructure was verified.

Cache layout is `[layer][position][kv_head * head_dim]`, contiguous in the last
dimension — attention reads one position's keys for one head at a time, so those
values sit adjacent in memory. At 1024 context the 120M model's cache is
**18.9 MB**; with MHA instead of GQA it would be 75.5 MB.

## CUDA kernels

```bash
cargo build --release --features cuda
./target/release/llm-engine gpu-validate
./target/release/llm-engine gpu-bench --rows 8192 --cols 4096
```

Kernels are compiled at runtime with **NVRTC**, not offline with nvcc. Two
reasons, one forced and one earned:

- CUDA 13's headers conflict with **glibc 2.43** on Ubuntu 26.04 — both declare
  `rsqrt`/`rsqrtf` with incompatible exception specifications, and nvcc injects
  host headers even for `--ptx`, so *every* compile fails including an empty
  kernel. No compiler flag fixes it; `__GLIBC_USE(IEC_60559_FUNCS_EXT_C23)` is
  not overridable. NVRTC never includes host headers, so the conflict cannot
  arise.
- As a side effect the engine builds without the CUDA toolkit present, and
  targets the exact GPU at runtime.

### Validation

Every kernel has a scalar CPU twin in `ops.rs`, and `gpu-validate` compares them
rather than assuming they agree:

| kernel | max relative difference |
|---|---:|
| gemv | 0 (bit-exact) |
| rmsnorm | 1.1e-7 |
| softmax | 3.7e-7 |
| silu_mul | 1.8e-7 |
| rope | 9.6e-7 |
| gemm (scalar, f32 / int8) | 3.3e-7 / 3.3e-7 |
| gemm (tensor core, f32 / int8) | 6.4e-5 / 3.4e-5 |

Exact equality is not the bar — the GPU reduces in a different order and
`use_fast_math` trades accuracy for speed. Rounding-level agreement is.

The tensor-core rows are held to a looser bound than the scalar ones because
they convert activations to half, but a looser bound is not no bound: half
carries 11 mantissa bits, so with K=768 accumulating in f32 the error must stay
under 2^-11 ≈ 4.9e-4. Measured at 6.4e-5, comfortably inside. Anything above
1e-2 would mean the tiling or the fragment layout is wrong, not that half is
imprecise.

### Two ways this test passed for the wrong reason

The generators feeding `gpu-validate` are not arbitrary, and both properties
were learned by getting them wrong.

The first version used `(i % 89 - 44) / 128` — small integers over a power of
two. half represents those *exactly*, so the tensor-core path and the scalar
path agreed bit-for-bit and the test reported `0.000e0`. A kernel that dropped
precision catastrophically would have passed identically. Test data has to be
mantissa-dense before a precision test means anything.

The fix after that used raw `sin`/`cos`, which is mantissa-dense but signed. A
768-term dot product of random signs cancels down to near zero, and relative
error against a near-zero result is meaningless — it reported ~2.0, which is
what a sign flip on noise looks like, not a broken kernel. Shifting both
operands positive keeps the mantissas full while making the sum accumulate
monotonically.

Both failures produced a confident-looking number. The first said the kernel was
perfect, the second said it was broken, and the kernel was the same kernel.

### Why decode is bandwidth-bound

Generating one token at a time makes every matmul a matrix-*vector* product:
weights are read once and reused for a single output element. Measured with
`scripts/bench_bandwidth.py` at a 135 W enforced limit:

**757 GB/s** median (spread 13.1%, best 759). Against the 120M model's weights:

| precision | GB/token | ceiling |
|---|---:|---:|
| f32 | 0.45 | 1,674 tok/s |
| f16 / bf16 | 0.23 | 3,348 tok/s |
| int8 | 0.11 | **6,696 tok/s** |

Two consequences:

**FP8 tensor cores are close to irrelevant for single-stream decode.** The
186 TFLOP/s figure only pays off in batched prefill, where weights are reused
across many tokens. For decode, quantisation helps because it moves fewer
bytes — not because it multiplies faster.

**The GEMV kernel is already done.** It reaches 723–770 GB/s, at the measured
device ceiling. No amount of kernel tuning raises decode throughput further in
f32; only reading fewer bytes does.

### Comparing kernels requires paired trials

`float4` vectorised loads were expected to beat the scalar kernel. Measuring
that took three attempts, and the first two were the interesting ones.

The first design ran all scalar trials, then all `float4` trials, and reported
a 120% difference against a 168% spread. The second, after correcting the power
profile, still showed 76% spread. Both were unusable, for the same structural
reason: **running kernel A to completion then kernel B cannot separate a real
difference from clock drift between the two phases**, and on a laptop the clocks
move far more than any kernel effect.

The fix is a **paired** design — both kernels run back to back inside each
trial, with the order alternating between trials, and the reported statistic is
the per-trial ratio. Drift then affects both kernels equally and cancels:

```
  scalar      723 GB/s   [462-745]  spread 39.1%
  float4      738 GB/s   [492-770]  spread 37.6%

  paired ratio float4/scalar: 1.034   [0.910-1.128]  spread 21.1%
  -> 3.4% difference against 21.1% paired spread: not distinguishable
```

Absolute throughput still swings ~39%, but the paired comparison narrows to
21% and gives a usable answer: **`float4` is not faster.** The scalar kernel
already saturates memory bandwidth, so there is nothing for wider loads to
recover.

### Every measurement must record its power envelope

Getting a stable number took longer than getting a fast one. An early run
spanned **167–505 GB/s, a 168% spread**, with clocks sampled at 180–300 MHz
during 100% utilisation while boosting to 1867 MHz when idle:

```
SM MHz  temp  watts  util
  1095    46   63.0   100
   180    45   45.4   100     <- 180 MHz at full utilisation
  1867    44   25.0     0     <- 1867 MHz once idle
```

That looked like a clock-governor fault and was written up here as one. It was
not. The laptop was in a **55 W eco power profile**, and the GPU was doing
exactly what it had been told. Temperature never exceeded 46 °C because the cap
was the binding constraint, not heat.

The lesson is methodological, and it now applies to every GPU number in this
repo: **this machine's power limit is user-switchable between roughly 55 W and
175 W, so a measurement that does not record its envelope is not
reproducible.** `gpu-bench` and `gpu-validate` now print
`enforced.power.limit` and the maximum SM clock alongside every result, and
`gpu-bench` warns when spread exceeds 30% — the signature of a throttled or
capped run.

The size of the error is worth stating: the eco-profile baseline measured
**479 GB/s**, against **757 GB/s** at 135 W. Every derived figure — the decode
ceiling, the value of quantisation, how close the GEMV kernel sits to peak —
was computed from a number that was 37% low.

### GPU forward pass

```bash
cargo build --release --features cuda
./target/release/llm-engine gpu-logits export/120m --decode 64
./target/release/llm-engine generate export/120m --tokenizer export/gpt2.tok     --prompt "The capital of France is" --gpu
```

Weights and the KV cache stay resident on the device; only the token id goes in
and the logits come out. Kernels are queued without synchronising between them,
so the host does not stall on each of the ~170 launches a token requires.

| | CPU (scalar reference) | GPU |
|---|---:|---:|
| decode | 9.66 tok/s | **471–501 tok/s** |
| per token | 103 ms | **2.00 ms** |
| resident | 18.9 MB cache | 485 MB weights + cache |

**~49× faster, and byte-identical output** under the same seed — which is the
check that matters, since a wrong kernel still produces fluent text. Against the
CPU reference, logits agree to `2.0e-4` maximum relative difference with the
top-10 identical in order; the residual comes from `use_fast_math` (`__expf`,
`rsqrtf`) and f32 rather than f64 accumulation.

Attention is fused into a single kernel per head — scores, softmax and the
weighted sum of values, with scores held in dynamic shared memory. Splitting
those into three kernels would mean 432 launches per token at 12 heads × 12
layers, and launch overhead alone would dominate a model this size.

### What limits it now

2.00 ms/token moves 0.45 GB of weights, an effective **225 GB/s against the
measured 757 GB/s ceiling — about 30%**. The gap is not the kernels: GEMV in
isolation reaches 723–770 GB/s. It is that ~170 separate launches per token
leave the GPU idle between small kernels.

Two levers, in order of expected value:

1. **Quantisation.** int8 weights cut bytes-per-token by 4×, raising the ceiling
   to ~6,700 tok/s. *(Implemented — and it delivered 4× on memory but only
   1.06× on speed. See the int8 section below: the ceiling rose, but the engine
   was never near it.)*
2. **CUDA graphs**, capturing the per-token launch sequence once and replaying
   it, to remove per-launch overhead. *(Implemented — 1.39x on its own, and it
   raised int8's contribution from 1.05x to 1.21x. See the CUDA graphs section
   below.)*

### int8 quantisation

```bash
./target/release/llm-engine gpu-eval export/120m --data data/fineweb-2b/val.bin     --tokens 1024 --quant f32,int8
```

Weight-only, symmetric, per-output-row scales. Activations stay f32 — they are
a negligible share of the bytes moved during decode, so quantising them would
cost accuracy for nothing. Per-row rather than per-tensor because one scale
across a whole matrix is set by its largest outlier, crushing the resolution of
every other row.

Measured on 1024 held-out tokens:

| | f32 | int8 |
|---|---:|---:|
| cross-entropy | 3.720299 | 3.720334 |
| perplexity | 41.2767 | 41.2782 |
| weights resident | 452 MB | **114 MB** |
| decode | 405 tok/s | 431 tok/s |

**Quality cost is +0.001% cross-entropy — free, within any reasonable
tolerance. Memory drops 4×. But speed rises only 1.06×, not the 4× the
bandwidth argument predicted.**

That gap is the interesting part, and it corrects an earlier claim in this
README. int8 moves 0.11 GB per token, which at 757 GB/s is 0.15 ms — yet a
token takes 2.32 ms. Roughly **2.2 ms is fixed overhead**, about 13 µs across
the ~170 kernel launches a token requires. Decode at this model size is
**launch-bound, not bandwidth-bound**, so removing bytes barely moves the wall
clock.

The bandwidth ceiling reasoning was not wrong, it was premature: the ceiling did
rise from ~1,674 to ~6,700 tok/s, but the engine sits at 431, nowhere near
either. Quantisation cashes in only once per-token overhead is gone.

**Revised priority.** CUDA graphs — capturing the per-token launch sequence once
and replaying it — now comes before further quantisation work, because it is
what makes quantisation pay. int8 is worth keeping regardless for the 4× memory
reduction, which is what determines how large a model fits in 16 GB.

### CUDA graphs

```bash
./target/release/llm-engine gpu-logits export/120m --quant int8 --decode 256 --graph
./target/release/llm-engine generate export/120m --tokenizer export/gpt2.tok \
    --prompt "The capital of France is" --graph
```

Decoding one token issues ~170 kernel launches, each costing microseconds of
driver work. A CUDA graph captures that sequence once and replays it as a single
submission.

| | eager | graph | graph gain |
|---|---:|---:|---:|
| f32 | 509 tok/s | 707 tok/s | 1.39x |
| int8 | 533 tok/s | **852 tok/s** | 1.60x |
| int8 gain | 1.05x | **1.21x** | |

Output is byte-identical to eager and to the CPU reference under the same seed,
and cross-entropy is unchanged to six decimals.

**The two optimisations compose, and one enables the other.** int8 alone was
worth 1.05x; once graphs removed the launch overhead it became 1.21x, because
only then was there enough bandwidth-bound time left for fewer bytes to matter.
Combined: 509 to 852 tok/s, and **88x over the 9.66 tok/s CPU reference**.

### Making capture work

Two failures, neither obvious from the error text, both worth recording.

**A captured graph freezes its kernel arguments.** `token`, `pos`, `seq_len`
and the KV cache slot offset all change every step, so passing them by value
would bake step 0 into the graph and every replay would recompute the same
token. Every per-step scalar therefore lives in a small device buffer that
kernels index into, updated with one host-to-device copy of 5 ints per token --
one transfer replacing ~170 launches. Slot 0 holds a permanent zero so call
sites needing a constant offset take the same code path instead of requiring a
second kernel variant. Shared memory for attention is sized for maximum context
rather than current length, for the same reason: a graph fixes its allocation at
capture.

**`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`.** CUDA forbids capturing the legacy
default stream, which is what `ctx.default_stream()` returns. Fixed by creating
a dedicated stream.

**`CUDA_ERROR_STREAM_CAPTURE_ISOLATION`.** cudarc records a CUDA event per
buffer and inserts `cuStreamWaitEvent` on every kernel touching it, to order
work across streams. A captured launch waiting on an event recorded by
uncaptured work is exactly what capture isolation forbids -- so the safety
mechanism made capture impossible. Creating a dedicated stream is also what
*activated* that tracking, since cudarc only engages it in multi-stream mode.

Resolved with `disable_event_tracking()`, which is sound here because the engine
uses a single stream and issues everything in program order, so the stream
itself provides the ordering those events exist to guarantee. It must be called
before any allocation: the flag is read when a buffer is created, so buffers
allocated earlier keep their events and still poison capture.

### What limits it now

1.17 ms/token at int8 moves 0.11 GB of weights, which at 757 GB/s is 0.15 ms.
So roughly **1.0 ms per token is still unaccounted for**, and decode sits at
about 13% of the bandwidth ceiling.

Launch overhead is no longer the main suspect -- graphs removed most of it. The
candidates were:

- **The attention kernel's inner loop is uncoalesced.** *(Confirmed by profiling
  and fixed -- see below. It was the largest stage at 32%.)*
- **Small-matrix inefficiency.** *(Confirmed and partly fixed: warp-per-row GEMV
  for int8.)*
- **Logits transfer.** *(Measured at 4.6% of decode -- real but minor, and still
  unaddressed.)*

### Profiling, and getting the profiler right

```bash
./target/release/llm-engine gpu-profile export/120m --quant int8 --warm 256
```

Nsight Systems reports no CUDA kernel data under WSL2 virtualisation, so
attribution is done in-engine: the stream is synchronised between stages and
each is timed on the host.

The first version of that produced a nonsense ranking. A 768-element `rmsnorm`
appeared to cost more than a 50304x768 matmul, because `rmsnorm` runs 24 times
per token and `lm_head` once, so it absorbed 24 syncs to the other's one. **The
profiler was measuring itself.**

The correction needs a per-block overhead estimate. Timing syncs on an idle
stream gave 70.8 us, which cannot be right: 111 blocks would then cost 7.9 ms
against a 3.9 ms measured total. Syncing an idle stream is simply a different
operation from syncing after queued work. The estimate now comes from the
cheapest stage that still launches a kernel -- `embed`, one 768-element row copy
-- giving ~21 us and an adjusted total of 1.37 ms against 0.87 ms of real
decode, which is close enough for ranking.

| stage | adjusted ms | share |
|---|---:|---:|
| mlp | 0.390 | 28.5% |
| attention | 0.362 | 26.4% |
| lm_head | 0.169 | 12.3% |
| qkv_proj | 0.168 | 12.2% |
| rope | 0.117 | 8.5% |
| logits_copy | 0.062 | 4.6% |
| rmsnorm | 0.060 | 4.4% |
| o_proj | 0.035 | 2.6% |
| residual | 0.006 | 0.4% |

### Two kernel fixes the profile found

**Attention: uncoalesced reads.** The original scoring loop gave each *thread* a
cached position and walked that key vector sequentially, so neighbouring threads
read addresses `cache_stride` floats apart -- 768 bytes here -- and every lane
issued its own memory transaction. Rewritten to one *warp* per position with
lanes striding across the key, reads become contiguous and coalesce. The value
accumulation was already coalesced but left three quarters of each block idle at
`head_dim` 64; each warp now sums a slice of positions into its own partial
vector in shared memory.

Attention fell from 0.531 to 0.362 ms, and decode went 852 to 1021 tok/s.

**GEMV: starved blocks.** A block per output row suits a long row and wastes a
short one. `gate_proj` is 2048x768, which as int8 with `char4` loads is 192
elements against 256 threads: a quarter of the block idle, one load per active
thread, then a full eight-warp block reduction to combine them. One warp per row
removes the idle threads and replaces the block reduction with shuffles alone.

int8 measured 1170-1299 tok/s against 852, a 1.42x gain well outside the spread.
**The same change did not help f32** -- 737/736/864 against 825, a possible
regression sitting inside its own 17% spread. Unproven in either direction, so
f32 keeps block-per-row and the switch is applied only where it was measured to
help. An f32 row carries four times the bytes, so its block is far less starved.

### Where decode stands

| path | tok/s | ms/token |
|---|---:|---:|
| CPU reference (scalar) | 9.7 | 103 |
| GPU eager, f32 | 509 | 1.96 |
| GPU graph, f32 | 825 | 1.21 |
| GPU graph, int8 | **~1149** | **0.87** |

**119x the CPU reference**, with cross-entropy unchanged at every step
(3.720334 for int8, before and after both kernel rewrites).

Still well under the bandwidth ceiling. The MLP was the largest stage and its
three projections are most of the model's weights, so the next gain looked like
it needed fusing the per-layer sequence rather than tuning kernels further --
which is what happened next.

### Kernel fusion

Graphs removed the CPU cost of launching, but each kernel still pays GPU-side
dispatch, so kernel *count* keeps mattering. Three fusions, all int8:

- **SwiGLU**: `silu(gate . x) * (up . x)` in one kernel instead of three, with
  no `hidden`-sized intermediates. MLP time halved, 0.349 to 0.174 ms.
- **Residual into projection**: `o_proj` and `down_proj` accumulate straight
  into the residual stream, removing the separate `add_inplace` at both sites.

Decode went 1149 to **1320 tok/s** (median of five), 0.76 ms/token. Cross-entropy
unchanged at 3.720334.

**The fusion shipped broken first, and speed alone would have passed it.**
`gemv_i8_at` routes to warp-per-row only when `cols/4 < 256`; `down_proj` has
2048 columns, so it took the block-per-row kernel, which had no `accumulate`
parameter. It overwrote the residual stream instead of adding to it. The engine
ran *faster* while doing this, generated fluent text, and perplexity went from
41.28 to **64,300**. Only the held-out cross-entropy check caught it. The f32
path now fails loudly rather than silently overwriting.

The profiler broke in the same way, more quietly: `profile_step` duplicates the
forward pass so it can sync between stages, and only `queue_token` was fused --
so it spent a round reporting a `residual` stage that no longer existed. Both
now mirror each other, and the duplication is called out in the code as
something that must be kept in step.

### Where decode stands

| path | tok/s | ms/token |
|---|---:|---:|
| CPU reference (scalar) | 9.7 | 103 |
| GPU eager, f32 | 509 | 1.96 |
| GPU graph, f32 | 825 | 1.21 |
| GPU graph, int8 | 1149 | 0.87 |
| GPU graph, int8, fused | **1320** | **0.76** |

**137x the CPU reference**, cross-entropy unchanged at every step.

Current stage breakdown (`gpu-profile`, position ~256):

| stage | adjusted ms | share |
|---|---:|---:|
| attention | 0.427 | 38.8% |
| mlp | 0.174 | 15.8% |
| qkv_proj | 0.173 | 15.7% |
| rmsnorm | 0.100 | 9.1% |
| rope | 0.073 | 6.7% |
| lm_head | 0.061 | 5.5% |
| logits_copy | 0.060 | 5.5% |
| o_proj | 0.031 | 2.8% |

### Split-position attention: implemented, exact, and it did not help

Attention dominated at 38.8%, and not because of kernel quality. At position 256
it reads roughly 4.7 MB of KV cache per token, which at 757 GB/s should take
about 6 us; it took 427. The kernel launches **one block per head -- 12 blocks**
on a GPU with dozens of SMs, so most of the machine sat idle. Coalescing the
reads (an earlier fix) helped, but no per-thread tuning fixes a grid that cannot
fill the device.

So the sequence was split across blocks too: grid `(n_head, n_chunks)`, each
block reducing one chunk into a partial softmax, with a second kernel rescaling
by `exp(m_chunk - m_global)` and merging -- flash-decoding. `n_chunks` is fixed
at capacity rather than current length, because a captured CUDA graph freezes
grid dimensions.

It is exact: cross-entropy is identical to the single-block path to six
decimals. It is also not faster.

| decode tok/s, int8 + graph | 256 tokens | 900 tokens |
|---|---:|---:|
| single block per head | **1484** | **1424** |
| split positions | 1305 | 1389 |

Median of three. Splitting costs a second kernel dispatch per layer, 12 more per
token, which at this size is about what the extra parallelism saves. Clearly
worse at short context, a wash at long.

Kept behind `CRUCIBLE_ATTN=split` rather than deleted: the trade should invert
with more heads, a larger `head_dim`, or context well beyond 1024, where
attention work grows while the extra dispatch does not. It is off by default
because on *this* model it loses.

### A limit of the profiler, found the hard way

The profiler said attention had halved -- 0.427 to 0.206 ms adjusted -- while
end-to-end decode did not move. Both cannot be true.

The profiler subtracts one launch-plus-sync overhead per timed block. The split
path runs **two** kernels inside that one block, so it was credited with one
subtraction where it should have had two, and looked better than it was. The
comparison it was built for -- ranking stages within one configuration -- is
still valid. Comparing configurations whose stages contain different numbers of
kernels is not something it can do.

This is the third time a measurement here has produced a confident wrong answer:
first the sync overhead attributed to whoever called most often, then the
idle-stream probe that contradicted its own arithmetic, now this. The pattern is
consistent -- the tool is fine for what it was built for and silently wrong just
outside it, and only an end-to-end number catches the difference.

### Tensor-core GEMM

Prefill multiplies a `[seq, n_embd]` activation block by every weight matrix, so
unlike decode it is compute-bound rather than waiting on memory. The scalar
16x16 tiled kernel sustained ~3.4 TFLOP/s — 4.5% of this GPU's BF16 tensor-core
peak — which was the entire remaining gap to llama.cpp.

Replacing it with a `wmma` kernel, at a 156 W enforced limit, 5 interleaved
trials per point:

| seq | scalar tiled | tensor core | speedup |
|---:|---:|---:|---:|
| 128 | 15,684 | 20,325 | 1.30x |
| 256 | 17,203 | 32,763 | 1.90x |
| 512 | 16,858 | **39,731** | 2.36x |
| 1024 | 15,016 | 30,779 | 2.05x |

Throughput falls off past 512 on both paths because attention is O(n^2): its
share of the work grows with sequence length while the GEMM's shrinks.

Two design choices carry the int8 path. Weights stay int8 in global memory and
convert to half **in shared memory** — keeping a half copy of the model would
cost 226 MB and give back most of what quantisation bought, and int8 -> half is
exact, so that conversion loses nothing. Activations are f32 and convert to half
on load, which does lose mantissa bits; that one is a real numerical change and
is priced below rather than assumed away.

### Bigger tiles are not better tiles

The obvious next move was a larger block tile. A 64x64 tile produces four times
the output per block for twice the loads — ~64 FLOP per element loaded against
~25 — so it should win. Measured against the 16x64 tile:

| seq | 16x64 | 64x64 |
|---:|---:|---:|
| 128 | 20,325 | 9,950 |
| 256 | 32,763 | 18,749 |
| 512 | 39,731 | 30,033 |
| 1024 | 30,779 | **32,170** |

It loses everywhere except 1024, by 2x at the short end. The intensity argument
is correct and was not the binding constraint: M is the sequence length, so a
64-row tile at seq=128 with `n_embd=768` launches `ceil(128/64) * ceil(768/64)`
= 24 blocks on a 60-SM GPU. The 16-row tile launches 96. Arithmetic intensity
only starts paying once there is enough work to fill the machine, which is why
the big tile wins at 1024 and only there.

Both tiles are generated from one templated device function so they cannot drift
apart, and the 16x64 one is the default: it wins outright below 1024 and gives up
4.5% above it.

`CRUCIBLE_GEMM=wmma-auto` selects per launch, using the big tile only when a
launch would still produce at least two blocks per SM. That should beat both
fixed tiles, because the projections in a single forward pass have very
different N — 768, 2048, and 50304 for the lm_head — and want different tiles.
It is opt-in rather than default because that is an argument and not a
measurement: **the benchmark comparing it against both fixed tiles has not been
run.** `scripts/bench_prefill.py` is what would settle it.

### What the speedup costs in accuracy

The tensor-core path converts activations to half, so it is a different
computation and not just a faster one. Two measurements, because neither is
inferable from the other.

`gpu-validate` bounds the kernel error against the CPU reference: 6.4e-5 (f32)
and 3.4e-5 (int8), against 3.3e-7 for the scalar path. half carries 11 mantissa
bits, so with K=768 and an f32 accumulator the error has to stay under 2^-11 ≈
4.9e-4 — measured well inside that, which makes it rounding rather than a
structural defect.

That bounds the kernel but says nothing about the model, and the existing
`gpu-eval` could not help: it fed one token per call, making every matmul a
matrix-*vector* product, so it never executed the GEMM at all and could not have
detected a prefill regression. `gpu-eval --prefill-ctx N` scores each position
from a fresh prefill of the preceding N tokens, which puts the batched GEMM on
the path that produces the number. Over 511 windows at ctx=256:

| | scalar tiled | tensor core | delta |
|---|---:|---:|---:|
| f32 cross-entropy | 3.089870 | 3.089860 | -1.0e-5 |
| int8 cross-entropy | 3.090417 | 3.090399 | -1.8e-5 |

The GEMM change moves cross-entropy by ~1e-5, thirty times less than int8
quantisation's own cost of +5.4e-4, and slightly downward — which is noise, not
an improvement. Ragged sequence lengths (7, 17, 100, 333) agree to within half
rounding, and `--prefill 1` is bit-identical because M=1 routes through GEMV,
confirming decode is untouched.

## Against llama.cpp and vLLM

```bash
python scripts/export_hf.py runs/120m-main --out export/120m-hf --verify
python llama.cpp/convert_hf_to_gguf.py export/120m-hf --outfile export/120m-f32.gguf --outtype f32
llama-quantize export/120m-f32.gguf export/120m-q8_0.gguf Q8_0
python scripts/bench_compare.py --tokens 256 --trials 7
```

Every other number in this README compares crucible against an earlier version
of itself, which answers "did that change help" and not "is this any good".

**Decode** — 256 tokens, batch 1, greedy, 7 interleaved rounds, 151 W enforced:

| engine | tok/s | spread | weights |
|---|---:|---:|---:|
| **crucible** | **1463.6** | 5.8% | int8, per-row scales, 114 MB |
| llama.cpp (b925e117, CUDA) | 862.1 | 15.6% | Q8_0, 122 MB |
| vLLM 0.28.0 | — | — | cannot run here, see below |

crucible led in all seven rounds with no overlap.

**Prefill** — 512-token prompt:

| engine | tok/s | |
|---|---:|---|
| llama.cpp | 109,459 ± 21% | |
| **crucible** (batched) | **15,221** | 7.2x slower |
| crucible (token at a time) | 888 | 123x slower |

The middle row is the point. crucible originally processed prompts **one token
at a time**, so 512 tokens cost ~77,000 kernel launches, each a matrix-*vector*
product — and prefill came out *slower* than decode, which is backwards and was
the tell. Prompt tokens have no sequential dependency on each other, so the
whole prompt can go through as a matrix-*matrix* multiply: compute-bound, with
arithmetic intensity that grows with prompt length.

Batching it is **17x faster** and costs ~14 launches per layer for the entire
prompt instead of ~150 per token. Logits agree with the serial path to 5e-6
relative, and generated text is unchanged.

**The prefill gap is now 2.8x, down from 7.2x.** That 7.2x was a tensor-core
gap: crucible's prefill sustained ~3.4 TFLOP/s, 4.5% of this GPU's BF16
tensor-core peak, on a plain 16x16 tiled f32 kernel, while llama.cpp dispatches
to cuBLAS. A hand-written `wmma` GEMM raised prefill to ~39,700 tok/s at seq
512, roughly 8 TFLOP/s or ~11% of that peak.

What remains is still a kernel-quality gap rather than a mystery. cuBLAS does
things this kernel does not: double-buffered shared-memory loads that overlap
the next tile's fetch with the current tile's math, `ldmatrix` for fragment
loads, and per-shape tuned tiles. ~11% of peak is a working tensor-core kernel,
not a tuned one.

### The comparison is valid because the outputs are identical

Greedy decoding, same prompt, both engines:

> The capital of France is the capital of the United Kingdom.
> The capital of the United Kingdom is the capital of the United Kingdom.
> The capital of the United

Token for token, from two independent implementations using different
quantisation schemes — block-wise Q8_0 against per-row int8. That is far
stronger evidence that the same model is being measured than any logit
tolerance would be. The HF export feeding the GGUF is separately logit-verified
against the reference implementation (`export_hf.py --verify`).

llama.cpp's own trace reports `graphs reused = 26`, so it is using CUDA graphs
as well; neither side has a structural advantage there.

### What this result is not

crucible implements **one** architecture, batch 1, greedy, one quantisation
scheme, one context length. llama.cpp supports dozens of architectures, CPU and
GPU backends, many quantisation formats, a server, batching and full sampling —
and is tuned for models above 7B, where a 120M model at batch 1 is nowhere near
its design point. A 1.7x decode margin on this workload says the specialised path is
faster on the case it was specialised for; the remaining 7.2x prefill deficit
says where tensor cores still earn their keep. It says nothing about the
projects.

### Three ways this benchmark was wrong before it was right

Each favoured crucible, and each was caught before publication.

**The `-ngl` default.** `llama-bench` does not put all layers on the GPU by
default: 583.7 tok/s. With `-ngl 99`, the same build does 846.5. Publishing the
default would have overstated the margin by 45%.

**Block ordering.** Running all of crucible's trials and then all of
llama.cpp's gave llama.cpp 614.6 tok/s at 20% spread against 846.5 standalone —
the second engine inherits a hot GPU. This is the same confound already fixed
for kernel comparisons and not applied here until it bit. The harness now
interleaves, one trial per engine per round, and llama.cpp's number moved back
to 862.

**An unverified model.** Until the greedy outputs were compared, there was no
evidence llama.cpp was computing the same thing. A subtly broken GGUF could
have been fast for entirely uninteresting reasons.

### vLLM cannot run under WSL2

`RuntimeError: UVA is not available`. WSL2's GPU paravirtualisation does not
expose Unified Virtual Addressing, which vLLM's memory management requires. It
is a platform limitation, not a configuration problem: native Linux would work,
WSL2 and Windows will not.

This also corrects a justification given earlier in this README. The choice of
WSL2 was defended partly on the grounds that "vLLM does not officially support
Windows" — but vLLM does not run under WSL2 either, so that argument never held.
The other reasons stand: Triton is first-class on Linux, and CUDA C++ needs MSVC
on Windows where gcc suffices here.

### A CUDA 13.3 note

Building llama.cpp needs nvcc, which could not compile *any* `.cu` file on this
machine: CUDA 13.0's headers conflict with glibc 2.43 over `rsqrt`. **CUDA 13.3
fixes it** — `cuda-nvcc-13-3` compiles cleanly and llama.cpp builds with native
`sm_120a`. crucible stays on NVRTC regardless, since runtime compilation also
removes the build-time toolkit dependency.

## Layout

```
model.py           # transformer; ablation axes are config flags
train.py           # training loop, BF16 autocast, MFU accounting
sweep.py           # ablation runner
analyze.py         # figures + significance-aware results table
data/
  prepare.py       # streams FineWeb-Edu into uint16 shards
scripts/
  setup_wsl.sh     # one-shot environment bootstrap
  verify_gpu.py    # device checks, FP8 probe, quick throughput baseline
  bench.py         # repeated-trial harness: median, IQR, thermal state
  bench_prefill.py # GEMM variants: scalar vs tensor-core tiles, interleaved
tests/
  test_attention.py  # GQA path equivalence, causality, param parity
export.py          # checkpoint -> safetensors for the Rust engine
engine/            # Rust inference engine
  src/config.rs      # architecture, refuses to guess
  src/weights.rs     # mmap'd safetensors, bf16/f16/f32
  src/ops.rs         # scalar reference kernels
  src/model.rs       # CPU forward pass
  src/tokenizer.rs   # GPT-2 BPE, pinned against tiktoken
  src/cache.rs       # KV cache for incremental decode
  src/gpu.rs         # CUDA backend, NVRTC compilation, validation
  src/gpu_model.rs   # full forward pass on device
  src/quant.rs       # int8 weight quantisation
  kernels/kernels.cu # gemv, rmsnorm, rope, softmax, silu
scripts/
  bench_bandwidth.py # memory bandwidth and the decode ceiling it implies
  export_hf.py       # checkpoint -> HuggingFace Llama, logit-verified
  bench_compare.py   # interleaved comparison against llama.cpp / vLLM
runs/              # per-run log.csv + best.pt checkpoint
figures/           # ablation plots
```

## Tests

```bash
python -m pytest tests/ -v
```

The load-bearing test is `test_gqa_paths_match`. Grouped-query attention can be
computed by materialising KV heads with `repeat_interleave` or by letting the
fused SDPA kernel group them internally, and the two are interchangeable only
if their query-head-to-KV-head mapping agrees. If the conventions differed,
training would still run and loss would still fall — the model would simply be
wrong. The paths are therefore pinned to each other numerically across
`n_rep` ∈ {2, 4, 12}, with and without an attention bias, rather than assumed
equivalent.

`test_causality` verifies that editing token *t* leaves every output before *t*
bit-identical, which catches a broken mask that a falling loss curve would hide.

## Roadmap

- [x] Environment bootstrap and hardware baseline
- [x] Data pipeline (FineWeb-Edu → uint16 shards)
- [x] Transformer with switchable architecture axes
- [x] Training loop with MFU accounting
- [x] Architecture ablations across seeds (27 runs, 5 axes)
- [x] Compute-budget analysis; 120M selected as compute-optimal for this GPU
- [x] Training at 120M (1.44B tokens, val loss 3.2839)
- [x] Rust inference engine: checkpoint loading, CPU forward pass validated against PyTorch
- [x] GPT-2 BPE tokenizer in Rust, generation end to end
- [x] KV cache (11.4x at 30 tokens, ~55x at 150)
- [x] CUDA kernels, validated against the CPU reference
- [x] GPU forward pass end to end (49x over CPU, identical output)
- [x] int8 quantisation: 4x smaller weights, +0.001% cross-entropy
- [x] CUDA graphs: 1.39x (f32) / 1.60x (int8), and int8's own gain rose 1.05x -> 1.21x
- [x] Profiler with sync-overhead correction; coalesced attention; warp-per-row int8 GEMV
- [x] Kernel fusion: SwiGLU in one kernel, residual folded into the projections
- [x] Split-position attention (flash-decoding) — exact, but slower here; kept opt-in
- [ ] Paged attention → continuous batching
- [x] Throughput comparison against llama.cpp (decode 1.7x faster, prefill 104x slower)
- [x] Batched prefill — 17x faster, prompt processed as a matrix
- [x] Tensor-core GEMM — prefill 2.4x faster, llama.cpp gap 7.2x -> 2.8x
- [ ] Tune the tensor-core GEMM — double buffering, `ldmatrix`; ~11% of BF16 peak
- [ ] Measure `wmma-auto` against both fixed tiles — implemented, unmeasured
- [ ] vLLM comparison — blocked: WSL2 does not expose UVA, needs native Linux

## License

Apache-2.0.
