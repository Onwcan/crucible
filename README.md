# crucible

Training small language models from scratch and building a custom inference
engine, targeting **NVIDIA Blackwell (`sm_120`)** hardware.

> **Status:** 120M model trained (1.44B tokens, val loss 3.28). Rust engine
> generates at **431 tok/s on GPU** with int8 weights (4x smaller, +0.001%
> cross-entropy), byte-identical output to the CPU reference path whose logits
> match PyTorch. Decode is launch-bound; CUDA graphs next.

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

- **vLLM does not officially support Windows**, and the project's final
  benchmark compares this engine against vLLM on identical hardware.
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

Exact equality is not the bar — the GPU reduces in a different order and
`use_fast_math` trades accuracy for speed. Rounding-level agreement is.

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
   it, to remove per-launch overhead. *(This turned out to be the real
   bottleneck, and should have come first.)*

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
- [ ] CUDA graphs to remove per-launch overhead — the actual bottleneck at 120M
- [ ] Paged attention → continuous batching
- [ ] Throughput comparison against llama.cpp and vLLM on identical hardware

## License

Apache-2.0.
