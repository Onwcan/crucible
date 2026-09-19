# Contributing to Crucible

Keep changes scoped to one engineering question. Explain the behavior changed,
the correctness checks run, and the limits of any performance claim. Use the
[pull request template](.github/pull_request_template.md); CUDA checks apply to
changes that affect CUDA or serving behavior, not to every documentation edit.

## Reproduce hosted CI

[CI](.github/workflows/ci.yml) runs on Ubuntu 24.04 for pull requests, pushes to
`main`, and manual dispatch. The repository has no declared MSRV or toolchain
file, so CI uses stable Rust. Helper checks use Python 3.12 and the standard
library. Install Git, a C linker, rustup, Python 3.12, and ShellCheck before
running these commands from the repository root:

```bash
rustup toolchain install stable --profile minimal --component rustfmt --component clippy
rustup default stable
rustc --version
cargo --version
export CARGO_TARGET_DIR=engine/target
export PYTHONDONTWRITEBYTECODE=1

# quality
cargo fmt --manifest-path engine/Cargo.toml --all -- --check
python scripts/ci_source.py
python -m unittest discover -s tests -p 'test_ci_*.py'
shellcheck --severity=warning scripts/setup_wsl.sh
cargo clippy --manifest-path engine/Cargo.toml --all-targets --locked -- -D clippy::correctness -D clippy::suspicious

# test
cargo test --manifest-path engine/Cargo.toml --locked

# feature-checks
cargo test --manifest-path engine/Cargo.toml --no-default-features --locked
cargo check --manifest-path engine/Cargo.toml --no-default-features --features tui --locked
CUDARC_CUDA_VERSION=13020 CUDA_VISIBLE_DEVICES=-1 cargo test --manifest-path engine/Cargo.toml --features cuda --locked
```

Use Python 3.12 as `python` for parity with the workflow. Source checks compile
tracked Python in memory without importing dependencies or writing bytecode.
They parse tracked JSON, check local Markdown file links and conflict markers,
and reject tracked model/profiler/build artifacts, oversized files, and literal
machine home paths in executable source/configuration. Documentation examples
and historical evidence are excluded from the home-path check. External URLs
and Markdown heading anchors are not checked. Newly added files must be tracked
in the Git index to be included by `ci_source.py`.

The default Rust feature is `tui`. Its unit and loopback mock-server tests run
headlessly. The no-default suite exercises the core without optional features;
the explicit TUI check preserves its independence from CUDA. The last command
uses cudarc's supported binding-version override to bypass `nvcc` detection.
It compiles the CUDA Rust code and tests host protocol logic with GPU devices
hidden. It does not initialize a CUDA device, compile NVRTC kernels, execute
GPU code, or validate SM120 correctness. No NVIDIA toolkit or model download
belongs in the hosted lane.

The Rust source is rustfmt-clean, and formatting is a blocking hosted CI gate.
Before submitting changes, run
`cargo fmt --manifest-path engine/Cargo.toml --all -- --check`.
To format intentional Rust edits locally, run
`cargo fmt --manifest-path engine/Cargo.toml --all`; CI only checks and never
rewrites source. The [formatting report](docs/rustfmt-results.md) records the
normalization and validation, while the [CI report](docs/ci-results.md) preserves
the earlier milestone's historical formatting-debt policy.
Strict Clippy `-D warnings`
still fails on existing style/performance warning debt. The gate therefore
denies `clippy::correctness` and `clippy::suspicious`, leaving other warnings
visible without adding global allow attributes or changing inference logic.

Cargo commands that resolve dependencies use `--locked`. CI never runs
`cargo update`. Keep dependency updates separate and explain any intended
lockfile change. Cargo caches contain registry/git data and build outputs;
their keys separate OS, architecture, job, compiler identity, and lockfile.
They contain no model weights, local benchmark output, or credentials.

Every hosted job also checks that a clean checkout stays clean:

```bash
test -z "$(git status --porcelain --untracked-files=all)"
```

Run that assertion on a clean checkout or isolated validation snapshot. It
will correctly fail in a development checkout containing intentional edits.
`scripts/setup_wsl.sh` is only inspected by ShellCheck in CI; its installation
commands are not executed. The hosted workflow installs ShellCheck from Ubuntu
packages if it is absent from the runner image.

## Optional GPU correctness

[GPU CI](.github/workflows/gpu-ci.yml) is prepared for a provisioned Linux x64
runner in the `crucible-trusted` organization/enterprise runner group. Before
registering any runner accessible to this public repository, restrict the
group's repository access to this repository and its **Selected workflows** to
`Onwcan/crucible/.github/workflows/gpu-ci.yml@refs/heads/main`. This server-side
workflow/ref restriction is required: PRs can edit workflow YAML, so labels and
an `if` condition alone are not an access boundary. Do not register an
unrestricted repository-level GPU runner here. If the account cannot enforce
this group policy, keep the Actions runner unregistered and run the same script
manually on reviewed local source. See GitHub's
[runner group access controls](https://docs.github.com/en/enterprise-cloud@latest/actions/how-tos/manage-runners/self-hosted-runners/manage-access).
The selected-workflow policy requires eligible Enterprise support; ordinary
Free/Team groups do not provide this workflow restriction. A personal repository
cannot directly attach an organization runner group. For that account setup,
use manual local validation and leave this Actions workflow prepared but inactive
until an eligible ownership/access configuration can enforce the policy.

After that policy is configured, assign labels `self-hosted`, `linux`, `x64`,
`nvidia`, and `crucible-gpu`, or update the workflow and
[actionlint labels](.github/actionlint.yaml) together if changing the custom
labels. No runner registration or successful GitHub GPU run is implied by
the workflow file.

The workflow accepts only manual dispatch on `main`; checkout uses the event's
trusted commit, with no PR ref input. It has no automatic push, pull-request,
`pull_request_target`, or `workflow_run` trigger. Keep the independent runner
group policy in place even when editing workflows. Provision a dedicated runner
without personal credentials or unrelated workloads, and use an isolated or
ephemeral runner where possible. Both workflows request only `contents: read`
and disable checkout credential persistence.

Provision these before dispatch:

- A visible SM120-capable NVIDIA GPU and compatible driver, `nvidia-smi`, CUDA
  `nvcc`, and NVRTC that reports support for `compute_120`. Driver and NVRTC
  shared libraries must be discoverable by the dynamic loader; `mma.h` and
  `cuda_fp16.h` must be in an include location supported by
  [Gpu::new](engine/src/gpu.rs), normally `/usr/local/cuda/include`.
- Stable Rust via rustup, Cargo, a C linker (`cc`), Git, and Python 3.12 or
  newer on the runner service's `PATH`. Allow at least 8 GiB free on the Cargo
  target filesystem. The workflow installs neither the toolchain nor assets.
- A local exported 120M model with `config.json` and `model.safetensors`:
  12 layers, 12 query heads, 3 KV heads, width 768, context 1024, and vocabulary
  50,304. The current correctness corpus is specific to that geometry.
- For full mode, its matching GPT-2 tokenizer, held-out little-endian uint16
  tokens (at least 16,385 tokens), and executable Python environments with the
  official `openai` and `anthropic` SDKs already installed. The model directory
  basename must contain `120m` because the existing TUI smoke corpus identifies
  the model through that name.
- For sanitizer mode, native Linux and NVIDIA Compute Sanitizer. WSL is
  explicitly rejected for this mode and cannot supply sanitizer coverage.

Set repository Actions variables to absolute local paths accessible to the
runner account. For direct local runs, export environment variables with the
same names. These are path settings, not download URLs:

| Variable | Required modes | Contents |
|---|---|---|
| `CRUCIBLE_MODEL_PATH` | All | Exported 120M model directory |
| `CRUCIBLE_TOKENIZER_PATH` | Full | Matching tokenizer file |
| `CRUCIBLE_HELDOUT_PATH` | Full | Held-out uint16 token file |
| `CRUCIBLE_OPENAI_PYTHON` | Full | Python executable with the OpenAI SDK |
| `CRUCIBLE_ANTHROPIC_PYTHON` | Full | Python executable with the Anthropic SDK |

Keep these assets outside the checkout. The script redacts configured paths
in its output, checks readability and prerequisites, and fails when something
is missing. It never downloads a model or silently skips requested coverage.
It clears inherited `CRUCIBLE_*` tuning controls before testing production
defaults. Full-mode servers bind only to loopback and are stopped after each
suite, including failure, timeout, or cancellation.

Select **GPU CI → Run workflow → main**, then choose one mode. The same entry
points work locally from the repository root:

```bash
python3 scripts/ci_gpu.py --mode smoke
python3 scripts/ci_gpu.py --mode full
python3 scripts/ci_gpu.py --mode sanitizer
```

Run one mode at a time. Smoke runs kernel launches and selection checks, graph
replay, batching, sampling, contiguous and paged prefill, packed generation,
and attention isolation. Full increases the graph/fuzz cases and adds f32
fallback, paired reference/exact held-out CE, a freshly recorded HTTP
reference, native/OpenAI/Anthropic HTTP and SDK tests, real-server TUI checks,
and page-pressure/cancellation tests. Sanitizer runs attention and packed
prefill under `compute-sanitizer --tool memcheck --error-exitcode 99`.
All modes build the CUDA release binary with `--locked` first. Existing
`gpu-validate` fails on launch errors and argmax/top-k mismatches, but its generic
floating-kernel error rows are diagnostics without pass/fail tolerances. The
separate attention check enforces its tensor comparison thresholds; do not
interpret `gpu-validate` success as blanket numerical validation of every kernel.
`gpu-batch` gates generated-token, admission and page-lifecycle comparisons;
its separate GEMV/forced-GEMM dispatch error summaries are diagnostic only.

Inspect the exact command plan without executing it, or check prerequisites
separately, with:

```bash
python3 scripts/ci_gpu.py --mode full --list-commands
python3 scripts/ci_gpu.py --mode full --preflight-only
```

A printed plan or successful preflight is not a correctness pass. Full mode
uses loopback port 18081 by default; local runs can select another unused port
with `--port`. The script rejects an occupied port so tests cannot accidentally
exercise an unrelated server. Workflow timeouts are 60 minutes for smoke/full
and 90 minutes for sanitizer, with additional per-command timeouts.

## Runtime and performance changes

Preserve or explicitly justify changes to token identity, numerical tolerances,
GQA head mapping, causality, request isolation, paged KV ownership/reuse,
cancellation, and CUDA graph pointer/topology lifetimes. Compare optimized
paths against retained references and exercise boundary lengths and mixed
requests. Describe which model, quantization, and serving controls were tested.
Small tensor-level error alone is insufficient: verify generated-token identity,
deterministic seeded sampling, request isolation, and page cancellation/reuse.
Preserve reference paths and fallbacks until correctness checks and benchmark
evidence justify their removal.

CI has no throughput or latency pass/fail threshold. For performance claims,
attach repeated paired A/B measurements with the same model, inputs, build,
warmup, and power/clock envelope. Record the enforced power limit, maximum SM
clock, and observed clock under load. Use matched A/B ordering over multiple
rounds and report medians and ranges, including noise and any profiling
limitations. Keep raw traces, model exports, and machine-local result
directories out of Git. Portable engineering evidence belongs in `docs/` when
it supports the change. See [attention results](docs/paged-attention-results.md)
for the reference and evidence discipline used by the current runtime.

## Dependency maintenance and GitHub rollout

[Dependabot](.github/dependabot.yml) checks Cargo (`/engine`) and Actions (`/`)
weekly on Monday, targets `main`, and caps open PRs at five per ecosystem.
Compatible Cargo minor/patch updates are grouped. Nothing is automatically
merged. Actions use verified full commit SHAs with readable version comments.

The separate `dependency-audit (informational)` job checks the lockfile against
RustSec with a pinned, checksum-verified cargo-audit binary. It is not a
required merge gate. The initial audit reports warning advisories for `paste`
and `lru`; default cargo-audit success does not mean the tree is advisory-free.
Review the output and address dependency remediation separately. The
[CI report](docs/ci-results.md) records the audit, Clippy baseline, optional
tooling decisions, local results, and remaining coverage limits.

GitHub execution is pending until the milestone is committed and pushed. After
pushing, open **Actions → CI**, inspect that commit's run, and verify `quality`,
`test`, and `feature-checks` are green. Capture logs before changing code for
any runner-specific failure. Only after a successful hosted run should branch
protection require those three check names. Keep the informational audit and
optional GPU workflow out of required branch checks. Do not mark GPU CI
operational until a provisioned runner has completed a real dispatch.
