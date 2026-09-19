# CI hardening milestone

## Follow-up: blocking Rust formatting, 2026-09-19

The Rust source has now been normalized by rustfmt, and
`cargo fmt --manifest-path engine/Cargo.toml --all -- --check` is a blocking
hosted quality gate. The informational exception and formatting-warning step
were removed. See the [formatting normalization report](rustfmt-results.md) for
the source-equivalence checks and validation results.

The original milestone record below is preserved as historical evidence. Its
references to pending formatting debt, informational rustfmt and the then-current
HEAD describe the earlier CI milestone, not the current formatting policy.

## Original CI milestone record

Prepared against the attention milestone on 2026-09-13 and validated through
2026-09-14. GitHub execution is
**pending**: no project commit, push, runner registration, or branch-protection
change was made. Rustfmt detects pre-existing drift in 27 Rust files. The
maintainer explicitly chose to keep runtime source unchanged, expose formatting
as an informational check, and defer its promotion to a blocking gate until a
dedicated formatting-only normalization commit. All other hosted quality/test
gates remain blocking. The repository is not represented as rustfmt-clean.

This report separates local command results, workflow inspection, and coverage
that only an actual GitHub or native-Linux GPU run can establish.

## 1. Starting HEAD

`a2d3ebcb8530bd9dcd52753bbc52885027d680ff`, verified with `git rev-parse HEAD`.

## 2. Starting commit title

`perf(cuda): add exact tiled GQA prefill attention`, verified with
`git log -1 --oneline`. This remains the project HEAD.

## 3. Initial Git status

`git status --short --untracked-files=all` was empty before CI implementation.
There was no existing `.github/` directory or `CONTRIBUTING.md`.

## 4. Repository and feature audit

The repository contains Python training/export/analysis helpers, a single Rust
package in `engine/`, runtime-compiled CUDA kernels, validation/benchmark scripts,
and portable engineering evidence in `docs/`. It is not a Python distribution.
There is no workspace-level Cargo manifest, toolchain file, or declared MSRV.

`engine/Cargo.toml` uses edition 2021, an inferred library and `llm-engine` binary,
and a checked-in version-4 lockfile with 188 packages. Registry dependencies use
crates.io; no model data is a Cargo dependency. Default features are `["tui"]`.
`tui` selects Ratatui, Crossterm, Reqwest, futures-util and Tokio independently of
`cuda`; `cuda` adds cudarc, Tokio, Axum, tokio-stream and async-stream. The TUI
integration corpus has 14 headless loopback mock-server tests. The CUDA-feature
unit tests exercise host protocol/state logic without constructing a GPU.

The initial script audit covered 28 tracked Python files, one WSL bootstrap
shell script, and three JSON evidence files. Existing Python runtime regressions
need a server/model, provisioned SDK interpreters, or PyTorch/pytest. Only the new
standard-library CI checker tests are added to hosted Python execution.

## 5. Hosted-compatible command classification

| Category A | Coverage and constraints |
|---|---|
| `cargo fmt --manifest-path engine/Cargo.toml --all -- --check` | Informational whole-crate check; fails committed baseline drift |
| Chosen `cargo clippy --all-targets --locked` invocation in item 17 | Default-feature static Rust analysis |
| `cargo test --manifest-path engine/Cargo.toml --locked` | 130 unit + 14 headless TUI integration tests |
| `cargo test --manifest-path engine/Cargo.toml --no-default-features --locked` | 99 core unit tests without optional dependencies |
| `cargo check --manifest-path engine/Cargo.toml --no-default-features --features tui --locked` | Explicit TUI feature isolation |
| CUDA-feature Cargo tests with binding-version override, item 21 | 176 host unit + 14 TUI tests; compile coverage only for CUDA |
| `python scripts/ci_source.py` | Tracked Python syntax, JSON, local documentation, hygiene |
| `python -m unittest discover -s tests -p 'test_ci_*.py'` | Portable checker/orchestration regression tests; no real GPU |
| `shellcheck --severity=warning scripts/setup_wsl.sh` | Static shell analysis only; never executes the installer |
| cargo-audit | Registry lockfile advisory lookup; informational |

Default `cargo test` already compiles TUI. Its tests are not repeated as a
second TUI-only suite. The explicit TUI check is retained as the requested
documented feature-isolation contract. No-default testing catches unintended
dependencies on optional features. CUDA-feature host tests compile the binding
path as well, avoiding a redundant extra `cargo check --features cuda` step.

## 6. CUDA-dependent classification

Category B: an ordinary CUDA-feature build uses cudarc's
`cuda-version-from-build-system`, which invokes `nvcc --version`. With a fresh
target and no toolkit in `PATH`, the ordinary check failed. Setting documented
`CUDARC_CUDA_VERSION=13020` selects its shipped bindings and permits compilation
without that detection. The `dynamic-loading` feature avoids link-time CUDA
driver/NVRTC dependencies. This does not compile any NVRTC kernel.

Category C: actual `gpu-validate`, `gpu-graph-check`, `gpu-batch`, `gpu-paged`,
`gpu-sampling`, `gpu-prefill-check`, `gpu-prefill-graph-check`,
`gpu-packed-prefill-check`, `gpu-prefill-attention-check`, `gpu-logits`, GPU
generation, `serve`, and GPU evaluation initialize or use CUDA. These require a
real device, driver, NVRTC and headers. The present engine targets `compute_120`.
`gpu-validate` and the attention tensor check use synthetic data and need no
exported model. GPU performance commands also need the runtime, even when their
inputs are synthetic. Native-Linux Compute Sanitizer is a separate runtime tool.

## 7. Model/artifact-dependent classification

Category D: `inspect`, CPU `logits` and CPU generation can run without CUDA but
need an exported model (and generation needs a tokenizer). `tokenize` needs a
tokenizer. These are not model-free hosted tests. Model-based GPU commands listed
above need `config.json` and `model.safetensors`; `gpu-eval` additionally needs
held-out uint16 tokens. HTTP/OpenAI/Anthropic/TUI live regressions need a real
server, matching model/tokenizer, and SDK interpreters for official SDK cases.
`tests/test_attention.py` can choose a CPU when no CUDA is available but imports
PyTorch and pytest, so it is syntax-checked rather than installing a heavy
inference environment in hosted CI. Training, exports, reference-logit and GPU
verification helpers likewise have artifact or third-party-library requirements.

## 8. Machine-specific benchmark classification

Category E: `gpu-bench`, `gpu-gemv-bench`, `gpu-topk-bench`, `gpu-sample-bench`,
`gpu-prefill-attention-bench`, `gpu-packed-prefill-bench`, `gpu-prefill-bench`,
`gpu-serve-bench`, `gpu-profile`, `gpu-profile-batch`, `gpu-prefill-trace`, and
`scripts/bench*.py`/`probe_batch.py` are measurement/profiling tools, with runtime
and model requirements according to their inputs. Power/clock orchestration,
Nsight traces, WSL setup, training sweeps and local baseline trees are outside
merge CI. `scripts/setup_wsl.sh` is an installer, never a CI execution step.
Portable historical JSON/Markdown evidence is parsed, not re-benchmarked.

## 9. Workflows and supporting files

Added hosted and manual GPU workflows, an actionlint custom-label configuration,
weekly Dependabot, a short PR template, practical contributor instructions, two
stdlib CI scripts and their unit tests, and this report. README has a small CI
section and one badge. `scripts/bench_prefill.py` has one portability correction:
its default model path now uses `export/120m` instead of a machine home path;
explicit `--model` usage is unchanged. No Rust/CUDA source, Cargo manifest or
lockfile change is part of this implementation.

## 10. Hosted triggers

`pull_request`, `push` to `main`, and `workflow_dispatch`, with Ubuntu 24.04
hosted runners. No path filters or decorative OS/Python matrices hide required
checks. Stable names are `quality`, `test`, and `feature-checks`, plus the
non-required `dependency-audit (informational)` job.

## 11. Permissions

Both workflows explicitly request only `contents: read`; checkout uses
`persist-credentials: false`. There are no secrets, write scopes, OIDC,
PR-comment bots, release/package uploads, Docker workflows, or branch-setting
mutations. The supplied PR jobs use hosted runners. Since PRs can modify workflow
files, the separate runner-group restriction in item 27 is also required before
any GPU runner is registered.

## 12. Concurrency and shell behavior

Hosted group:
`${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}`.
A newer run cancels obsolete work for the same workflow and PR/ref; different
PRs and branches remain separate. GPU group:
`${{ github.workflow }}-${{ github.ref }}`; only manual main dispatch is eligible.
The chosen mode is intentionally absent so a new mode replaces obsolete work
on the same single GPU resource. Nontrivial shell blocks use `set -euo pipefail`;
simple steps use Actions' explicit Bash error/pipefail behavior.

## 13. Rust toolchain

Stable Rust, installed through runner-provided rustup with a minimal profile;
quality additionally installs rustfmt and Clippy. No invented MSRV or new
runtime toolchain policy. The initial baseline used stable 1.98.0; exact final
command validation uses updated stable 1.98.1. Compiler identity is printed and
hashed into cache keys, so later stable updates cannot reuse incompatible
compiler caches. Stable is deliberately a moving supported channel.

## 14. Action pinning

Only three first-party actions are used. Full SHAs were resolved against
authoritative upstream tags and their action metadata inspected:

| Action | Version | Immutable revision |
|---|---|---|
| [actions/checkout](https://github.com/actions/checkout) | v7.0.1 | `3d3c42e5aac5ba805825da76410c181273ba90b1` |
| [actions/cache](https://github.com/actions/cache) | v5.1.0 | `caa296126883cff596d87d8935842f9db880ef25` |
| [actions/setup-python](https://github.com/actions/setup-python) | v6.3.0 | `ece7cb06caefa5fff74198d8649806c4678c61a1` |

Readable version comments and weekly Actions Dependabot updates make pins
maintainable. No floating `main` action reference or third-party toolchain action
is used. Future self-hosted runners must use a current Actions runner compatible
with these actions. This follows GitHub's
[secure-use guidance](https://docs.github.com/en/actions/reference/security/secure-use).

## 15. Formatting command and accepted debt policy

`cargo fmt --manifest-path engine/Cargo.toml --all -- --check` is informational.
It checks source without rewriting it. The clean starting tree fails: rustfmt
would normalize 27 Rust files, with 4,287 added and 1,382 removed diff lines.
A concrete formatting-only preview was prepared outside the repository; none
of it has been applied. The maintainer explicitly declined whole-crate formatting
within this CI milestone and authorized an informational check so known debt
does not intentionally leave main red. The step uses `continue-on-error: true`,
is labeled informational, and emits a warning when its actual outcome is failure.
No baseline allowlist, changed-files-only shortcut, or auto-fix hides the diff.
The repository is not rustfmt-clean and formatting is not a passing mandatory gate.

The follow-up is a separate formatting-only normalization commit, verified to
contain no semantic changes. Once its whole-crate check passes, remove the
formatting step's `continue-on-error` and the reporting-only warning step to
promote it to a blocking part of the required `quality` check.

## 16. Clippy baseline

Clean-tree strict command:
`cargo clippy --manifest-path engine/Cargo.toml --all-targets --locked -- -D warnings`
failed with exit 101. The first library pass reported eight existing warnings:
two `manual_is_multiple_of`, two `neg_cmp_op_on_partial_ord`, one
`excessive_precision`, and three `chunks_exact_to_as_chunks`. The passing
non-global gate also exposes a `too_many_arguments` warning in `generate`.
Changing floating-point comparisons casually could alter NaN behavior; numerical
source cleanup is outside this milestone. No global allow attributes were added.

## 17. Final Clippy command

```bash
cargo clippy --manifest-path engine/Cargo.toml --all-targets --locked -- -D clippy::correctness -D clippy::suspicious
```

This denies the correctness and suspicious-code groups while leaving existing
style/complexity/performance warnings visible. `-D warnings` is not used because
it cannot honestly pass the committed baseline without unrelated source work.

## 18. CPU test command

`cargo test --manifest-path engine/Cargo.toml --locked`: default core and
headless TUI coverage, including loopback mock servers. No interactive TUI is
launched on hosted runners.

## 19. Feature-isolation commands

```bash
cargo test --manifest-path engine/Cargo.toml --no-default-features --locked
cargo check --manifest-path engine/Cargo.toml --no-default-features --features tui --locked
CUDARC_CUDA_VERSION=13020 CUDA_VISIBLE_DEVICES=-1 cargo test --manifest-path engine/Cargo.toml --features cuda --locked
```

## 20. TUI check

The explicit no-default, `--features tui` check preserves the requested
standalone TUI build contract. The present default feature set resolves to TUI
already; no second equivalent test suite is added. The 14 integration tests
exercise protocol handling through local mock servers without a GPU.

## 21. Hosted CUDA decision

Include CUDA Rust compilation and host tests with `CUDARC_CUDA_VERSION=13020`.
The initial fresh-target ordinary check failed without `nvcc` in 1.745 seconds;
the explicit-binding check passed in 8.283 seconds and host tests in 12.777
seconds. The override is implemented by the locked cudarc 0.19.9 build script;
see its [official build source](https://docs.rs/crate/cudarc/0.19.9/source/build.rs).
No driver, NVRTC, CUDA toolkit, or model is installed in the hosted workflow.

Local reproduction used WSL on a machine that physically has an NVIDIA GPU.
Toolkit paths and loader overrides were removed; the final CUDA host step also
sets `CUDA_VISIBLE_DEVICES=-1`. Source inspection confirms these tests do not
initialize the driver. That evidence supports compile/host coverage, not a claim
that local WSL was a physically GPU-free GitHub runner or that kernels passed.

## 22. Python validation and lint decision

Python 3.12 is a supported version compatible with the helpers; see the
[Python support table](https://devguide.python.org/versions/). One version is
sufficient. `ci_source.py` calls `compile` in memory on tracked Python bytes,
without imports, virtualenv traversal or `.pyc` writes. New checker/orchestration
tests use only stdlib and disposable local fixtures.

Ruff 0.16.7 was evaluated outside the repository: default isolated lint produced
91 findings, including existing undefined-name issues in benchmark cleanup.
Those remain recorded debt, not silently fixed through a broad harness rewrite.
Ruff/Pylint are deferred; no Python packaging metadata or heavy dependencies
were introduced. ShellCheck 0.11.0 reported three informational diagnostics in
the WSL installer; warning-or-higher gating passes. The workflow installs
ShellCheck from Ubuntu packages only if absent, then performs static analysis.

## 23. JSON validation

Every tracked `.json` is parsed by Python's standard library, rejecting invalid
syntax, duplicate object keys and non-JSON NaN/Infinity constants. The three
existing evidence JSON files pass. No invented semantic benchmark schema or
performance threshold is imposed.

## 24. Documentation and hygiene

`ci_source.py` checks local Markdown inline/image/reference destinations against
tracked files, ignoring code blocks, remote URLs and heading anchors. It
detects merge-conflict edges and separator markers without rejecting ordinary
Markdown underline headings. It rejects obvious tracked model/profiler/binary
outputs, build/cache/baseline directories, and files over 10 MiB. Python syntax,
JSON corruption, path escaping and fixture regressions are unit-tested.

Literal machine home paths are checked in executable source/configuration,
including escaped Windows strings, while historical prose/evidence is exempt.
This is repository hygiene, not a general secret scanner or full CommonMark
parser. Every hosted job asserts `git status --porcelain --untracked-files=all`
is empty after checks; ignored Cargo targets are deliberately allowed.

## 25. Cache design

Cache paths are only Cargo registry index/cache/source, Cargo Git database and
`engine/target`. Keys include runner OS/architecture, job identity, a SHA-256 of
`rustc -vV`, and the lockfile hash. Restore prefixes retain compiler/job isolation
while allowing dependency reuse after compatible lockfile changes. Cargo itself
fingerprints feature builds within the feature job. No GPU/model/evidence/home
directory cache exists. PR cache scopes do not flow back into main; see GitHub's
[cache access restrictions](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows#restrictions-for-accessing-a-cache).

## 26. Local hosted command timings

Initial clean-baseline checks, stable 1.98.0, warm pre-existing target:

| Check | Exit | Seconds |
|---|---:|---:|
| Whole-crate fmt check | 1 | 0.603 |
| Strict Clippy `-D warnings` | 101 | 1.394 |
| Chosen Clippy groups | 0 | 0.940 |
| Default tests | 0 | 0.870 |
| No-default tests | 0 | 4.511 |
| Explicit TUI check | 0 | 0.143 |

The exact workflow shell commands were also run under stable 1.98.1 and Python
3.12.14 in an isolated clean Git test fixture made from the intended files.
This fixture had its own disposable Git metadata; the project HEAD/index were
not staged or committed. It started with a fresh target directory and available
Cargo registry downloads; serial execution allowed later checks to reuse earlier
builds. It is not a prediction of three independent GitHub VMs' cold times.

| Workflow command/step | Exit | Seconds | Tracked/untracked mutations |
|---|---:|---:|---|
| Stable toolchain setup, quality | 0 | 0.819 | None |
| Whole-crate fmt (informational) | 1 | 0.318 | None |
| Visible formatting warning | 0 | 0.005 | None |
| Python/JSON/docs/hygiene | 0 | 0.166 | None |
| CI helper unit tests, 41 passed after restart fix | 0 | 0.317 | None |
| ShellCheck conditional setup and check | 0 | 0.034 | None |
| Chosen Clippy gate | 0 | 6.539 | None |
| Quality cleanliness assertion | 0 | 0.005 | None |
| Stable toolchain setup, test | 0 | 0.469 | None |
| Default tests, 144 passed | 0 | 8.295 | None |
| Test cleanliness assertion | 0 | 0.010 | None |
| Stable toolchain setup, feature checks | 0 | 0.468 | None |
| No-default tests, 99 passed | 0 | 5.986 | None |
| Explicit TUI check | 0 | 0.769 | None |
| CUDA binding/host tests, 190 passed | 0 | 14.616 | None |
| Feature cleanliness assertion | 0 | 0.005 | None |
| Download, checksum, extraction and RustSec audit | 0 | 4.078 | None |
| Audit cleanliness assertion | 0 | 0.005 | None |

Local validation continued through independent checks after the known fmt
failure to establish their results. The final policy permits this formatting
step to fail and emits a warning, allowing the remaining blocking checks to run.
Checkout/setup-python/cache actions were syntax/metadata-reviewed, not
executed as GitHub services locally. ShellCheck was already installed, so the
conditional apt-install branch remains a hosted setup check pending first run.

The table includes the final formatting policy. After the server restart fix,
the final 41-test/source checks were rerun in another clean snapshot: source
0.166s, unit tests 0.317s, cleanliness 0.006s, all exit 0 with no file mutations.
The remaining hosted commands were unaffected by that Python-only change. The
blocking commands passed; the explicitly informational formatting command failed
as expected on unchanged baseline source. This validates command behavior
locally; only GitHub can confirm the final job conclusions and annotations.

These are local wall times, not GitHub timing guarantees. Jobs allow 15 minutes
for quality/default tests, 20 for feature checks, and 10 for the informational
audit, accounting for cold compilation and network variance. Independent jobs
surface useful failures without a combinatorial matrix; cheap quality checks
precede Clippy dependency compilation.

## 27. Self-hosted GPU security model

Manual main-only dispatch is the chosen model. The job requires
`github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main'`.
Checkout uses the event's commit with no arbitrary PR/ref input. There is no
automatic push/PR, `pull_request_target`, or `workflow_run` trigger. GPU testing
of the merged, reviewed main revision is an explicit maintainer action.

The workflow selects runner group `crucible-trusted` and labels `self-hosted`,
`linux`, `x64`, `nvidia`, `crucible-gpu`. Before registration, an operator must
independently restrict that group's repository access and its **Selected
workflows** to `Onwcan/crucible/.github/workflows/gpu-ci.yml@refs/heads/main`.
Labels and the job's main-only condition alone are insufficient: a fork PR can
modify another workflow to target the same labels. The group policy must reject
that different workflow/ref at scheduling time. Do not attach an unrestricted
repository-level runner to this public repository. Where the account cannot
enforce the policy, leave the Actions runner unregistered and use the same
script manually on reviewed local source. GitHub documents
[workflow/ref restrictions for runner groups](https://docs.github.com/en/enterprise-cloud@latest/actions/how-tos/manage-runners/self-hosted-runners/manage-access).
This selected-workflow policy needs eligible Enterprise support, not merely
Free/Team group creation. A personal repository cannot directly attach an
organization group. Keep the workflow inactive and run local reviewed source
until ownership/access capabilities can enforce that scheduling boundary.

Custom labels are recognized by actionlint. Use a dedicated,
isolated, preferably disposable account/runner with no personal files,
privileged desktop access or long-lived cloud credentials. No runner was
registered or GitHub GPU execution observed during this milestone.

The only dispatch input is the enumerated mode. It is passed through a quoted
environment variable and validated by argparse. Asset paths remain argv data,
not shell fragments. No event title/body or untrusted PR value enters a run
script. The supplied hosted workflow uses disposable hosted machines with
read-only permissions and no supplied secrets. The independent group policy,
not a claim about PR-modifiable YAML, keeps a future GPU runner out of PR reach.

## 28. GPU preflight

Before building, the script verifies Linux, mode-specific local assets and
120M geometry; `nvidia-smi`, `nvcc`, rustup/rustc/cargo and `cc`; stable Rust;
at least 8 GiB free on the target filesystem; CUDA driver initialization and a
visible device supporting compute capability 12.0 or later; NVRTC architecture
support for `compute_120`; and CUDA headers in the engine's supported include
locations. Full mode additionally imports both provisioned SDKs. Sanitizer
rejects WSL and verifies Compute Sanitizer. Missing prerequisites fail clearly.

Logs include GPU name, driver, compute capability, maximum SM clock, VRAM,
NVRTC/nvcc and Rust/Cargo versions. Configured asset paths are redacted. The
wrapper clears inherited `CRUCIBLE_*` tuning overrides, uses loopback servers,
and terminates process groups on failure, timeout or cancellation. Preflight
success is explicitly not reported as a CUDA correctness pass.

## 29. GPU smoke command list

Entry: `python3 scripts/ci_gpu.py --mode smoke`. After preflight it executes:

```bash
cargo build --manifest-path engine/Cargo.toml --release --features cuda --locked
"$BIN" gpu-validate
"$BIN" gpu-graph-check "$MODEL"
"$BIN" gpu-batch "$MODEL"
"$BIN" gpu-sampling "$MODEL"
"$BIN" gpu-prefill-check "$MODEL"
"$BIN" gpu-paged "$MODEL" --graph
"$BIN" gpu-prefill-graph-check "$MODEL" --lengths 15,16,17,127,128,129,941 --chunks 32,37,256 --steps 8
"$BIN" gpu-packed-prefill-check "$MODEL" --quant int8 --steps 16 --fuzz 4
"$BIN" gpu-prefill-attention-check --fuzz 8 --seed 20260908
```

Here `BIN` denotes the freshly built Cargo target release executable and
`MODEL` the configured asset. These are explanatory placeholders; the script
constructs argument lists directly. It adds no timing acceptance threshold.

`gpu-validate` gates launch failures and its argmax/top-k comparisons. Its
general floating-kernel error rows are printed diagnostics, without numerical
failure thresholds in the existing implementation. The attention check separately
enforces tensor tolerances. This milestone does not invent new thresholds or
claim `gpu-validate` proves every kernel numerically correct.
Likewise, `gpu-batch` gates token equivalence/admission/page lifecycle but its
GEMV-versus-forward and forced-GEMM numerical summaries are diagnostic only.
`gpu-paged --graph` intentionally uses reference attention for storage parity;
the dedicated packed and attention checks supply exact-path comparisons.

## 30. GPU full command list

Entry: `python3 scripts/ci_gpu.py --mode full`. The release build and smoke
commands are retained, with these changes/additions:

- `cargo test --manifest-path engine/Cargo.toml --features cuda --locked`.
- `gpu-prefill-graph-check "$MODEL"` uses its complete default corpus.
- Packed int8 uses `--steps 16 --fuzz 24`; attention uses `--fuzz 128 --seed 20260908`.
- `gpu-packed-prefill-check "$MODEL" --quant f32 --steps 4 --fuzz 0`.
- Paired `gpu-eval "$MODEL" --data "$HELDOUT" --tokens N --quant int8 --graph --paged --prefill-ctx C`, first with `CRUCIBLE_PREFILL_ATTN=reference`, then `exact-q4-k64`, for `(C,N)` of `(32,1024)`, `(256,16384)`, `(941,16384)`. CE/perplexity/position rows must be finite, have positive counts, and match at printed precision; this is not a hardcoded model-quality or performance threshold.
- Start a fresh reference server with singleton prefill and reference attention; record `test_packed_prefill.py --record-reference "$FIXTURE" --steps 16 --fuzz-rounds 24 --port "$PORT"`.
- Restart with production defaults; run `test_packed_prefill.py --reference "$FIXTURE" --steps 16 --fuzz-rounds 24 --port "$PORT"`, `test_serve.py --port "$PORT"`, `test_openai.py --port "$PORT" --sdk "$OPENAI_PYTHON"`, `test_anthropic.py --port "$PORT" --sdk "$ANTHROPIC_PYTHON"`, and `smoke_tui.py --port "$PORT" --binary "$BIN"`.
- Restart with `--kv-pages 32 --max-queue 4`; run `test_packed_pressure.py --port "$PORT" --pages 32 --queue-limit 4 --requests 24`.

All client files are under `scripts/` and use the invoking Python interpreter.
Servers use `serve "$MODEL" --tokenizer "$TOKENIZER" --host 127.0.0.1 --port
"$PORT" --max-prompt-tokens 1000 --max-new-tokens 1000`; the existing full TUI
corpus additionally requires the model basename to contain `120m`. Temporary
reference/server logs stay outside the checkout and are cleaned on exit.
`--list-commands` prints the exact expanded plan without executing it.

## 31. Sanitizer mode

`python3 scripts/ci_gpu.py --mode sanitizer` requires native Linux. After the
locked release build it runs attention tensors with `--fuzz 128 --seed 20260908`
and exact packed int8 with `--steps 16 --fuzz 24`, each prefixed by
`compute-sanitizer --tool memcheck --error-exitcode 99`. It is an explicit
90-minute manual job, not part of smoke/full or hosted checks. WSL rejection
is validated; successful sanitizer coverage remains pending on an eligible
native-Linux runner.

## 32. Model artifact configuration

Repository Actions variables or local environment variables configure
`CRUCIBLE_MODEL_PATH` for every mode; full also requires
`CRUCIBLE_TOKENIZER_PATH`, `CRUCIBLE_HELDOUT_PATH`, `CRUCIBLE_OPENAI_PYTHON`, and
`CRUCIBLE_ANTHROPIC_PYTHON`. All are absolute local paths read by the runner
account. Model files are external; no URL input, downloads, uploads, secrets
or committed weights are introduced. Geometry and file requirements are in
[CONTRIBUTING](../CONTRIBUTING.md).

## 33. Dependabot decision

Implemented version-2 configuration: Cargo ecosystem at `/engine`, Actions at
`/`, both target `main`, weekly Monday, maximum five open PRs per ecosystem.
Cargo minor/patch updates are grouped; major updates remain separate. No
automatic merge, CUDA/model update configuration, or daily notification spam.

## 34. cargo-audit and cargo-deny

Selected a separate informational cargo-audit 0.22.2 job with job-level
`continue-on-error: true`; no advisory ignore IDs. The workflow fetches the
official Linux musl archive, checks SHA-256
`7fb9497f8594b389e5fce5ef9b92db08432996895b2e0c5a0167a69ed445c428`, extracts only
the binary, then runs `cargo-audit audit --file engine/Cargo.lock`. The binary
download took about 1.02 seconds and baseline audit about 4.61 seconds including
database fetch. Tool upgrades require review of both URL and digest; Dependabot
does not update this shell-pinned release. The RustSec database intentionally
updates so newly published advisories become visible.

At database revision `b50980aad8b8f14f77e25a97b32dd94bf008b0af` (2026-09-09),
188 dependencies yielded zero vulnerability-class findings and **three warning
advisories**, not an advisory-clean dependency tree:

- `paste 1.0.15`: unmaintained, [RUSTSEC-2024-0436](https://rustsec.org/advisories/RUSTSEC-2024-0436.html).
- `lru 0.12.5`: unsound iterator behavior, [RUSTSEC-2026-0002](https://rustsec.org/advisories/RUSTSEC-2026-0002.html).
- `lru 0.12.5`: unsoundness/potential use-after-free, [RUSTSEC-2026-0253](https://rustsec.org/advisories/RUSTSEC-2026-0253.html).

Default audit exited zero; `--deny warnings` did not. Informational policy
keeps output visible without declaring a remediation/merge policy implicitly.
Maintainers should review advisories and resolve dependency changes separately;
future vulnerability or network failures also remain informational until the
policy is deliberately tightened. cargo-deny 0.20.2 advisories-only was evaluated
and failed on existing dependency debt; adding duplicate advisory tooling and an
invented license/ban policy has no justified value here, so it is deferred.

## 35. CodeQL decision

Current [official CodeQL documentation](https://codeql.github.com/docs/codeql-overview/supported-languages-and-frameworks/)
includes Rust support. It is not rejected as unsupported. A meaningful separate
security-analysis setup, SARIF permissions, and evaluation of runtime-generated
CUDA boundaries deserve a later security milestone. No ceremonial CodeQL file
or `security-events: write` permission is added here.

## 36. PR template

Added a compact problem/behavior, correctness and performance template. It asks
for numerical effects, relevant reference/seeded tests, paged KV/cancellation,
graph pointer/topology lifetime, and paired performance evidence or an explicit
no-performance-claim statement. No generic issue-template collection was added.

## 37. CONTRIBUTING

Added exact local commands, feature/coverage limits, runner labels/assets,
manual trust boundaries, native-Linux sanitizer requirements, numerical/reference
policy and paired power/clock-controlled benchmark discipline. It distinguishes
small tensor error from intended token/seed equivalence and requires attention
to request isolation, page reuse and cancellation.

## 38. README and badge

One hosted `ci.yml` badge targeting `main`; no GPU badge. The CI section and
roadmap describe prepared hosted/manual GPU workflows and pending GitHub
execution. The test section corrects the old claim that CUDA-feature host Rust
tests inherently need a GPU/server. Performance roadmap and production defaults
remain unchanged.

## 39. Local validation results

The hosted snapshot results are in item 26. Final source validation includes 92
tracked/intended files, 32 Python files, three JSON files, nine Markdown files,
45 local links, and **41** Python 3.12 unit tests (22 source-checker and 19 GPU
orchestration tests). No successful GitHub Actions execution is claimed.

| GPU wrapper validation | Exit | Seconds | Meaning |
|---|---:|---:|---|
| Plan-only smoke/full/sanitizer | 0 each | 0.084 / 0.081 / 0.081 | Plans only, no GPU claim |
| Missing assets | 1 expected | 0.095 | Fails instead of silently skipping |
| Real smoke preflight | 0 | 2.893 | Driver/NVRTC/toolchain/model requirements verified |
| Real full preflight | 0 | 2.537 | Also validates tokenizer/heldout/SDK prerequisites |
| Sanitizer on WSL | 1 expected | 0.086 | Correctly refuses ineligible coverage |
| Real local GPU smoke | 0 | 160.605 | Release rebuild 57.4s plus all nine selected GPU commands |
| Full plan remainder after the interrupted prefix | 0 | 128.048 | All 15 remaining steps, including attention/f32/CE and all three server configurations |

The local GPU is an RTX PRO 4000 Blackwell Laptop, driver 597.06, compute
capability 12.0, maximum SM clock 3090 MHz, 16,303 MiB reported VRAM. Preflight
reported nvcc 13.3.73, loaded NVRTC 13.0, and Rust/Cargo 1.98.1. These are
diagnostic environment facts; no controlled performance result is claimed.
Full validation was completed **in segments**, not as one uninterrupted
`--mode full` invocation. Usage/WSL interruptions stopped runs after already
successful graph and packed checks. The final unchanged full-plan prefix passed
release build, all 190 CUDA-feature host tests, kernel smoke, graph/batch/sampling,
prefill and paged checks, the complete prefill-graph corpus (252.3s), and packed
int8 (35.6s). The remaining 15 steps were then executed by the same preflight,
plan construction and executor against the built binary; they all passed in
128.048s. No omitted command or interrupted invocation is counted as a pass.

An earlier complete attempt reached all HTTP/SDK/TUI checks, then failed at the
pressure-server restart after 458.001s: the CI port probe rejected TCP TIME_WAIT.
The wrapper now sets `SO_REUSEADDR` on that probe, with two regression tests
proving closed connections permit reuse while a live listener is still rejected.
The successful remainder run exercised that exact default-to-pressure restart,
as well as the reference-to-default restart, and passed cancellation/page reuse.
No inference source change was required.

Both attention implementations reported matching finite held-out results at
printed precision:

| Context | Scored positions | CE | Perplexity |
|---|---:|---:|---:|
| 32 | 31 | 3.319376 | 27.6431 |
| 256 | 63 | 3.424866 | 30.7185 |
| 941 | 17 | 4.048477 | 57.3101 |

Independent process probes verified descendant cleanup after success, failure
and timeout, and loopback-server cleanup after success, failure and cancellation.
After the final GPU run, no GPU test processes remained and the test port was
closed. An uninterrupted full invocation, actual GitHub dispatch, and native
Linux sanitizer execution remain unverified; the segmented results above are
the local coverage claim.

## 40. YAML and expression validation

Both workflows pass actionlint 1.7.12 using `.github/actionlint.yaml` for the two
custom runner labels. The official actionlint archive was digest-verified before
local use. Existing PyYAML 6.0.3 provides an additional syntax parse for all YAML;
generic YAML parsing alone is not treated as Actions expression validation.
Manual review covers event-specific concurrency fallbacks, main-only dispatch,
mode timeout/input expressions, asset variables, runner labels and cache scopes.

## 41. GitHub-run status

**Pending** for every job. Workflow YAML and local commands are validated only
as described here. No push or dispatch was performed; self-hosted GPU CI is
prepared, not operationally proven.

## 42. Recommended required branch checks

After the first successful GitHub run, select the actual emitted `quality`,
`test`, and `feature-checks` check names from the CI workflow in branch protection.
Do not require the informational audit or optional GPU job on public community
PRs. No branch-protection setting was changed.

## 43. Known limitations and unchanged scope

The formatting gate is explicitly informational pending a dedicated normalization
commit; a passing `quality` job does not mean rustfmt-clean source. Strict Clippy
and Ruff debt remain visible/documented.
RustSec warning advisories remain unremediated. Stable Rust and the advisory
database intentionally evolve; lockfile resolution and action/tool archives
are pinned. Actual hosted runner setup/cache behavior and native-Linux sanitizer
coverage remain pending. Local timings use WSL and available dependency caches.
All full-plan checks passed in segments because usage/WSL interruptions prevented
one uninterrupted full run; no end-to-end full-run success is claimed.
The inherited graph/paged CLI checks compare numeric differences; they do not
establish raw-bit equality for signed zeros or exhaustive NaN detection. The
dedicated attention validator separately checks finite elements and exact bits
for exact variants. Kernel and batch diagnostic-only rows are identified in
item 29 rather than promoted to stronger CI coverage claims.

No inference source/kernel/default changed: packed prefill stays on, aggregate
budget 1024, extra chunk cap off, singleton prefill graphs on, packed serving
graph cache off, supported exact-q4-k64 attention on with reference retained,
hybrid current-K/V off and page-table cache off. No packed graph cache, prefix
caching, speculation, packaging, Docker or release milestone was started.

## 44. Exact changed files

```text
.github/actionlint.yaml
.github/dependabot.yml
.github/pull_request_template.md
.github/workflows/ci.yml
.github/workflows/gpu-ci.yml
CONTRIBUTING.md
README.md
docs/ci-results.md
scripts/bench_prefill.py
scripts/ci_gpu.py
scripts/ci_source.py
tests/test_ci_gpu.py
tests/test_ci_source.py
```

## 45. Diff whitespace check

`git diff --check` passed with exit 0. Final source/hygiene checking also included
all new files through a temporary external index; the real index hash remained
unchanged. Windows checkout CRLF normalization is distinguished from source
edits when comparing against the committed tree.

## 46. Final Git status

Only the 13 intentional files in item 44 appear: README and the benchmark
path fix are modified, the remaining files are new. No real index staging or
project commit is performed. Tool downloads, raw logs, formatting preview and
validation snapshots are outside the repository. The project HEAD remains the
starting attention commit. Status:

```text
 M README.md
 M scripts/bench_prefill.py
?? .github/actionlint.yaml
?? .github/dependabot.yml
?? .github/pull_request_template.md
?? .github/workflows/ci.yml
?? .github/workflows/gpu-ci.yml
?? CONTRIBUTING.md
?? docs/ci-results.md
?? scripts/ci_gpu.py
?? scripts/ci_source.py
?? tests/test_ci_gpu.py
?? tests/test_ci_source.py
```

## 47. Exact post-push verification

1. Review and commit the intended CI milestone changes yourself. Formatting debt
   remains explicitly informational; no source normalization or commit was made here.
2. Push the commit to `main` (`git push origin main`) or push the reviewed branch
   and open a PR targeting `main`.
3. Open [Actions → CI](https://github.com/Onwcan/crucible/actions/workflows/ci.yml),
   select the run for that exact commit and inspect `quality`, `test`, and
   `feature-checks`. Verify all three are green. Confirm rustfmt's known failure
   is labeled informational with a warning, and inspect advisory output separately.
4. If a runner-specific setup, compilation, test or cache error occurs, download
   the failed job logs before changing source; reproduce the exact failed step.
5. Only after a successful hosted run, consider requiring those three emitted
   checks. Do not require the optional GPU workflow.
6. Configure and verify the independent runner-group workflow/ref restriction
   in CONTRIBUTING before registering the dedicated runner. If unavailable,
   use manual local validation instead. Then provision labels and repository
   variables. On trusted merged `main`, open
   [GPU CI](https://github.com/Onwcan/crucible/actions/workflows/gpu-ci.yml), choose
   **Run workflow → main → smoke**, then inspect real preflight and every check.
   Run full separately; run sanitizer only on eligible native Linux. Record the
   run URLs before calling GPU CI operational.

## 48. Proposed single commit title

`ci: add hosted quality gates and manual NVIDIA validation`

## 49. Proposed complete commit body

```text
ci: add hosted quality gates and manual NVIDIA validation

Add read-only Ubuntu CI for pull requests, main pushes and manual runs:
visible informational rustfmt debt, blocking Clippy correctness/suspicious gates,
locked CPU tests, no-default/TUI isolation, and CUDA binding/host tests without
toolkit installation or GPU runtime claims. Pin first-party actions to verified
SHAs, isolate Cargo caches by compiler/job/lockfile, and enforce timeouts and
checkout cleanliness.

Validate tracked Python syntax, JSON, local documentation links and repository
hygiene with standard-library checks and focused regression tests. Statically
check the WSL bootstrap script and replace one machine-specific benchmark
model default with a relative path.

Prepare trusted-main-only manual GPU smoke/full/native-Linux sanitizer modes
with real CUDA and asset preflight, local model configuration, reference/token
and held-out checks, protocol/SDK/TUI regressions, and process cleanup. Require
independent selected-workflow/ref runner-group restrictions before registration;
use manual local checks where that policy is unavailable. Keep performance
thresholds out of CI.

Add weekly Cargo/Actions Dependabot, an informational checksum-pinned RustSec
audit, a compact PR template, contributor commands and CI evidence. Preserve
all inference sources, dependencies, defaults and the performance roadmap.

Validation: blocking hosted commands, 41 CI helper tests, actionlint, local GPU
smoke and every full-plan check (in segments) passed. Evidence and limitations
are documented in docs/ci-results.md. GitHub runs, one uninterrupted full GPU
invocation and native-Linux sanitizer remain pending.
Per maintainer direction, keep Rust source unchanged and defer whole-crate
format normalization to a separate commit. Its check remains visibly
informational until that commit enables promotion to a blocking quality gate.
```
