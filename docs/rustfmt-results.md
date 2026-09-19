# Rust formatting normalization

Baseline and normalization: 2026-09-14. Final verification: 2026-09-19.
This milestone applies rustfmt to existing Rust source and promotes the existing
hosted formatting check to blocking. It makes no performance claim and changes
no dependency, CUDA kernel, model behavior or production configuration.

## 1. Starting HEAD

`29ebaaeb17cd2a63802fe9c7dba5eb8da219e88d`, titled
`ci: add hosted quality gates and manual NVIDIA validation`.
The exact requested commit was verified before editing and remains HEAD.

## 2. Starting tree

`git status --short --untracked-files=all` was empty. The pre-format command
suite changed no tracked file bytes. No project commit, staging, push or GitHub
setting change was performed during this milestone.

## 3. Original formatting failure

```bash
cargo fmt --manifest-path engine/Cargo.toml --all -- --check
```

The committed baseline returned **exit 1**, in 0.465 seconds, with formatting
drift in 27 files. The CI policy was left unchanged until the pre-format
correctness baseline passed and rustfmt normalization had been applied.

## 4. Files normalized and formatter

Exactly **27 of 33 tracked Rust files** changed. Source changes were generated
only by:

```bash
cargo fmt --manifest-path engine/Cargo.toml --all
```

Local toolchain: rustc/Cargo 1.98.1, rustfmt
`1.9.0-stable (48a229ceae 2026-09-01)`, edition 2021. No manual Rust edit, Clippy
fix, generic formatter, or permanent parser dependency was introduced.

## 5. Rust-only diff statistics

`git diff --stat -- engine` reports **27 files changed, 4,248 insertions and
1,343 deletions**. The large line count is predominantly wrapping and indentation.
The exact changed-file list is in item 24. Diff statistics depend on the diff
algorithm; they are not a count of semantic changes.

## 6. Verification method

All changed files are reviewed against the committed baseline, including a
word-oriented diff and a temporary Rust parser/token comparison. Verification
tools, parser dependencies, raw logs, source snapshots and binary hashes are
kept outside the repository. Source strings are parsed as Rust tokens rather
than compared with a naive whitespace-deletion algorithm.

The final comparison details and results are recorded in item 7. Compilation,
the same before/after correctness tests, and a focused real-GPU smoke run provide
independent checks; none alone is presented as a formal proof for every possible
runtime execution.

## 7. Token/AST comparison

The temporary tool was built offline and locked with already available
`syn 2.0.119`, `proc-macro2 1.0.107` and `quote 1.0.47`. All **33** Rust files
were parsed before and after. Thirteen have equal raw ASTs; all **33 have equal
normalized ASTs**, with **zero unresolved differences**. Baseline copies
byte-match the requested HEAD, and every current Rust file byte-matches a second,
independently generated `cargo fmt` preview.

Normalization is limited to the observed formatter transformations:

- Parsed optional comma punctuation, retaining expression/tuple structure and
  element ordering, including the distinction between `(value)` and `(value,)`.
- Ordering within use groups, adjacent imports, and external module declarations;
  the reordered modules have no `macro_use` attributes.
- Unlabelled, attribute-free single-expression wrappers in match-arm and closure
  bodies, plus optional match-arm commas.
- Exactly the two observed `let ... else { break[;] }` cases, with no label,
  break value, attributes or extra statements. Value-producing expression
  semicolons are not discarded.
- Parsed argument/body structure for the known `vec!` and local `timed!` macros;
  other macro bodies remain opaque and must stay identical.

Independent per-file inventories preserve **56,198 identifier tokens**,
**8,001 literal tokens** (including **3,485 numeric literals**) and **2,883
attributes**. String contents and numeric spellings are preserved; ordered AST
comparison additionally protects expression order, cfg conditions and test
expectations rather than relying on inventory equality alone.

Of **1,543 macro nodes**, exactly **25** have changed raw token streams: seven
`vec!` and eighteen `timed!` calls. The changes are comma punctuation; their
parsed argument/body ASTs match. Macro definitions and all opaque macro bodies
are unchanged. Independent word-oriented diff review covered every changed
file, including import/module ordering and the two bare-break semicolons.

Ten deliberate negative controls were rejected: changed string contents,
numeric literal, identifier order, cfg expression, test expectation, opaque
macro comma, singleton tuple, value semicolon, vector ordering and timed-body
expression. This checks that normalization does not silently erase those
meaningful mutations.

Source spans, ordinary comments and diagnostic locations are outside AST
equality. No `line!`, `column!`, `file!`, `stringify!`, `track_caller` or
`macro_use` appears in these sources; the only include macro reads the unchanged
CUDA kernel. These limits are stated explicitly rather than claiming a formal
compiler-equivalence proof.

## 8. Cargo manifest

`git diff -- engine/Cargo.toml` is empty. Features, defaults, profiles and runtime
dependencies are unchanged. Its pre/post file SHA-256 is
`bb612a174416e9223775cd524a49dc329b98268abfc464f4dc5aeba13497cf73`.

## 9. Cargo lockfile

`git diff -- engine/Cargo.lock` is empty. Its pre/post file SHA-256 is
`78349daa26609018d92623bd92f8d18e5e077cbdd3cbff185d77d3b99f060b28`.
Every Cargo build/test/check/Clippy command uses `--locked`; no dependency update
or RustSec advisory remediation is included.

## 10. CUDA source

`git diff -- engine/kernels/kernels.cu` is empty. Its pre/post file SHA-256 is
`40de2b47c79ba5cef0adbefe0daa907211c40328260b7df58968befc1e3bc869`.
The source's only explicit include macro still embeds that same file.

## 11. Production defaults

The source comparison and scope audit preserve packed prefill on, aggregate
prefill budget 1024, extra chunk cap off, singleton prefill graphs on, packed
serving graphs off, supported exact-q4-k64 paged attention on, reference
attention retained, hybrid current-K/V off, and page-table cache off. No adaptive
scheduler or other new runtime feature was added.

## 12. Post-format rustfmt

The exact check in item 3 now returns **exit 0** (0.516 seconds in the post-format
suite). A clean format check is required; the command has not been narrowed.

## 13. Clippy

```bash
cargo clippy --manifest-path engine/Cargo.toml --all-targets --locked -- -D clippy::correctness -D clippy::suspicious
```

Exit 0 before and after (0.917s / 0.918s). Existing style/performance warnings
remain; no global allows, warning-policy changes or opportunistic fixes were
made. `-D warnings` is still outside the current baseline-compatible policy.

## 14. Default Rust tests

`cargo test --manifest-path engine/Cargo.toml --locked` passed **144 tests**
(130 unit and 14 headless TUI integration tests) before and after. Exit 0;
8.494s / 3.424s.

## 15. No-default Rust tests

`cargo test --manifest-path engine/Cargo.toml --no-default-features --locked`
passed **99 tests** before and after. Exit 0; 4.528s / 0.968s.

## 16. TUI feature isolation

`cargo check --manifest-path engine/Cargo.toml --no-default-features --features tui --locked`
passed before and after. Exit 0; 4.233s / 0.867s.

## 17. CUDA-feature host tests

```bash
CUDARC_CUDA_VERSION=13020 CUDA_VISIBLE_DEVICES=-1 cargo test --manifest-path engine/Cargo.toml --features cuda --locked
```

Passed **190 host tests** (176 unit and 14 TUI integration tests) before and
after. Exit 0; 6.286s / 6.082s. The binding-version override permits CUDA Rust
compilation without toolkit discovery; these tests do not initialize a GPU or
compile NVRTC kernels. Real device coverage is separate in item 20.

## 18. Source/helper and shell checks

The unchanged source checker, all **41 CI helper tests**, and the workflow's
ShellCheck command pass with exit 0. Source checking covers 93 tracked/intended
files, 32 Python files, three JSON files, ten Markdown files and 48 local links.
The exact commands are:

```bash
python scripts/ci_source.py
python -m unittest discover -s tests -p 'test_ci_*.py'
shellcheck --severity=warning scripts/setup_wsl.sh
```

No Python, shell or benchmark/test helper source was changed. The shell
installer is only statically checked, never executed.

The actual workflow shell blocks ran in a disposable clean Git fixture containing
the intended files. The project index remained unchanged. Source checking took
0.166s, helper tests 0.367s, ShellCheck 0.066s, and the final clean-checkout
assertion 0.005s; every step left the fixture clean. ShellCheck was already
installed, so its conditional installation branch was not needed.

## 19. Workflow validation

Both final workflows pass actionlint 1.7.12 with the existing custom-label
configuration. Parsed YAML comparison verifies that only the hosted formatting
policy changes; all remaining jobs and workflow-level settings are identical.
The actionlint command exited 0 in 0.066s. No GitHub run is implied by local
workflow validation.

## 20. Focused real-GPU smoke and optional binary comparison

The existing, unchanged wrapper is used:

```bash
python3 scripts/ci_gpu.py --mode smoke
```

It passed with **exit 0 in 98.536 seconds** on the provisioned local SM120
NVIDIA GPU and exported 120M model. The freshly built binary had the post-format
hash listed below. All nine selected checks completed: `gpu-validate`,
`gpu-graph-check`, `gpu-batch`, `gpu-sampling`, `gpu-prefill-check`,
`gpu-paged --graph`, the bounded prefill-graph corpus, packed int8 with
`--steps 16 --fuzz 4`, and attention with `--fuzz 8 --seed 20260908`.
The attention corpus reported 53 fixed plus eight fuzz cases and 2,928 reference
tensor comparisons. No performance matrix or throughput acceptance benchmark
was run. The existing wrapper's documented numerical-coverage limits remain
unchanged; printed diagnostic-only kernel measurements are not new gates.

For additional evidence, the same workspace, target directory, toolchain and
command built CUDA release binaries before and after:

```bash
cargo build --manifest-path engine/Cargo.toml --release --features cuda --locked
```

Both builds exited 0 (0.265s before, warm; 46.561s after). SHA-256 values:

- Before: `97001237b3a35350ce8e562e9cf7629703832eb39bdf026b4b8dfdfe2ed571c0`.
- After: `eb9d51618fe5246aff8281f04a36660217b25069073bd1f3053591b96adc6f50`.

They differ, so binary identity is **not** claimed. Rust binaries may include
source-location and other build metadata; the differing hash alone neither
proves nor disproves a behavioral change. No time was spent attributing every
binary byte. Source structure, literal/macro inspection and correctness checks
are the meaningful evidence here. Timings are local validation wall times,
not a performance comparison; build-cache states differ.

## 21. Documentation updates

README now describes enforced Rust formatting. CONTRIBUTING says the source is
rustfmt-clean, makes the exact pre-submission check explicit, and retains the
existing Clippy policy. The old [CI report](ci-results.md) has a dated follow-up
note linking here; its original milestone record is preserved verbatim below
that note. Historical informational-formatting results are not rewritten as
successful checks. This report records the separate normalization milestone.

## 22. Exact CI promotion

In `.github/workflows/ci.yml`, the format step becomes `Check Rust formatting`.
Its `continue-on-error: true` and now-unused `id: fmt` are removed, along with
the obsolete debt comments and warning-only step. The actual Cargo command is
unchanged. A formatting failure now fails `quality` normally.

Workflow/job names, triggers, permissions, checkout credential policy, action
SHAs, cache keys/paths, Rust setup, test/feature commands and dependency-audit
semantics are unchanged. GPU CI is byte-for-byte unchanged. No badge was added.

## 23. Diff whitespace validation

`git diff --check` returned **0**. The final tree audit found only the three
allowed change categories in item 24.

## 24. Exact changed-file classification

**A. Rustfmt-generated Rust source (27 files):**

```text
engine/src/anthropic/messages.rs
engine/src/chat_template.rs
engine/src/config.rs
engine/src/gpu.rs
engine/src/gpu_attention_validation.rs
engine/src/gpu_model.rs
engine/src/gpu_packed_validation.rs
engine/src/lib.rs
engine/src/main.rs
engine/src/model.rs
engine/src/openai/chat.rs
engine/src/openai/completions.rs
engine/src/openai/mod.rs
engine/src/ops.rs
engine/src/paged.rs
engine/src/prefill.rs
engine/src/protocol.rs
engine/src/quant.rs
engine/src/runtime.rs
engine/src/sampling.rs
engine/src/server.rs
engine/src/tokenizer.rs
engine/src/tui/app.rs
engine/src/tui/client.rs
engine/src/tui/ui.rs
engine/src/weights.rs
engine/tests/tui_client.rs
```

**B. Hosted formatting-gate promotion:** `.github/workflows/ci.yml`.

**C. Documentation directly related to promotion:** `README.md`,
`CONTRIBUTING.md`, `docs/ci-results.md`, and new `docs/rustfmt-results.md`.

Nothing else belongs to this diff.

## 25. Final Git status

The final tree contains only the 32 intentional files listed in item 24:
31 modified tracked files and this new report. No project commit was made.

```text
 M .github/workflows/ci.yml
 M CONTRIBUTING.md
 M README.md
 M docs/ci-results.md
 M engine/src/anthropic/messages.rs
 M engine/src/chat_template.rs
 M engine/src/config.rs
 M engine/src/gpu.rs
 M engine/src/gpu_attention_validation.rs
 M engine/src/gpu_model.rs
 M engine/src/gpu_packed_validation.rs
 M engine/src/lib.rs
 M engine/src/main.rs
 M engine/src/model.rs
 M engine/src/openai/chat.rs
 M engine/src/openai/completions.rs
 M engine/src/openai/mod.rs
 M engine/src/ops.rs
 M engine/src/paged.rs
 M engine/src/prefill.rs
 M engine/src/protocol.rs
 M engine/src/quant.rs
 M engine/src/runtime.rs
 M engine/src/sampling.rs
 M engine/src/server.rs
 M engine/src/tokenizer.rs
 M engine/src/tui/app.rs
 M engine/src/tui/client.rs
 M engine/src/tui/ui.rs
 M engine/src/weights.rs
 M engine/tests/tui_client.rs
?? docs/rustfmt-results.md
```

## 26. Known limitations

Parser normalization and review provide strong practical evidence, not a
formal theorem of all observable behavior. Source line/column positions and
diagnostic/panic locations naturally move. Release binary hashes differ.
Existing Clippy, Python lint and RustSec debt remains out of scope. GitHub
execution of the newly blocking gate is pending commit/push; local validation
is not represented as a GitHub run. No new GPU sanitizer or performance claim
is made.

## 27. Proposed single commit title

`style: normalize Rust formatting and enforce rustfmt`

## 28. Proposed complete commit body

```text
Normalize 27 existing Rust files using cargo fmt and promote the unchanged
whole-crate rustfmt check from informational to blocking in hosted quality CI.
Remove only its failure exception and obsolete formatting-debt warning step.

Update contributor/README guidance and add a dated follow-up to the historical
CI report without rewriting that milestone's original results. Record source
comparison methods, validation results and limitations in docs/rustfmt-results.md.

Keep Cargo manifests/lockfile, CUDA kernels, inference defaults, Python helpers,
GPU CI, Clippy policy, action pins, permissions, caches and dependency-audit
semantics unchanged. No performance or dependency work is included.

Validation: all 33 Rust files match an independent cargo-fmt preview and pass
normalized AST/literal/macro comparison. Before/after default and feature tests,
Clippy, source checks, all 41 CI helper tests, ShellCheck, actionlint and focused
real-GPU smoke pass. Release binary hashes differ, so binary identity is not
claimed. Full evidence and limitations are in docs/rustfmt-results.md.

No project commit or GitHub workflow execution was performed during preparation.
```
