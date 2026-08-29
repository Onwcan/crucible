use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use llm_engine::{Config, Tokenizer, Weights};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "llm-engine", about = "Inference engine for locally trained models")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Load a checkpoint and report what is in it.
    Inspect {
        /// Directory containing model.safetensors and config.json
        model: PathBuf,
        /// List every tensor rather than a summary
        #[arg(long)]
        verbose: bool,
    },
    /// Run a forward pass and print logits, for comparison against PyTorch.
    Logits {
        model: PathBuf,
        /// Comma-separated token ids, e.g. "464,2159,318"
        #[arg(long, default_value = "464,2159,318,1719")]
        tokens: String,
        /// How many top logits to print
        #[arg(long, default_value_t = 10)]
        top: usize,
    },
    /// Check every CUDA kernel against the CPU reference.
    #[cfg(feature = "cuda")]
    GpuValidate,
    /// Measure GEMV bandwidth on the GPU.
    #[cfg(feature = "cuda")]
    GpuBench {
        #[arg(long, default_value_t = 4096)]
        rows: usize,
        #[arg(long, default_value_t = 4096)]
        cols: usize,
        #[arg(long, default_value_t = 200)]
        iters: usize,
    },
    /// Run the forward pass on the GPU and compare against the CPU reference.
    #[cfg(feature = "cuda")]
    GpuLogits {
        model: PathBuf,
        #[arg(long, default_value = "464,2159,318,1719")]
        tokens: String,
        #[arg(long, default_value_t = 10)]
        top: usize,
        /// Decode this many tokens to measure throughput (0 to skip)
        #[arg(long, default_value_t = 64)]
        decode: usize,
        /// Weight precision: f32 or int8
        #[arg(long, default_value = "f32")]
        quant: String,
        /// Replay decode from a captured CUDA graph
        #[arg(long)]
        graph: bool,
        /// Prefill this many synthetic tokens and report prompt throughput.
        ///
        /// Passing a long token list on the command line is impractical, and
        /// prefill throughput is the number that reveals whether the prompt is
        /// processed in parallel or one token at a time.
        #[arg(long, default_value_t = 0)]
        prefill: usize,
    },
    /// Per-stage timing breakdown for one decode step.
    #[cfg(feature = "cuda")]
    GpuProfile {
        model: PathBuf,
        #[arg(long, default_value = "int8")]
        quant: String,
        #[arg(long, default_value_t = 100)]
        iters: usize,
        /// Cache positions to fill before profiling, so attention sees a
        /// realistic sequence length rather than an empty cache.
        #[arg(long, default_value_t = 256)]
        warm: usize,
    },
    /// Cross-entropy on held-out tokens, per precision.
    ///
    /// The honest way to price quantisation: speed is easy to measure and easy
    /// to be pleased by, but it is only worth having if quality holds.
    /// Check the paged KV cache against the contiguous one.
    ///
    /// Paging changes where keys and values live, not what is computed, so the
    /// two paths must agree bit-for-bit. Anything less means a page is being
    /// read at the wrong offset, and the symptom of that is fluent text with
    /// quietly wrong attention -- the failure mode this repo has been bitten by
    /// before and does not detect by reading output.
    #[cfg(feature = "cuda")]
    GpuPaged {
        model: PathBuf,
        #[arg(long, default_value = "int8")]
        quant: String,
        /// Sequence lengths to check. Defaults straddle every page boundary
        /// that matters at PAGE_TOKENS=16.
        #[arg(long, default_value = "1,7,15,16,17,31,32,33,64,127,128,129,256,511,512,1023")]
        lengths: String,
        #[arg(long)]
        graph: bool,
    },
    /// Check batched decode and the continuous-batching scheduler against
    /// independent single-request execution.
    ///
    /// The only result that matters is that a request's output does not depend
    /// on who it was batched with. Heterogeneous lengths are the point: padding
    /// everything to the longest sequence would pass a same-length test and
    /// still be wrong.
    /// Concurrent-decode throughput against N independent single-request runs.
    ///
    /// Aggregate throughput and per-request throughput are different claims and
    /// are reported separately: batching raises the first while lowering the
    /// second, and calling that a latency improvement would be wrong.
    #[cfg(feature = "cuda")]
    GpuServeBench {
        model: PathBuf,
        #[arg(long, default_value = "int8")]
        quant: String,
        #[arg(long, default_value = "1,2,4,8,16")]
        batches: String,
        /// Prompt lengths, cycled across requests so a batch is heterogeneous.
        #[arg(long, default_value = "32,128,256,512")]
        lengths: String,
        #[arg(long, default_value_t = 64)]
        steps: usize,
        #[arg(long, default_value_t = 3)]
        trials: usize,
    },
    #[cfg(feature = "cuda")]
    GpuBatch {
        model: PathBuf,
        #[arg(long, default_value = "int8")]
        quant: String,
        /// Prompt lengths, one request each. Chosen to straddle page
        /// boundaries at PAGE_TOKENS=16.
        #[arg(long, default_value = "7,63,129,511")]
        lengths: String,
        /// Tokens to generate per request, counting the one prefill produces.
        #[arg(long, default_value_t = 24)]
        steps: usize,
        #[arg(long, default_value_t = 8)]
        max_batch: usize,
    },
    #[cfg(feature = "cuda")]
    GpuEval {
        model: PathBuf,
        /// Path to val.bin (flat uint16 token ids)
        #[arg(long)]
        data: PathBuf,
        #[arg(long, default_value_t = 2048)]
        tokens: usize,
        /// Comma-separated precisions to compare
        #[arg(long, default_value = "f32,int8")]
        quant: String,
        /// Score through the prefill path using a context of this many tokens.
        ///
        /// Default 0 scores token-at-a-time, which routes every matmul through
        /// GEMV and never touches the GEMM. With a context set, each scored
        /// position is predicted from a fresh prefill of the preceding tokens,
        /// so the batched GEMM is what produces the number.
        #[arg(long, default_value_t = 0)]
        prefill_ctx: usize,
        /// Replay decode from a captured CUDA graph
        #[arg(long)]
        graph: bool,
    },
    /// Encode text and print token ids, to compare against tiktoken.
    Tokenize {
        /// Path to gpt2.tok
        tokenizer: PathBuf,
        text: String,
    },
    /// Generate text from a prompt.
    Generate {
        model: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long, default_value = "The meaning of life is")]
        prompt: String,
        #[arg(long, default_value_t = 40)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.8)]
        temperature: f32,
        #[arg(long, default_value_t = 40)]
        top_k: usize,
        #[arg(long, default_value_t = 1234)]
        seed: u64,
        /// Run on the GPU instead of the CPU reference path
        #[arg(long)]
        gpu: bool,
        /// Replay decode from a captured CUDA graph (implies --gpu)
        #[arg(long)]
        graph: bool,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Inspect { model, verbose } => inspect(model, verbose),
        Command::Logits { model, tokens, top } => logits(model, &tokens, top),
        Command::Tokenize { tokenizer, text } => tokenize(tokenizer, &text),
        #[cfg(feature = "cuda")]
        Command::GpuValidate => llm_engine::gpu::validate(),
        #[cfg(feature = "cuda")]
        Command::GpuBench { rows, cols, iters } => llm_engine::gpu::bench(rows, cols, iters),
        #[cfg(feature = "cuda")]
        Command::GpuLogits { model, tokens, top, decode, quant, graph, prefill } => {
            gpu_logits(model, &tokens, top, decode, &quant, graph, prefill)
        }
        #[cfg(feature = "cuda")]
        #[cfg(feature = "cuda")]
        Command::GpuProfile { model, quant, iters, warm } => gpu_profile(model, &quant, iters, warm),
        #[cfg(feature = "cuda")]
        Command::GpuServeBench { model, quant, batches, lengths, steps, trials } => {
            gpu_serve_bench(model, &quant, &batches, &lengths, steps, trials)
        }
        #[cfg(feature = "cuda")]
        Command::GpuBatch { model, quant, lengths, steps, max_batch } => {
            gpu_batch(model, &quant, &lengths, steps, max_batch)
        }
        #[cfg(feature = "cuda")]
        Command::GpuPaged { model, quant, lengths, graph } => {
            gpu_paged(model, &quant, &lengths, graph)
        }
        #[cfg(feature = "cuda")]
        Command::GpuEval { model, data, tokens, quant, graph, prefill_ctx } => {
            gpu_eval(model, data, tokens, &quant, graph, prefill_ctx)
        }
        Command::Generate {
            model,
            tokenizer,
            prompt,
            max_tokens,
            temperature,
            top_k,
            seed,
            gpu,
            graph,
        } => generate(model, tokenizer, &prompt, max_tokens, temperature, top_k, seed, gpu || graph, graph),
    }
}

fn tokenize(path: PathBuf, text: &str) -> Result<()> {
    let tok = Tokenizer::load(&path)?;
    let ids = tok.encode(text)?;
    println!("vocab   : {}", tok.vocab_size());
    println!("text    : {text:?}");
    println!("ids     : {ids:?}");
    println!("count   : {}", ids.len());
    println!("decoded : {:?}", tok.decode(&ids));
    if tok.decode(&ids) != text {
        println!();
        println!("WARNING: round-trip does not match the input");
    }
    Ok(())
}

/// xorshift64*, so sampling is reproducible without pulling in a rand crate.
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        // Top 24 bits give a uniform float in [0, 1).
        ((self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / (1u32 << 24) as f32
    }
}

fn sample(logits: &[f32], temperature: f32, top_k: usize, rng: &mut Rng) -> u32 {
    if temperature <= 0.0 {
        // Greedy: argmax, deterministic.
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    let k = top_k.clamp(1, ranked.len());
    // Only the top k matter; partial selection avoids sorting 50k entries.
    ranked.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
    ranked.truncate(k);

    let max = ranked.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for (_, v) in ranked.iter_mut() {
        *v = ((*v - max) / temperature).exp();
        sum += *v as f64;
    }

    let threshold = rng.next_f32() as f64 * sum;
    let mut acc = 0.0f64;
    for (id, weight) in &ranked {
        acc += *weight as f64;
        if acc >= threshold {
            return *id as u32;
        }
    }
    ranked.last().map(|(i, _)| *i as u32).unwrap_or(0)
}

fn generate(
    model_dir: PathBuf,
    tokenizer_path: PathBuf,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    seed: u64,
    use_gpu: bool,
    use_graph: bool,
) -> Result<()> {
    let cfg = Config::from_file(model_dir.join("config.json"))?;
    let weights = Weights::open(model_dir.join("model.safetensors"))?;
    let tok = Tokenizer::load(&tokenizer_path)?;

    enum Backend {
        Cpu(llm_engine::Model, llm_engine::KvCache),
        #[cfg(feature = "cuda")]
        Gpu(llm_engine::gpu_model::GpuModel),
    }

    let block_size = cfg.block_size;
    let mut backend = if use_gpu {
        #[cfg(feature = "cuda")]
        {
            let mut m = llm_engine::gpu_model::GpuModel::load(
                cfg.clone(),
                &weights,
                block_size,
            )?;
            m.enable_graph(use_graph);
            Backend::Gpu(m)
        }
        #[cfg(not(feature = "cuda"))]
        {
            anyhow::bail!("built without the cuda feature; rebuild with --features cuda")
        }
    } else {
        let m = llm_engine::Model::load(cfg.clone(), &weights)?;
        let c = m.new_cache(block_size);
        Backend::Cpu(m, c)
    };

    let step = |backend: &mut Backend, toks: &[usize]| -> Result<Vec<f32>> {
        match backend {
            Backend::Cpu(m, c) => m.forward_cached(toks, c),
            #[cfg(feature = "cuda")]
            Backend::Gpu(m) => m.forward(toks),
        }
    };
    let used = |backend: &Backend| -> usize {
        match backend {
            Backend::Cpu(_, c) => c.len(),
            #[cfg(feature = "cuda")]
            Backend::Gpu(m) => m.cache_len(),
        }
    };

    let mut ids: Vec<usize> = tok.encode(prompt)?.into_iter().map(|v| v as usize).collect();
    if ids.is_empty() {
        anyhow::bail!("prompt encoded to zero tokens");
    }
    let prompt_len = ids.len();
    let mut rng = Rng(seed | 1);

    println!("prompt      : {prompt:?} ({prompt_len} tokens)");
    println!("backend     : {}{}", if use_gpu { "gpu" } else { "cpu" },
             if use_graph { " (cuda graph)" } else { "" });
    println!("sampling    : temperature {temperature}, top-k {top_k}, seed {seed}");
    println!();
    print!("{prompt}");
    use std::io::Write;
    std::io::stdout().flush().ok();

    // Bytes are buffered because one UTF-8 character can span several tokens;
    // printing each token's bytes alone would emit replacement characters.
    let mut pending: Vec<u8> = Vec::new();

    // Prefill: run the prompt once, filling the cache. Timed separately from
    // decode because the two scale differently -- prefill is a single pass over
    // the prompt, decode is one token at a time against a growing cache.
    let prefill_start = std::time::Instant::now();
    let mut logits = step(&mut backend, &ids)?;
    let prefill_s = prefill_start.elapsed().as_secs_f64();

    let started = std::time::Instant::now();
    for _ in 0..max_tokens {
        if used(&backend) >= block_size {
            println!("\n[context full]");
            break;
        }
        let next = sample(&logits, temperature, top_k, &mut rng);
        if next == tok.eot {
            println!("\n[end of text]");
            break;
        }
        if let Some(bytes) = tok.decode_piece(next) {
            pending.extend_from_slice(bytes);
            if let Ok(text) = std::str::from_utf8(&pending) {
                print!("{text}");
                std::io::stdout().flush().ok();
                pending.clear();
            }
        }
        ids.push(next as usize);
        // Only the new token is processed; earlier positions are read from the
        // cache rather than recomputed.
        logits = step(&mut backend, &[next as usize])?;
    }

    let elapsed = started.elapsed().as_secs_f64();
    let generated = ids.len() - prompt_len;
    println!();
    println!();
    println!(
        "prefill  {prompt_len} tokens in {prefill_s:.2}s ({:.1} tok/s)",
        prompt_len as f64 / prefill_s
    );
    println!(
        "decode   {generated} tokens in {elapsed:.2}s ({:.2} tok/s, {:.0} ms/token)",
        generated as f64 / elapsed,
        elapsed * 1000.0 / generated.max(1) as f64
    );
    match &backend {
        Backend::Cpu(_, c) => println!("kv cache {:.1} MB (host)", c.bytes() as f64 / 1e6),
        #[cfg(feature = "cuda")]
        Backend::Gpu(m) => println!("device   {:.0} MB (weights + cache)",
                                    m.device_bytes() as f64 / 1e6),
    }
    Ok(())
}

fn logits(dir: PathBuf, tokens: &str, top: usize) -> Result<()> {
    let cfg = Config::from_file(dir.join("config.json"))?;
    let weights = Weights::open(dir.join("model.safetensors"))?;

    let ids: Vec<usize> = tokens
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .context("parsing --tokens as comma-separated integers")?;

    let started = std::time::Instant::now();
    let model = llm_engine::Model::load(cfg, &weights)?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;

    let started = std::time::Instant::now();
    let out = model.forward(&ids)?;
    let fwd_ms = started.elapsed().as_secs_f64() * 1000.0;

    println!("tokens {ids:?}");
    println!("load {load_ms:.0} ms, forward {fwd_ms:.0} ms");

    // A checksum over all logits catches a discrepancy anywhere in the
    // vocabulary, not just in the handful printed below.
    let sum: f64 = out.iter().map(|v| *v as f64).sum();
    let max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min = out.iter().copied().fold(f32::INFINITY, f32::min);
    println!("logits: sum {sum:.4}, min {min:.6}, max {max:.6}");

    let mut ranked: Vec<(usize, f32)> = out.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!();
    println!("top {top}:");
    for (id, value) in ranked.iter().take(top) {
        println!("  {id:6} {value:10.6}");
    }
    Ok(())
}

fn inspect(dir: PathBuf, verbose: bool) -> Result<()> {
    let cfg = Config::from_file(dir.join("config.json"))?;
    let weights = Weights::open(dir.join("model.safetensors")).context("opening weights")?;
    let inventory = weights.inventory()?;

    println!("config");
    println!("  layers      {}", cfg.n_layer);
    println!("  d_model     {}", cfg.n_embd);
    println!(
        "  heads       {} query / {} kv  (n_rep {}, {})",
        cfg.n_head,
        cfg.n_kv_head,
        cfg.n_rep(),
        cfg.attention
    );
    println!("  head_dim    {}", cfg.head_dim());
    println!("  vocab       {}", cfg.vocab_size);
    println!("  context     {}", cfg.block_size);
    println!(
        "  arch        {} / {} / {}-{}",
        cfg.pos_encoding, cfg.activation, cfg.norm_placement, cfg.norm
    );
    if let Some(loss) = cfg.val_loss {
        println!("  val loss    {loss:.4}");
    }

    let params: usize = inventory
        .iter()
        .map(|(_, s, _)| s.iter().product::<usize>())
        .sum();
    println!();
    println!("checkpoint");
    println!("  tensors     {}", inventory.len());
    println!("  parameters  {:.1}M", params as f64 / 1e6);
    println!("  mapped      {:.1} MB", weights.total_bytes() as f64 / 1e6);
    println!(
        "  kv cache    {:.1} MB at full context (f32, one sequence)",
        cfg.kv_cache_bytes(cfg.block_size) as f64 / 1e6
    );

    if verbose {
        println!();
        for (name, shape, dtype) in &inventory {
            println!("  {name:38} {shape:?} {dtype:?}");
        }
    }

    // Prove the mapping is readable, not merely parseable.
    let emb = weights.get("tok_emb.weight")?;
    println!();
    println!(
        "tok_emb.weight {:?}, first row starts {:.5} {:.5} {:.5}",
        emb.shape, emb.data[0], emb.data[1], emb.data[2]
    );
    if cfg.tie_word_embeddings {
        println!("lm_head is tied to tok_emb (not stored separately)");
    }

    Ok(())
}


#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn gpu_logits(dir: PathBuf, tokens: &str, top: usize, decode: usize, quant: &str, graph: bool, prefill: usize) -> Result<()> {
    use llm_engine::gpu_model::{GpuModel, Precision};

    let precision = Precision::parse(quant)
        .ok_or_else(|| anyhow::anyhow!("unknown precision {quant:?}; expected f32 or int8"))?;

    let cfg = Config::from_file(dir.join("config.json"))?;
    let weights = Weights::open(dir.join("model.safetensors"))?;

    let ids: Vec<usize> = if prefill > 0 {
        // Varied ids rather than a repeated token, so nothing can be skipped
        // by a cache or short-circuited by identical embeddings.
        (0..prefill).map(|i| 1000 + (i * 7) % 20000).collect()
    } else {
        tokens
            .split(',')
            .map(|s| s.trim().parse::<usize>())
            .collect::<Result<Vec<_>, _>>()
            .context("parsing --tokens")?
    };

    let started = std::time::Instant::now();
    let mut gpu_model = GpuModel::load_with(cfg.clone(), &weights, cfg.block_size, precision)?;
    gpu_model.enable_graph(graph);
    println!("precision   {quant}, graph {}", if graph { "on" } else { "off" });
    println!("load        {:.0} ms, {:.0} MB weights + {:.0} MB cache",
             started.elapsed().as_secs_f64() * 1000.0,
             gpu_model.weight_bytes() as f64 / 1e6,
             gpu_model.cache_bytes() as f64 / 1e6);

    // Prefill, timed after a warm-up pass so the measurement is not dominated
    // by first-launch costs (module load, allocator warm-up, clock ramp).
    gpu_model.forward(&ids)?;
    gpu_model.reset();

    let started = std::time::Instant::now();
    let gpu_out = gpu_model.forward(&ids)?;
    let prefill_ms = started.elapsed().as_secs_f64() * 1000.0;

    // The CPU path is the reference: same weights, same order, scalar kernels.
    let cpu_model = llm_engine::Model::load(cfg.clone(), &weights)?;
    let cpu_out = cpu_model.forward(&ids)?;

    let max_rel = gpu_out
        .iter()
        .zip(&cpu_out)
        .map(|(g, c)| {
            let scale = (g.abs().max(c.abs()) as f64).max(1e-6);
            ((*g as f64) - (*c as f64)).abs() / scale
        })
        .fold(0.0f64, f64::max);

    let sum: f64 = gpu_out.iter().map(|v| *v as f64).sum();
    println!("prefill     {} tokens in {prefill_ms:.1} ms  ({:.0} tok/s)",
             ids.len(), ids.len() as f64 / (prefill_ms / 1000.0));
    println!();
    println!("logits: sum {sum:.4}, min {:.6}, max {:.6}",
             gpu_out.iter().copied().fold(f32::INFINITY, f32::min),
             gpu_out.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    println!("max relative difference vs CPU reference: {max_rel:.3e}");

    let mut ranked: Vec<(usize, f32)> = gpu_out.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut cpu_ranked: Vec<(usize, f32)> = cpu_out.iter().copied().enumerate().collect();
    cpu_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!();
    println!("top {top}:            GPU              CPU");
    for i in 0..top.min(ranked.len()) {
        let flag = if ranked[i].0 == cpu_ranked[i].0 { " " } else { " <- ORDER DIFFERS" };
        println!("  {:6} {:10.6}   {:6} {:10.6}{flag}",
                 ranked[i].0, ranked[i].1, cpu_ranked[i].0, cpu_ranked[i].1);
    }

    if decode > 0 {
        // Greedy decode, so throughput is measured without sampling noise.
        let mut next = ranked[0].0;
        let started = std::time::Instant::now();
        for _ in 0..decode {
            let out = gpu_model.forward(&[next])?;
            next = out
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
        let secs = started.elapsed().as_secs_f64();
        println!();
        println!("decode      {decode} tokens in {secs:.2} s  ({:.1} tok/s, {:.2} ms/token){}",
                 decode as f64 / secs,
                 secs * 1000.0 / decode as f64,
                 if gpu_model.graph_active() { "  [graph]" } else { "" });
    }

    Ok(())
}


/// Deterministic pseudo-random token ids, so a failure reproduces exactly.
#[cfg(feature = "cuda")]
fn probe_tokens(n: usize, vocab: usize) -> Vec<usize> {
    let mut state: u64 = 0x9E3779B97F4A7C15;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % vocab as u64) as usize
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn gpu_paged(dir: PathBuf, quant: &str, lengths: &str, graph: bool) -> Result<()> {
    use llm_engine::gpu_model::{GpuModel, Precision};
    use llm_engine::paged::PAGE_TOKENS;

    let cfg = Config::from_file(dir.join("config.json"))?;
    let weights = Weights::open(dir.join("model.safetensors"))?;
    let precision = Precision::parse(quant)
        .ok_or_else(|| anyhow::anyhow!("unknown precision {quant:?}"))?;

    let mut model = GpuModel::load_with(cfg.clone(), &weights, cfg.block_size, precision)?;
    model.enable_graph(graph);

    // Enough pages for one full-context sequence, which is the parity case
    // against the contiguous cache.
    let n_pages = cfg.block_size.div_ceil(PAGE_TOKENS);
    model.enable_paging(n_pages, 1)?;
    model.set_paged(false);

    println!("page size   {PAGE_TOKENS} tokens");
    println!("pool        {n_pages} pages, {:.2} MB",
             model.page_pool().total_bytes() as f64 / 1e6);
    println!("graph       {graph}");
    println!();

    let lens: Vec<usize> = lengths
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .filter(|v| *v > 0 && *v <= cfg.block_size)
        .collect();

    let mut worst_prefill = 0.0f64;
    let mut worst_decode = 0.0f64;
    let mut failures = 0usize;

    println!("{:>6}  {:>6}  {:>14}  {:>14}  {:>7}",
             "len", "pages", "prefill maxdiff", "decode maxdiff", "top-1");
    println!("{}", "-".repeat(58));

    for &len in &lens {
        let toks = probe_tokens(len, cfg.vocab_size);

        // Whole prompt at once: exercises cache_store and prefill attention.
        model.set_paged(false);
        model.reset();
        let want_prefill = model.forward(&toks)?;
        model.set_paged(true);
        model.reset();
        let got_prefill = model.forward(&toks)?;
        let pages = model.seq_pages();
        let d_prefill = max_abs_diff(&want_prefill, &got_prefill);

        // One token at a time: exercises the decode projection writing into a
        // page and the paged decode attention, including graph replay.
        model.set_paged(false);
        model.reset();
        let mut want_decode = Vec::new();
        for &t in &toks {
            want_decode = model.forward(&[t])?;
        }
        model.set_paged(true);
        model.reset();
        let mut got_decode = Vec::new();
        for &t in &toks {
            got_decode = model.forward(&[t])?;
        }
        let d_decode = max_abs_diff(&want_decode, &got_decode);

        let top_ok = argmax(&want_prefill) == argmax(&got_prefill)
            && argmax(&want_decode) == argmax(&got_decode);
        if d_prefill != 0.0 || d_decode != 0.0 || !top_ok {
            failures += 1;
        }
        worst_prefill = worst_prefill.max(d_prefill);
        worst_decode = worst_decode.max(d_decode);

        println!("{len:>6}  {pages:>6}  {d_prefill:>14.3e}  {d_decode:>14.3e}  {:>7}",
                 if top_ok { "ok" } else { "MISMATCH" });
    }

    model.set_paged(true);
    model.reset();

    println!();
    println!("worst prefill difference {worst_prefill:.3e}");
    println!("worst decode  difference {worst_decode:.3e}");
    println!("pages free after reset: {} of {}",
             model.page_pool().free_pages(), model.page_pool().n_pages());

    if failures > 0 {
        anyhow::bail!("{failures} length(s) disagreed with the contiguous cache");
    }
    // Bit-exact is the bar, not "close". Paging moves storage, not arithmetic:
    // the same values are summed in the same order, so any nonzero difference
    // means an offset is wrong somewhere.
    if worst_prefill != 0.0 || worst_decode != 0.0 {
        anyhow::bail!("paged path is not bit-identical to the contiguous path");
    }
    println!();
    println!("Paged and contiguous agree bit-for-bit at every length.");
    Ok(())
}

#[cfg(feature = "cuda")]
fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| ((*x as f64) - (*y as f64)).abs())
        .fold(0.0f64, f64::max)
}

#[cfg(feature = "cuda")]
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

/// GPU name and enforced power limit, recorded with every benchmark.
///
/// This machine's limit is user-switchable between ~55 W and ~175 W and an
/// early measurement in this repo swung 168%% purely from that.
#[cfg(feature = "cuda")]
fn envelope() -> String {
    std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,enforced.power.limit,clocks.max.sm",
            "--format=csv,noheader",
        ])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unavailable".into())
}

#[cfg(feature = "cuda")]
fn gpu_serve_bench(
    dir: PathBuf,
    quant: &str,
    batches: &str,
    lengths: &str,
    steps: usize,
    trials: usize,
) -> Result<()> {
    use llm_engine::gpu_model::{GpuModel, Precision};
    use llm_engine::paged::PAGE_TOKENS;
    use llm_engine::runtime::{Request, Runtime};
    use std::time::Instant;

    let cfg = Config::from_file(dir.join("config.json"))?;
    let weights = Weights::open(dir.join("model.safetensors"))?;
    let precision = Precision::parse(quant)
        .ok_or_else(|| anyhow::anyhow!("unknown precision {quant:?}"))?;

    let sizes: Vec<usize> = batches
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .filter(|v: &usize| *v > 0)
        .collect();
    let lens: Vec<usize> = lengths
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .filter(|v: &usize| *v > 0)
        .collect();
    let max_n = *sizes.iter().max().unwrap_or(&1);

    println!("gpu          {}", envelope());
    println!("page size    {PAGE_TOKENS} tokens");
    println!("prompts      {lens:?} (cycled), {steps} decode steps, {trials} trials");
    println!();

    // --- baseline: today's engine, one request at a time -------------------
    // Legacy contiguous cache with graph replay, which is the fastest
    // single-request configuration the engine has.
    // Three single-request baselines, so a regression can be attributed
    // rather than just reported: graph replay and the GEMV decode path are
    // separate advantages, and the batched runtime gives up both.
    let mut legacy = GpuModel::load_with(cfg.clone(), &weights, cfg.block_size, precision)?;
    let mut baseline = |model: &mut GpuModel, label: &str| -> Result<f64> {
        let mut secs = Vec::new();
        for _ in 0..trials {
            model.reset();
            let prompt = probe_tokens(lens[0], cfg.vocab_size);
            let mut tok = argmax(&model.forward(&prompt)?);
            let t0 = Instant::now();
            for _ in 1..steps {
                tok = argmax(&model.forward(&[tok])?);
            }
            secs.push(t0.elapsed().as_secs_f64());
        }
        secs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let t = secs[secs.len() / 2];
        let tps = (steps - 1) as f64 / t;
        println!("  {label:<44} {tps:>8.1} tok/s");
        Ok(tps)
    };

    println!("single-request baselines");
    legacy.set_paged(false);
    legacy.enable_graph(true);
    let single_tps = baseline(&mut legacy, "contiguous cache + CUDA graph (today's engine)")?;
    legacy.enable_graph(false);
    baseline(&mut legacy, "contiguous cache, eager")?;
    legacy.enable_paging(cfg.block_size.div_ceil(PAGE_TOKENS), 1)?;
    legacy.enable_graph(true);
    baseline(&mut legacy, "paged cache + CUDA graph, single-request path")?;
    drop(legacy);
    println!();

    // --- paged runtime at each batch size ---------------------------------
    // One pool sized for the largest batch, reused across sizes so the memory
    // configuration does not change underneath the comparison.
    let per_req_pages: usize = lens
        .iter()
        .cycle()
        .take(max_n)
        .map(|l| (l + steps).div_ceil(PAGE_TOKENS) + 1)
        .sum();
    let mut model = GpuModel::load_with(cfg.clone(), &weights, cfg.block_size, precision)?;
    model.enable_paging(per_req_pages, max_n)?;
    let total_pages = model.page_pool().n_pages();
    let pool_bytes = model.page_pool().total_bytes();
    let weight_bytes = model.weight_bytes();

    println!("weights      {:.1} MB", weight_bytes as f64 / 1e6);
    println!("page pool    {total_pages} pages, {:.2} MB, {} resident tokens",
             pool_bytes as f64 / 1e6, total_pages * PAGE_TOKENS);
    println!("bytes/page   {} ({} per token across all layers, K and V)",
             model.page_pool().page_bytes() * 2,
             model.page_pool().page_bytes() * 2 / PAGE_TOKENS);
    println!();

    let header = format!("{:>6}  {:>12}  {:>12}  {:>11}  {:>8}  {:>7}  {:>7}",
                         "batch", "aggregate", "per-request", "step ms", "vs 1x", "pages", "wasted");
    println!("{header}");
    println!("{}", "-".repeat(header.len()));

    let mut base_agg = 0.0f64;
    let mut rt = Runtime::new(model)?;
    for &n in &sizes {
        let mut secs = Vec::new();
        let mut pages_used = 0usize;
        let mut wasted = 0usize;
        for _ in 0..trials {
            for i in 0..n {
                rt.submit(Request {
                    id: i as u64,
                    prompt: probe_tokens(lens[i % lens.len()], cfg.vocab_size),
                    max_new_tokens: steps,
                });
            }
            // Admission (which prefills) is one step; time the decode steps
            // only, so prompt processing is not counted as decode throughput.
            rt.step()?;
            let (p, w) = rt.residency();
            pages_used = pages_used.max(p);
            wasted = wasted.max(w);

            let t0 = Instant::now();
            let mut done = 0;
            while !rt.is_idle() {
                rt.step()?;
                done += 1;
                if done > steps * 4 + 16 {
                    anyhow::bail!("runtime did not drain");
                }
            }
            secs.push(t0.elapsed().as_secs_f64());
            let _ = rt.completed();
        }
        secs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let t = secs[secs.len() / 2];
        let decode_steps = (steps - 1) as f64;
        let agg = n as f64 * decode_steps / t;
        let per = decode_steps / t;
        let step_ms = t / decode_steps * 1000.0;
        if n == 1 {
            base_agg = agg;
        }
        println!("{n:>6}  {agg:>10.0} t/s  {per:>10.0} t/s  {step_ms:>9.2}  {:>7.2}x  {pages_used:>7}  {wasted:>7}",
                 if base_agg > 0.0 { agg / base_agg } else { 1.0 });
    }

    println!();
    println!("Aggregate is total tokens per second across the batch; per-request is");
    println!("what one client sees. Batching raises the first and lowers the second:");
    println!("that is a throughput result, not a latency result.");
    Ok(())
}

/// Gap between the top two logits: how close this step came to a tie.
///
/// A greedy decoder is only reproducible across two numerically different
/// implementations when this gap exceeds their disagreement.
#[cfg(feature = "cuda")]
fn top_gap(v: &[f32]) -> f32 {
    let (mut best, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &x in v {
        if x > best {
            second = best;
            best = x;
        } else if x > second {
            second = x;
        }
    }
    best - second
}

#[cfg(feature = "cuda")]
fn gpu_batch(
    dir: PathBuf,
    quant: &str,
    lengths: &str,
    steps: usize,
    max_batch: usize,
) -> Result<()> {
    use llm_engine::gpu_model::{GpuModel, Precision};
    use llm_engine::paged::PAGE_TOKENS;
    use llm_engine::runtime::{Request, Runtime};

    let cfg = Config::from_file(dir.join("config.json"))?;
    let weights = Weights::open(dir.join("model.safetensors"))?;
    let precision = Precision::parse(quant)
        .ok_or_else(|| anyhow::anyhow!("unknown precision {quant:?}"))?;

    let lens: Vec<usize> = lengths
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .filter(|v: &usize| *v > 0)
        .collect();
    if lens.is_empty() {
        anyhow::bail!("no valid prompt lengths");
    }

    // Enough pages for every request at its full final length, plus slack.
    let needed: usize = lens
        .iter()
        .map(|l| (l + steps).div_ceil(PAGE_TOKENS) + 1)
        .sum();

    let prompts: Vec<Vec<usize>> = lens
        .iter()
        .map(|l| probe_tokens(*l, cfg.vocab_size))
        .collect();

    // --- reference ---------------------------------------------------------
    //
    // Two references, because they answer different questions.
    //
    // `solo` runs each request alone through the *same* batched code path
    // (a runtime with max_batch = 1). Any difference between that and a shared
    // batch is contamination between requests, which is the property this test
    // exists to check, and it must be exact.
    //
    // `fwd` runs each request through the single-request `forward` path. That
    // one is not expected to agree bit-for-bit: its lm_head is a GEMV while the
    // batched path uses a GEMM, and the two sum in different orders. Greedy
    // argmax turns a ~1e-4 logit difference into a different token whenever the
    // top two are that close, and one different token diverges the rest. It is
    // reported rather than asserted, with the tie gap, so a rounding flip is
    // not mistaken for a cache bug.
    let mut solo: Vec<Vec<usize>> = Vec::new();
    {
        let mut m = GpuModel::load_with(cfg.clone(), &weights, cfg.block_size, precision)?;
        m.enable_paging(needed, 1)?;
        let mut solo_rt = Runtime::new(m)?;
        for (i, prompt) in prompts.iter().enumerate() {
            solo_rt.submit(Request {
                id: i as u64,
                prompt: prompt.clone(),
                max_new_tokens: steps,
            });
            solo_rt.run_to_completion(steps * 4 + 16)?;
            let mut c = solo_rt.completed();
            c.sort_by_key(|x| x.id);
            solo.push(c.pop().expect("one completion per request").tokens);
        }
    }

    let mut fwd: Vec<Vec<usize>> = Vec::new();
    let mut fwd_gap: Vec<f32> = Vec::new();
    {
        let mut model = GpuModel::load_with(cfg.clone(), &weights, cfg.block_size, precision)?;
        model.enable_paging(cfg.block_size.div_ceil(PAGE_TOKENS), 1)?;
        for prompt in &prompts {
            model.reset();
            let mut out = Vec::new();
            let mut worst_gap = f32::INFINITY;
            let logits = model.forward(prompt)?;
            let mut tok = argmax(&logits);
            worst_gap = worst_gap.min(top_gap(&logits));
            out.push(tok);
            while out.len() < steps {
                let logits = model.forward(&[tok])?;
                tok = argmax(&logits);
                worst_gap = worst_gap.min(top_gap(&logits));
                out.push(tok);
            }
            fwd.push(out);
            fwd_gap.push(worst_gap);
        }
    }
    let reference = solo;

    // --- batched: all requests resident together ---------------------------
    let mut model = GpuModel::load_with(cfg.clone(), &weights, cfg.block_size, precision)?;
    model.enable_paging(needed, max_batch)?;
    let total_pages = model.page_pool().n_pages();
    let pool_mb = model.page_pool().total_bytes() as f64 / 1e6;
    let mut rt = Runtime::new(model)?;

    println!("page size    {PAGE_TOKENS} tokens");
    println!("pool         {total_pages} pages, {pool_mb:.2} MB, {} tokens",
             total_pages * PAGE_TOKENS);
    println!("max batch    {max_batch}");
    println!("requests     {:?} prompt tokens, {steps} generated each", lens);
    println!();

    for (i, prompt) in prompts.iter().enumerate() {
        rt.submit(Request {
            id: i as u64,
            prompt: prompt.clone(),
            max_new_tokens: steps,
        });
    }
    let all_at_once = rt.run_to_completion(steps * 4 + 16)?;
    let mut got = rt.completed();
    got.sort_by_key(|c| c.id);

    let mut failures = 0usize;
    println!("{:>4}  {:>7}  {:>9}  {:>12}", "req", "prompt", "generated", "vs reference");
    println!("{}", "-".repeat(40));
    for c in &got {
        let want = &reference[c.id as usize];
        let ok = &c.tokens == want;
        if !ok {
            failures += 1;
        }
        println!("{:>4}  {:>7}  {:>9}  {:>12}",
                 c.id, c.prompt_len, c.tokens.len(),
                 if ok { "identical" } else { "MISMATCH" });
        if !ok {
            let first = want
                .iter()
                .zip(&c.tokens)
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            println!("        diverged at step {first}: want {:?} got {:?}",
                     &want[first..(first + 3).min(want.len())],
                     &c.tokens[first..(first + 3).min(c.tokens.len())]);
        }
    }
    println!();
    println!("simultaneous admission: {} steps, all pages returned: {}",
             all_at_once.len(),
             rt.free_pages() == total_pages);

    println!();
    println!("against the single-request `forward` path (GEMV lm_head, so exact");
    println!("agreement is not expected -- see the note in the source):");
    println!("{:>4}  {:>12}  {:>18}", "req", "vs forward", "closest top-2 gap");
    for (i, f) in fwd.iter().enumerate() {
        let same = f == &reference[i];
        println!("{:>4}  {:>12}  {:>18.3e}",
                 i, if same { "identical" } else { "diverges" }, fwd_gap[i]);
    }

    // --- staggered: requests enter and leave at different times ------------
    // The scheduler must not assume requests start together, and a request
    // admitted after another has already retired must reuse its pages without
    // seeing any of its KV.
    println!();
    println!("staggered admission");
    let schedule: Vec<(usize, usize)> = prompts
        .iter()
        .enumerate()
        .map(|(i, _)| (i * 3, i))
        .collect();
    for (at, id) in &schedule {
        println!("  t={at}: submit request {id}");
    }

    let mut pending = schedule.clone();
    let mut t = 0usize;
    let mut staggered: Vec<llm_engine::runtime::Completion> = Vec::new();
    let mut peak_active = 0usize;
    while t < steps * 8 {
        while let Some(pos) = pending.iter().position(|(at, _)| *at == t) {
            let (_, id) = pending.remove(pos);
            rt.submit(Request {
                id: id as u64,
                prompt: prompts[id].clone(),
                max_new_tokens: steps,
            });
        }
        if rt.is_idle() && pending.is_empty() {
            break;
        }
        let info = rt.step()?;
        peak_active = peak_active.max(info.active_after);
        staggered.extend(rt.completed());
        t += 1;
    }
    staggered.sort_by_key(|c| c.id);

    println!();
    println!("{:>4}  {:>9}  {:>12}", "req", "generated", "vs reference");
    println!("{}", "-".repeat(30));
    for c in &staggered {
        let want = &reference[c.id as usize];
        let ok = &c.tokens == want;
        if !ok {
            failures += 1;
        }
        println!("{:>4}  {:>9}  {:>12}", c.id, c.tokens.len(),
                 if ok { "identical" } else { "MISMATCH" });
    }

    let (pages, wasted) = rt.residency();
    println!();
    println!("peak active batch     {peak_active}");
    println!("steps taken           {t}");
    println!("pages held after      {pages} (wasted slots {wasted})");
    println!("pages free after      {} of {}", rt.free_pages(), total_pages);

    if staggered.len() != prompts.len() {
        anyhow::bail!("staggered run produced {} completions, expected {}",
                      staggered.len(), prompts.len());
    }
    if rt.free_pages() != total_pages {
        anyhow::bail!("pages leaked: {} of {} free after every request finished",
                      rt.free_pages(), total_pages);
    }
    if failures > 0 {
        anyhow::bail!("{failures} request(s) differed from independent execution");
    }

    println!();
    println!("Every request produced identical output batched and alone,");
    println!("under both simultaneous and staggered admission.");
    Ok(())
}

#[cfg(feature = "cuda")]
fn gpu_eval(dir: PathBuf, data: PathBuf, n_tokens: usize, quant: &str, graph: bool,
            prefill_ctx: usize) -> Result<()> {
    use llm_engine::gpu_model::{GpuModel, Precision};

    let cfg = Config::from_file(dir.join("config.json"))?;
    let weights = Weights::open(dir.join("model.safetensors"))?;

    // val.bin is a flat little-endian uint16 stream, the same format train.py
    // memory-maps. These tokens were held out of training.
    let raw = std::fs::read(&data)
        .with_context(|| format!("reading {}", data.display()))?;
    let all: Vec<usize> = raw
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]) as usize)
        .collect();

    let limit = if prefill_ctx > 0 {
        n_tokens.min(all.len().saturating_sub(1))
    } else {
        n_tokens.min(cfg.block_size).min(all.len() - 1)
    };
    let ids = &all[..limit + 1];
    println!("data        {} ({} tokens evaluated)", data.display(), limit);
    println!();

    let mut results = Vec::new();

    for name in quant.split(',') {
        let name = name.trim();
        let precision = Precision::parse(name)
            .ok_or_else(|| anyhow::anyhow!("unknown precision {name:?}"))?;

        let mut model = GpuModel::load_with(cfg.clone(), &weights, cfg.block_size, precision)?;
        model.enable_graph(graph);

        // Teacher forcing: feed the true token at every step and score the
        // model's prediction of the next one. Feeding its own samples back
        // would measure something else entirely.
        let mut total_nll = 0.0f64;
        let mut scored = 0usize;
        let started = std::time::Instant::now();

        // log softmax, max-subtracted for stability.
        fn nll_of(logits: &[f32], target: usize) -> f64 {
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f64 = logits.iter().map(|v| ((*v - max) as f64).exp()).sum();
            sum_exp.ln() - ((logits[target] - max) as f64)
        }

        if prefill_ctx > 0 {
            // Each scored position gets a fresh prefill of the preceding
            // `prefill_ctx` tokens. Windows are disjoint so no position is
            // counted twice, and the cache is reset between them so a window
            // never inherits state from the previous one.
            let ctx = prefill_ctx.min(cfg.block_size - 1);
            let mut p = ctx;
            while p < limit {
                model.reset();
                let logits = model.forward(&ids[p - ctx..p])?;
                total_nll += nll_of(&logits, ids[p]);
                scored += 1;
                p += ctx;
            }
        } else {
            let mut logits = model.forward(&[ids[0]])?;
            for i in 1..=limit {
                total_nll += nll_of(&logits, ids[i]);
                scored += 1;
                if i < limit {
                    logits = model.forward(&[ids[i]])?;
                }
            }
        }

        let secs = started.elapsed().as_secs_f64();
        let ce = total_nll / scored as f64;
        results.push((name.to_string(), ce, model.weight_bytes(), scored as f64 / secs));

        println!("{name:6}  cross-entropy {ce:.6}   perplexity {:.4}   weights {:.0} MB   {scored} positions in {secs:.1}s",
                 ce.exp(), model.weight_bytes() as f64 / 1e6);
    }

    if results.len() > 1 {
        let (base_name, base_ce, base_bytes, base_tps) = results[0].clone();
        println!();
        println!("relative to {base_name}:");
        for (name, ce, bytes, tps) in results.iter().skip(1) {
            println!("  {name:6}  cross-entropy {:+.6} ({:+.4}%)   weights {:.2}x   speed {:.2}x",
                     ce - base_ce,
                     (ce - base_ce) / base_ce * 100.0,
                     *bytes as f64 / base_bytes as f64,
                     tps / base_tps);
        }
        println!();
        println!("Cross-entropy is the number that decides whether the speed is worth having.");
    }

    Ok(())
}


#[cfg(feature = "cuda")]
fn gpu_profile(dir: PathBuf, quant: &str, iters: usize, warm: usize) -> Result<()> {
    use llm_engine::gpu_model::{GpuModel, Precision};

    let cfg = Config::from_file(dir.join("config.json"))?;
    let weights = Weights::open(dir.join("model.safetensors"))?;
    let precision = Precision::parse(quant)
        .ok_or_else(|| anyhow::anyhow!("unknown precision {quant:?}"))?;

    let mut model = GpuModel::load_with(cfg.clone(), &weights, cfg.block_size, precision)?;

    // Fill the cache first: attention cost grows with sequence length, so
    // profiling against an empty cache would understate it.
    for i in 0..warm.min(cfg.block_size - iters - 1) {
        model.forward(&[(i % cfg.vocab_size).max(1)])?;
    }

    let report = model.profile_step(464, iters)?;
    let raw_total: f64 = report.stages.iter().map(|s| s.raw).sum();
    let adj_total: f64 = report.stages.iter().map(|s| s.adjusted).sum();

    println!("precision {quant}, {iters} steps at position ~{warm}");
    println!("launch+sync overhead ~{:.1} us/block (from the cheapest stage); subtracted",
             report.sync_cost * 1e6);
    println!();
    println!("{:<14} {:>6} {:>10} {:>10} {:>8}",
             "stage", "calls", "raw ms", "adjusted", "share");
    println!("{}", "-".repeat(52));

    let mut sorted: Vec<&llm_engine::gpu_model::Stage> = report.stages.iter().collect();
    sorted.sort_by(|a, b| b.adjusted.partial_cmp(&a.adjusted).unwrap());
    for st in &sorted {
        println!("{:<14} {:>6} {:>10.3} {:>10.3} {:>7.1}%",
                 st.name, st.calls, st.raw * 1000.0, st.adjusted * 1000.0,
                 st.adjusted / adj_total * 100.0);
    }
    println!("{}", "-".repeat(52));
    println!("{:<14} {:>6} {:>10.3} {:>10.3}", "total", "", raw_total * 1000.0, adj_total * 1000.0);
    println!();
    println!("Raw includes one stream sync per timed block, so frequently-called");
    println!("stages absorb the most overhead. Adjusted removes it. Act on the");
    println!("adjusted column.");
    Ok(())
}
