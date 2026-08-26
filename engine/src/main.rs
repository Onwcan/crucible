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
        Command::GpuLogits { model, tokens, top, decode, quant, graph } => {
            gpu_logits(model, &tokens, top, decode, &quant, graph)
        }
        #[cfg(feature = "cuda")]
        #[cfg(feature = "cuda")]
        Command::GpuProfile { model, quant, iters, warm } => gpu_profile(model, &quant, iters, warm),
        #[cfg(feature = "cuda")]
        Command::GpuEval { model, data, tokens, quant, graph } => {
            gpu_eval(model, data, tokens, &quant, graph)
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
fn gpu_logits(dir: PathBuf, tokens: &str, top: usize, decode: usize, quant: &str, graph: bool) -> Result<()> {
    use llm_engine::gpu_model::{GpuModel, Precision};

    let precision = Precision::parse(quant)
        .ok_or_else(|| anyhow::anyhow!("unknown precision {quant:?}; expected f32 or int8"))?;

    let cfg = Config::from_file(dir.join("config.json"))?;
    let weights = Weights::open(dir.join("model.safetensors"))?;

    let ids: Vec<usize> = tokens
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .context("parsing --tokens")?;

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
    println!("prefill     {} tokens in {prefill_ms:.1} ms", ids.len());
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


#[cfg(feature = "cuda")]
fn gpu_eval(dir: PathBuf, data: PathBuf, n_tokens: usize, quant: &str, graph: bool) -> Result<()> {
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

    let limit = n_tokens.min(cfg.block_size).min(all.len() - 1);
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
        let started = std::time::Instant::now();
        let mut logits = model.forward(&[ids[0]])?;

        for i in 1..=limit {
            let target = ids[i];
            // log softmax, max-subtracted for stability.
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f64 = logits.iter().map(|v| ((*v - max) as f64).exp()).sum();
            total_nll += sum_exp.ln() - ((logits[target] - max) as f64);

            if i < limit {
                logits = model.forward(&[target])?;
            }
        }

        let secs = started.elapsed().as_secs_f64();
        let ce = total_nll / limit as f64;
        results.push((name.to_string(), ce, model.weight_bytes(), limit as f64 / secs));

        println!("{name:6}  cross-entropy {ce:.6}   perplexity {:.4}   weights {:.0} MB   {:.0} tok/s",
                 ce.exp(), model.weight_bytes() as f64 / 1e6, limit as f64 / secs);
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
