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
    /// Terminal client for a running Crucible server.
    ///
    /// Speaks HTTP and SSE only. It never loads the model, so the server must
    /// already be running -- start `llm-engine serve` in another terminal.
    #[cfg(feature = "tui")]
    Tui {
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        server: String,
        /// Tokens to request per prompt.
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
    },
    /// Run the HTTP inference service.
    ///
    /// Binds loopback unless `--host` says otherwise: this service has no
    /// authentication, so reaching the network must be a deliberate act.
    #[cfg(feature = "cuda")]
    Serve {
        model: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        #[arg(long, default_value = "int8")]
        quant: String,
        #[arg(long, default_value_t = 16)]
        max_batch: usize,
        /// Requests admitted to the waiting queue before 429.
        #[arg(long, default_value_t = 64)]
        max_queue: usize,
        #[arg(long, default_value_t = 512)]
        max_prompt_tokens: usize,
        #[arg(long, default_value_t = 512)]
        max_new_tokens: usize,
        /// KV pages. Defaults to enough for max_batch full-context sequences.
        #[arg(long)]
        kv_pages: Option<usize>,
        /// Prompt tokens consumed per scheduler step.
        ///
        /// Smaller values interleave prefill with decode more finely, at the
        /// cost of prefill GEMM efficiency. The default is measured; see the
        /// chunk-size table in the README.
        #[arg(long)]
        prefill_chunk_tokens: Option<usize>,
        /// Public model id for the OpenAI-compatible endpoints.
        ///
        /// Published by /v1/models and echoed in every compatibility response.
        /// Set it when serving a checkpoint other than the 120M one, so clients
        /// are not told they are talking to a model that is not loaded.
        #[arg(long, default_value = llm_engine::openai::DEFAULT_MODEL_ID)]
        model_id: String,
    },
    /// What sampling costs, against the greedy fast path, both ways.
    ///
    /// Each sampled mode runs with device top-k and with the full-logit path it
    /// replaces, interleaved within a trial.
    #[cfg(feature = "cuda")]
    GpuSampleBench {
        model: PathBuf,
        #[arg(long, default_value = "int8")]
        quant: String,
        #[arg(long, default_value = "1,4,8,16")]
        batches: String,
        #[arg(long, default_value = "32,128,256,512")]
        lengths: String,
        #[arg(long, default_value_t = 64)]
        steps: usize,
        #[arg(long, default_value_t = 3)]
        trials: usize,
    },
    /// Microbenchmark the top-k extraction kernel against the full-logit path.
    ///
    /// No model needed: this is the selection stage in isolation, on synthetic
    /// logits of the production shape.
    #[cfg(feature = "cuda")]
    GpuTopkBench {
        #[arg(long, default_value = "1,4,8,16")]
        rows: String,
        #[arg(long, default_value = "5,10,20,40,128")]
        top_k: String,
        #[arg(long, default_value_t = 50304)]
        vocab: usize,
        #[arg(long, default_value_t = 200)]
        iters: usize,
        #[arg(long, default_value_t = 5)]
        trials: usize,
    },
    /// Where a prefill's time goes: submission, execution, or the logits copy.
    #[cfg(feature = "cuda")]
    GpuPrefillBench {
        model: PathBuf,
        #[arg(long, default_value = "int8")]
        quant: String,
        #[arg(long, default_value = "1,33,67,134,268,535,941")]
        lengths: String,
        #[arg(long, default_value_t = 30)]
        iters: usize,
    },
    /// Check graph-replayed prefill against issuing every launch eagerly.
    ///
    /// The claim under test is that a captured prefill graph holds nothing
    /// request-specific, so one graph serves any request of that shape.
    #[cfg(feature = "cuda")]
    GpuPrefillGraphCheck {
        model: PathBuf,
        #[arg(long, default_value = "int8")]
        quant: String,
        #[arg(long, default_value = "1,2,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257,511,512,941")]
        lengths: String,
        #[arg(long, default_value = "32,64,128,256,37,73,131")]
        chunks: String,
        #[arg(long, default_value_t = 24)]
        steps: usize,
    },
    /// Check chunked prefill against prefilling each prompt in one piece.
    ///
    /// The claim under test is that where a chunk boundary falls cannot change
    /// what a request generates. Page boundaries are 16 tokens, so boundaries
    /// that align and boundaries that do not are both exercised.
    #[cfg(feature = "cuda")]
    GpuPrefillCheck {
        model: PathBuf,
        #[arg(long, default_value = "int8")]
        quant: String,
        /// Prompt lengths to check. Defaults straddle every page boundary.
        #[arg(long, default_value = "1,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257,511,512")]
        lengths: String,
        /// Chunk sizes to compare against monolithic prefill.
        #[arg(long, default_value = "32,64,128,256")]
        chunks: String,
        #[arg(long, default_value_t = 24)]
        steps: usize,
    },
    /// Check that a request's sampled output does not depend on who it is
    /// batched with, or on which selection path served it.
    ///
    /// The invariant: a request run alone with seed S must produce exactly the
    /// same tokens when run concurrently with unrelated requests. If another
    /// request's presence changes the sequence, the RNG is keyed on the wrong
    /// thing.
    #[cfg(feature = "cuda")]
    GpuSampling {
        model: PathBuf,
        #[arg(long, default_value = "int8")]
        quant: String,
        #[arg(long, default_value_t = 24)]
        steps: usize,
        #[arg(long, default_value_t = 16)]
        max_batch: usize,
    },
    /// Eager versus graph replay on identical inputs, across graph-cache
    /// transitions and slot permutations.
    #[cfg(feature = "cuda")]
    GpuGraphCheck {
        model: PathBuf,
        #[arg(long, default_value = "int8")]
        quant: String,
        /// Active counts to execute in order. Repeats exercise cache hits;
        /// changes exercise capture of a new shape.
        #[arg(long, default_value = "1,4,8,3,16,2,16,1,8,4")]
        shapes: String,
        /// Prompt lengths, cycled. Chosen around page boundaries.
        #[arg(long, default_value = "15,16,17,31,32,33,127,128,129,255,256,511,512,513,700,900")]
        lengths: String,
    },
    /// Batched GEMV against the tiled GEMM, per projection shape and batch.
    ///
    /// Also checks the two agree numerically before reporting any timing: a
    /// faster kernel that computes something else is not a result.
    #[cfg(feature = "cuda")]
    GpuGemvBench {
        #[arg(long, default_value = "1,2,4,8,16")]
        batches: String,
        #[arg(long, default_value_t = 200)]
        iters: usize,
    },
    /// Stage breakdown of a batched decode step, at several batch sizes.
    #[cfg(feature = "cuda")]
    GpuProfileBatch {
        model: PathBuf,
        #[arg(long, default_value = "int8")]
        quant: String,
        #[arg(long, default_value = "1,2,4,8,16")]
        batches: String,
        /// Sequence length every request sits at, so the attention cost is
        /// comparable across batch sizes.
        #[arg(long, default_value_t = 256)]
        context: usize,
        #[arg(long, default_value_t = 40)]
        iters: usize,
    },
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
    /// Check batched decode and the continuous-batching scheduler against
    /// independent single-request execution.
    ///
    /// The only result that matters is that a request's output does not depend
    /// on who it was batched with. Heterogeneous lengths are the point: padding
    /// everything to the longest sequence would pass a same-length test and
    /// still be wrong.
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
        #[cfg(feature = "tui")]
        Command::Tui { server, max_tokens } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(llm_engine::tui::run(server, max_tokens))
        }
        #[cfg(feature = "cuda")]
        Command::Serve {
            model,
            tokenizer,
            host,
            port,
            quant,
            max_batch,
            max_queue,
            max_prompt_tokens,
            max_new_tokens,
            kv_pages,
            model_id,
            prefill_chunk_tokens,
        } => {
            use llm_engine::paged::PAGE_TOKENS;
            use llm_engine::server::{serve, Limits, ServeOptions};

            let cfg = Config::from_file(model.join("config.json"))?;
            let host: std::net::IpAddr = host
                .parse()
                .with_context(|| format!("invalid --host {host:?}"))?;
            let pages = kv_pages
                .unwrap_or_else(|| max_batch * cfg.block_size.div_ceil(PAGE_TOKENS));
            serve(ServeOptions {
                host,
                port,
                model_dir: model,
                tokenizer,
                quant,
                kv_pages: pages,
                model_id,
                prefill_chunk: prefill_chunk_tokens,
                limits: Limits {
                    max_batch,
                    max_queue,
                    max_prompt_tokens,
                    max_new_tokens,
                    context: cfg.block_size,
                },
            })
        }
        #[cfg(feature = "cuda")]
        Command::GpuSampleBench { model, quant, batches, lengths, steps, trials } => {
            gpu_sample_bench(model, &quant, &batches, &lengths, steps, trials)
        }
        #[cfg(feature = "cuda")]
        Command::GpuTopkBench { rows, top_k, vocab, iters, trials } => {
            gpu_topk_bench(&rows, &top_k, vocab, iters, trials)
        }
        #[cfg(feature = "cuda")]
        Command::GpuPrefillBench { model, quant, lengths, iters } => {
            gpu_prefill_bench(model, &quant, &lengths, iters)
        }
        #[cfg(feature = "cuda")]
        Command::GpuPrefillGraphCheck { model, quant, lengths, chunks, steps } => {
            gpu_prefill_graph_check(model, &quant, &lengths, &chunks, steps)
        }
        #[cfg(feature = "cuda")]
        Command::GpuPrefillCheck { model, quant, lengths, chunks, steps } => {
            gpu_prefill_check(model, &quant, &lengths, &chunks, steps)
        }
        #[cfg(feature = "cuda")]
        Command::GpuSampling { model, quant, steps, max_batch } => {
            gpu_sampling(model, &quant, steps, max_batch)
        }
        #[cfg(feature = "cuda")]
        Command::GpuGraphCheck { model, quant, shapes, lengths } => {
            gpu_graph_check(model, &quant, &shapes, &lengths)
        }
        #[cfg(feature = "cuda")]
        Command::GpuGemvBench { batches, iters } => gpu_gemv_bench(&batches, iters),
        #[cfg(feature = "cuda")]
        Command::GpuProfileBatch { model, quant, batches, context, iters } => {
            gpu_profile_batch(model, &quant, &batches, context, iters)
        }
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

// Token selection lives in `llm_engine::sampling`, shared with the runtime.
// It used to be duplicated here, which meant the CLI and the service could
// disagree about the same prompt and seed. Two behaviours changed in the move
// and both are deliberate: greedy ties now resolve to the lowest index, the
// same rule the GPU argmax kernel uses, and a NaN logit no longer panics.

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
    let mut rng = llm_engine::sampling::Rng::new(seed);

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
        let cfg = llm_engine::sampling::GenerationConfig {
            max_tokens,
            temperature,
            top_k,
            seed,
        };
        let next = llm_engine::sampling::sample(&logits, &cfg, &mut rng) as u32;
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

/// Execute a sequence of active counts, returning each step's logits.
///
/// State is rebuilt from scratch at the start, so two calls with the same
/// arguments see identical KV history and their outputs are directly
/// comparable. Slots are permuted every step: the metadata arrays are rebuilt
/// each time, so which request sits in which slot must not matter, and this is
/// what would catch it if a captured graph had tied itself to slot contents.
#[cfg(feature = "cuda")]
fn run_shape_sequence(
    model: &mut llm_engine::gpu_model::GpuModel,
    prompts: &[Vec<usize>],
    shapes: &[usize],
) -> Result<Vec<Vec<f32>>> {
    use llm_engine::paged::SequencePages;

    let stride = model.table_stride();
    let vocab = model.cfg.vocab_size;

    let mut seqs: Vec<SequencePages> = Vec::new();
    for p in prompts {
        let mut sq = SequencePages::new();
        sq.grow(model.page_pool_mut(), p.len())?;
        let table = sq.table_padded(stride);
        model.prefill_request(p, &table, 0)?;
        seqs.push(sq);
    }

    let mut out = Vec::new();
    for (step, &n) in shapes.iter().enumerate() {
        let n = n.min(seqs.len()).min(model.max_batch());
        // Rotate which sequence occupies which slot.
        let order: Vec<usize> = (0..n).map(|i| (i + step) % seqs.len()).collect();

        let mut tables = vec![0i32; model.max_batch() * stride];
        let (mut toks, mut pos, mut lens) = (Vec::new(), Vec::new(), Vec::new());
        for (slot, &si) in order.iter().enumerate() {
            let p = seqs[si].len();
            seqs[si].grow(model.page_pool_mut(), 1)?;
            tables[slot * stride..(slot + 1) * stride]
                .copy_from_slice(&seqs[si].table_padded(stride));
            toks.push((si * 31 + step * 7 + 3) % vocab);
            pos.push(p);
            lens.push((p + 1) as i32);
        }
        // Both paths run the same graph; compare what each returns. The
        // device ids must equal a host scan of the full logits, or generated
        // text would diverge the first time two logits tie.
        let ids = model.decode_batch_tokens(&toks, &pos, &tables, &lens)?;
        let logits = model.decode_batch(&toks, &pos, &tables, &lens)?;
        let n_rows = ids.len();
        for i in 0..n_rows {
            let host = argmax(&logits[i * vocab..(i + 1) * vocab]);
            if host != ids[i] {
                anyhow::bail!(
                    "step {step} row {i}: device argmax {} != host argmax {host}",
                    ids[i]
                );
            }
        }
        out.push(logits);
    }

    for mut sq in seqs {
        sq.release(model.page_pool_mut())?;
    }
    Ok(out)
}

#[cfg(feature = "cuda")]
fn gpu_sample_bench(
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
    use llm_engine::sampling::GenerationConfig;
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
    let pages: usize = (0..max_n)
        .map(|i| (lens[i % lens.len()] + steps + 2).div_ceil(PAGE_TOKENS) + 1)
        .sum();

    let mut model = GpuModel::load_with(cfg.clone(), &weights, cfg.block_size, precision)?;
    model.enable_paging(pages, max_n)?;
    let vocab = cfg.vocab_size;
    let mut rt = Runtime::new(model)?;

    println!("gpu        {}", envelope());
    println!("workload   prompts {lens:?} cycled, {steps} tokens, {trials} trials");
    println!("sampling   temperature 0.8, top-k 40");
    println!();

    // greedy / sampled / half sampled, each sampled mode run both ways --
    // device top-k and the full-logit path it replaces. All five interleaved
    // within a trial, so thermal drift cannot masquerade as a cost difference
    // between them.
    let modes: [(&str, fn(usize) -> bool, bool); 5] = [
        ("greedy", |_| false, true),
        ("sampled", |_| true, false),
        ("sampled+tk", |_| true, true),
        ("mixed", |i| i % 2 == 1, false),
        ("mixed+tk", |i| i % 2 == 1, true),
    ];

    let header = format!("{:>6}  {:>11}  {:>11}  {:>11}  {:>9}  {:>11}  {:>8}",
                         "batch", "mode", "aggregate", "per-request", "step ms",
                         "D2H/step", "vs greedy");
    println!("{header}");
    println!("{}", "-".repeat(header.len()));

    for &n in &sizes {
        let mut greedy_agg = 0.0f64;
        let mut full_agg: std::collections::HashMap<&str, f64> = Default::default();
        for (label, is_sampled, device_topk) in modes {
            rt.model_mut().set_device_topk(device_topk);
            let mut secs = Vec::new();
            let mut d2h_seen = 0usize;
            for _ in 0..trials {
                for i in 0..n {
                    let config = if is_sampled(i) {
                        GenerationConfig {
                            max_tokens: steps,
                            temperature: 0.8,
                            top_k: 40,
                            seed: 1000 + i as u64,
                        }
                    } else {
                        GenerationConfig::greedy(steps)
                    };
                    rt.submit(Request {
                        id: i as u64,
                        prompt: probe_tokens(lens[i % lens.len()], vocab),
                        config,
                    });
                }
                rt.step()?; // admission + prefill, not timed as decode
                let t0 = Instant::now();
                let mut done = 0;
                while !rt.is_idle() {
                    let info = rt.step()?;
                    // The first full-width step is representative; later ones
                    // shrink as requests retire.
                    if done == 0 {
                        d2h_seen = info.d2h_bytes;
                    }
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
            if label == "greedy" {
                greedy_agg = agg;
            }
            if !device_topk {
                full_agg.insert(label, agg);
            }
            // Measured, not derived: the point of the change is this number.
            let against_full = label
                .strip_suffix("+tk")
                .and_then(|base| full_agg.get(base))
                .map(|f| format!("   {:.2}x vs full-logit", agg / f))
                .unwrap_or_default();
            println!("{n:>6}  {label:>11}  {agg:>9.0} t/s  {:>9.0} t/s  {:>9.2}  {d2h_seen:>9} B  {:>7}{}",
                     agg / n as f64,
                     t / decode_steps * 1000.0,
                     if label == "greedy" { "-".to_string() }
                     else { format!("{:.2}x", agg / greedy_agg) },
                     against_full);
        }
        println!();
    }
    rt.model_mut().set_device_topk(true);

    println!("D2H/step is what the first full-width decode step actually copied back.");
    println!("+tk rows take candidates from the device top-k kernel; the rows without");
    println!("it copy a full {vocab}-float logits row per sampled request, which is the");
    println!("path the kernel replaces and which stays available as the reference.");
    Ok(())
}

/// Time the selection stage on its own: kernel, transfer, and the host work
/// each path leaves behind.
///
/// Separate from the end-to-end benchmark because the end-to-end number folds
/// the transformer in with it, and a 2% change in selection cost disappears
/// inside that. This one measures only what changed.
#[cfg(feature = "cuda")]
fn gpu_topk_bench(
    rows: &str,
    top_k: &str,
    vocab: usize,
    iters: usize,
    trials: usize,
) -> Result<()> {
    use llm_engine::gpu::{Gpu, TOPK_MAX};
    use llm_engine::sampling::{self, GenerationConfig, Rng};
    use std::time::Instant;

    let counts: Vec<usize> = rows
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .filter(|v: &usize| *v > 0)
        .collect();
    let ks: Vec<usize> = top_k
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .filter(|v: &usize| *v > 0 && *v <= TOPK_MAX)
        .collect();
    let max_rows = *counts.iter().max().unwrap_or(&1);

    let gpu = Gpu::new(0)?;
    println!("gpu        {}", envelope());
    println!("workload   vocab {vocab}, {iters} iterations, median of {trials}");
    println!();

    // Continuous, in the range a real lm_head produces. Deliberately not the
    // tie-heavy distribution the correctness fuzz uses: exact ties are the
    // interesting case for correctness and a misleading one for timing, since
    // they change how much work the host comparison-based selection does.
    let mut rng = Rng::new(20260903);
    let host: Vec<f32> = (0..max_rows * vocab)
        .map(|_| rng.next_f32() * 22.0 - 11.0)
        .collect();
    let d_logits = gpu.to_device(&host)?;
    let mut d_cv = gpu.alloc(max_rows * TOPK_MAX)?;
    let mut d_ci = gpu.to_device_i32(&vec![-1i32; max_rows * TOPK_MAX])?;

    // Median of `trials`, each an `iters`-long run. Interleaving is not needed
    // here the way it is end-to-end: these are microseconds apart, so drift
    // cannot separate them.
    let median = |mut v: Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };

    // Bring the clocks up before measuring anything. The GPU idles between
    // runs, and the first configuration in the table would otherwise be timed
    // during the boost ramp -- which showed up as one 286 us outlier in a
    // column that is otherwise flat at 23.
    {
        let warm = gpu.to_device_i32(&vec![40i32; max_rows])?;
        for _ in 0..2000 {
            gpu.topk_rows(&d_logits, &warm, &mut d_cv, &mut d_ci, max_rows, vocab)?;
        }
        gpu.sync()?;
    }

    let header = format!("{:>6} {:>6} {:>11} {:>11} {:>11} {:>11} {:>9} {:>9}",
                         "rows", "k", "kernel us", "cand D2H", "full D2H",
                         "host topk", "new us", "speedup");
    println!("{header}");
    println!("{}", "-".repeat(header.len()));

    for &n in &counts {
        for &k in &ks {
            let d_k = gpu.to_device_i32(&vec![k as i32; n])?;

            // Kernel alone.
            gpu.topk_rows(&d_logits, &d_k, &mut d_cv, &mut d_ci, n, vocab)?;
            gpu.sync()?;
            let kernel = median((0..trials).map(|_| {
                let t0 = Instant::now();
                for _ in 0..iters {
                    gpu.topk_rows(&d_logits, &d_k, &mut d_cv, &mut d_ci, n, vocab).unwrap();
                }
                gpu.sync().unwrap();
                t0.elapsed().as_secs_f64() / iters as f64
            }).collect::<Vec<_>>());

            // Candidate transfer: two blocks covering every active row.
            let cand_d2h = median((0..trials).map(|_| {
                let t0 = Instant::now();
                for _ in 0..iters {
                    gpu.to_host_n(&d_cv, n * TOPK_MAX).unwrap();
                    gpu.to_host_i32_n(&d_ci, n * TOPK_MAX).unwrap();
                }
                t0.elapsed().as_secs_f64() / iters as f64
            }).collect::<Vec<_>>());

            // The path this replaces: one full logits row per sampled request.
            let full_d2h = median((0..trials).map(|_| {
                let t0 = Instant::now();
                for _ in 0..iters {
                    for r in 0..n {
                        gpu.to_host_range(&d_logits, r * vocab, vocab).unwrap();
                    }
                }
                t0.elapsed().as_secs_f64() / iters as f64
            }).collect::<Vec<_>>());

            // ...and the host selection it forced, which is the larger half.
            let cfg = GenerationConfig { max_tokens: 1, temperature: 0.8, top_k: k, seed: 1 };
            let host_topk = median((0..trials).map(|_| {
                let iters_h = iters.min(20);
                let t0 = Instant::now();
                for _ in 0..iters_h {
                    for r in 0..n {
                        let row = &host[r * vocab..(r + 1) * vocab];
                        let mut rg = Rng::new(1);
                        std::hint::black_box(sampling::sample(row, &cfg, &mut rg));
                    }
                }
                t0.elapsed().as_secs_f64() / iters_h as f64
            }).collect::<Vec<_>>());

            let new = kernel + cand_d2h;
            let old = full_d2h + host_topk;
            println!("{n:>6} {k:>6} {:>10.1}u {:>10.1}u {:>10.1}u {:>10.1}u {:>8.1}u {:>8.2}x",
                     kernel * 1e6, cand_d2h * 1e6, full_d2h * 1e6, host_topk * 1e6,
                     new * 1e6, old / new);
        }
    }

    // What a greedy-only step pays for the launch being in the graph at all.
    println!();
    let zero = gpu.to_device_i32(&vec![0i32; max_rows])?;
    for &n in &counts {
        let skip = median((0..trials).map(|_| {
            let t0 = Instant::now();
            for _ in 0..iters {
                gpu.topk_rows(&d_logits, &zero, &mut d_cv, &mut d_ci, n, vocab).unwrap();
            }
            gpu.sync().unwrap();
            t0.elapsed().as_secs_f64() / iters as f64
        }).collect::<Vec<_>>());
        println!("rows {n:>2}: {:.2} us for an all-greedy launch (every block exits on row_k)",
                 skip * 1e6);
    }
    println!();
    println!("cand D2H is two blocks of rows*{TOPK_MAX} regardless of k; full D2H is one");
    println!("{vocab}-float row per sampled request. host topk is the selection the");
    println!("full-logit path then has to do, which the kernel removes as well.");
    Ok(())
}




/// Split a prefill into submission cost, device execution and the logits copy.
///
/// The point is to tell two very different problems apart. If eager and graph
/// wall times differ a lot, prefill is launch-bound and graphs fix it. If they
/// agree and both sit well above nothing, the kernels themselves are the cost
/// and no amount of graph work will help.
#[cfg(feature = "cuda")]
fn gpu_prefill_bench(dir: PathBuf, quant: &str, lengths: &str, iters: usize) -> Result<()> {
    use llm_engine::gpu_model::{GpuModel, Precision};
    use llm_engine::paged::PAGE_TOKENS;
    use llm_engine::paged::SequencePages;
    use std::time::Instant;

    let cfg = Config::from_file(dir.join("config.json"))?;
    let weights = Weights::open(dir.join("model.safetensors"))?;
    let precision = Precision::parse(quant)
        .ok_or_else(|| anyhow::anyhow!("unknown precision {quant:?}"))?;
    let lens: Vec<usize> = lengths
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .filter(|v: &usize| *v > 0 && *v < cfg.block_size)
        .collect();
    let max_len = *lens.iter().max().unwrap_or(&1);
    let pages = max_len.div_ceil(PAGE_TOKENS) + 4;

    let mut model = GpuModel::load_with(cfg.clone(), &weights, cfg.block_size, precision)?;
    model.enable_paging(pages, 4)?;

    println!("gpu       {}", envelope());
    println!("workload  median of {iters} iterations, final chunks (logits produced)");
    println!();
    let header = format!("{:>7} {:>11} {:>11} {:>12} {:>11} {:>9}",
                         "tokens", "eager ms", "graph ms", "replay ms", "submit ms", "of total");
    println!("{header}");
    println!("{}", "-".repeat(header.len()));

    for &len in &lens {
        let tokens = probe_tokens(len, cfg.vocab_size);
        let mut seq = SequencePages::new();
        seq.grow(model.page_pool_mut(), len)?;
        let table = seq.table_padded(model.table_stride());

        let time_it = |model: &mut GpuModel, graph: bool| -> Result<f64> {
            model.set_prefill_graph(graph);
            // Warm: first call captures, and a cold clock is not the subject.
            for _ in 0..3 {
                model.prefill_chunk(&tokens, &table, 0, true)?;
            }
            let mut samples = Vec::new();
            for _ in 0..iters {
                let t0 = Instant::now();
                model.prefill_chunk(&tokens, &table, 0, true)?;
                samples.push(t0.elapsed().as_secs_f64());
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            Ok(samples[samples.len() / 2])
        };

        let eager = time_it(&mut model, false)?;
        let graph = time_it(&mut model, true)?;
        // Pure device time for the same sequence: no submission, no copy.
        let replay = model.time_prefill_replay(len, true, iters)?;
        let submit = eager - graph;

        println!("{len:>7} {:>10.3} {:>10.3} {:>11.3} {:>10.3} {:>8.0}%",
                 eager * 1000.0, graph * 1000.0, replay * 1000.0, submit * 1000.0,
                 replay / graph * 100.0);
        seq.release(model.page_pool_mut())?;
    }

    println!();
    println!("replay ms is the GPU executing the same kernels with zero submission");
    println!("cost. 'of total' is how much of a graph-mode prefill that accounts for:");
    println!("the closer to 100%, the less there is left for any launch-side fix.");
    Ok(())
}

/// Graph-replayed prefill against eager prefill, on generated tokens.
///
/// Four independent paths are compared -- eager/graph crossed with
/// monolithic/chunked -- because a defect in graph capture and a defect in
/// chunk handling would each show up in only some of them, and comparing two
/// paths could not tell them apart.
#[cfg(feature = "cuda")]
fn gpu_prefill_graph_check(
    dir: PathBuf,
    quant: &str,
    lengths: &str,
    chunks: &str,
    steps: usize,
) -> Result<()> {
    use llm_engine::gpu_model::{GpuModel, Precision};
    use llm_engine::paged::PAGE_TOKENS;
    use llm_engine::runtime::{Request, Runtime};
    use llm_engine::sampling::GenerationConfig;

    let cfg = Config::from_file(dir.join("config.json"))?;
    let weights = Weights::open(dir.join("model.safetensors"))?;
    let precision = Precision::parse(quant)
        .ok_or_else(|| anyhow::anyhow!("unknown precision {quant:?}"))?;

    let lens: Vec<usize> = lengths
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .filter(|v: &usize| *v > 0 && *v + steps + 2 < cfg.block_size)
        .collect();
    let chunk_sizes: Vec<usize> = chunks
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .filter(|v: &usize| *v > 0)
        .collect();
    let max_len = *lens.iter().max().unwrap_or(&1);
    let pages = (max_len + steps + 2).div_ceil(PAGE_TOKENS) * 8 + 8;

    // graph: replay captured prefill. chunked: consume the prompt in pieces.
    let build = |graph: bool, chunked: bool, chunk: usize| -> Result<Runtime> {
        let mut m = GpuModel::load_with(cfg.clone(), &weights, cfg.block_size, precision)?;
        m.enable_paging(pages, 8)?;
        m.set_prefill_graph(graph);
        let mut rt = Runtime::new(m)?;
        rt.set_chunked_prefill(chunked);
        rt.set_prefill_chunk(chunk.max(1));
        Ok(rt)
    };
    let run = |rt: &mut Runtime, id: u64, prompt: &[usize], config: GenerationConfig|
     -> Result<Vec<usize>> {
        rt.submit(Request { id, prompt: prompt.to_vec(), config });
        rt.run_to_completion(steps * 8 + 64)?;
        Ok(rt.completed().pop().expect("one completion").tokens)
    };

    let configs: [(&str, GenerationConfig); 5] = [
        ("greedy", GenerationConfig::greedy(steps)),
        ("top_k 5", GenerationConfig { max_tokens: steps, temperature: 0.9, top_k: 5, seed: 11 }),
        ("top_k 40", GenerationConfig { max_tokens: steps, temperature: 0.8, top_k: 40, seed: 4242 }),
        ("top_k 128", GenerationConfig { max_tokens: steps, temperature: 0.7, top_k: 128, seed: 7 }),
        ("top_k 500", GenerationConfig { max_tokens: steps, temperature: 0.8, top_k: 500, seed: 99 }),
    ];

    println!("prompts   {lens:?}");
    println!("chunks    {chunk_sizes:?}");
    println!();

    let mut failures = 0usize;

    // --- eager vs graph, monolithic ----------------------------------------
    println!("monolithic prefill: eager vs graph");
    let header = format!("{:>7}  {}", "prompt",
                         configs.iter().map(|(l, _)| format!("{l:>11}"))
                             .collect::<Vec<_>>().join(""));
    println!("{header}");
    println!("{}", "-".repeat(header.len()));
    let mut eager = build(false, false, 1)?;
    let mut graph = build(true, false, 1)?;
    for &len in &lens {
        let prompt = probe_tokens(len, cfg.vocab_size);
        print!("{len:>7}  ");
        for (_, config) in &configs {
            let want = run(&mut eager, 0, &prompt, config.clone())?;
            let got = run(&mut graph, 0, &prompt, config.clone())?;
            let ok = got == want;
            if !ok {
                failures += 1;
            }
            print!("{:>11}", if ok { "same" } else { "DIFFERS" });
        }
        println!();
    }
    let (captured, replays, secs) = graph.model().prefill_graph_stats();
    println!("  {captured} graphs captured in {:.1} ms total, {replays} replays",
             secs * 1000.0);

    // --- the four-way matrix ------------------------------------------------
    println!();
    println!("four paths, greedy and sampled, per chunk size");
    let header = format!("{:>7} {:>6}  {:>12} {:>12} {:>12}",
                         "prompt", "chunk", "mono-graph", "chunk-eager", "chunk-graph");
    println!("{header}");
    println!("{}", "-".repeat(header.len()));
    for &len in &lens {
        if len < 32 {
            continue;
        }
        let prompt = probe_tokens(len, cfg.vocab_size);
        for &c in &chunk_sizes {
            let mut row = Vec::new();
            for (g, ch) in [(true, false), (false, true), (true, true)] {
                let mut rt = build(g, ch, c)?;
                let mut ok = true;
                for (_, config) in &configs {
                    let want = run(&mut eager, 0, &prompt, config.clone())?;
                    let got = run(&mut rt, 0, &prompt, config.clone())?;
                    if got != want {
                        ok = false;
                    }
                }
                if !ok {
                    failures += 1;
                }
                row.push(if ok { "same" } else { "DIFFERS" });
            }
            println!("{len:>7} {c:>6}  {:>12} {:>12} {:>12}", row[0], row[1], row[2]);
        }
    }

    // --- a graph captured for one request must serve another ---------------
    println!();
    println!("request isolation: one graph, different requests");
    let mut rt = build(true, true, 128)?;
    let mut isolation_ok = true;
    for i in 0..6u64 {
        // Same shape every time so the same graph key is reused, but different
        // token ids, different pages and -- through chunking -- different
        // offsets. Anything request-specific baked into the graph shows here.
        let prompt = probe_tokens(300, cfg.vocab_size)
            .iter()
            .map(|t| (t + i as usize * 977) % cfg.vocab_size)
            .collect::<Vec<_>>();
        let config = GenerationConfig { max_tokens: steps, temperature: 0.8, top_k: 40,
                                        seed: 1000 + i };
        let want = run(&mut eager, i, &prompt, config.clone())?;
        let got = run(&mut rt, i, &prompt, config)?;
        if got != want {
            isolation_ok = false;
            failures += 1;
        }
        // Cancel a request mid-prefill between reuses, so the next one inherits
        // a pool that has been released and re-taken.
        rt.submit(Request { id: 500 + i, prompt: prompt.clone(),
                            config: GenerationConfig::greedy(steps) });
        rt.step()?;
        rt.cancel(500 + i)?;
        let _ = rt.completed();
    }
    let (captured, replays, _) = rt.model().prefill_graph_stats();
    println!("  six different requests through the same graphs: {}",
             if isolation_ok { "identical to eager" } else { "MISMATCH" });
    println!("  {captured} graphs served {replays} replays");
    println!("  pages free: {} of {}", rt.free_pages(), rt.model().page_pool().n_pages());
    if rt.free_pages() != rt.model().page_pool().n_pages() {
        anyhow::bail!("pages leaked");
    }

    // --- non-zero offsets must not be captured -----------------------------
    println!();
    println!("non-zero chunk offsets");
    let prompt = probe_tokens(700, cfg.vocab_size);
    let want = run(&mut eager, 0, &prompt, GenerationConfig::greedy(steps))?;
    for c in [64usize, 128, 256] {
        let mut rt = build(true, true, c)?;
        // Every chunk after the first replays the same graph at a different
        // offset, which is only correct if the offset was never captured.
        let got = run(&mut rt, 0, &prompt, GenerationConfig::greedy(steps))?;
        let ok = got == want;
        if !ok {
            failures += 1;
        }
        let (g, r, _) = rt.model().prefill_graph_stats();
        println!("  chunk {c:>4}: {} ({g} graphs, {r} replays for {} chunks)",
                 if ok { "same as eager" } else { "DIFFERS" },
                 prompt.len().div_ceil(c));
    }

    println!();
    if failures > 0 {
        anyhow::bail!("{failures} comparison(s) differed between eager and graph prefill");
    }
    println!("Graph-replayed prefill generates exactly what eager prefill generates,");
    println!("across prompt lengths, chunk sizes, sampling policies and offsets, and a");
    println!("graph captured for one request serves unrelated ones unchanged.");
    Ok(())
}

/// Chunked prefill against monolithic prefill, on generated tokens.
///
/// Comparing prose would prove nothing, so this compares token id sequences:
/// the same prompt, the same seed, the same everything except where the prompt
/// was cut. Greedy and sampled are both checked, because sampling turns a tiny
/// logit difference into a visibly different sequence and is therefore the more
/// sensitive test of the two.
#[cfg(feature = "cuda")]
fn gpu_prefill_check(
    dir: PathBuf,
    quant: &str,
    lengths: &str,
    chunks: &str,
    steps: usize,
) -> Result<()> {
    use llm_engine::gpu_model::{GpuModel, Precision};
    use llm_engine::paged::PAGE_TOKENS;
    use llm_engine::runtime::{Request, Runtime};
    use llm_engine::sampling::GenerationConfig;

    let cfg = Config::from_file(dir.join("config.json"))?;
    let weights = Weights::open(dir.join("model.safetensors"))?;
    let precision = Precision::parse(quant)
        .ok_or_else(|| anyhow::anyhow!("unknown precision {quant:?}"))?;

    let lens: Vec<usize> = lengths
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .filter(|v: &usize| *v > 0 && *v + steps + 2 < cfg.block_size)
        .collect();
    let chunk_sizes: Vec<usize> = chunks
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .filter(|v: &usize| *v > 0)
        .collect();

    let max_len = *lens.iter().max().unwrap_or(&1);
    let pages = (max_len + steps + 2).div_ceil(PAGE_TOKENS) * 8 + 8;

    let build = |chunked: bool, chunk: usize| -> Result<Runtime> {
        let mut m = GpuModel::load_with(cfg.clone(), &weights, cfg.block_size, precision)?;
        m.enable_paging(pages, 8)?;
        let mut rt = Runtime::new(m)?;
        rt.set_chunked_prefill(chunked);
        rt.set_prefill_chunk(chunk);
        Ok(rt)
    };

    // One request at a time, so the only variable is where the prompt was cut.
    let run_one = |rt: &mut Runtime, prompt: &[usize], config: GenerationConfig|
     -> Result<Vec<usize>> {
        rt.submit(Request { id: 0, prompt: prompt.to_vec(), config });
        rt.run_to_completion(steps * 8 + 64)?;
        Ok(rt.completed().pop().expect("one completion").tokens)
    };

    let configs: [(&str, GenerationConfig); 2] = [
        ("greedy", GenerationConfig::greedy(steps)),
        ("sampled", GenerationConfig {
            max_tokens: steps, temperature: 0.8, top_k: 40, seed: 4242,
        }),
    ];

    println!("page size    {PAGE_TOKENS} tokens");
    println!("chunks       {chunk_sizes:?}");
    println!("prompts      {lens:?}");
    println!();

    let mut failures = 0usize;
    let header = format!("{:>7}  {:>9}  {}", "prompt", "mode",
                         chunk_sizes.iter().map(|c| format!("{:>9}", format!("chunk {c}")))
                             .collect::<Vec<_>>().join(""));
    println!("{header}");
    println!("{}", "-".repeat(header.len()));

    let mut mono = build(false, 0)?;
    let mut chunked: Vec<Runtime> = chunk_sizes.iter().map(|c| build(true, *c)).collect::<Result<_>>()?;

    for &len in &lens {
        let prompt = probe_tokens(len, cfg.vocab_size);
        for (label, config) in &configs {
            let want = run_one(&mut mono, &prompt, config.clone())?;
            print!("{len:>7}  {label:>9}  ");
            for rt in chunked.iter_mut() {
                let got = run_one(rt, &prompt, config.clone())?;
                let ok = got == want;
                if !ok {
                    failures += 1;
                }
                print!("{:>9}", if ok { "same" } else { "DIFFERS" });
            }
            println!();
        }
    }

    // Boundary alignment: a chunk size that divides the page size and one that
    // does not, against a prompt length that is neither.
    println!();
    println!("misaligned chunk sizes against a misaligned prompt");
    let odd_prompt = probe_tokens(251, cfg.vocab_size);
    let want = run_one(&mut mono, &odd_prompt, GenerationConfig::greedy(steps))?;
    for c in [7usize, 13, 17, 100, 251, 300] {
        let mut rt = build(true, c)?;
        let got = run_one(&mut rt, &odd_prompt, GenerationConfig::greedy(steps))?;
        let ok = got == want;
        if !ok {
            failures += 1;
        }
        println!("  chunk {c:>4} vs monolithic: {}", if ok { "same" } else { "DIFFERS" });
    }

    // Concurrency: a request's output must not depend on other prompts being
    // prefilled at the same time.
    println!();
    println!("independence from concurrent prefill");
    let mut rt = build(true, 128)?;
    let solo_prompt = probe_tokens(300, cfg.vocab_size);
    let solo_cfg = GenerationConfig { max_tokens: steps, temperature: 0.9, top_k: 20, seed: 77 };
    let solo = run_one(&mut rt, &solo_prompt, solo_cfg.clone())?;

    let mut rt = build(true, 128)?;
    rt.submit(Request { id: 0, prompt: solo_prompt.clone(), config: solo_cfg });
    for i in 1..6u64 {
        rt.submit(Request {
            id: i,
            prompt: probe_tokens(120 * i as usize + 37, cfg.vocab_size),
            config: GenerationConfig::greedy(steps),
        });
    }
    rt.run_to_completion(steps * 16 + 256)?;
    let mut done = rt.completed();
    done.sort_by_key(|c| c.id);
    let together = done.iter().find(|c| c.id == 0).expect("request 0").tokens.clone();
    let ok = together == solo;
    if !ok {
        failures += 1;
    }
    println!("  alone vs prefilled alongside five other prompts: {}",
             if ok { "identical" } else { "MISMATCH" });
    println!("  pages free: {} of {}", rt.free_pages(), rt.model().page_pool().n_pages());
    if rt.free_pages() != rt.model().page_pool().n_pages() {
        anyhow::bail!("pages leaked");
    }

    // Cancellation at four points in a long prompt's prefill.
    println!();
    println!("cancellation during prefill");
    let long = probe_tokens(600, cfg.vocab_size);
    for (label, chunks_before) in [("before the first chunk", 0usize),
                                   ("after one chunk", 1),
                                   ("midway", 3),
                                   ("one chunk before the end", 4)] {
        let mut rt = build(true, 128)?;
        let total = rt.model().page_pool().n_pages();
        rt.submit(Request { id: 900, prompt: long.clone(), config: GenerationConfig::greedy(steps) });
        for _ in 0..chunks_before {
            rt.step()?;
        }
        let was_prefilling = rt.prefilling_len();
        rt.cancel(900)?;
        let _ = rt.completed();
        // The pages must come back, and the runtime must still work afterwards.
        let freed = rt.free_pages() == total;
        rt.submit(Request { id: 901, prompt: long.clone(), config: GenerationConfig::greedy(steps) });
        rt.run_to_completion(steps * 16 + 256)?;
        let reused = rt.completed().pop().map(|c| c.tokens.len()) == Some(steps);
        let clean = rt.free_pages() == total;
        if !(freed && reused && clean) {
            failures += 1;
        }
        println!("  {label:<26} prefilling {was_prefilling}, pages freed {freed}, \
                  reuse {reused}, clean {clean}");
    }

    println!();
    if failures > 0 {
        anyhow::bail!("{failures} comparison(s) differed between chunked and monolithic prefill");
    }
    println!("Chunked prefill generates exactly what monolithic prefill generates,");
    println!("at every prompt length, chunk size and alignment tested, and a");
    println!("request cancelled mid-prompt returns every page it held.");
    Ok(())
}

#[cfg(feature = "cuda")]
fn gpu_sampling(dir: PathBuf, quant: &str, steps: usize, max_batch: usize) -> Result<()> {
    use llm_engine::gpu::TOPK_MAX;
    use llm_engine::gpu_model::{GpuModel, Precision};
    use llm_engine::paged::PAGE_TOKENS;
    use llm_engine::runtime::{Completion, Request, Runtime};
    use llm_engine::sampling::GenerationConfig;

    let cfg = Config::from_file(dir.join("config.json"))?;
    let weights = Weights::open(dir.join("model.safetensors"))?;
    let precision = Precision::parse(quant)
        .ok_or_else(|| anyhow::anyhow!("unknown precision {quant:?}"))?;

    // Deliberately heterogeneous in every dimension at once: prompt length,
    // decoding policy, temperature, top-k and seed. A bug that only shows up
    // when two sampled requests share a batch would survive a uniform test.
    //
    // The last request asks for more candidates than the device kernel holds,
    // so it takes the full-logit path while its neighbours take the device one.
    // That mixture inside a single batch is the case a per-batch dispatch would
    // get wrong.
    let specs: Vec<(usize, GenerationConfig)> = vec![
        (15, GenerationConfig { max_tokens: steps, temperature: 0.0, top_k: 40, seed: 1 }),
        (16, GenerationConfig { max_tokens: steps, temperature: 0.7, top_k: 40, seed: 11 }),
        (17, GenerationConfig { max_tokens: steps, temperature: 1.0, top_k: 5, seed: 22 }),
        (63, GenerationConfig { max_tokens: steps, temperature: 0.0, top_k: 40, seed: 2 }),
        (129, GenerationConfig { max_tokens: steps, temperature: 0.3, top_k: 20, seed: 33 }),
        (511, GenerationConfig { max_tokens: steps, temperature: 0.9, top_k: 10, seed: 44 }),
        (33, GenerationConfig { max_tokens: steps, temperature: 0.8, top_k: 500, seed: 55 }),
    ];
    let prompts: Vec<Vec<usize>> = specs
        .iter()
        .map(|(len, _)| probe_tokens(*len, cfg.vocab_size))
        .collect();

    let pages: usize = specs
        .iter()
        .map(|(len, _)| (len + steps + 2).div_ceil(PAGE_TOKENS) + 1)
        .sum();

    let load = |mb: usize, pg: usize, device_topk: bool| -> Result<Runtime> {
        let mut m = GpuModel::load_with(cfg.clone(), &weights, cfg.block_size, precision)?;
        m.enable_paging(pg, mb)?;
        m.set_device_topk(device_topk);
        Runtime::new(m)
    };

    let run_alone = |device_topk: bool| -> Result<Vec<Vec<usize>>> {
        let mut out = Vec::new();
        let mut rt = load(1, pages, device_topk)?;
        for (i, prompt) in prompts.iter().enumerate() {
            rt.submit(Request {
                id: i as u64,
                prompt: prompt.clone(),
                config: specs[i].1.clone(),
            });
            rt.run_to_completion(steps * 4 + 16)?;
            let mut c = rt.completed();
            c.sort_by_key(|x| x.id);
            out.push(c.pop().expect("one completion").tokens);
        }
        Ok(out)
    };

    println!("steps        {steps} tokens per request");
    println!("pool         {pages} pages");
    println!("capacity     device top-k holds {} candidates per row", TOPK_MAX);
    println!();

    let mut failures = 0usize;

    // --- the two selection paths must generate the same text ----------------
    //
    // This is the claim the whole optimisation rests on. The device kernel and
    // the host reference find the same candidates in the same order, so a
    // request cannot tell which one served it.
    let alone = run_alone(true)?;
    let alone_full = run_alone(false)?;
    println!("device top-k against the full-logit reference, each request alone");
    for i in 0..specs.len() {
        let ok = alone[i] == alone_full[i];
        if !ok {
            failures += 1;
        }
        println!("  request {i}: {}",
                 if ok { "identical tokens" } else { "MISMATCH" });
    }

    // Everything below is checked on both paths against the same reference, so
    // the fallback is held to the isolation guarantee too.
    for device_topk in [true, false] {
        let path = if device_topk { "device top-k" } else { "full logits" };
        println!();
        println!("=== {path} ===");
        let mut rt = load(max_batch, pages, device_topk)?;

        // --- all together ---------------------------------------------------
        for (i, prompt) in prompts.iter().enumerate() {
            rt.submit(Request {
                id: i as u64,
                prompt: prompt.clone(),
                config: specs[i].1.clone(),
            });
        }
        rt.run_to_completion(steps * 8 + 32)?;
        let mut together = rt.completed();
        together.sort_by_key(|c| c.id);

        println!("{:>4}  {:>7}  {:>12}  {:>6}  {:>5}  {:>14}",
                 "req", "prompt", "mode", "top-k", "seed", "vs alone");
        println!("{}", "-".repeat(60));
        for c in &together {
            let (len, gc) = &specs[c.id as usize];
            let want = &alone[c.id as usize];
            let ok = &c.tokens == want;
            if !ok {
                failures += 1;
            }
            let mode = if gc.is_greedy() {
                "greedy".to_string()
            } else {
                format!("temp {:.1}", gc.temperature)
            };
            println!("{:>4}  {len:>7}  {mode:>12}  {:>6}  {:>5}  {:>14}",
                     c.id, gc.top_k, gc.seed, if ok { "identical" } else { "MISMATCH" });
        }

        // --- staggered admission, forcing slot reordering -------------------
        println!();
        println!("staggered admission (forces swap_remove reordering)");
        let mut staggered: Vec<Completion> = Vec::new();
        let mut pending: Vec<(usize, usize)> =
            (0..prompts.len()).map(|i| (i * 3, i)).collect();
        let mut t = 0usize;
        while t < steps * 12 {
            while let Some(pos) = pending.iter().position(|(at, _)| *at == t) {
                let (_, i) = pending.remove(pos);
                rt.submit(Request {
                    id: i as u64,
                    prompt: prompts[i].clone(),
                    config: specs[i].1.clone(),
                });
            }
            if rt.is_idle() && pending.is_empty() {
                break;
            }
            rt.step()?;
            staggered.extend(rt.completed());
            t += 1;
        }
        staggered.sort_by_key(|c| c.id);
        for c in &staggered {
            let want = &alone[c.id as usize];
            let ok = &c.tokens == want;
            if !ok {
                failures += 1;
            }
            println!("{:>4}  {:>14}", c.id, if ok { "identical" } else { "MISMATCH" });
        }

        // --- cancellation must not disturb a later identical request --------
        println!();
        println!("cancellation does not perturb a later request with the same seed");
        let victim = 1usize;
        rt.submit(Request {
            id: 900,
            prompt: prompts[victim].clone(),
            config: specs[victim].1.clone(),
        });
        rt.step()?;
        rt.step()?;
        rt.cancel(900)?;
        let _ = rt.completed();
        rt.submit(Request {
            id: 901,
            prompt: prompts[victim].clone(),
            config: specs[victim].1.clone(),
        });
        rt.run_to_completion(steps * 4 + 16)?;
        let after = rt.completed().pop().expect("one completion").tokens;
        let ok = after == alone[victim];
        if !ok {
            failures += 1;
        }
        println!("  request after a cancelled twin: {}",
                 if ok { "identical to running alone" } else { "MISMATCH" });

        println!();
        println!("pages free: {} of {}", rt.free_pages(), rt.model().page_pool().n_pages());
        if rt.free_pages() != rt.model().page_pool().n_pages() {
            anyhow::bail!("pages leaked on the {path} path");
        }
    }

    if failures > 0 {
        anyhow::bail!("{failures} request(s) changed output when batched");
    }
    println!();
    println!("Every request produced identical tokens alone and batched, under");
    println!("simultaneous and staggered admission and across a cancellation, on");
    println!("both the device top-k path and the full-logit path it replaces.");
    Ok(())
}

#[cfg(feature = "cuda")]
fn gpu_graph_check(dir: PathBuf, quant: &str, shapes: &str, lengths: &str) -> Result<()> {
    use llm_engine::gpu_model::{GpuModel, Precision};
    use llm_engine::paged::PAGE_TOKENS;

    let cfg = Config::from_file(dir.join("config.json"))?;
    let weights = Weights::open(dir.join("model.safetensors"))?;
    let precision = Precision::parse(quant)
        .ok_or_else(|| anyhow::anyhow!("unknown precision {quant:?}"))?;

    let shape_list: Vec<usize> = shapes
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .filter(|v: &usize| *v > 0)
        .collect();
    let lens: Vec<usize> = lengths
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .filter(|v: &usize| *v > 0)
        .collect();
    let max_batch = *shape_list.iter().max().unwrap_or(&1);
    let prompts: Vec<Vec<usize>> = lens
        .iter()
        .take(max_batch)
        .map(|l| probe_tokens(*l, cfg.vocab_size))
        .collect();

    let steps = shape_list.len();
    let pages: usize = prompts
        .iter()
        .map(|p| (p.len() + steps + 2).div_ceil(PAGE_TOKENS) + 1)
        .sum();

    let mut model = GpuModel::load_with(cfg.clone(), &weights, cfg.block_size, precision)?;
    model.enable_paging(pages, max_batch)?;

    println!("shapes     {shape_list:?}");
    println!("prompts    {:?}", prompts.iter().map(|p| p.len()).collect::<Vec<_>>());
    println!("pool       {pages} pages, {:.2} MB",
             model.page_pool().total_bytes() as f64 / 1e6);
    println!();

    model.set_batch_graph(false);
    let eager = run_shape_sequence(&mut model, &prompts, &shape_list)?;
    let free_after_eager = model.page_pool().free_pages();

    model.set_batch_graph(true);
    let graphed = run_shape_sequence(&mut model, &prompts, &shape_list)?;
    let free_after_graph = model.page_pool().free_pages();

    println!("{:>5}  {:>6}  {:>14}  {:>10}", "step", "batch", "max abs diff", "verdict");
    println!("{}", "-".repeat(42));
    let mut failures = 0usize;
    for (i, (a, b)) in eager.iter().zip(&graphed).enumerate() {
        let diff = a
            .iter()
            .zip(b)
            .map(|(x, y)| ((*x as f64) - (*y as f64)).abs())
            .fold(0.0f64, f64::max);
        let ok = diff == 0.0 && a.len() == b.len();
        if !ok {
            failures += 1;
        }
        println!("{:>5}  {:>6}  {diff:>14.3e}  {:>10}",
                 i, shape_list[i], if ok { "identical" } else { "MISMATCH" });
    }

    println!();
    println!("graphs captured        {}", model.graphs_captured());
    println!("capture time total     {:.1} ms", model.graph_capture_secs() * 1e3);
    println!("kernels per step       {}", model.batch_step_kernels());
    println!("pages free, eager pass {free_after_eager}");
    println!("pages free, graph pass {free_after_graph}");

    if failures > 0 {
        anyhow::bail!("{failures} step(s) differed between eager and graph replay");
    }
    if free_after_eager != free_after_graph {
        anyhow::bail!("page accounting differed between eager and graph passes");
    }
    println!();
    println!("Graph replay is bit-identical to eager execution at every shape,");
    println!("across cache transitions and slot permutations.");
    Ok(())
}

#[cfg(feature = "cuda")]
fn gpu_gemv_bench(batches: &str, iters: usize) -> Result<()> {
    use llm_engine::gpu::{Gpu, Proj2};
    use std::time::Instant;

    let sizes: Vec<usize> = batches
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .filter(|v: &usize| *v > 0 && *v <= 16)
        .collect();

    // The model's actual decode shapes. There is no fused QKV here: q is
    // 768x768 and k/v are 192x768 each, because n_kv_head=3 with head_dim=64.
    let shapes: [(&str, usize, usize); 6] = [
        ("q_proj / o_proj", 768, 768),
        ("k_proj / v_proj", 192, 768),
        ("gate / up_proj", 2048, 768),
        ("down_proj", 768, 2048),
        ("lm_head", 50304, 768),
        ("lm_head (f32 cols)", 50304, 768),
    ];

    let gpu = Gpu::new(0)?;
    println!("gpu      {}", envelope());
    println!("workload int8 weights, {iters} iterations, median of 3");
    println!();

    let header = format!("{:<20} {:>6} {:>10} {:>10} {:>9} {:>10}",
                         "shape", "batch", "gemm us", "gemv us", "speedup", "max reldiff");
    println!("{header}");
    println!("{}", "-".repeat(header.len()));

    for (name, rows, cols) in shapes.iter().take(5) {
        let (rows, cols) = (*rows, *cols);
        // Deterministic, mantissa-dense, single-sign -- the properties the
        // GEMM validation had to learn the hard way.
        let wf: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) * 0.7391).sin() * 0.4 + 0.6)
            .collect();
        let mut qw = vec![0i8; rows * cols];
        let mut qs = vec![0.0f32; rows];
        for r in 0..rows {
            let row = &wf[r * cols..(r + 1) * cols];
            let amax = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            qs[r] = scale;
            for (j, v) in row.iter().enumerate() {
                qw[r * cols + j] = (v / scale).round().clamp(-127.0, 127.0) as i8;
            }
        }
        let d_w = gpu.to_device_i8(&qw)?;
        let d_s = gpu.to_device(&qs)?;

        for &n in &sizes {
            let x: Vec<f32> = (0..n * cols)
                .map(|i| ((i as f32) * 1.2113).cos() * 0.3 + 0.5)
                .collect();
            let d_x = gpu.to_device(&x)?;
            let mut d_gemm = gpu.alloc(n * rows)?;
            let mut d_gemv = gpu.alloc(n * rows)?;

            gpu.gemm(&Proj2::Int8(&d_w, &d_s), &d_x, &mut d_gemm, n, rows, cols, false)?;
            gpu.gemv_batch_i8(&d_w, &d_s, &d_x, &mut d_gemv, rows, cols, n, false)?;
            gpu.sync()?;
            let a = gpu.to_host(&d_gemm)?;
            let b = gpu.to_host(&d_gemv)?;
            let reldiff = a
                .iter()
                .zip(&b)
                .map(|(p, q)| {
                    let scale = (p.abs().max(q.abs()) as f64).max(1e-6);
                    ((*p as f64) - (*q as f64)).abs() / scale
                })
                .fold(0.0f64, f64::max);

            let mut timings = [0.0f64; 2];
            for (slot, gemm_path) in [(0usize, true), (1usize, false)] {
                let mut runs = Vec::new();
                for _ in 0..3 {
                    // Warm, then time.
                    for _ in 0..5 {
                        if gemm_path {
                            gpu.gemm(&Proj2::Int8(&d_w, &d_s), &d_x, &mut d_gemm,
                                     n, rows, cols, false)?;
                        } else {
                            gpu.gemv_batch_i8(&d_w, &d_s, &d_x, &mut d_gemv,
                                              rows, cols, n, false)?;
                        }
                    }
                    gpu.sync()?;
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        if gemm_path {
                            gpu.gemm(&Proj2::Int8(&d_w, &d_s), &d_x, &mut d_gemm,
                                     n, rows, cols, false)?;
                        } else {
                            gpu.gemv_batch_i8(&d_w, &d_s, &d_x, &mut d_gemv,
                                              rows, cols, n, false)?;
                        }
                    }
                    gpu.sync()?;
                    runs.push(t0.elapsed().as_secs_f64() / iters as f64);
                }
                runs.sort_by(|p, q| p.partial_cmp(q).unwrap());
                timings[slot] = runs[1];
            }

            println!("{name:<20} {n:>6} {:>10.1} {:>10.1} {:>8.2}x {:>10.3e}",
                     timings[0] * 1e6, timings[1] * 1e6,
                     timings[0] / timings[1], reldiff);
        }
        println!();
    }

    println!("GEMV keeps one warp per output row and carries the batch inside the");
    println!("warp, so weight traffic does not grow with batch; GEMM amortises the");
    println!("weight read but has almost no parallelism at these M.");
    Ok(())
}

#[cfg(feature = "cuda")]
fn gpu_profile_batch(
    dir: PathBuf,
    quant: &str,
    batches: &str,
    context: usize,
    iters: usize,
) -> Result<()> {
    use llm_engine::gpu_model::{GpuModel, Precision};
    use llm_engine::paged::{SequencePages, PAGE_TOKENS};

    let cfg = Config::from_file(dir.join("config.json"))?;
    let weights = Weights::open(dir.join("model.safetensors"))?;
    let precision = Precision::parse(quant)
        .ok_or_else(|| anyhow::anyhow!("unknown precision {quant:?}"))?;

    let sizes: Vec<usize> = batches
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .filter(|v: &usize| *v > 0)
        .collect();
    let max_n = *sizes.iter().max().unwrap_or(&1);
    let pages_each = context.div_ceil(PAGE_TOKENS) + 1;

    let mut model = GpuModel::load_with(cfg.clone(), &weights, cfg.block_size, precision)?;
    model.enable_paging(pages_each * max_n, max_n)?;

    println!("gpu        {}", envelope());
    println!("context    {context} positions per request, {iters} iterations");
    println!();

    for &n in &sizes {
        // Give every request its own pages at the same length, so the batch
        // dimension is the only thing changing between rows.
        let mut seqs: Vec<SequencePages> = Vec::new();
        for _ in 0..n {
            let mut sq = SequencePages::new();
            sq.grow(model.page_pool_mut(), context)?;
            seqs.push(sq);
        }
        let stride = model.table_stride();
        let mut tables = vec![0i32; model.max_batch() * stride];
        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        let mut lens = Vec::new();
        for (i, sq) in seqs.iter().enumerate() {
            tables[i * stride..(i + 1) * stride].copy_from_slice(&sq.table_padded(stride));
            tokens.push((i * 37 + 11) % cfg.vocab_size);
            positions.push(context - 1);
            lens.push(context as i32);
        }

        let rep = model.profile_batch(&tokens, &positions, &tables, &lens, iters)?;
        let total: f64 = rep.stages.iter().map(|s| s.adjusted).sum();
        let raw_total: f64 = rep.stages.iter().map(|s| s.raw).sum();

        println!("batch {n}   raw step {:.3} ms, adjusted {:.3} ms, sync {:.1} us/call",
                 raw_total * 1e3, total * 1e3, rep.sync_cost * 1e6);
        println!("  {:<14} {:>6} {:>10} {:>10} {:>7}",
                 "stage", "calls", "raw ms", "adj ms", "% adj");
        let mut ranked: Vec<_> = rep.stages.iter().collect();
        ranked.sort_by(|a, b| b.adjusted.partial_cmp(&a.adjusted).unwrap());
        for st in ranked {
            println!("  {:<14} {:>6} {:>10.3} {:>10.3} {:>6.1}%",
                     st.name, st.calls, st.raw * 1e3, st.adjusted * 1e3,
                     if total > 0.0 { st.adjusted / total * 100.0 } else { 0.0 });
        }
        println!();

        for mut sq in seqs {
            sq.release(model.page_pool_mut())?;
        }
    }

    println!("Adjusted removes one sync per timed block, estimated from the");
    println!("cheapest kernel-launching stage. Raw is what the wall clock saw.");
    Ok(())
}

/// GPU memory currently in use, in MB, for reporting graph storage overhead.
#[cfg(feature = "cuda")]
fn gpu_mem_used_mb() -> Option<i64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    String::from_utf8(out.stdout).ok()?.trim().parse().ok()
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

    let header = format!("{:>6}  {:>11}  {:>11}  {:>9}  {:>9}  {:>8}  {:>7}  {:>10}",
                         "batch", "logits agg", "argmax agg", "argmax/req",
                         "step ms", "speedup", "spread", "D2H/step");
    println!("{header}");
    println!("{}", "-".repeat(header.len()));

    let mut rt = Runtime::new(model)?;
    let mem_before = gpu_mem_used_mb();
    let mut step_ms: Vec<f64> = Vec::new();
    for &n in &sizes {
        // Full-logit and device-argmax trials are interleaved, not run in
        // blocks: measuring one path fully and then the other lets thermal
        // drift masquerade as a real difference, which this repo has been
        // caught by once. Graph replay is on for both, so the only variable is
        // what crosses PCIe.
        //
        // A warm-up round runs first at each size so the graph for that shape
        // is already captured; capture cost is reported separately rather than
        // charged to steady-state throughput.
        let mut secs = [Vec::new(), Vec::new()];
        let mut pages_used = 0usize;
        let mut wasted = 0usize;
        for warm in 0..(trials + 1) {
            for (slot, dev_argmax) in [(0usize, false), (1usize, true)] {
                rt.model_mut().set_device_argmax(dev_argmax);
                for i in 0..n {
                    rt.submit(Request {
                        id: i as u64,
                        prompt: probe_tokens(lens[i % lens.len()], cfg.vocab_size),
                        config: llm_engine::sampling::GenerationConfig::greedy(steps),
                    });
                }
                // Admission prefills; time only the decode steps, so prompt
                // processing is not counted as decode throughput.
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
                if warm > 0 {
                    secs[slot].push(t0.elapsed().as_secs_f64());
                }
                let _ = rt.completed();
            }
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let spread = |v: &Vec<f64>| {
            let (lo, hi) = (v.iter().cloned().fold(f64::INFINITY, f64::min),
                            v.iter().cloned().fold(0.0f64, f64::max));
            (hi - lo) / v[v.len() / 2] * 100.0
        };
        let sp = spread(&secs[0]).max(spread(&secs[1]));
        let t_logits = med(&mut secs[0]);
        let t_argmax = med(&mut secs[1]);
        let decode_steps = (steps - 1) as f64;
        let agg_logits = n as f64 * decode_steps / t_logits;
        let agg_argmax = n as f64 * decode_steps / t_argmax;
        let per_argmax = decode_steps / t_argmax;
        let step = t_argmax / decode_steps * 1000.0;
        let bytes = rt.model().d2h_bytes(n, true);
        println!("{n:>6}  {agg_logits:>9.0} t/s  {agg_argmax:>9.0} t/s  {per_argmax:>7.0} t/s  {step:>9.2}  {:>7.2}x  {sp:>6.1}%  {bytes:>8} B",
                 agg_argmax / agg_logits);
        step_ms.push(step);
        let _ = (pages_used, wasted);
    }
    rt.model_mut().set_device_argmax(true);
    let mem_after = gpu_mem_used_mb();

    // Host-side cost per step, which graph replay does not touch: the
    // scheduler reads a full logits row per request and scans it for argmax.
    // Worth measuring rather than assuming, because at 50304 entries per
    // request it is not obviously small next to a 0.67 ms step.
    {
        let vocab = cfg.vocab_size;
        let probe: Vec<f32> = (0..vocab).map(|i| ((i as f32) * 0.7391).sin()).collect();
        let t0 = Instant::now();
        let mut sink = 0usize;
        for _ in 0..1000 {
            sink ^= argmax(&probe);
        }
        let per = t0.elapsed().as_secs_f64() / 1000.0;
        std::hint::black_box(sink);
        println!();
        println!("host argmax          {:.3} ms per request-step", per * 1e3);
        for &n in &sizes {
            println!("  batch {n:<2}           {:.3} ms per step", per * n as f64 * 1e3);
        }
    }

    // Pure replay: kernels only, no upload, no copy-back, no host work. The
    // gap to the measured step is everything graph replay cannot remove.
    println!();
    println!("{:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>9}",
             "batch", "replay ms", "ids D2H", "logits D2H", "host/sched", "step ms");
    for (i, &n) in sizes.iter().enumerate() {
        let replay = rt.model_mut().time_graph_replay(n, 200).unwrap_or(0.0);
        let d_ids = rt.model_mut().time_d2h(n, true, 200).unwrap_or(0.0);
        let d_log = rt.model_mut().time_d2h(n, false, 50).unwrap_or(0.0);
        let step = step_ms[i] / 1e3;
        println!("{n:>6}  {:>10.3}  {:>10.3}  {:>10.3}  {:>10.3}  {:>9.3}",
                 replay * 1e3, d_ids * 1e3, d_log * 1e3,
                 (step - replay - d_ids).max(0.0) * 1e3, step * 1e3);
    }
    println!("(logits D2H is what the old path paid; it is not in the step total)");

    println!();
    println!("graphs captured      {}", rt.model().graphs_captured());
    println!("capture time total   {:.1} ms ({:.1} ms per shape)",
             rt.model().graph_capture_secs() * 1e3,
             rt.model().graph_capture_secs() * 1e3
                 / rt.model().graphs_captured().max(1) as f64);
    println!("kernels per step     {}", rt.model().batch_step_kernels());
    println!();
    println!("{:>6}  {:>14}  {:>14}  {:>10}", "batch", "logits D2H", "ids D2H", "reduction");
    for &n in &sizes {
        let a = rt.model().d2h_bytes(n, false);
        let b = rt.model().d2h_bytes(n, true);
        println!("{n:>6}  {:>12} B  {:>12} B  {:>9.0}x", a, b, a as f64 / b as f64);
    }
    match (mem_before, mem_after) {
        (Some(a), Some(b)) => println!("VRAM used            {a} MB -> {b} MB (graph storage {} MB)", b - a),
        _ => println!("VRAM used            unavailable"),
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
                config: llm_engine::sampling::GenerationConfig::greedy(steps),
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
            config: llm_engine::sampling::GenerationConfig::greedy(steps),
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
                config: llm_engine::sampling::GenerationConfig::greedy(steps),
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

    // --- dispatch numerics -------------------------------------------------
    // GEMV and GEMM sum in different orders, so bit-identical logits are not
    // mathematically available. What matters is that the difference stays at
    // rounding level and never changes a decision that was not already a
    // near-tie.
    {
        use llm_engine::paged::SequencePages;
        let mut m = GpuModel::load_with(cfg.clone(), &weights, cfg.block_size, precision)?;
        m.enable_paging(needed, prompts.len())?;
        let n = prompts.len().min(m.max_batch());

        let mut seqs = Vec::new();
        for p in prompts.iter().take(n) {
            let mut sq = SequencePages::new();
            sq.grow(m.page_pool_mut(), p.len())?;
            seqs.push(sq);
        }
        let stride = m.table_stride();
        let mut tables = vec![0i32; m.max_batch() * stride];
        let (mut toks, mut pos, mut lens2) = (Vec::new(), Vec::new(), Vec::new());
        for (i, sq) in seqs.iter().enumerate() {
            tables[i * stride..(i + 1) * stride].copy_from_slice(&sq.table_padded(stride));
            toks.push(prompts[i][0]);
            pos.push(sq.len() - 1);
            lens2.push(sq.len() as i32);
        }

        m.set_force_decode_gemm(true);
        let a = m.decode_batch(&toks, &pos, &tables, &lens2)?;
        m.set_force_decode_gemm(false);
        let b = m.decode_batch(&toks, &pos, &tables, &lens2)?;

        let vocab = cfg.vocab_size;
        let mut max_abs = 0.0f64;
        let mut max_rel = 0.0f64;
        let mut top1_agree = 0usize;
        let mut worst_margin = f32::INFINITY;
        for i in 0..n {
            let (ra, rb) = (&a[i * vocab..(i + 1) * vocab], &b[i * vocab..(i + 1) * vocab]);
            for (x, y) in ra.iter().zip(rb) {
                let d = ((*x as f64) - (*y as f64)).abs();
                max_abs = max_abs.max(d);
                let scale = (x.abs().max(y.abs()) as f64).max(1e-6);
                max_rel = max_rel.max(d / scale);
            }
            if argmax(ra) == argmax(rb) {
                top1_agree += 1;
            } else {
                worst_margin = worst_margin.min(top_gap(ra));
            }
        }
        println!();
        println!("dispatch numerics, GEMV vs forced GEMM, {n} rows.");
        println!("Note the GEMM here is the tensor-core path, which rounds");
        println!("activations to half; GEMV accumulates in f32 throughout, so");
        println!("this difference is mostly the GEMM's, not the GEMV's:");
        println!("  max absolute error   {max_abs:.3e}");
        println!("  max relative error   {max_rel:.3e}");
        println!("  top-1 agreement      {top1_agree}/{n}");
        if top1_agree < n {
            println!("  min top-2 margin on disagreement  {worst_margin:.3e}");
        }
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
