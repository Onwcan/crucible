//! HTTP inference service over the continuous-batching runtime.
//!
//! # Ownership
//!
//! The GPU runtime has exactly one owner: a dedicated OS thread. HTTP handlers
//! never touch it. They send a job down a channel and read tokens back from a
//! per-request channel:
//!
//! ```text
//! axum handlers  --jobs-->  inference thread  --tokens-->  per-request channel
//!                                  |
//!                          Runtime (scheduler, paged KV, graphs)
//! ```
//!
//! A dedicated thread rather than a Tokio task, for two reasons. The CUDA
//! context and its buffers are not `Sync`, so they must not be shared across a
//! work-stealing scheduler; and a decode step is a blocking GPU call, which
//! would stall an async worker for the whole step. The channel boundary also
//! means no mutex is ever held across a GPU launch.
//!
//! This is what keeps batching intact. If each handler called the model behind
//! a lock, concurrent requests would serialise and the batching engine would be
//! bypassed by the very layer meant to feed it. Instead every in-flight request
//! is submitted to one scheduler, which decides how they share a step.
//!
//! # Scope
//!
//! Local development service: bound to loopback unless told otherwise, no auth,
//! no TLS. Input sizes are validated and queues bounded, because those protect
//! the runtime rather than the network.
//!
//! # Two wire protocols, one runtime
//!
//! This module serves Crucible's native protocol. `openai` serves an
//! OpenAI-compatible subset by translating into the same `Job` and reading back
//! the same `StreamItem` stream -- it is a second *protocol*, never a second
//! engine. The plumbing it needs (`AppState`, `StreamItem`, `submit_openai`) is
//! `pub(crate)` for that reason and for no other; there is deliberately no way
//! for a compatibility handler to reach `GpuModel`, the scheduler or a queue of
//! its own.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::gpu_model::{GpuModel, Precision};
use crate::protocol::{
    ErrorBody, GenerateRequest, GenerateResponse, Health as HealthBody, Metrics as MetricsBody,
    SamplingCapabilities,
};
use crate::paged::PAGE_TOKENS;
use crate::runtime::{FinishReason, Request as RtRequest, Runtime};
use crate::sampling::{self, GenerationConfig, DEFAULT_SEED, DEFAULT_TOP_K};
use crate::tokenizer::{IncrementalDecoder, Tokenizer};
use crate::weights::Weights;
use crate::Config;

/// Service limits. These bound what HTTP can ask the runtime to allocate; the
/// page pool remains the final authority on memory.
#[derive(Debug, Clone)]
pub struct Limits {
    pub max_batch: usize,
    pub max_queue: usize,
    pub max_prompt_tokens: usize,
    pub max_new_tokens: usize,
    pub context: usize,
}

#[derive(Debug, Clone)]
pub struct ServeOptions {
    pub host: IpAddr,
    pub port: u16,
    pub model_dir: std::path::PathBuf,
    pub tokenizer: std::path::PathBuf,
    pub quant: String,
    pub limits: Limits,
    pub kv_pages: usize,
    /// Public identifier for the served checkpoint, used by the
    /// OpenAI-compatible endpoints.
    pub model_id: String,
}

// --- wire types -------------------------------------------------------------

// --- inference thread plumbing ---------------------------------------------

/// What a handler sends to the inference thread.
pub(crate) struct Job {
    id: u64,
    prompt: Vec<usize>,
    config: GenerationConfig,
    events: mpsc::Sender<StreamItem>,
}

/// What the inference thread sends back per request.
#[derive(Debug)]
pub(crate) enum StreamItem {
    Token { id: usize, text: String },
    Done { reason: FinishReason, generated: usize, tail: String },
    Failed(String),
}

/// Shared counters. Updated once per scheduler step, never inside the GPU path.
#[derive(Debug, Default)]
struct Stats {
    active: usize,
    queued: usize,
    completed: u64,
    cancelled: u64,
    failed: u64,
    pages_used: usize,
    pages_free: usize,
    last_batch: usize,
    steps: u64,
    tokens: u64,
    batch_sum: u64,
    greedy_requests: u64,
    sampled_requests: u64,
}

#[derive(Clone)]
pub(crate) struct AppState {
    jobs: mpsc::Sender<Job>,
    stats: Arc<Mutex<Stats>>,
    next_id: Arc<AtomicU64>,
    pub(crate) limits: Limits,
    vocab: usize,
    health: Arc<HealthBody>,
    started: Instant,
    /// Set when the inference thread dies, so /health can report it.
    fatal: Arc<Mutex<Option<String>>>,
    /// The public model id, as published by `/v1/models` and echoed in every
    /// compatibility response.
    pub(crate) model_id: Arc<str>,
    /// Unix time the checkpoint was written, for the `created` field of the
    /// model object. Stable across restarts, unlike server start time.
    pub(crate) model_created: i64,
}

/// What the inference thread reports once the model is resident.
///
/// The runtime cannot be built on the main thread and moved here: a captured
/// `CudaGraph` holds raw pointers and is not `Send`. That is the right
/// constraint rather than an obstacle -- a CUDA context belongs to the thread
/// that drives it -- so the thread loads the model and sends back the facts the
/// HTTP layer needs to describe itself.
struct InitInfo {
    device: String,
    pages: usize,
    pool_bytes: usize,
    weight_bytes: usize,
    vocab: usize,
}

/// Per-request state held by the inference thread.
struct Live {
    events: mpsc::Sender<StreamItem>,
    decoder: IncrementalDecoder,
    generated: usize,
}

// --- request validation -----------------------------------------------------

/// Reject what the runtime cannot serve, with a reason the caller can act on.
///
/// Pure so it can be tested without a GPU, and so the inference thread never
/// has to defend itself against malformed input.
pub fn validate(
    prompt_tokens: usize,
    max_tokens: usize,
    limits: &Limits,
) -> std::result::Result<(), String> {
    if prompt_tokens == 0 {
        return Err("prompt is empty after tokenisation".into());
    }
    if max_tokens == 0 {
        return Err("max_tokens must be at least 1".into());
    }
    if max_tokens > limits.max_new_tokens {
        return Err(format!(
            "max_tokens {max_tokens} exceeds the server limit of {}",
            limits.max_new_tokens
        ));
    }
    if prompt_tokens > limits.max_prompt_tokens {
        return Err(format!(
            "prompt of {prompt_tokens} tokens exceeds the server limit of {}",
            limits.max_prompt_tokens
        ));
    }
    if prompt_tokens + max_tokens > limits.context {
        return Err(format!(
            "prompt ({prompt_tokens}) plus max_tokens ({max_tokens}) exceeds the \
             model context of {}",
            limits.context
        ));
    }
    Ok(())
}

/// Whether a validation failure was about size rather than shape.
///
/// The compatibility layer needs to answer with `context_length_exceeded`
/// rather than a generic invalid-value code, and sniffing the message string
/// for that would break the first time somebody rewords it.
pub(crate) fn is_size_failure(prompt_tokens: usize, max_tokens: usize, limits: &Limits) -> bool {
    max_tokens > limits.max_new_tokens
        || prompt_tokens > limits.max_prompt_tokens
        || prompt_tokens + max_tokens > limits.context
}

/// The finish reason as the OpenAI schema spells it.
///
/// `Length` is the only value a client can actually observe. Generation stops
/// when the token budget runs out and nothing else: this model has no trained
/// stop token, and stop sequences are not implemented, so reporting `stop`
/// would assert a natural ending that never happened. A cancelled request has
/// already lost its connection, so its finish reason is never delivered --
/// mapping it here keeps the function total rather than describing a case a
/// client can see.
pub(crate) fn finish_reason_str(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Length => "length",
        FinishReason::Cancelled => "length",
    }
}

// --- handlers ---------------------------------------------------------------

async fn health(State(st): State<AppState>) -> Response {
    if let Some(err) = st.fatal.lock().unwrap().clone() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorBody {
                error: format!("inference runtime stopped: {err}"),
            }),
        )
            .into_response();
    }
    Json(&*st.health).into_response()
}

async fn metrics(State(st): State<AppState>) -> Json<MetricsBody> {
    let s = st.stats.lock().unwrap();
    Json(MetricsBody {
        active_requests: s.active,
        queued_requests: s.queued,
        completed_requests: s.completed,
        cancelled_requests: s.cancelled,
        failed_requests: s.failed,
        kv_pages_used: s.pages_used,
        kv_pages_free: s.pages_free,
        last_batch_size: s.last_batch,
        decode_steps: s.steps,
        aggregate_tokens_generated: s.tokens,
        greedy_requests: s.greedy_requests,
        sampled_requests: s.sampled_requests,
        average_batch_size: if s.steps > 0 {
            s.batch_sum as f64 / s.steps as f64
        } else {
            0.0
        },
        uptime_seconds: st.started.elapsed().as_secs_f64(),
    })
}

/// Turn a request body into a generation config, rejecting nonsense.
///
/// Pure, so the parameter semantics can be tested without a GPU or a server.
pub fn config_from_request(
    req: &GenerateRequest,
    vocab: usize,
) -> std::result::Result<GenerationConfig, String> {
    // Absent temperature means greedy, which is what this service did before
    // sampling existed.
    let temperature = req.temperature.unwrap_or(0.0);
    if let Some(t) = req.temperature {
        if t.is_nan() {
            return Err("temperature must be a number".into());
        }
        if t < 0.0 {
            return Err(format!("temperature {t} must not be negative"));
        }
    }
    if let Some(k) = req.top_k {
        if k == 0 {
            return Err("top_k must be at least 1".into());
        }
    }
    if req.top_k.is_some() && !req.wants_sampling() {
        return Err(
            "top_k has no effect without a positive temperature; omit it for greedy".into(),
        );
    }
    if req.seed.is_some() && !req.wants_sampling() {
        return Err("seed has no effect without a positive temperature".into());
    }

    let cfg = GenerationConfig {
        max_tokens: req.max_tokens,
        temperature,
        top_k: req.top_k.unwrap_or(DEFAULT_TOP_K),
        seed: req.seed.unwrap_or(DEFAULT_SEED),
    };
    sampling::validate(&cfg, vocab)?;
    Ok(cfg)
}

/// Tokenise, validate and submit. Shared by both generate endpoints so there is
/// only ever one inference implementation.
async fn submit(
    st: &AppState,
    req: &GenerateRequest,
    tokenizer: &Tokenizer,
) -> std::result::Result<mpsc::Receiver<StreamItem>, (StatusCode, String)> {
    let ids = tokenizer
        .encode(&req.prompt)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("could not tokenise prompt: {e}")))?;
    let prompt: Vec<usize> = ids.into_iter().map(|v| v as usize).collect();

    validate(prompt.len(), req.max_tokens, &st.limits)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let config = config_from_request(req, st.vocab)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    // Bounded channel: a client that stops reading cannot make the inference
    // thread buffer without limit.
    let (tx, rx) = mpsc::channel(st.limits.max_new_tokens.min(512) + 8);
    let job = Job {
        id: st.next_id.fetch_add(1, Ordering::Relaxed),
        prompt,
        config,
        events: tx,
    };

    // try_send rather than send: a full queue must answer 429 immediately
    // rather than park the connection until capacity appears.
    st.jobs.try_send(job).map_err(|e| match e {
        mpsc::error::TrySendError::Full(_) => (
            StatusCode::TOO_MANY_REQUESTS,
            format!(
                "server queue is full ({} waiting); retry shortly",
                st.limits.max_queue
            ),
        ),
        mpsc::error::TrySendError::Closed(_) => (
            StatusCode::SERVICE_UNAVAILABLE,
            "inference runtime is not running".into(),
        ),
    })?;
    Ok(rx)
}

/// Tokenise, validate and submit on behalf of a compatibility handler.
///
/// The same three steps `submit` does, with two differences: it takes the
/// generation parameters directly rather than through the native request DTO,
/// and it reports failures in the OpenAI error shape. It goes through the same
/// bounded queue to the same inference thread -- a compatibility request has no
/// privileged path and no separate scheduler.
///
/// Returns the token channel and the prompt's token count, which the caller
/// needs for `usage.prompt_tokens` and which must come from the tokenizer
/// rather than from the string's length.
pub(crate) async fn submit_openai(
    st: &AppState,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
) -> std::result::Result<(mpsc::Receiver<StreamItem>, usize), crate::openai::ApiError> {
    use crate::openai::ApiError;

    let ids = tokenizer.encode(prompt).map_err(|e| {
        ApiError::invalid(format!("Could not tokenise the prompt: {e}"), Some("prompt"))
    })?;
    let tokens: Vec<usize> = ids.into_iter().map(|v| v as usize).collect();
    let prompt_tokens = tokens.len();

    if let Err(msg) = validate(prompt_tokens, max_tokens, &st.limits) {
        return Err(if is_size_failure(prompt_tokens, max_tokens, &st.limits) {
            ApiError::context_length(msg)
        } else {
            ApiError::invalid(msg, Some("prompt"))
        });
    }

    // Crucible's sampling semantics, unchanged: absent or non-positive
    // temperature is greedy. That differs from OpenAI's documented default of
    // 1.0 and is deliberate -- it keeps this endpoint and the native one
    // answering the same prompt with the same text, and it is documented in the
    // README rather than left for a user to discover.
    let sampling = temperature.is_some_and(|t| t > 0.0);
    if top_k.is_some() && !sampling {
        return Err(ApiError::invalid(
            "'top_k' has no effect without a positive 'temperature'; omit it for              greedy decoding."
                .to_string(),
            Some("top_k"),
        ));
    }
    let config = GenerationConfig {
        max_tokens,
        temperature: temperature.unwrap_or(0.0),
        top_k: top_k.unwrap_or(DEFAULT_TOP_K),
        // A seed given without a positive temperature is accepted and unused:
        // greedy decoding already satisfies the reproducibility the field asks
        // for, so refusing it would break clients that always send one.
        seed: seed.unwrap_or(DEFAULT_SEED),
    };
    if let Err(msg) = sampling::validate(&config, st.vocab) {
        return Err(ApiError::invalid(msg, Some("temperature")));
    }

    let (tx, rx) = mpsc::channel(st.limits.max_new_tokens.min(512) + 8);
    let job = Job {
        id: st.next_id.fetch_add(1, Ordering::Relaxed),
        prompt: tokens,
        config,
        events: tx,
    };
    st.jobs.try_send(job).map_err(|e| match e {
        mpsc::error::TrySendError::Full(_) => ApiError::rate_limited(format!(
            "The server queue is full ({} waiting). Retry shortly.",
            st.limits.max_queue
        )),
        mpsc::error::TrySendError::Closed(_) => {
            ApiError::unavailable("The inference runtime is not running.")
        }
    })?;
    Ok((rx, prompt_tokens))
}

async fn generate(
    State(st): State<AppState>,
    tokenizer: axum::Extension<Arc<Tokenizer>>,
    Json(req): Json<GenerateRequest>,
) -> Response {
    let mut rx = match submit(&st, &req, &tokenizer).await {
        Ok(rx) => rx,
        Err((code, msg)) => return (code, Json(ErrorBody { error: msg })).into_response(),
    };

    let mut text = String::new();
    let mut generated = 0usize;
    let mut reason = FinishReason::Length;
    while let Some(item) = rx.recv().await {
        match item {
            StreamItem::Token { text: t, .. } => text.push_str(&t),
            StreamItem::Done {
                reason: r,
                generated: g,
                tail,
            } => {
                text.push_str(&tail);
                generated = g;
                reason = r;
                break;
            }
            StreamItem::Failed(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorBody { error: e }),
                )
                    .into_response()
            }
        }
    }

    Json(GenerateResponse {
        text,
        tokens_generated: generated,
        finish_reason: reason.as_str().into(),
        prompt_tokens: req.prompt.len(),
    })
    .into_response()
}

async fn generate_stream(
    State(st): State<AppState>,
    tokenizer: axum::Extension<Arc<Tokenizer>>,
    Json(req): Json<GenerateRequest>,
) -> Response {
    let rx = match submit(&st, &req, &tokenizer).await {
        Ok(rx) => rx,
        Err((code, msg)) => return (code, Json(ErrorBody { error: msg })).into_response(),
    };

    // Dropping this stream drops the receiver, which closes the channel. The
    // inference thread notices on its next step and cancels the request -- that
    // is the whole disconnect path.
    let stream = ReceiverStream::new(rx).map(|item| {
        let ev = match item {
            StreamItem::Token { id, text } => Event::default()
                .event("token")
                .json_data(serde_json::json!({ "token_id": id, "text": text })),
            StreamItem::Done {
                reason,
                generated,
                tail,
            } => Event::default().event("done").json_data(serde_json::json!({
                "finish_reason": reason.as_str(),
                "tokens_generated": generated,
                "text": tail,
            })),
            StreamItem::Failed(e) => Event::default()
                .event("error")
                .json_data(serde_json::json!({ "error": e })),
        };
        ev
    });

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

// --- inference thread -------------------------------------------------------

/// Owns the runtime for the process lifetime.
///
/// Structured so that a per-request failure is reported to that request only,
/// while a runtime failure stops the loop and marks the service unhealthy.
fn inference_thread(
    opts: ServeOptions,
    tokenizer: Arc<Tokenizer>,
    mut jobs: mpsc::Receiver<Job>,
    stats: Arc<Mutex<Stats>>,
    fatal: Arc<Mutex<Option<String>>>,
    init: std::sync::mpsc::Sender<Result<InitInfo>>,
) {
    let mut rt = match build_runtime(&opts) {
        Ok((rt, info)) => {
            let _ = init.send(Ok(info));
            rt
        }
        Err(e) => {
            // Report the failure to the waiting caller rather than panicking on
            // a thread nobody is watching.
            let _ = init.send(Err(e));
            return;
        }
    };

    let mut live: HashMap<u64, Live> = HashMap::new();

    loop {
        // Take whatever has arrived without blocking.
        loop {
            match jobs.try_recv() {
                Ok(job) => {
                    {
                        let mut s = stats.lock().unwrap();
                        if job.config.is_greedy() {
                            s.greedy_requests += 1;
                        } else {
                            s.sampled_requests += 1;
                        }
                    }
                    rt.submit(RtRequest {
                        id: job.id,
                        prompt: job.prompt,
                        config: job.config,
                    });
                    live.insert(
                        job.id,
                        Live {
                            events: job.events,
                            decoder: IncrementalDecoder::new(),
                            generated: 0,
                        },
                    );
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                // Every handler dropped: the server is shutting down.
                Err(mpsc::error::TryRecvError::Disconnected) => return,
            }
        }

        // Nothing to do: block until work arrives rather than spinning on the
        // GPU. This is also where shutdown is noticed when idle.
        if rt.is_idle() {
            match jobs.blocking_recv() {
                Some(job) => {
                    {
                        let mut s = stats.lock().unwrap();
                        if job.config.is_greedy() {
                            s.greedy_requests += 1;
                        } else {
                            s.sampled_requests += 1;
                        }
                    }
                    rt.submit(RtRequest {
                        id: job.id,
                        prompt: job.prompt,
                        config: job.config,
                    });
                    live.insert(
                        job.id,
                        Live {
                            events: job.events,
                            decoder: IncrementalDecoder::new(),
                            generated: 0,
                        },
                    );
                }
                None => return,
            }
        }

        // Withdraw anything whose client has gone. Checked between steps: a
        // step is one fused graph launch and cannot be interrupted partway.
        let gone: Vec<u64> = live
            .iter()
            .filter(|(_, l)| l.events.is_closed())
            .map(|(id, _)| *id)
            .collect();
        for id in gone {
            match rt.cancel(id) {
                Ok(_) => {
                    live.remove(&id);
                    stats.lock().unwrap().cancelled += 1;
                }
                Err(e) => {
                    *fatal.lock().unwrap() = Some(format!("cancel failed: {e}"));
                    return;
                }
            }
            // cancel() records a Completion; drop it so it is not double-counted.
            rt.completed().retain(|_| false);
        }

        let info = match rt.step() {
            Ok(i) => i,
            Err(e) => {
                // A runtime failure is fatal: the GPU state is no longer
                // trustworthy. Tell everyone still waiting, then stop.
                let msg = format!("{e}");
                for (_, l) in live.drain() {
                    let _ = l.events.try_send(StreamItem::Failed(msg.clone()));
                }
                stats.lock().unwrap().failed += 1;
                *fatal.lock().unwrap() = Some(msg);
                return;
            }
        };

        // Route tokens. A send failure means the client vanished between the
        // disconnect check and now, which the next iteration will clean up.
        for (id, token) in &info.tokens {
            if let Some(l) = live.get_mut(id) {
                let piece = tokenizer
                    .decode_piece(*token as u32)
                    .map(|b| l.decoder.push(b))
                    .unwrap_or_default();
                l.generated += 1;
                let _ = l.events.try_send(StreamItem::Token {
                    id: *token,
                    text: piece,
                });
            }
        }

        for c in rt.completed() {
            if let Some(mut l) = live.remove(&c.id) {
                let tail = l.decoder.finish();
                let _ = l.events.try_send(StreamItem::Done {
                    reason: c.reason,
                    generated: l.generated,
                    tail,
                });
            }
            let mut s = stats.lock().unwrap();
            match c.reason {
                FinishReason::Length => s.completed += 1,
                FinishReason::Cancelled => {}
            }
        }

        {
            let mut s = stats.lock().unwrap();
            s.active = info.active_after;
            s.queued = info.pending_after;
            s.last_batch = info.decoded;
            s.steps += 1;
            s.batch_sum += info.decoded as u64;
            s.tokens += info.tokens.len() as u64;
            s.pages_free = info.free_pages;
            s.pages_used = rt.model().page_pool().n_pages() - info.free_pages;
        }
    }
}

// --- entry point ------------------------------------------------------------

/// Load the model and wrap it in a runtime. Runs on the inference thread.
fn build_runtime(opts: &ServeOptions) -> Result<(Runtime, InitInfo)> {
    let cfg = Config::from_file(opts.model_dir.join("config.json"))?;
    let weights = Weights::open(opts.model_dir.join("model.safetensors"))?;
    let precision = Precision::parse(&opts.quant)
        .ok_or_else(|| anyhow::anyhow!("unknown precision {:?}", opts.quant))?;

    let mut model = GpuModel::load_with(cfg.clone(), &weights, cfg.block_size, precision)?;
    let device = model.gpu.name()?;
    model.enable_paging(opts.kv_pages, opts.limits.max_batch)?;

    let info = InitInfo {
        device,
        pages: model.page_pool().n_pages(),
        pool_bytes: model.page_pool().total_bytes(),
        weight_bytes: model.weight_bytes(),
        vocab: cfg.vocab_size,
    };
    Ok((Runtime::new(model)?, info))
}

pub fn serve(opts: ServeOptions) -> Result<()> {
    let tokenizer = Arc::new(
        Tokenizer::load(&opts.tokenizer)
            .with_context(|| format!("loading tokenizer {}", opts.tokenizer.display()))?,
    );

    let limits = opts.limits.clone();
    let stats = Arc::new(Mutex::new(Stats::default()));
    let fatal = Arc::new(Mutex::new(None));
    let (job_tx, job_rx) = mpsc::channel::<Job>(limits.max_queue);
    let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<InitInfo>>();

    let model_name = opts
        .model_dir
        .file_name()
        .map(|v| v.to_string_lossy().into_owned())
        .unwrap_or_else(|| opts.model_dir.display().to_string());
    let model_dir = opts.model_dir.clone();
    let quant = opts.quant.clone();

    let model_id: Arc<str> = Arc::from(opts.model_id.as_str());
    crate::openai::validate_model_id(&model_id).map_err(|e| anyhow::anyhow!(e))?;
    // The checkpoint's own timestamp, so `created` survives a restart. Model
    // objects are cached by clients; a value that moved every boot would look
    // like a different model each time.
    let model_created = std::fs::metadata(model_dir.join("config.json"))
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    let worker = {
        let tokenizer = tokenizer.clone();
        let stats = stats.clone();
        let fatal = fatal.clone();
        let opts = opts.clone();
        std::thread::Builder::new()
            .name("crucible-inference".into())
            .spawn(move || {
                inference_thread(opts, tokenizer, job_rx, stats, fatal, init_tx)
            })?
    };

    // Wait for the model before binding a port: a server that accepts requests
    // it cannot serve is worse than one that has not started.
    let info = init_rx
        .recv()
        .context("inference thread exited before reporting readiness")??;
    let vocab = info.vocab;
    let pool_mb = info.pool_bytes as f64 / 1e6;
    let weight_mb = info.weight_bytes as f64 / 1e6;
    let pages = info.pages;
    let device = info.device.clone();
    stats.lock().unwrap().pages_free = pages;

    let health_body = Arc::new(HealthBody {
        status: "ok".into(),
        model: model_name,
        device: device.clone(),
        max_batch: limits.max_batch,
        context: limits.context,
        kv_pages: pages,
        sampling: SamplingCapabilities::default(),
    });

    let state = AppState {
        jobs: job_tx,
        vocab,
        stats,
        next_id: Arc::new(AtomicU64::new(1)),
        limits: limits.clone(),
        health: health_body,
        started: Instant::now(),
        fatal,
        model_id: model_id.clone(),
        model_created,
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/generate", post(generate))
        .route("/v1/generate/stream", post(generate_stream))
        // OpenAI-compatible surface. Adapters over the handlers above: same
        // queue, same inference thread, same scheduler.
        .route("/v1/models", get(crate::openai::models_list))
        .route("/v1/models/:model", get(crate::openai::models_get))
        .route("/v1/completions", post(crate::openai::completions::completions))
        .route(
            "/v1/chat/completions",
            post(crate::openai::chat::chat_completions),
        )
        .layer(axum::Extension(tokenizer))
        .with_state(state);

    let addr = SocketAddr::new(opts.host, opts.port);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    println!("crucible inference server");
    println!("  model        {}", model_dir.display());
    println!("  device       {device}");
    println!("  precision    {quant}");
    println!("  weights      {weight_mb:.1} MB");
    println!("  kv pool      {pages} pages, {pool_mb:.1} MB, {} tokens",
             pages * PAGE_TOKENS);
    println!("  max batch    {}", limits.max_batch);
    println!("  max queue    {}", limits.max_queue);
    println!("  context      {}", limits.context);
    println!("  sampling     greedy");
    println!("  model id     {model_id}  (OpenAI-compatible endpoints)");
    println!("  listening    http://{addr}");
    if !opts.host.is_loopback() {
        println!();
        println!("  WARNING: bound to a non-loopback address. This service has no");
        println!("  authentication and is intended for local development only.");
    }
    println!();

    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .with_context(|| format!("binding {addr}"))?;
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
                println!("\nshutting down: no longer accepting requests");
            })
            .await?;
        Ok::<_, anyhow::Error>(())
    })?;

    // `axum::serve` consumed the router, which owned the only job sender, so
    // by here the channel is closed. That is how the inference thread learns to
    // stop. Joining it releases the GPU context before the process exits rather
    // than leaving it to teardown.
    let _ = worker.join();
    println!("inference runtime stopped");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> Limits {
        Limits {
            max_batch: 16,
            max_queue: 32,
            max_prompt_tokens: 512,
            max_new_tokens: 256,
            context: 1024,
        }
    }

    #[test]
    fn rejects_an_empty_prompt() {
        assert!(validate(0, 16, &limits()).unwrap_err().contains("empty"));
    }

    #[test]
    fn rejects_zero_and_oversized_max_tokens() {
        assert!(validate(4, 0, &limits()).is_err());
        assert!(validate(4, 257, &limits()).unwrap_err().contains("server limit"));
        assert!(validate(4, 256, &limits()).is_ok());
    }

    #[test]
    fn rejects_an_oversized_prompt() {
        assert!(validate(513, 8, &limits()).unwrap_err().contains("prompt of 513"));
        assert!(validate(512, 8, &limits()).is_ok());
    }

    #[test]
    fn rejects_a_request_that_would_exceed_context() {
        // Each part is individually allowed; together they do not fit.
        let l = limits();
        let err = validate(500, 600, &l);
        assert!(err.is_err());
        assert!(validate(500, 256, &l).is_ok());
    }
}
