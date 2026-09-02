//! Ratatui client for the Crucible inference service.
//!
//! Talks to the server over HTTP and SSE and nothing else. It does not link
//! `GpuModel`, `Runtime`, `PagePool` or any CUDA type, and it compiles without
//! the `cuda` feature — which is the real test of the boundary, not a comment
//! claiming one exists.
//!
//! # Event architecture
//!
//! ```text
//! keyboard task ─┐
//! SSE task ──────┤
//! metrics task ──┼──> AppEvent channel ──> app loop ──> render
//! health task ───┘
//! ```
//!
//! Tasks only send events. One task owns `App` and applies them; nothing else
//! touches application state, and no lock is held across an await.
//!
//! # Rendering cadence
//!
//! Token ingestion is decoupled from drawing. The backend can emit well over a
//! thousand tokens a second and a terminal cannot usefully repaint that fast,
//! so events are applied as they arrive and the screen is redrawn on a ~30 FPS
//! tick when something changed. Every token is still applied exactly once: the
//! coalescing is in the *drawing*, never in the data.

pub mod app;
pub mod client;
pub mod ui;

use std::io::{self, Stdout};
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::event::{
    DisableMouseCapture, Event as CtEvent, EventStream, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use futures_util::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use app::{App, ConnState};
use client::{Client, StreamMessage};

/// Maximum redraws per second. A terminal cannot usefully show more, and
/// drawing per token would make the client the slowest part of the system.
const FRAME_MS: u64 = 33;
/// Metrics poll interval. Frequent enough to feel live, rare enough that the
/// client is not a load generator.
const METRICS_MS: u64 = 700;
/// Reconnect probe interval while disconnected. Deliberately unhurried.
const RECONNECT_MS: u64 = 1500;
/// SSE events buffered between frames. Large enough that a fast generation is
/// never throttled by the terminal.
const STREAM_BUFFER: usize = 8192;

#[derive(Debug)]
enum AppEvent {
    Key(KeyEvent),
    Resize,
    Health(Result<crate::protocol::Health, String>),
    Metrics(Result<crate::protocol::Metrics, String>),
    Stream(StreamMessage),
}

/// Restores the terminal on every exit path, including panics.
///
/// A TUI that leaves raw mode enabled makes the user's shell unusable, so this
/// is a guard rather than a cleanup call at the end of `run`: an early return or
/// a panic must not be able to skip it.
struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl TerminalGuard {
    fn new() -> Result<Self> {
        enable_raw_mode().context("entering raw mode")?;
        let mut out = io::stdout();
        execute!(out, EnterAlternateScreen).context("entering alternate screen")?;
        let terminal = Terminal::new(CrosstermBackend::new(out))?;
        Ok(Self { terminal })
    }

    fn restore() {
        // Best effort and order matters: leave the alternate screen before
        // disabling raw mode so the shell is drawn on the real screen.
        let _ = execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture);
        let _ = disable_raw_mode();
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        Self::restore();
        let _ = self.terminal.show_cursor();
    }
}

pub async fn run(server: String, max_tokens: usize) -> Result<()> {
    let client = Client::new(&server).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut app = App::new(client.base().to_string(), max_tokens);

    // Install the panic hook before touching the terminal, so a panic during
    // setup still restores it.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        TerminalGuard::restore();
        default_hook(info);
    }));

    let mut guard = TerminalGuard::new()?;
    let (tx, mut rx) = mpsc::channel::<AppEvent>(STREAM_BUFFER);

    // Keyboard. Its own task so a slow frame never drops input.
    let key_tx = tx.clone();
    let keys: JoinHandle<()> = tokio::spawn(async move {
        let mut events = EventStream::new();
        while let Some(Ok(ev)) = events.next().await {
            let msg = match ev {
                CtEvent::Key(k) if k.kind == KeyEventKind::Press => AppEvent::Key(k),
                CtEvent::Resize(_, _) => AppEvent::Resize,
                _ => continue,
            };
            if key_tx.send(msg).await.is_err() {
                break;
            }
        }
    });

    // Health and metrics polling. Both degrade rather than fail: a missed poll
    // changes the status indicator, it does not end the session.
    let poll_tx = tx.clone();
    let poll_client = client.clone();
    let polls: JoinHandle<()> = tokio::spawn(async move {
        let mut have_health = false;
        let mut metrics_tick = tokio::time::interval(Duration::from_millis(METRICS_MS));
        let mut retry_tick = tokio::time::interval(Duration::from_millis(RECONNECT_MS));
        loop {
            tokio::select! {
                _ = metrics_tick.tick() => {
                    if have_health {
                        let r = poll_client.metrics().await.map_err(|e| e.to_string());
                        if r.is_err() { have_health = false; }
                        if poll_tx.send(AppEvent::Metrics(r)).await.is_err() { return; }
                    }
                }
                _ = retry_tick.tick() => {
                    // Fetch health on connect and after any failure, not on a
                    // fast loop: it is stable information.
                    if !have_health {
                        let r = poll_client.health().await.map_err(|e| e.to_string());
                        have_health = r.is_ok();
                        if poll_tx.send(AppEvent::Health(r)).await.is_err() { return; }
                    }
                }
            }
        }
    });

    let mut stream_task: Option<JoinHandle<()>> = None;
    let mut ticker = tokio::time::interval(Duration::from_millis(FRAME_MS));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut dirty = true;

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                if dirty {
                    guard.terminal.draw(|f| ui::draw(f, &mut app))?;
                    dirty = false;
                }
            }
            Some(ev) = rx.recv() => {
                dirty = true;
                match ev {
                    AppEvent::Resize => {}
                    AppEvent::Health(Ok(h)) => app.on_health(h),
                    AppEvent::Health(Err(e)) => app.on_poll_failure(e),
                    AppEvent::Metrics(Ok(m)) => app.on_metrics(m),
                    AppEvent::Metrics(Err(e)) => app.on_poll_failure(e),
                    AppEvent::Stream(msg) => match msg {
                        StreamMessage::Token { text, .. } => app.on_token(&text),
                        StreamMessage::Done { finish_reason, tokens_generated, text } => {
                            app.on_done(finish_reason, tokens_generated, &text);
                            stream_task = None;
                        }
                        StreamMessage::Failed(e) => {
                            app.on_stream_error(e.to_string());
                            stream_task = None;
                        }
                        StreamMessage::Ended => {
                            app.on_stream_ended();
                            stream_task = None;
                        }
                    },
                    AppEvent::Key(k) => {
                        if handle_key(k, &mut app, &client, &tx, &mut stream_task) {
                            break;
                        }
                    }
                }
            }
        }
        if app.should_quit {
            break;
        }
    }

    // Abort outstanding work before the guard restores the terminal, so no task
    // can write to a screen that is being torn down.
    if let Some(t) = stream_task.take() {
        t.abort();
    }
    keys.abort();
    polls.abort();
    drop(guard);
    Ok(())
}

/// Apply one key. Returns true to quit.
fn handle_key(
    k: KeyEvent,
    app: &mut App,
    client: &Client,
    tx: &mpsc::Sender<AppEvent>,
    stream_task: &mut Option<JoinHandle<()>>,
) -> bool {
    let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
    let alt = k.modifiers.contains(KeyModifiers::ALT);

    // Help swallows the next keypress so it can be dismissed with anything
    // obvious rather than only the key that opened it.
    if app.show_help {
        match k.code {
            KeyCode::Char('c') if ctrl => return true,
            _ => app.show_help = false,
        }
        return false;
    }

    // While the settings panel is open the arrow keys belong to it.
    if app.show_settings {
        match k.code {
            KeyCode::Char('c') if ctrl => return true,
            KeyCode::F(3) | KeyCode::Esc | KeyCode::Enter => app.show_settings = false,
            KeyCode::Up => app.settings_field = app.settings_field.prev(),
            KeyCode::Down => app.settings_field = app.settings_field.next(),
            KeyCode::Left => app.adjust_setting(false),
            KeyCode::Right => app.adjust_setting(true),
            _ => {}
        }
        return false;
    }

    match k.code {
        KeyCode::Char('c') if ctrl => return true,
        KeyCode::Char('u') if ctrl => app.input.clear(),
        KeyCode::F(1) => app.toggle_help(),
        KeyCode::F(2) => app.toggle_telemetry(),
        KeyCode::F(3) => app.toggle_settings(),

        KeyCode::Esc => {
            // Dropping the task drops the HTTP response, which closes the
            // connection. The server sees the disconnect and cancels at its
            // next scheduler boundary; there is no second cancel protocol.
            if app.begin_cancel() {
                if let Some(t) = stream_task.take() {
                    t.abort();
                }
                app.on_stream_ended();
            }
        }

        KeyCode::Enter if alt => app.input.insert('\n'),
        KeyCode::Enter => {
            if let Some(prompt) = app.submit() {
                let c = client.clone();
                let out = tx.clone();
                let max = app.max_tokens;
                let sampling = app.settings.request_params();
                *stream_task = Some(tokio::spawn(async move {
                    let (stx, mut srx) = mpsc::channel::<StreamMessage>(STREAM_BUFFER);
                    let pump =
                        tokio::spawn(async move { c.stream(prompt, max, sampling, stx).await });
                    while let Some(m) = srx.recv().await {
                        if out.send(AppEvent::Stream(m)).await.is_err() {
                            break;
                        }
                    }
                    let _ = pump.await;
                }));
            } else if app.conn != ConnState::Connected {
                app.status = Some(format!("Not connected to {}", app.server));
            }
        }

        KeyCode::Backspace => app.input.backspace(),
        KeyCode::Delete => app.input.delete(),
        KeyCode::Left => app.input.left(),
        KeyCode::Right => app.input.right(),
        KeyCode::Home => app.input.home(),
        KeyCode::End => {
            // End doubles as "return to the newest text" when scrolled away,
            // which is the more useful meaning at that moment.
            if !app.follow {
                app.scroll_to_bottom();
            } else {
                app.input.end();
            }
        }
        KeyCode::PageUp => app.scroll_up(10),
        KeyCode::PageDown => app.scroll_down(10),

        KeyCode::Char(c) => app.input.insert(c),
        _ => {}
    }
    false
}
