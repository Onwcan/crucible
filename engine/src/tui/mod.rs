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
    Stream {
        generation: u64,
        message: StreamMessage,
    },
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
    let mut generation = 0;
    let mut ticker = tokio::time::interval(Duration::from_millis(FRAME_MS));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut dirty = true;

    let result = async {
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
                    AppEvent::Stream { generation: source, message } => {
                        apply_stream_message(source, generation, message, &mut app, &mut stream_task);
                    }
                    AppEvent::Key(k) => {
                        if handle_key(k, &mut app, &client, &tx, &mut stream_task, &mut generation) {
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
    Ok::<(), anyhow::Error>(())
    }.await;

    // Wait for cancellation before terminal teardown, including on draw errors.
    stop_generation(&mut stream_task).await;
    keys.abort();
    polls.abort();
    let _ = keys.await;
    let _ = polls.await;
    drop(guard);
    result
}

async fn stop_generation(stream_task: &mut Option<JoinHandle<()>>) {
    if let Some(task) = stream_task.take() {
        task.abort();
        let _ = task.await;
    }
}

fn apply_stream_message(
    source: u64,
    generation: u64,
    message: StreamMessage,
    app: &mut App,
    stream_task: &mut Option<JoinHandle<()>>,
) {
    // A cancelled task may already have queued events. They must not resume
    // the cancelled message or finish a subsequent prompt's generation.
    if source != generation || stream_task.is_none() {
        return;
    }
    match message {
        StreamMessage::Token { text, .. } => app.on_token(&text),
        StreamMessage::Done {
            finish_reason,
            tokens_generated,
            text,
        } => {
            app.on_done(finish_reason, tokens_generated, &text);
            *stream_task = None;
        }
        StreamMessage::Failed(e) => {
            app.on_stream_error(e.to_string());
            *stream_task = None;
        }
        StreamMessage::Ended => {
            app.on_stream_ended();
            *stream_task = None;
        }
    }
}

/// Poll the HTTP producer and its forwarding channel in the same task. Aborting
/// this task drops the producer (and its response), even with no body bytes.
async fn forward_stream(
    producer: impl std::future::Future<Output = ()>,
    mut messages: mpsc::Receiver<StreamMessage>,
    out: mpsc::Sender<AppEvent>,
    generation: u64,
) {
    tokio::pin!(producer);
    loop {
        tokio::select! {
            _ = out.closed() => return,
            _ = &mut producer => break,
            message = messages.recv() => {
                let Some(message) = message else { return };
                if out.send(AppEvent::Stream { generation, message }).await.is_err() {
                    return;
                }
            }
        }
    }
    // Completion may have queued Done/Failed immediately before returning.
    // The producer cannot send again, so drain the finite backlog before exit.
    while let Ok(message) = messages.try_recv() {
        if out
            .send(AppEvent::Stream {
                generation,
                message,
            })
            .await
            .is_err()
        {
            return;
        }
    }
}

/// Apply one key. Returns true to quit.
fn handle_key(
    k: KeyEvent,
    app: &mut App,
    client: &Client,
    tx: &mpsc::Sender<AppEvent>,
    stream_task: &mut Option<JoinHandle<()>>,
    generation: &mut u64,
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
            // This task directly owns the HTTP future. Aborting it drops the
            // response even while body.next() is stalled; no nested pump is
            // left waiting for a token to discover cancellation.
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
                *generation += 1;
                let current = *generation;
                *stream_task = Some(tokio::spawn(async move {
                    let (stx, srx) = mpsc::channel::<StreamMessage>(STREAM_BUFFER);
                    let producer = c.stream(prompt, max, sampling, stx);
                    forward_stream(producer, srx, out, current).await;
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

#[cfg(test)]
mod tests {
    use super::*;
    use app::{MessageState, RequestState};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};
    use tokio::sync::oneshot;

    const LIMIT: Duration = Duration::from_secs(2);

    async fn bounded<T>(future: impl std::future::Future<Output = T>) -> T {
        tokio::time::timeout(LIMIT, future)
            .await
            .expect("TUI task did not finish promptly")
    }

    async fn read_request(socket: &mut TcpStream) {
        let mut request = Vec::new();
        loop {
            let mut bytes = [0; 1024];
            let count = bounded(socket.read(&mut bytes)).await.unwrap();
            assert_ne!(count, 0);
            request.extend_from_slice(&bytes[..count]);
            assert!(request.len() < 16 * 1024);
            if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = std::str::from_utf8(&request[..end]).unwrap();
                assert!(headers.starts_with("POST /v1/generate/stream HTTP/1.1"));
                let length = headers
                    .lines()
                    .find_map(|line| {
                        let (key, value) = line.split_once(':')?;
                        key.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap();
                if request.len() >= end + 4 + length {
                    return;
                }
            }
        }
    }

    struct Stall {
        base: String,
        ready: oneshot::Receiver<()>,
        closed: oneshot::Receiver<()>,
        server: JoinHandle<()>,
    }

    async fn stall(initial: &'static str, reuse: bool) -> Stall {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let (ready_tx, ready) = oneshot::channel();
        let (closed_tx, closed) = oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut socket, _) = bounded(listener.accept()).await.unwrap();
            // Consume the JSON body: the next read must measure EOF, not payload.
            read_request(&mut socket).await;
            socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n").await.unwrap();
            if !initial.is_empty() {
                socket
                    .write_all(format!("{:x}\r\n{initial}\r\n", initial.len()).as_bytes())
                    .await
                    .unwrap();
            }
            ready_tx.send(()).unwrap();
            // No more bytes are sent. Cancellation must close this idle body.
            let mut byte = [0];
            assert_eq!(bounded(socket.read(&mut byte)).await.unwrap(), 0);
            closed_tx.send(()).unwrap();
            if reuse {
                let (mut socket, _) = bounded(listener.accept()).await.unwrap();
                read_request(&mut socket).await;
                let body = "event: done\ndata: {\"finish_reason\":\"length\",\"tokens_generated\":0,\"text\":\"reused\"}\n\n";
                socket
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
            }
        });
        Stall {
            base,
            ready,
            closed,
            server,
        }
    }

    struct Session {
        app: App,
        client: Client,
        tx: mpsc::Sender<AppEvent>,
        rx: mpsc::Receiver<AppEvent>,
        task: Option<JoinHandle<()>>,
        generation: u64,
    }

    impl Session {
        fn new(base: &str) -> Self {
            let mut app = App::new(base.into(), 8);
            app.conn = ConnState::Connected;
            let (tx, rx) = mpsc::channel(16);
            Self {
                app,
                client: Client::new(base).unwrap(),
                tx,
                rx,
                task: None,
                generation: 0,
            }
        }

        fn key(&mut self, code: KeyCode, modifiers: KeyModifiers) -> bool {
            handle_key(
                KeyEvent::new(code, modifiers),
                &mut self.app,
                &self.client,
                &self.tx,
                &mut self.task,
                &mut self.generation,
            )
        }

        fn submit(&mut self) {
            self.app.input.insert_str("test prompt");
            assert!(!self.key(KeyCode::Enter, KeyModifiers::NONE));
            assert!(self.task.is_some());
            assert_eq!(self.app.request, RequestState::Submitting);
        }

        async fn receive(&mut self) {
            let event = bounded(self.rx.recv())
                .await
                .expect("generation channel closed");
            if let AppEvent::Stream {
                generation,
                message,
            } = event
            {
                apply_stream_message(
                    generation,
                    self.generation,
                    message,
                    &mut self.app,
                    &mut self.task,
                );
            } else {
                panic!("unexpected event {event:?}");
            }
        }
    }

    async fn escape_and_reuse(initial: &'static str) {
        let server = stall(initial, true).await;
        let mut session = Session::new(&server.base);
        session.submit();
        bounded(server.ready).await.unwrap();
        if !initial.is_empty() {
            session.receive().await;
            assert_eq!(session.app.messages.last().unwrap().text, "first");
        }
        let cancelled = session.generation;
        assert!(!session.key(KeyCode::Esc, KeyModifiers::NONE));
        assert!(session.task.is_none());
        assert_eq!(session.app.request, RequestState::Idle);
        assert_eq!(
            session.app.messages.last().unwrap().state,
            MessageState::Cancelled
        );
        assert!(session.app.messages.last().unwrap().error.is_none());
        assert_eq!(session.app.conn, ConnState::Connected);
        bounded(server.closed).await.unwrap();

        // Already-buffered updates are ignored both while idle and after a new
        // submission; an old terminal event must not clear the new task handle.
        apply_stream_message(
            cancelled,
            session.generation,
            StreamMessage::Token {
                token_id: 99,
                text: "late".into(),
            },
            &mut session.app,
            &mut session.task,
        );
        assert_eq!(session.app.request, RequestState::Idle);
        session.submit();
        apply_stream_message(
            cancelled,
            session.generation,
            StreamMessage::Token {
                token_id: 99,
                text: "late".into(),
            },
            &mut session.app,
            &mut session.task,
        );
        apply_stream_message(
            cancelled,
            session.generation,
            StreamMessage::Done {
                finish_reason: "length".into(),
                tokens_generated: 1,
                text: "late".into(),
            },
            &mut session.app,
            &mut session.task,
        );
        assert!(session.task.is_some());
        assert!(session.app.messages.last().unwrap().text.is_empty());
        session.receive().await;
        assert_eq!(session.app.messages.last().unwrap().text, "reused");
        assert_eq!(
            session.app.messages.last().unwrap().state,
            MessageState::Complete
        );
        assert_eq!(session.app.request, RequestState::Idle);
        assert!(session.task.is_none());
        bounded(server.server).await.unwrap();
        assert!(session.rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn escape_closes_a_header_only_stream_and_allows_the_next_prompt() {
        escape_and_reuse("").await;
    }

    #[tokio::test]
    async fn escape_keeps_partial_text_cancelled_and_rejects_stale_events() {
        escape_and_reuse("event: token\ndata: {\"token_id\":1,\"text\":\"first\"}\n\n").await;
    }

    #[tokio::test]
    async fn quit_waits_for_a_stalled_generation_to_drop_its_socket() {
        let server = stall("", false).await;
        let mut session = Session::new(&server.base);
        session.submit();
        bounded(server.ready).await.unwrap();
        assert!(session.key(KeyCode::Char('c'), KeyModifiers::CONTROL));
        // The same cleanup invoked by run(), before terminal teardown.
        bounded(stop_generation(&mut session.task)).await;
        bounded(server.closed).await.unwrap();
        bounded(server.server).await.unwrap();
        drop(session.tx);
        assert!(
            bounded(session.rx.recv()).await.is_none(),
            "generation retained an event sender"
        );
    }

    #[tokio::test]
    async fn losing_the_app_receiver_drops_a_stalled_generation() {
        let server = stall("", false).await;
        let mut session = Session::new(&server.base);
        session.submit();
        bounded(server.ready).await.unwrap();
        drop(session.rx);
        bounded(session.task.take().unwrap()).await.unwrap();
        bounded(server.closed).await.unwrap();
        bounded(server.server).await.unwrap();
    }

    #[tokio::test]
    async fn producer_completion_drains_tokens_and_every_terminal_message() {
        for terminal in [
            StreamMessage::Done {
                finish_reason: "length".into(),
                tokens_generated: 2,
                text: "終".into(),
            },
            StreamMessage::Failed(client::ClientError::Protocol("bad event".into())),
            StreamMessage::Ended,
        ] {
            let expected = format!("{terminal:?}");
            let (tx, rx) = mpsc::channel(4);
            let (out, mut events) = mpsc::channel(4);
            let producer = async move {
                tx.send(StreamMessage::Token {
                    token_id: 1,
                    text: "é".into(),
                })
                .await
                .unwrap();
                tx.send(StreamMessage::Token {
                    token_id: 2,
                    text: "雪🦀".into(),
                })
                .await
                .unwrap();
                tx.send(terminal).await.unwrap();
            };
            // All sends fit immediately. Producer completion wins while the
            // forwarding channel still holds tokens and its terminal message.
            bounded(tokio::spawn(forward_stream(producer, rx, out, 7)))
                .await
                .unwrap();
            let mut messages = Vec::new();
            while let Some(event) = bounded(events.recv()).await {
                match event {
                    AppEvent::Stream {
                        generation: 7,
                        message,
                    } => messages.push(format!("{message:?}")),
                    other => panic!("wrong generation: {other:?}"),
                }
            }
            assert_eq!(
                messages,
                vec![
                    "Token { token_id: 1, text: \"é\" }".to_string(),
                    "Token { token_id: 2, text: \"雪🦀\" }".to_string(),
                    expected
                ]
            );
        }
    }
}
