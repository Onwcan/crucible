//! Application state for the TUI.
//!
//! Deliberately free of network, terminal and CUDA types: every transition here
//! is a pure function of the previous state and one event, so the whole of it
//! is testable with `cargo test` and no GPU, no server and no terminal.
//!
//! Rendering reads this; tasks never mutate widgets.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::protocol::{Health, Metrics};

/// Who produced a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    System,
}

/// Where a message is in its lifecycle. An enum rather than a pile of booleans
/// because "streaming and also failed" is not a state that should be
/// representable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageState {
    Complete,
    Streaming,
    Cancelled,
    Failed,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub text: String,
    pub state: MessageState,
    pub tokens: usize,
    pub finish_reason: Option<String>,
    pub error: Option<String>,
}

impl Message {
    pub fn user(text: String) -> Self {
        Self {
            role: Role::User,
            text,
            state: MessageState::Complete,
            tokens: 0,
            finish_reason: None,
            error: None,
        }
    }

    pub fn assistant_streaming() -> Self {
        Self {
            role: Role::Assistant,
            text: String::new(),
            state: MessageState::Streaming,
            tokens: 0,
            finish_reason: None,
            error: None,
        }
    }

    pub fn system(text: String) -> Self {
        Self {
            role: Role::System,
            text,
            state: MessageState::Complete,
            tokens: 0,
            finish_reason: None,
            error: None,
        }
    }
}

/// Connection to the service.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnState {
    Connected,
    Reconnecting,
    Disconnected,
}

impl ConnState {
    pub fn marker(self) -> &'static str {
        match self {
            ConnState::Connected => "●",
            ConnState::Reconnecting => "○",
            ConnState::Disconnected => "×",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ConnState::Connected => "connected",
            ConnState::Reconnecting => "reconnecting",
            ConnState::Disconnected => "disconnected",
        }
    }
}

/// Where the current generation is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestState {
    Idle,
    Submitting,
    Streaming,
    /// The stream has been dropped and the server has been left to notice.
    Cancelling,
    Error,
}

impl RequestState {
    pub fn is_busy(&self) -> bool {
        matches!(self, RequestState::Submitting | RequestState::Streaming | RequestState::Cancelling)
    }
}

/// Client-observed timing for one generation.
///
/// Everything here is measured at the HTTP/SSE boundary, not on the GPU. TTFT
/// includes queueing, prompt prefill and the network round trip, so it is
/// deliberately not comparable to a kernel timing.
#[derive(Debug, Clone, Default)]
pub struct GenStats {
    pub submitted_at: Option<Instant>,
    pub first_token_at: Option<Instant>,
    pub last_token_at: Option<Instant>,
    pub tokens: usize,
    /// Recent inter-token gaps. Bounded so a long generation cannot grow it
    /// without limit.
    gaps: VecDeque<Duration>,
    pub finish_reason: Option<String>,
}

const GAP_WINDOW: usize = 64;

impl GenStats {
    pub fn start(&mut self) {
        *self = Self {
            submitted_at: Some(Instant::now()),
            ..Default::default()
        };
    }

    pub fn on_token(&mut self) {
        let now = Instant::now();
        if self.first_token_at.is_none() {
            self.first_token_at = Some(now);
        } else if let Some(prev) = self.last_token_at {
            if self.gaps.len() == GAP_WINDOW {
                self.gaps.pop_front();
            }
            self.gaps.push_back(now - prev);
        }
        self.last_token_at = Some(now);
        self.tokens += 1;
    }

    /// Time to first token, client-observed.
    pub fn ttft(&self) -> Option<Duration> {
        match (self.submitted_at, self.first_token_at) {
            (Some(a), Some(b)) => Some(b - a),
            _ => None,
        }
    }

    /// Elapsed generation time, measured from the first token so it excludes
    /// prefill and queueing.
    pub fn generating(&self) -> Option<Duration> {
        match (self.first_token_at, self.last_token_at) {
            (Some(a), Some(b)) => Some(b - a),
            _ => None,
        }
    }

    /// Tokens per second, averaged over the generation so far rather than
    /// computed from the last gap.
    ///
    /// An instantaneous rate at these speeds jumps between hundreds and
    /// thousands between adjacent frames and reads as noise; a running average
    /// after the first token is stable and still responsive.
    pub fn tokens_per_second(&self) -> Option<f64> {
        let elapsed = self.generating()?.as_secs_f64();
        if elapsed <= 0.0 || self.tokens < 2 {
            return None;
        }
        Some((self.tokens - 1) as f64 / elapsed)
    }

    /// Median inter-token gap over the recent window.
    pub fn median_gap(&self) -> Option<Duration> {
        if self.gaps.is_empty() {
            return None;
        }
        let mut v: Vec<Duration> = self.gaps.iter().copied().collect();
        v.sort();
        Some(v[v.len() / 2])
    }
}

/// A single-line text editor with char-indexed cursor.
///
/// Cursor positions are character indices, never byte offsets: indexing a Rust
/// string by bytes splits multi-byte characters and panics, and a prompt with
/// an accent or an emoji in it is not an edge case.
#[derive(Debug, Default, Clone)]
pub struct Input {
    text: String,
    cursor: usize,
}

impl Input {
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn char_len(&self) -> usize {
        self.text.chars().count()
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    fn byte_at(&self, char_idx: usize) -> usize {
        self.text
            .char_indices()
            .nth(char_idx)
            .map(|(b, _)| b)
            .unwrap_or(self.text.len())
    }

    pub fn insert(&mut self, c: char) {
        let b = self.byte_at(self.cursor);
        self.text.insert(b, c);
        self.cursor += 1;
    }

    pub fn insert_str(&mut self, s: &str) {
        for c in s.chars() {
            self.insert(c);
        }
    }

    pub fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let start = self.byte_at(self.cursor - 1);
        let end = self.byte_at(self.cursor);
        self.text.replace_range(start..end, "");
        self.cursor -= 1;
    }

    pub fn delete(&mut self) {
        if self.cursor >= self.char_len() {
            return;
        }
        let start = self.byte_at(self.cursor);
        let end = self.byte_at(self.cursor + 1);
        self.text.replace_range(start..end, "");
    }

    pub fn left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    pub fn right(&mut self) {
        if self.cursor < self.char_len() {
            self.cursor += 1;
        }
    }

    pub fn home(&mut self) {
        self.cursor = 0;
    }

    pub fn end(&mut self) {
        self.cursor = self.char_len();
    }

    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
    }

    pub fn take(&mut self) -> String {
        let out = std::mem::take(&mut self.text);
        self.cursor = 0;
        out
    }
}

pub struct App {
    pub server: String,
    pub messages: Vec<Message>,
    pub input: Input,
    pub conn: ConnState,
    pub request: RequestState,
    pub health: Option<Health>,
    pub metrics: Option<Metrics>,
    pub stats: GenStats,
    /// Transient status or error line. Not a modal: the app stays usable.
    pub status: Option<String>,
    pub show_help: bool,
    pub show_telemetry: bool,
    /// Lines scrolled up from the bottom. 0 means pinned to the newest text.
    pub scroll: usize,
    /// Whether new tokens should keep the view at the bottom.
    pub follow: bool,
    pub should_quit: bool,
    /// Index of the assistant message currently being written into.
    active: Option<usize>,
    pub max_tokens: usize,
}

impl App {
    pub fn new(server: String, max_tokens: usize) -> Self {
        Self {
            server,
            messages: Vec::new(),
            input: Input::default(),
            conn: ConnState::Reconnecting,
            request: RequestState::Idle,
            health: None,
            metrics: None,
            stats: GenStats::default(),
            status: None,
            show_help: false,
            show_telemetry: false,
            scroll: 0,
            follow: true,
            should_quit: false,
            active: None,
            max_tokens,
        }
    }

    pub fn can_submit(&self) -> bool {
        !self.request.is_busy()
            && !self.input.text().trim().is_empty()
            && self.conn == ConnState::Connected
    }

    /// Move the typed text into the conversation and open an assistant message.
    ///
    /// Returns the prompt to send, or None when submission is not allowed.
    pub fn submit(&mut self) -> Option<String> {
        if !self.can_submit() {
            return None;
        }
        let prompt = self.input.take();
        self.messages.push(Message::user(prompt.clone()));
        self.messages.push(Message::assistant_streaming());
        self.active = Some(self.messages.len() - 1);
        self.request = RequestState::Submitting;
        self.stats.start();
        self.status = None;
        // A new prompt always returns the view to the bottom.
        self.follow = true;
        self.scroll = 0;
        Some(prompt)
    }

    pub fn on_token(&mut self, text: &str) {
        self.request = RequestState::Streaming;
        self.stats.on_token();
        if let Some(i) = self.active {
            let m = &mut self.messages[i];
            m.text.push_str(text);
            m.tokens += 1;
        }
    }

    pub fn on_done(&mut self, finish_reason: String, tokens: usize, tail: &str) {
        if let Some(i) = self.active {
            let m = &mut self.messages[i];
            m.text.push_str(tail);
            m.tokens = tokens;
            m.state = if finish_reason == "cancelled" {
                MessageState::Cancelled
            } else {
                MessageState::Complete
            };
            m.finish_reason = Some(finish_reason.clone());
        }
        self.stats.finish_reason = Some(finish_reason);
        self.request = RequestState::Idle;
        self.active = None;
    }

    /// A stream-level failure. Whatever was already streamed is kept: losing
    /// the user's output to report an error would be a worse outcome than the
    /// error itself.
    pub fn on_stream_error(&mut self, err: String) {
        if let Some(i) = self.active {
            let m = &mut self.messages[i];
            m.state = MessageState::Failed;
            m.error = Some(err.clone());
        }
        self.status = Some(err);
        self.request = RequestState::Idle;
        self.active = None;
    }

    /// The user asked to cancel. The HTTP stream is dropped by the caller; the
    /// server notices the disconnect and reclaims the request at its next
    /// scheduler boundary, so there is no second cancellation protocol.
    pub fn begin_cancel(&mut self) -> bool {
        if !matches!(self.request, RequestState::Streaming | RequestState::Submitting) {
            return false;
        }
        self.request = RequestState::Cancelling;
        true
    }

    /// The stream ended without a `done` event, which is what a cancellation or
    /// a dropped connection looks like from here.
    pub fn on_stream_ended(&mut self) {
        let cancelling = self.request == RequestState::Cancelling;
        if let Some(i) = self.active {
            let m = &mut self.messages[i];
            m.state = if cancelling {
                MessageState::Cancelled
            } else {
                MessageState::Failed
            };
            m.finish_reason = Some(if cancelling { "cancelled" } else { "disconnected" }.into());
            if !cancelling {
                m.error = Some("stream disconnected".into());
            }
        }
        self.status = Some(
            if cancelling { "Generation cancelled" } else { "Stream disconnected" }.into(),
        );
        self.request = RequestState::Idle;
        self.active = None;
    }

    pub fn on_health(&mut self, health: Health) {
        self.health = Some(health);
        self.conn = ConnState::Connected;
        if self.status.as_deref() == Some("Server unavailable") {
            self.status = None;
        }
    }

    pub fn on_metrics(&mut self, metrics: Metrics) {
        self.metrics = Some(metrics);
        // Metrics arriving is itself evidence the server is reachable, but it
        // must not promote a connection that has never completed a health
        // fetch: the client would then claim to know a model it has not seen.
        if self.conn == ConnState::Reconnecting && self.health.is_some() {
            self.conn = ConnState::Connected;
        }
    }

    /// A polling failure. One miss must not take the app down or spam the user.
    pub fn on_poll_failure(&mut self, err: String) {
        if self.conn == ConnState::Connected {
            self.conn = ConnState::Reconnecting;
            self.status = Some(err);
        } else {
            self.conn = ConnState::Disconnected;
        }
    }

    pub fn scroll_up(&mut self, lines: usize) {
        self.scroll = self.scroll.saturating_add(lines);
        // Manual scrolling turns off auto-follow, otherwise the next token
        // would yank the view back down mid-read.
        self.follow = false;
    }

    pub fn scroll_down(&mut self, lines: usize) {
        self.scroll = self.scroll.saturating_sub(lines);
        if self.scroll == 0 {
            self.follow = true;
        }
    }

    pub fn scroll_to_bottom(&mut self) {
        self.scroll = 0;
        self.follow = true;
    }

    pub fn toggle_help(&mut self) {
        self.show_help = !self.show_help;
    }

    pub fn toggle_telemetry(&mut self) {
        self.show_telemetry = !self.show_telemetry;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn connected_app() -> App {
        let mut a = App::new("http://127.0.0.1:8080".into(), 128);
        a.on_health(Health {
            status: "ok".into(),
            model: "120m".into(),
            device: "test".into(),
            max_batch: 16,
            context: 1024,
            kv_pages: 312,
            sampling: "greedy".into(),
        });
        a
    }

    #[test]
    fn submission_requires_a_connection_and_some_text() {
        let mut a = App::new("s".into(), 64);
        a.input.insert_str("hello");
        assert!(a.submit().is_none(), "submitted while disconnected");

        let mut a = connected_app();
        assert!(a.submit().is_none(), "submitted an empty prompt");
        a.input.insert_str("   ");
        assert!(a.submit().is_none(), "submitted whitespace only");
    }

    #[test]
    fn submitting_adds_both_messages_and_arms_the_stream() {
        let mut a = connected_app();
        a.input.insert_str("hi there");
        let prompt = a.submit().unwrap();
        assert_eq!(prompt, "hi there");
        assert_eq!(a.messages.len(), 2);
        assert_eq!(a.messages[0].role, Role::User);
        assert_eq!(a.messages[1].role, Role::Assistant);
        assert_eq!(a.messages[1].state, MessageState::Streaming);
        assert_eq!(a.request, RequestState::Submitting);
        assert!(a.input.is_empty(), "input not cleared on submit");
    }

    #[test]
    fn cannot_submit_while_a_generation_is_running() {
        let mut a = connected_app();
        a.input.insert_str("one");
        a.submit().unwrap();
        a.input.insert_str("two");
        assert!(a.submit().is_none());
    }

    #[test]
    fn tokens_accumulate_into_the_active_message() {
        let mut a = connected_app();
        a.input.insert_str("q");
        a.submit().unwrap();
        a.on_token("Hello");
        a.on_token(", ");
        a.on_token("world");
        assert_eq!(a.request, RequestState::Streaming);
        assert_eq!(a.messages[1].text, "Hello, world");
        assert_eq!(a.messages[1].tokens, 3);
        assert_eq!(a.stats.tokens, 3);
    }

    #[test]
    fn done_completes_the_message_and_appends_the_tail() {
        let mut a = connected_app();
        a.input.insert_str("q");
        a.submit().unwrap();
        a.on_token("ab");
        a.on_done("length".into(), 3, "!");
        assert_eq!(a.messages[1].text, "ab!");
        assert_eq!(a.messages[1].state, MessageState::Complete);
        assert_eq!(a.messages[1].finish_reason.as_deref(), Some("length"));
        assert_eq!(a.request, RequestState::Idle);
        // Ready for another prompt.
        a.input.insert_str("next");
        assert!(a.can_submit());
    }

    #[test]
    fn a_cancelled_finish_is_visually_distinct_from_a_normal_one() {
        let mut a = connected_app();
        a.input.insert_str("q");
        a.submit().unwrap();
        a.on_token("partial");
        a.on_done("cancelled".into(), 1, "");
        assert_eq!(a.messages[1].state, MessageState::Cancelled);
        assert_eq!(a.messages[1].text, "partial");
    }

    #[test]
    fn stream_error_keeps_text_already_received() {
        let mut a = connected_app();
        a.input.insert_str("q");
        a.submit().unwrap();
        a.on_token("kept");
        a.on_stream_error("server exploded".into());
        assert_eq!(a.messages[1].text, "kept", "streamed text was discarded");
        assert_eq!(a.messages[1].state, MessageState::Failed);
        assert_eq!(a.messages[1].error.as_deref(), Some("server exploded"));
        assert_eq!(a.request, RequestState::Idle, "app left unusable after error");
    }

    #[test]
    fn cancel_then_stream_end_marks_cancelled_not_failed() {
        let mut a = connected_app();
        a.input.insert_str("q");
        a.submit().unwrap();
        a.on_token("some");
        assert!(a.begin_cancel());
        assert_eq!(a.request, RequestState::Cancelling);
        a.on_stream_ended();
        assert_eq!(a.messages[1].state, MessageState::Cancelled);
        assert!(a.messages[1].error.is_none());
        assert_eq!(a.request, RequestState::Idle);
    }

    #[test]
    fn an_unexpected_stream_end_is_a_failure() {
        let mut a = connected_app();
        a.input.insert_str("q");
        a.submit().unwrap();
        a.on_token("some");
        a.on_stream_ended();
        assert_eq!(a.messages[1].state, MessageState::Failed);
        assert_eq!(a.messages[1].error.as_deref(), Some("stream disconnected"));
    }

    #[test]
    fn cancelling_when_idle_does_nothing() {
        let mut a = connected_app();
        assert!(!a.begin_cancel());
        assert_eq!(a.request, RequestState::Idle);
    }

    #[test]
    fn a_single_poll_failure_degrades_rather_than_disconnects() {
        let mut a = connected_app();
        assert_eq!(a.conn, ConnState::Connected);
        a.on_poll_failure("connection refused".into());
        assert_eq!(a.conn, ConnState::Reconnecting);
        a.on_poll_failure("connection refused".into());
        assert_eq!(a.conn, ConnState::Disconnected);
    }

    #[test]
    fn health_after_a_failure_restores_the_connection() {
        let mut a = connected_app();
        a.on_poll_failure("boom".into());
        a.on_poll_failure("boom".into());
        assert_eq!(a.conn, ConnState::Disconnected);
        a.on_health(Health {
            status: "ok".into(),
            model: "120m".into(),
            device: "test".into(),
            max_batch: 16,
            context: 1024,
            kv_pages: 312,
            sampling: "greedy".into(),
        });
        assert_eq!(a.conn, ConnState::Connected);
    }

    #[test]
    fn metrics_alone_do_not_claim_a_connection() {
        // Without a health fetch the client knows nothing about the model, so
        // reporting "connected" would be claiming more than it has.
        let mut a = App::new("s".into(), 64);
        a.on_metrics(Metrics::default());
        assert_eq!(a.conn, ConnState::Reconnecting);
    }

    #[test]
    fn scrolling_up_disables_follow_and_returning_to_bottom_restores_it() {
        let mut a = connected_app();
        assert!(a.follow);
        a.scroll_up(5);
        assert!(!a.follow);
        assert_eq!(a.scroll, 5);
        a.scroll_down(2);
        assert!(!a.follow, "follow re-enabled before reaching the bottom");
        a.scroll_down(10);
        assert_eq!(a.scroll, 0);
        assert!(a.follow);
    }

    #[test]
    fn a_new_prompt_re_enables_follow() {
        let mut a = connected_app();
        a.scroll_up(20);
        assert!(!a.follow);
        a.input.insert_str("hello");
        a.submit().unwrap();
        assert!(a.follow);
        assert_eq!(a.scroll, 0);
    }

    // --- input editor ---

    #[test]
    fn input_edits_by_character_not_byte() {
        let mut i = Input::default();
        // Accents, CJK and an emoji: 2, 3 and 4 byte characters.
        i.insert_str("aé中🚀");
        assert_eq!(i.char_len(), 4);
        assert_eq!(i.cursor(), 4);

        i.backspace();
        assert_eq!(i.text(), "aé中");
        i.left();
        i.left();
        assert_eq!(i.cursor(), 1);
        i.insert('X');
        assert_eq!(i.text(), "aXé中");
        i.delete();
        assert_eq!(i.text(), "aX中");
    }

    #[test]
    fn input_cursor_movement_is_bounded() {
        let mut i = Input::default();
        i.insert_str("ab");
        i.right();
        i.right();
        assert_eq!(i.cursor(), 2);
        i.home();
        assert_eq!(i.cursor(), 0);
        i.left();
        assert_eq!(i.cursor(), 0, "cursor moved before the start");
        i.backspace();
        assert_eq!(i.text(), "ab", "backspace at the start deleted something");
        i.end();
        assert_eq!(i.cursor(), 2);
        i.delete();
        assert_eq!(i.text(), "ab", "delete at the end removed something");
    }

    #[test]
    fn clearing_input_resets_the_cursor() {
        let mut i = Input::default();
        i.insert_str("hello");
        i.clear();
        assert!(i.is_empty());
        assert_eq!(i.cursor(), 0);
    }

    // --- stats ---

    #[test]
    fn throughput_is_unavailable_until_there_are_two_tokens() {
        let mut s = GenStats::default();
        s.start();
        assert!(s.tokens_per_second().is_none());
        s.on_token();
        assert!(s.ttft().is_some());
        assert!(s.tokens_per_second().is_none(), "rate reported from one token");
        s.on_token();
        assert_eq!(s.tokens, 2);
    }

    #[test]
    fn the_gap_window_is_bounded() {
        let mut s = GenStats::default();
        s.start();
        for _ in 0..(GAP_WINDOW + 50) {
            s.on_token();
        }
        assert!(s.gaps.len() <= GAP_WINDOW);
        assert_eq!(s.tokens, GAP_WINDOW + 50);
    }
}
