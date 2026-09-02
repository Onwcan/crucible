//! Rendering. Reads `App`, never mutates it.
//!
//! Styling is kept to bold, dim and the sixteen base colours. Truecolour is not
//! assumed: a terminal that lacks it should render a slightly plainer version
//! of the same layout, not an unreadable one.

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};

use super::app::{App, ConnState, MessageState, RequestState, Role, SettingField};

/// Below this the layout stops being useful and a message is shown instead.
const MIN_WIDTH: u16 = 40;
const MIN_HEIGHT: u16 = 12;

pub fn draw(f: &mut Frame, app: &mut App) {
    let area = f.area();
    if area.width < MIN_WIDTH || area.height < MIN_HEIGHT {
        let msg = format!(
            "Terminal too small\n{}x{} — need at least {MIN_WIDTH}x{MIN_HEIGHT}",
            area.width, area.height
        );
        f.render_widget(
            Paragraph::new(msg)
                .alignment(Alignment::Center)
                .wrap(Wrap { trim: true }),
            area,
        );
        return;
    }

    let telemetry_height = if app.show_telemetry { 6 } else { 0 };
    let chunks = Layout::vertical([
        Constraint::Length(1),                  // header
        Constraint::Min(3),                     // conversation
        Constraint::Length(3),                  // input
        Constraint::Length(telemetry_height),   // telemetry
        Constraint::Length(1),                  // status / hints
    ])
    .split(area);

    draw_header(f, chunks[0], app);
    draw_conversation(f, chunks[1], app);
    draw_input(f, chunks[2], app);
    if app.show_telemetry {
        draw_telemetry(f, chunks[3], app);
    }
    draw_status(f, chunks[4], app);

    if app.show_settings {
        draw_settings(f, area, app);
    }
    if app.show_help {
        draw_help(f, area, app);
    }
}

/// Generation settings overlay. Small and modal-ish rather than a screen of
/// its own: the chat is the application, this is a corner of it.
fn draw_settings(f: &mut Frame, area: Rect, app: &App) {
    let w = 48.min(area.width.saturating_sub(4));
    let h = 9.min(area.height.saturating_sub(2));
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(w)) / 2,
        y: area.y + (area.height.saturating_sub(h)) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, popup);

    let s = &app.settings;
    let row = |field: SettingField, label: &str, value: String| -> Line<'static> {
        let selected = app.settings_field == field;
        let marker = if selected { ">" } else { " " };
        let style = if selected {
            Style::default().add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        Line::from(vec![
            Span::styled(format!(" {marker} {label:<14}"), style.fg(Color::Cyan)),
            Span::styled(value, style),
        ])
    };

    let lines = vec![
        row(SettingField::Mode, "mode", if s.sample { "sample".into() } else { "greedy".into() }),
        row(SettingField::Temperature, "temperature", format!("{:.2}", s.temperature)),
        row(SettingField::TopK, "top-k", s.top_k.to_string()),
        row(SettingField::Seed, "seed", s.seed.to_string()),
        Line::from(""),
        Line::from(Span::styled(
            "  Up/Down select   Left/Right change   F3 close",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    f.render_widget(
        Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title(" generation ")),
        popup,
    );
}

fn conn_style(c: ConnState) -> Style {
    match c {
        ConnState::Connected => Style::default().fg(Color::Green),
        ConnState::Reconnecting => Style::default().fg(Color::Yellow),
        ConnState::Disconnected => Style::default().fg(Color::Red),
    }
}

fn draw_header(f: &mut Frame, area: Rect, app: &App) {
    let mut spans = vec![
        Span::styled("Crucible", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw("  "),
        Span::styled(app.conn.marker(), conn_style(app.conn)),
        Span::raw(" "),
        Span::styled(app.conn.label(), conn_style(app.conn)),
    ];

    if let Some(h) = &app.health {
        spans.push(Span::raw("   "));
        spans.push(Span::styled(
            h.model.clone(),
            Style::default().add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::raw("  "));
        let _ = &h.sampling;
        spans.push(Span::styled(
            format!("  max batch {}", h.max_batch),
            Style::default().fg(Color::DarkGray),
        ));
    }
    // What this client will actually ask for, which is more useful in a header
    // than what the server could in principle do.
    {
        let s = app.settings.summary();
        let style = if app.settings.sample {
            Style::default().fg(Color::Magenta)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        spans.push(Span::raw("  "));
        spans.push(Span::styled(s, style));
    }
    if let Some(m) = &app.metrics {
        spans.push(Span::styled(
            format!("  batch {}", m.last_batch_size),
            Style::default().fg(Color::DarkGray),
        ));
    }

    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// Build the conversation as wrapped lines.
///
/// Wrapping is done here rather than delegated to `Wrap`, because scrolling
/// needs to know how many lines exist: a Paragraph that wraps internally cannot
/// report that, so "scroll to the bottom" would be guesswork.
fn conversation_lines(app: &App, width: u16) -> Vec<Line<'static>> {
    let width = width.max(8) as usize;
    let mut out: Vec<Line> = Vec::new();

    for msg in &app.messages {
        let (label, label_style, body_style) = match msg.role {
            Role::User => (
                "You",
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                Style::default(),
            ),
            Role::Assistant => (
                "Crucible",
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
                Style::default(),
            ),
            Role::System => (
                "System",
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                Style::default().fg(Color::Yellow),
            ),
        };

        let mut header = vec![Span::styled(label, label_style)];
        match msg.state {
            MessageState::Cancelled => header.push(Span::styled(
                "  cancelled",
                Style::default().fg(Color::Yellow),
            )),
            MessageState::Failed => header.push(Span::styled(
                "  failed",
                Style::default().fg(Color::Red),
            )),
            _ => {}
        }
        out.push(Line::from(header));

        for para in msg.text.split('\n') {
            if para.is_empty() {
                out.push(Line::from(""));
                continue;
            }
            for chunk in wrap_text(para, width) {
                out.push(Line::from(Span::styled(chunk, body_style)));
            }
        }

        if let Some(err) = &msg.error {
            for chunk in wrap_text(err, width) {
                out.push(Line::from(Span::styled(
                    chunk,
                    Style::default().fg(Color::Red),
                )));
            }
        }
        out.push(Line::from(""));
    }
    out
}

/// Wrap on character count, breaking at spaces where possible.
///
/// Character count rather than byte length: a prompt containing accents or CJK
/// would otherwise wrap at the wrong column or split a character.
pub fn wrap_text(s: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![s.to_string()];
    }
    let mut lines = Vec::new();
    let mut line = String::new();
    let mut count = 0usize;

    for word in s.split_inclusive(' ') {
        let wlen = word.chars().count();
        if count + wlen > width && count > 0 {
            lines.push(std::mem::take(&mut line));
            count = 0;
        }
        // A single word longer than the line: hard-split it rather than
        // overflowing the pane.
        if wlen > width {
            for c in word.chars() {
                if count == width {
                    lines.push(std::mem::take(&mut line));
                    count = 0;
                }
                line.push(c);
                count += 1;
            }
        } else {
            line.push_str(word);
            count += wlen;
        }
    }
    if !line.is_empty() || lines.is_empty() {
        lines.push(line);
    }
    lines
}

fn draw_conversation(f: &mut Frame, area: Rect, app: &mut App) {
    let block = Block::default().borders(Borders::ALL).title(" conversation ");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let lines = conversation_lines(app, inner.width);
    let total = lines.len();
    let visible = inner.height as usize;

    // Clamp the scroll so it cannot run past the top of the history.
    let max_scroll = total.saturating_sub(visible);
    if app.scroll > max_scroll {
        app.scroll = max_scroll;
    }
    let offset = if app.follow {
        max_scroll
    } else {
        max_scroll.saturating_sub(app.scroll)
    };

    if total == 0 {
        let hint = Paragraph::new(Line::from(Span::styled(
            "Type a prompt and press Enter. F1 for help.",
            Style::default().fg(Color::DarkGray),
        )));
        f.render_widget(hint, inner);
        return;
    }

    f.render_widget(
        Paragraph::new(lines).scroll((offset as u16, 0)),
        inner,
    );

    // Only say the view is detached when it actually is.
    if !app.follow && app.scroll > 0 {
        let tag = " scrolled — End to follow ";
        let w = tag.chars().count() as u16;
        if inner.width > w {
            let r = Rect {
                x: inner.x + inner.width - w,
                y: inner.y,
                width: w,
                height: 1,
            };
            f.render_widget(
                Paragraph::new(Span::styled(tag, Style::default().fg(Color::Yellow))),
                r,
            );
        }
    }
}

fn draw_input(f: &mut Frame, area: Rect, app: &App) {
    let busy = app.request.is_busy();
    let title = match app.request {
        RequestState::Submitting => " waiting for first token ",
        RequestState::Streaming => " generating — Esc to cancel ",
        RequestState::Cancelling => " cancelling ",
        _ => " prompt ",
    };
    let border = if busy {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default()
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border)
        .title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let text = app.input.text();
    let shown: String = if text.is_empty() && !busy {
        String::new()
    } else {
        text.to_string()
    };
    let style = if busy {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default()
    };
    f.render_widget(Paragraph::new(Span::styled(shown, style)), inner);

    // Place the real terminal cursor so editing feels native.
    if !busy {
        let col = inner.x + (app.input.cursor() as u16).min(inner.width.saturating_sub(1));
        f.set_cursor_position((col, inner.y));
    }
}

fn draw_telemetry(f: &mut Frame, area: Rect, app: &App) {
    let cols = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    let mut left = vec![Line::from(Span::styled(
        "this request (client-observed)",
        Style::default().add_modifier(Modifier::BOLD),
    ))];
    let s = &app.stats;
    left.push(kv("TTFT", match s.ttft() {
        Some(d) => format!("{:.1} ms", d.as_secs_f64() * 1e3),
        None => "—".into(),
    }));
    left.push(kv("tok/s", match s.tokens_per_second() {
        Some(v) => format!("{v:.0}"),
        None => "—".into(),
    }));
    left.push(kv("gap median", match s.median_gap() {
        Some(d) => format!("{:.2} ms", d.as_secs_f64() * 1e3),
        None => "—".into(),
    }));
    left.push(kv("generated", s.tokens.to_string()));

    let mut right = vec![Line::from(Span::styled(
        "service",
        Style::default().add_modifier(Modifier::BOLD),
    ))];
    match &app.metrics {
        Some(m) => {
            right.push(kv("active / queued", format!("{} / {}", m.active_requests, m.queued_requests)));
            right.push(kv(
                "kv pages",
                format!("{} / {} ({:.0}%)", m.kv_pages_used, m.kv_total(), m.kv_usage() * 100.0),
            ));
            right.push(kv("batch (last / avg)", format!("{} / {:.1}", m.last_batch_size, m.average_batch_size)));
            right.push(kv(
                "tokens / uptime",
                format!("{} / {:.0}s", m.aggregate_tokens_generated, m.uptime_seconds),
            ));
        }
        None => right.push(Line::from(Span::styled(
            "no metrics yet",
            Style::default().fg(Color::DarkGray),
        ))),
    }

    f.render_widget(
        Paragraph::new(left).block(Block::default().borders(Borders::ALL)),
        cols[0],
    );
    f.render_widget(
        Paragraph::new(right).block(Block::default().borders(Borders::ALL)),
        cols[1],
    );
}

fn kv(k: &str, v: String) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{k:<20}"), Style::default().fg(Color::DarkGray)),
        Span::raw(v),
    ])
}

fn draw_status(f: &mut Frame, area: Rect, app: &App) {
    let line = match &app.status {
        Some(err) => Line::from(Span::styled(
            err.clone(),
            Style::default().fg(Color::Red),
        )),
        None => Line::from(Span::styled(
            "Enter send   Esc cancel   PgUp/PgDn scroll   F1 help   F2 telemetry   F3 settings   Ctrl+C quit",
            Style::default().fg(Color::DarkGray),
        )),
    };
    f.render_widget(Paragraph::new(line), area);
}

fn draw_help(f: &mut Frame, area: Rect, app: &App) {
    let w = 56.min(area.width.saturating_sub(4));
    let h = 16.min(area.height.saturating_sub(2));
    let popup = Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, popup);

    let rows = vec![
        ("Enter", "send prompt"),
        ("Alt+Enter", "insert newline"),
        ("Esc", "cancel active generation"),
        ("Ctrl+U", "clear input"),
        ("Left/Right", "move cursor"),
        ("Home/End", "start / end of input"),
        ("PgUp/PgDn", "scroll conversation"),
        ("F2", "toggle telemetry"),
        ("F3", "generation settings"),
        ("F1", "close this help"),
        ("Ctrl+C", "quit"),
    ];
    let mut lines: Vec<Line> = rows
        .into_iter()
        .map(|(k, v)| {
            Line::from(vec![
                Span::styled(format!("  {k:<12}"), Style::default().fg(Color::Cyan)),
                Span::raw(v),
            ])
        })
        .collect();
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!("  server  {}", app.server),
        Style::default().fg(Color::DarkGray),
    )));

    f.render_widget(
        Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title(" help ")),
        popup,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapping_breaks_at_spaces_within_the_width() {
        let out = wrap_text("the quick brown fox jumps", 10);
        assert!(out.iter().all(|l| l.chars().count() <= 10), "{out:?}");
        assert_eq!(out.concat().replace("  ", " ").trim(), "the quick brown fox jumps");
    }

    #[test]
    fn wrapping_hard_splits_a_word_longer_than_the_line() {
        let out = wrap_text("supercalifragilistic", 6);
        assert!(out.iter().all(|l| l.chars().count() <= 6), "{out:?}");
        assert_eq!(out.concat(), "supercalifragilistic");
    }

    #[test]
    fn wrapping_counts_characters_not_bytes() {
        // Each of these is multi-byte; wrapping by bytes would break early and
        // could split a character.
        let out = wrap_text("ééééé ééééé", 5);
        assert!(out.iter().all(|l| l.chars().count() <= 5 + 1), "{out:?}");
        assert!(out.concat().contains('é'));
    }

    #[test]
    fn wrapping_preserves_empty_input() {
        assert_eq!(wrap_text("", 10), vec!["".to_string()]);
    }
}
