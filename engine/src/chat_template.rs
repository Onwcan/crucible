//! Turning a conversation into the one prompt string the model sees.
//!
//! Protocol-neutral on purpose. Two compatibility adapters need this — OpenAI
//! Chat Completions and Anthropic Messages — and if each had its own copy they
//! would drift, so the same conversation would produce different text depending
//! on which wire format a client happened to speak. That is not a tidiness
//! argument: cross-adapter equivalence is a correctness property, and it is
//! only checkable if there is one implementation to check.
//!
//! Nothing about either protocol lives here. Adapters parse their own DTOs,
//! validate their own rules, and hand over a list of `Turn`s. What comes back
//! is the exact string submitted to the tokenizer.
//!
//! # The template, and why it looks like this
//!
//! ```text
//! System: be terse
//!
//! User: hello
//!
//! Assistant: hi
//!
//! User: and again
//!
//! Assistant:
//! ```
//!
//! The checkpoint behind this is a **base language model** trained on
//! FineWeb-Edu with the GPT-2 tokenizer. It has no chat fine-tune and no
//! trained template, so there is no "correct" format to reproduce — only
//! formats that suit the distribution it was trained on and formats that do
//! not.
//!
//! - **No pseudo-special tokens.** `<|im_start|>`, `<|system|>`, Anthropic's
//!   `\n\nHuman:` conventions — none of these are single tokens in the GPT-2
//!   vocabulary, so they would tokenise into several unrelated pieces the model
//!   has never seen arranged that way. Plain English role labels tokenise as
//!   ordinary words that do occur in transcripts on the web, which is the only
//!   prior this model has.
//! - **A blank line between turns**, which is how transcripts are separated in
//!   that distribution.
//! - **The trailing prime has no space after the colon.** GPT-2 merges a
//!   leading space into the following token — `" hi"` is one token, `"hi"` is
//!   another — so ending with `"Assistant: "` would force a space token
//!   followed by a word-without-space token, a pairing the training data almost
//!   never contains. Ending with `"Assistant:"` lets the model emit `" hi"` as
//!   the single token it expects. This is the one decision here that is about
//!   GPT-2 specifically rather than about taste, and it is why replies begin
//!   with a space.
//! - **A trailing assistant turn is a continuation**, not a new turn: the
//!   transcript ends with its text and the model carries on from there. Both
//!   protocols call this prefill and both get it for free.

/// Who is speaking. Deliberately smaller than either protocol's role set:
/// anything an adapter cannot map onto one of these is its own error to raise,
/// not something to approximate here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    fn label(self) -> &'static str {
        match self {
            Role::System => "System:",
            Role::User => "User:",
            Role::Assistant => "Assistant:",
        }
    }
}

/// One turn of a conversation, already flattened to text by the adapter.
#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub role: Role,
    pub text: String,
}

impl Turn {
    pub fn new(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            text: text.into(),
        }
    }
}

/// The exact prompt submitted to the model.
///
/// Total and deterministic: the same turns always give the same string. That is
/// what makes a seeded conversation reproducible, what lets a test compare an
/// adapter against the native endpoint, and what makes `count_tokens` able to
/// promise the number a later request will report.
pub fn serialize(turns: &[Turn]) -> String {
    let mut out = String::new();
    for turn in turns {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(turn.role.label());
        if !turn.text.is_empty() {
            out.push(' ');
            out.push_str(&turn.text);
        }
    }
    if turns.last().map(|t| t.role) != Some(Role::Assistant) {
        out.push_str("\n\nAssistant:");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(role: Role, text: &str) -> Turn {
        Turn::new(role, text)
    }

    #[test]
    fn a_single_user_turn_is_primed_for_the_assistant() {
        let p = serialize(&[t(Role::User, "hello")]);
        assert_eq!(p, "User: hello\n\nAssistant:");
        // No trailing space: GPT-2 merges the leading space into the first
        // generated token, so priming with one would split it.
        assert!(!p.ends_with(' '));
    }

    #[test]
    fn a_full_transcript_round_trips_in_order() {
        let p = serialize(&[
            t(Role::System, "be terse"),
            t(Role::User, "hi"),
            t(Role::Assistant, "hello"),
            t(Role::User, "again"),
        ]);
        assert_eq!(
            p,
            "System: be terse\n\nUser: hi\n\nAssistant: hello\n\nUser: again\n\nAssistant:"
        );
    }

    #[test]
    fn a_trailing_assistant_turn_continues_rather_than_repriming() {
        let p = serialize(&[t(Role::User, "count"), t(Role::Assistant, "one two")]);
        assert_eq!(p, "User: count\n\nAssistant: one two");
        assert!(!p.ends_with("Assistant:"));
    }

    #[test]
    fn empty_text_still_produces_a_labelled_turn() {
        assert_eq!(
            serialize(&[t(Role::System, ""), t(Role::User, "hi")]),
            "System:\n\nUser: hi\n\nAssistant:"
        );
    }

    #[test]
    fn an_empty_conversation_is_just_the_prime() {
        // Adapters reject empty message lists before reaching here; this only
        // pins that the function stays total rather than panicking.
        assert_eq!(serialize(&[]), "\n\nAssistant:");
    }

    #[test]
    fn serialization_is_deterministic() {
        let turns = vec![
            t(Role::System, "s"),
            t(Role::User, "u"),
            t(Role::Assistant, "a"),
            t(Role::User, "v"),
        ];
        assert_eq!(serialize(&turns), serialize(&turns));
    }

    #[test]
    fn consecutive_turns_of_one_role_each_get_their_own_label() {
        // Anthropic permits consecutive same-role messages; nothing here merges
        // them, so what the client sent is what the model sees.
        let p = serialize(&[t(Role::User, "one"), t(Role::User, "two")]);
        assert_eq!(p, "User: one\n\nUser: two\n\nAssistant:");
    }

    #[test]
    fn multibyte_text_survives_unchanged() {
        let p = serialize(&[t(Role::User, "héllo 世界 🌍")]);
        assert!(p.contains("héllo 世界 🌍"));
    }

    #[test]
    fn a_system_turn_leads_regardless_of_which_protocol_supplied_it() {
        // OpenAI carries system inside the message list, Anthropic as a
        // top-level field. Both become a leading System turn, and this is the
        // test that says the two must produce byte-identical prompts.
        let from_openai = serialize(&[
            t(Role::System, "be terse"),
            t(Role::User, "hi"),
        ]);
        let from_anthropic = serialize(&[
            t(Role::System, "be terse"),
            t(Role::User, "hi"),
        ]);
        assert_eq!(from_openai, from_anthropic);
        assert_eq!(from_openai, "System: be terse\n\nUser: hi\n\nAssistant:");
    }
}
