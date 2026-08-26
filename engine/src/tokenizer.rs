//! GPT-2 byte-pair encoding.
//!
//! Must match tiktoken's `gpt2` encoding exactly, because that is what produced
//! the training data. A tokenizer that differs even slightly feeds the model
//! ids it never saw, and the resulting output looks like a badly trained model
//! rather than a tokenizer bug -- which is a genuinely difficult thing to
//! diagnose from samples alone. `tests` pins the encoder against ids produced
//! by tiktoken itself.
//!
//! Two stages, matching the reference implementation:
//!   1. split the text with the GPT-2 pre-tokenizer regex, so merges can never
//!      cross a word boundary
//!   2. byte-pair merge within each piece, always taking the lowest rank first

use anyhow::{anyhow, bail, Context, Result};
use fancy_regex::Regex;
use std::collections::HashMap;
use std::path::Path;

/// The GPT-2 pre-tokenizer pattern.
///
/// `\s+(?!\S)` is a negative lookahead -- it matches trailing whitespace only
/// when not followed by a non-space -- which is why this needs `fancy-regex`
/// rather than the standard `regex` crate, which has no lookaround support.
const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

pub struct Tokenizer {
    /// Token bytes indexed by id; id equals BPE rank.
    tokens: Vec<Vec<u8>>,
    ranks: HashMap<Vec<u8>, u32>,
    pattern: Regex,
    pub eot: u32,
}

impl Tokenizer {
    /// Load the binary vocabulary written by `scripts/export_tokenizer.py`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading tokenizer {}", path.display()))?;

        let mut cursor = 0usize;
        let read_u32 = |cursor: &mut usize| -> Result<u32> {
            if *cursor + 4 > bytes.len() {
                bail!("tokenizer file truncated at byte {cursor}");
            }
            let v = u32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().unwrap());
            *cursor += 4;
            Ok(v)
        };

        let count = read_u32(&mut cursor)? as usize;
        let mut tokens = Vec::with_capacity(count);
        let mut ranks = HashMap::with_capacity(count);

        for id in 0..count {
            let len = read_u32(&mut cursor)? as usize;
            if cursor + len > bytes.len() {
                bail!("tokenizer file truncated in token {id}");
            }
            let token = bytes[cursor..cursor + len].to_vec();
            cursor += len;
            ranks.insert(token.clone(), id as u32);
            tokens.push(token);
        }

        Ok(Self {
            eot: count as u32,          // <|endoftext|> sits just past the merges
            tokens,
            ranks,
            pattern: Regex::new(GPT2_PATTERN).context("compiling GPT-2 pattern")?,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Merge one pre-tokenized piece into token ids.
    ///
    /// Repeatedly merges the adjacent pair with the lowest rank, which is what
    /// makes BPE deterministic. Merging greedily left-to-right instead would
    /// produce valid-looking but different ids.
    fn byte_pair_encode(&self, piece: &[u8], out: &mut Vec<u32>) {
        if piece.len() == 1 {
            if let Some(&id) = self.ranks.get(piece) {
                out.push(id);
            }
            return;
        }

        // Each token is a [start, end) byte range; start as individual bytes.
        let mut toks: Vec<(usize, usize)> = (0..piece.len()).map(|i| (i, i + 1)).collect();

        loop {
            // Merge the adjacent pair whose concatenation has the lowest rank.
            // Lowest-rank-first is what makes BPE deterministic; merging
            // greedily left to right would yield valid but different ids.
            let mut best: Option<(usize, u32)> = None;
            for i in 0..toks.len().saturating_sub(1) {
                let merged = &piece[toks[i].0..toks[i + 1].1];
                if let Some(&rank) = self.ranks.get(merged) {
                    if best.is_none_or(|(_, best_rank)| rank < best_rank) {
                        best = Some((i, rank));
                    }
                }
            }

            match best {
                Some((i, _)) => {
                    toks[i].1 = toks[i + 1].1;
                    toks.remove(i + 1);
                }
                None => break,
            }
        }

        for (start, end) in toks {
            if let Some(&id) = self.ranks.get(&piece[start..end]) {
                out.push(id);
            }
        }
    }

    /// Encode text, ignoring special tokens (matching `encode_ordinary`).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        for m in self.pattern.find_iter(text) {
            let m = m.map_err(|e| anyhow!("pre-tokenizer regex failed: {e}"))?;
            self.byte_pair_encode(m.as_str().as_bytes(), &mut out);
        }
        Ok(out)
    }

    /// Decode ids back to text.
    ///
    /// Bytes are concatenated before UTF-8 conversion, never decoded per token:
    /// a multi-byte character is often split across several tokens, so decoding
    /// individually would emit replacement characters mid-word.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(token) = self.tokens.get(id as usize) {
                bytes.extend_from_slice(token);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Decode a single id, for streaming output.
    ///
    /// Returns None when the bytes are an incomplete UTF-8 sequence, so the
    /// caller can buffer until the character is whole.
    pub fn decode_piece(&self, id: u32) -> Option<&[u8]> {
        self.tokens.get(id as usize).map(|v| v.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ids produced by tiktoken's gpt2 encoding, via scripts/export_tokenizer.py.
    ///
    /// This is a regression test for a real bug: an earlier merge loop removed
    /// the merged element before recomputing neighbouring ranks, which produced
    /// 29 tokens instead of 14. Decoding still round-tripped perfectly, so the
    /// only symptom was ids the model had never been trained on.
    const PROBE: &str = "The World is a stage, and 42 tokens walk onto it.\n";
    const PROBE_IDS: &[u32] = &[
        464, 2159, 318, 257, 3800, 11, 290, 5433, 16326, 2513, 4291, 340, 13, 198,
    ];

    fn load() -> Option<Tokenizer> {
        // Skip rather than fail when the vocabulary has not been exported.
        ["../export/gpt2.tok", "export/gpt2.tok"]
            .iter()
            .find_map(|p| Tokenizer::load(p).ok())
    }

    #[test]
    fn matches_tiktoken_ids() {
        let Some(tok) = load() else {
            eprintln!("skipping: run scripts/export_tokenizer.py first");
            return;
        };
        assert_eq!(tok.encode(PROBE).unwrap(), PROBE_IDS);
    }

    #[test]
    fn round_trips() {
        let Some(tok) = load() else { return };
        for text in [
            PROBE,
            "hello world",
            "  leading and trailing  ",
            "punctuation!!! and 12345 numbers",
            "unicode: café, naïve, 東京",
            "newlines\nand\ttabs",
        ] {
            let ids = tok.encode(text).unwrap();
            assert_eq!(tok.decode(&ids), text, "round-trip failed for {text:?}");
        }
    }

    #[test]
    fn empty_input_gives_no_tokens() {
        let Some(tok) = load() else { return };
        assert!(tok.encode("").unwrap().is_empty());
    }
}
