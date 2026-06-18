//! Low-level whitespace and punctuation tokenizer.

use crate::Token;

/// Tokenize `text` into [`Token`]s by splitting on whitespace and
/// stripping trailing punctuation (`.`, `,`, `;`, `:`, `!`, `?`).
///
/// Each token carries its byte-offset span in the original string.
pub fn tokenize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    for mat in text.match_indices(|c: char| !c.is_whitespace())
    {
        // skip whitespace runs
        let _ = mat;
    }
    // Use a simple state machine: accumulate characters until whitespace.
    let mut start = None;
    let bytes = text.as_bytes();
    for i in 0..bytes.len()
    {
        if bytes[i].is_ascii_whitespace()
        {
            if let Some(s) = start.take()
            {
                tokens.push(Token {
                    text: text[s..i].to_string(),
                    start: s,
                    end: i,
                });
            }
        }
        else if start.is_none()
        {
            start = Some(i);
        }
    }
    if let Some(s) = start
    {
        tokens.push(Token {
            text: text[s..].to_string(),
            start: s,
            end: text.len(),
        });
    }
    tokens
}

/// Normalize a token: lowercase and strip trailing punctuation.
pub fn normalize_token(raw: &str) -> String {
    let mut s = raw.to_lowercase();
    while s.ends_with(['.', ',', ';', ':', '!', '?', ')', ']', '"'])
    {
        s.pop();
    }
    // strip leading punctuation / quotes
    while s.starts_with(['(', '[', '"', '\'', '`'])
    {
        s.remove(0);
    }
    s
}

/// Tokenize and normalize: returns lowercased, punctuation-stripped tokens.
pub fn tokenize_normalized(text: &str) -> Vec<String> {
    tokenize(text)
        .into_iter()
        .map(|t| normalize_token(&t.text))
        .filter(|t| !t.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Hello, world!");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "Hello,");
        assert_eq!(tokens[0].start, 0);
        assert_eq!(tokens[1].text, "world!");
    }

    #[test]
    fn test_normalize_token() {
        assert_eq!(normalize_token("Hello,"), "hello");
        assert_eq!(normalize_token("(world)"), "world");
        assert_eq!(normalize_token("\"test\""), "test");
    }

    #[test]
    fn test_tokenize_normalized() {
        let result = tokenize_normalized("The quick, brown fox.");
        assert_eq!(result, vec!["the", "quick", "brown", "fox"]);
    }

    #[test]
    fn test_empty_input() {
        let tokens = tokenize("");
        assert!(tokens.is_empty());
    }
}
