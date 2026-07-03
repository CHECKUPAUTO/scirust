//! A small indentation-aware lexer for the supported Python subset.
//!
//! It emits `Newline`/`Indent`/`Dedent` tokens (Python is indentation-based)
//! and suppresses newlines inside brackets so multi-line calls parse. Tabs are
//! rejected (ambiguous width) — the subset requires spaces.

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Int(i64),
    Float(f64),
    Name(String),
    /// An operator or punctuation symbol: `+ - * / ** ( ) [ ] , : . = < <= > >= ==`
    Sym(String),
    Newline,
    Indent,
    Dedent,
    Eof,
}

pub fn lex(src: &str) -> Result<Vec<Tok>, String> {
    let chars: Vec<char> = src.chars().collect();
    let n = chars.len();
    let mut i = 0usize;
    let mut toks = Vec::new();
    let mut indent_stack = vec![0usize];
    let mut bracket_depth = 0i32;
    let mut at_line_start = true;

    while i < n
    {
        if at_line_start && bracket_depth == 0
        {
            // Measure indentation of this logical line.
            let mut col = 0usize;
            let line_begin = i;
            while i < n && (chars[i] == ' ' || chars[i] == '\t')
            {
                if chars[i] == '\t'
                {
                    return Err("tabs are not supported in indentation; use spaces".into());
                }
                col += 1;
                i += 1;
            }
            // Blank line or comment-only line: skip without emitting indent tokens.
            if i >= n || chars[i] == '\n' || chars[i] == '#'
            {
                while i < n && chars[i] != '\n'
                {
                    i += 1;
                }
                if i < n
                {
                    i += 1; // consume '\n'
                }
                continue;
            }
            let _ = line_begin;
            let top = *indent_stack.last().unwrap();
            if col > top
            {
                indent_stack.push(col);
                toks.push(Tok::Indent);
            }
            else if col < top
            {
                while *indent_stack.last().unwrap() > col
                {
                    indent_stack.pop();
                    toks.push(Tok::Dedent);
                }
                if *indent_stack.last().unwrap() != col
                {
                    return Err("inconsistent dedent".into());
                }
            }
            at_line_start = false;
            continue;
        }

        let c = chars[i];
        match c
        {
            ' ' | '\t' =>
            {
                i += 1;
            },
            '#' =>
            {
                while i < n && chars[i] != '\n'
                {
                    i += 1;
                }
            },
            '\n' =>
            {
                i += 1;
                if bracket_depth == 0
                {
                    // Avoid duplicate/leading newlines.
                    if !matches!(toks.last(), Some(Tok::Newline) | None)
                    {
                        toks.push(Tok::Newline);
                    }
                    at_line_start = true;
                }
            },
            '(' | '[' =>
            {
                bracket_depth += 1;
                toks.push(Tok::Sym(c.to_string()));
                i += 1;
            },
            ')' | ']' =>
            {
                bracket_depth -= 1;
                if bracket_depth < 0
                {
                    return Err("unbalanced closing bracket".into());
                }
                toks.push(Tok::Sym(c.to_string()));
                i += 1;
            },
            '*' =>
            {
                // '**' or '*'
                if i + 1 < n && chars[i + 1] == '*'
                {
                    toks.push(Tok::Sym("**".into()));
                    i += 2;
                }
                else
                {
                    toks.push(Tok::Sym("*".into()));
                    i += 1;
                }
            },
            '<' | '>' | '=' | '!' =>
            {
                if i + 1 < n && chars[i + 1] == '='
                {
                    toks.push(Tok::Sym(format!("{}=", c)));
                    i += 2;
                }
                else
                {
                    toks.push(Tok::Sym(c.to_string()));
                    i += 1;
                }
            },
            '+' | '-' | '/' | ',' | ':' | '.' =>
            {
                toks.push(Tok::Sym(c.to_string()));
                i += 1;
            },
            '"' | '\'' =>
            {
                // String literal — only meaningful as a type hint like "np.ndarray".
                let quote = c;
                i += 1;
                let start = i;
                while i < n && chars[i] != quote
                {
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                if i < n
                {
                    i += 1; // closing quote
                }
                // Represent as a Name so the parser can read it as a hint.
                toks.push(Tok::Name(format!("\u{1}STR:{}", s)));
            },
            c if c.is_ascii_digit() || (c == '.' && i + 1 < n && chars[i + 1].is_ascii_digit()) =>
            {
                let start = i;
                let mut is_float = false;
                while i < n
                    && (chars[i].is_ascii_digit()
                        || chars[i] == '.'
                        || chars[i] == 'e'
                        || chars[i] == 'E'
                        || ((chars[i] == '+' || chars[i] == '-')
                            && i > start
                            && (chars[i - 1] == 'e' || chars[i - 1] == 'E')))
                {
                    if chars[i] == '.' || chars[i] == 'e' || chars[i] == 'E'
                    {
                        is_float = true;
                    }
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                if is_float
                {
                    let v: f64 = s.parse().map_err(|_| format!("bad float literal: {}", s))?;
                    toks.push(Tok::Float(v));
                }
                else
                {
                    let v: i64 = s.parse().map_err(|_| format!("bad int literal: {}", s))?;
                    toks.push(Tok::Int(v));
                }
            },
            c if c.is_alphabetic() || c == '_' =>
            {
                let start = i;
                while i < n && (chars[i].is_alphanumeric() || chars[i] == '_')
                {
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                toks.push(Tok::Name(s));
            },
            other => return Err(format!("unexpected character: {:?}", other)),
        }
    }

    // Close out: final newline, then dangling dedents.
    if !matches!(toks.last(), Some(Tok::Newline) | None)
    {
        toks.push(Tok::Newline);
    }
    while indent_stack.len() > 1
    {
        indent_stack.pop();
        toks.push(Tok::Dedent);
    }
    toks.push(Tok::Eof);
    Ok(toks)
}
