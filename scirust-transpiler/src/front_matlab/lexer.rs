//! Token lexer for the supported MATLAB/Octave subset.
//!
//! Unlike Python, MATLAB is not indentation-based (blocks close with `end`), so
//! this lexer only needs statement terminators: a newline or `;` becomes a
//! [`MTok::Term`]. Newlines inside `( )` are suppressed. Comments are `%`.

#[derive(Debug, Clone, PartialEq)]
pub enum MTok {
    Num(f64),
    Ident(String),
    /// Operator / punctuation: `+ - * / ^ .* ./ .^ ( ) , : = < <= > >= == ~=`
    Sym(String),
    /// Statement terminator (`;` or newline).
    Term,
    Eof,
}

pub fn lex(src: &str) -> Result<Vec<MTok>, String> {
    let chars: Vec<char> = src.chars().collect();
    let n = chars.len();
    let mut i = 0usize;
    let mut toks: Vec<MTok> = Vec::new();
    let mut depth = 0i32;

    let push_term = |toks: &mut Vec<MTok>| {
        if !matches!(toks.last(), Some(MTok::Term) | None)
        {
            toks.push(MTok::Term);
        }
    };

    while i < n
    {
        let c = chars[i];
        match c
        {
            ' ' | '\t' | '\r' => i += 1,
            '%' =>
            {
                while i < n && chars[i] != '\n'
                {
                    i += 1;
                }
            },
            '\n' =>
            {
                i += 1;
                if depth == 0
                {
                    push_term(&mut toks);
                }
            },
            ';' | ',' if depth == 0 =>
            {
                // Statement terminator at top level.
                i += 1;
                push_term(&mut toks);
            },
            '(' =>
            {
                depth += 1;
                toks.push(MTok::Sym("(".into()));
                i += 1;
            },
            ')' =>
            {
                depth -= 1;
                if depth < 0
                {
                    return Err("unbalanced `)`".into());
                }
                toks.push(MTok::Sym(")".into()));
                i += 1;
            },
            ',' =>
            {
                // Inside parentheses: argument separator.
                toks.push(MTok::Sym(",".into()));
                i += 1;
            },
            '.' if i + 1 < n && matches!(chars[i + 1], '*' | '/' | '^') =>
            {
                toks.push(MTok::Sym(format!(".{}", chars[i + 1])));
                i += 2;
            },
            '~' if i + 1 < n && chars[i + 1] == '=' =>
            {
                toks.push(MTok::Sym("~=".into()));
                i += 2;
            },
            '<' | '>' | '=' if i + 1 < n && chars[i + 1] == '=' =>
            {
                toks.push(MTok::Sym(format!("{}=", c)));
                i += 2;
            },
            '+' | '-' | '*' | '/' | '^' | ':' | '=' | '<' | '>' =>
            {
                toks.push(MTok::Sym(c.to_string()));
                i += 1;
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
                let v: f64 = s
                    .parse()
                    .map_err(|_| format!("bad number literal: {}", s))?;
                let _ = is_float;
                toks.push(MTok::Num(v));
            },
            c if c.is_alphabetic() || c == '_' =>
            {
                let start = i;
                while i < n && (chars[i].is_alphanumeric() || chars[i] == '_')
                {
                    i += 1;
                }
                toks.push(MTok::Ident(chars[start..i].iter().collect()));
            },
            other => return Err(format!("unexpected character: {:?}", other)),
        }
    }

    push_term(&mut toks);
    toks.push(MTok::Eof);
    Ok(toks)
}
