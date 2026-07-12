//! MANDATORY CONSTRAINT guard: no floating point in the money path.
//!
//! The migration protocol forbids `f32`/`f64` anywhere in the accrual/amort
//! logic (audit_report.md Gap-1). This test greps the crate sources at compile
//! time and fails if a float type appears, so the constraint cannot regress
//! unnoticed. Doc examples and comments are the source too, so we match the
//! Rust type tokens specifically.

/// Source files that make up the money path.
const SOURCES: &[(&str, &str)] = &[
    ("src/lib.rs", include_str!("../src/lib.rs")),
    ("src/amort.rs", include_str!("../src/amort.rs")),
    ("src/paycalc.rs", include_str!("../src/paycalc.rs")),
    ("src/daycount.rs", include_str!("../src/daycount.rs")),
    ("src/brktcalc.rs", include_str!("../src/brktcalc.rs")),
    ("src/currcvt.rs", include_str!("../src/currcvt.rs")),
];

#[test]
fn no_float_in_money_path() {
    // Match the float type tokens with word boundaries so we don't trip on,
    // e.g., an identifier that merely contains the letters.
    for (name, src) in SOURCES
    {
        for (lineno, line) in src.lines().enumerate()
        {
            // Comments (incl. `//!`, `///`) are prose, not the money path.
            if line.trim_start().starts_with("//")
            {
                continue;
            }
            for tok in ["f32", "f64"]
            {
                if contains_token(line, tok)
                {
                    panic!(
                        "floating point `{tok}` found in {name}:{} — the money path must be \
                         decimal-only (audit_report.md Gap-1):\n  {}",
                        lineno + 1,
                        line.trim()
                    );
                }
            }
        }
    }
}

/// True if `tok` occurs in `line` not surrounded by identifier characters.
fn contains_token(line: &str, tok: &str) -> bool {
    let bytes = line.as_bytes();
    let mut from = 0;
    while let Some(pos) = line[from..].find(tok)
    {
        let start = from + pos;
        let end = start + tok.len();
        let before_ok = start == 0 || !is_ident(bytes[start - 1]);
        let after_ok = end >= bytes.len() || !is_ident(bytes[end]);
        if before_ok && after_ok
        {
            return true;
        }
        from = end;
    }
    false
}

fn is_ident(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphanumeric()
}
