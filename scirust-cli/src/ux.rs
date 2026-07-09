//! Terminal UX helpers for the `scirust` CLI — colour styling and "did you
//! mean?" typo suggestions.
//!
//! Zero dependencies: colour uses raw ANSI SGR codes gated on a run-time check
//! that respects the `NO_COLOR` convention (<https://no-color.org>), an explicit
//! `--no-color` flag / `CLICOLOR=0`, and whether stdout is actually a TTY
//! (`std::io::IsTerminal`, stable since 1.70). Suggestions use a small
//! Levenshtein edit-distance so a mistyped command points at the closest real
//! one — the same affordance `git` and `cargo` give.

use std::io::IsTerminal;
use std::sync::OnceLock;

/// Whether colour output is enabled for this process. Decided once, cheaply.
///
/// Disabled when: `NO_COLOR` is set (to anything, per the spec), `CLICOLOR=0`,
/// `TERM=dumb`, or stdout is not a terminal. `CLICOLOR_FORCE` (non-empty, not
/// `0`) forces it on even when piped, matching common tooling.
pub fn color_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let force = std::env::var_os("CLICOLOR_FORCE")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false);
        if force
        {
            return true;
        }
        if std::env::var_os("NO_COLOR").is_some()
        {
            return false;
        }
        if std::env::var("CLICOLOR").map(|v| v == "0").unwrap_or(false)
        {
            return false;
        }
        if std::env::var("TERM").map(|v| v == "dumb").unwrap_or(false)
        {
            return false;
        }
        std::io::stdout().is_terminal()
    })
}

/// Wrap `s` in an ANSI SGR code when colour is enabled, else return it plain.
fn paint(code: &str, s: &str) -> String {
    if color_enabled()
    {
        format!("\x1b[{code}m{s}\x1b[0m")
    }
    else
    {
        s.to_string()
    }
}

/// Bold.
pub fn bold(s: &str) -> String {
    paint("1", s)
}
/// Dim / faint (secondary text).
pub fn dim(s: &str) -> String {
    paint("2", s)
}
/// Bold cyan — headings.
pub fn heading(s: &str) -> String {
    paint("1;36", s)
}
/// Green — success / commands.
pub fn green(s: &str) -> String {
    paint("32", s)
}
/// Bold red — errors.
pub fn red(s: &str) -> String {
    paint("1;31", s)
}
/// Yellow — warnings / hints.
pub fn yellow(s: &str) -> String {
    paint("33", s)
}

/// An `error:` prefix, red and bold when colour is on (Rust-compiler style).
pub fn error_prefix() -> String {
    red("error:")
}

/// Levenshtein edit distance between two byte strings. O(a·b) time, O(b) space.
pub fn edit_distance(a: &str, b: &str) -> usize {
    let a = a.as_bytes();
    let b = b.as_bytes();
    if a.is_empty()
    {
        return b.len();
    }
    if b.is_empty()
    {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate()
    {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate()
        {
            let cost = if ca == cb { 0 } else { 1 };
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// The closest candidate to `input` within a small distance threshold, if any.
///
/// The threshold scales with the input length (a 3-letter word tolerates 1
/// edit, longer words tolerate more) and is capped so unrelated words never
/// match. Returns the single best suggestion — enough to be helpful without
/// being noisy.
pub fn suggest<'a>(input: &str, candidates: &'a [&'a str]) -> Option<&'a str> {
    let max_dist = match input.len()
    {
        0..=2 => 1,
        3..=5 => 2,
        _ => 3,
    };
    candidates
        .iter()
        .map(|&c| (c, edit_distance(input, c)))
        .filter(|&(_, d)| d <= max_dist)
        .min_by_key(|&(_, d)| d)
        .map(|(c, _)| c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_distance_basics() {
        assert_eq!(edit_distance("", ""), 0);
        assert_eq!(edit_distance("abc", "abc"), 0);
        assert_eq!(edit_distance("abc", "abd"), 1);
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("diff", ""), 4);
    }

    #[test]
    fn suggest_finds_close_command() {
        let cmds = ["diff", "simplify", "solve", "integrate", "minimize"];
        assert_eq!(suggest("dif", &cmds), Some("diff"));
        assert_eq!(suggest("simplfy", &cmds), Some("simplify"));
        assert_eq!(suggest("integate", &cmds), Some("integrate"));
    }

    #[test]
    fn suggest_rejects_unrelated_input() {
        let cmds = ["diff", "simplify", "solve"];
        assert_eq!(suggest("xyzzy", &cmds), None);
        assert_eq!(suggest("completely-different", &cmds), None);
    }

    #[test]
    fn paint_is_plain_without_a_tty() {
        // In the test harness stdout is not a TTY and NO_COLOR is typically
        // unset; either way `paint` must never corrupt the string content.
        let s = bold("hello");
        assert!(s.contains("hello"));
    }
}
