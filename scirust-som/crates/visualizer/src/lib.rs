//! Rendering of SOM ownership analyses for humans.
//!
//! Turns an oracle [`Analysis`] into a markdown table (token, ownership,
//! borrow, fault) plus a diagnostics section — the quickest way to inspect
//! what the oracle labelled and what a dataset actually contains. The PCG
//! itself already renders via `Pcg::to_dot()` in `scirust-som-pcg`.

use scirust_som_symbolic::{Analysis, borrow_name, ownership_name};
use scirust_som_tokenizer::SomToken;

fn token_text(token: &SomToken) -> String {
    match token
    {
        SomToken::FnDecl(n) => format!("fn {n}"),
        SomToken::Param(n) => format!("param {n}"),
        SomToken::VarDecl(n) => format!("let {n}"),
        SomToken::Assign(n) => format!("{n} = …"),
        SomToken::Use(n) => format!("use {n}"),
        SomToken::Ref(n) => format!("&{n}"),
        SomToken::MutRef(n) => format!("&mut {n}"),
        SomToken::Drop(n) => format!("drop {n}"),
        SomToken::Return => "return".to_string(),
        SomToken::ScopeStart => "{".to_string(),
        SomToken::ScopeEnd => "}".to_string(),
        SomToken::Sep => "—".to_string(),
        SomToken::Node(n) => format!("{n:?}"),
        SomToken::Edge(e) => format!("{e:?}"),
    }
}

/// Render an analysis as a markdown table with a diagnostics section.
pub fn render_markdown(analysis: &Analysis) -> String {
    let mut out = String::new();
    out.push_str("| # | token | ownership | borrow | fault |\n");
    out.push_str("|---|-------|-----------|--------|-------|\n");
    for (i, (token, label)) in analysis.tokens.iter().zip(&analysis.labels).enumerate()
    {
        let fault = if label.invalid { "FAULT" } else { "" };
        out.push_str(&format!(
            "| {} | `{}` | {} | {} | {} |\n",
            i,
            token_text(token),
            ownership_name(label.ownership),
            borrow_name(label.borrow),
            fault
        ));
    }
    if analysis.diagnostics.is_empty()
    {
        out.push_str("\nNo diagnostics — program is clean.\n");
    }
    else
    {
        out.push_str("\nDiagnostics:\n");
        for d in &analysis.diagnostics
        {
            out.push_str(&format!(
                "- token {} `{}`: {:?}\n",
                d.token_index, d.var, d.kind
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_som_pcg::ast::{Expression, Function, Literal, SomAst, Statement, Type};
    use scirust_som_pcg::{PcgEdge, PcgNode};
    use scirust_som_symbolic::{
        BORROW_MUT, BORROW_NA, BORROW_NONE, BORROW_SHARED, Diagnostic, FaultKind,
        OWNERSHIP_DROPPED, OWNERSHIP_MOVED, OWNERSHIP_NA, OWNERSHIP_OWNED, OwnershipOracle,
        TokenLabel,
    };

    /// Lines that make up the table body of a rendered analysis: every line
    /// from the header down to (but excluding) the blank line that precedes
    /// the diagnostics section.
    fn table_lines(md: &str) -> Vec<&str> {
        md.lines().take_while(|l| !l.is_empty()).collect()
    }

    #[test]
    fn renders_states_and_diagnostics() {
        // Owner-typed (move semantics): the second use of `x` is a
        // use-after-move and must render as a FAULT.
        let ast = SomAst::Program(vec![Function {
            name: "main".to_string(),
            params: vec![],
            body: vec![
                Statement::VarDecl {
                    name: "x".to_string(),
                    ty: Type::Str,
                    init: Some(Expression::Literal(Literal::Str("s".to_string()))),
                },
                Statement::VarDecl {
                    name: "y".to_string(),
                    ty: Type::Str,
                    init: Some(Expression::Variable("x".to_string())),
                },
                Statement::VarDecl {
                    name: "z".to_string(),
                    ty: Type::Str,
                    init: Some(Expression::Variable("x".to_string())),
                },
            ],
        }]);
        let analysis = OwnershipOracle::new().analyze(&ast);
        let md = render_markdown(&analysis);

        assert!(md.contains("`use x`"));
        assert!(md.contains("Moved"));
        assert!(md.contains("FAULT"));
        assert!(md.contains("UseAfterMove"));
        assert!(md.contains("`drop y`"));
    }

    #[test]
    fn clean_program_says_so() {
        let ast = SomAst::Program(vec![Function {
            name: "main".to_string(),
            params: vec![],
            body: vec![Statement::VarDecl {
                name: "x".to_string(),
                ty: Type::Int,
                init: Some(Expression::Literal(Literal::Int(1))),
            }],
        }]);
        let analysis = OwnershipOracle::new().analyze(&ast);
        let md = render_markdown(&analysis);
        assert!(md.contains("No diagnostics"));
    }

    /// Whole-output oracle: the move/use-after-move program below renders to
    /// *exactly* this table. Every row, glyph, label, the FAULT marker, the
    /// diagnostics bullet and the table framing are hand-derived from the
    /// oracle's documented semantics (`let x = owner; let y = x; let z = x;`):
    ///
    /// - `Use(x)` #1 is the legal move → ownership becomes Moved, no fault.
    /// - `Use(x)` #2 is use-after-move → Moved + FAULT (UseAfterMove).
    /// - drops run in reverse declaration order z, y, x; `z`/`y` are Dropped,
    ///   `x` was moved out so its drop is labelled Moved.
    ///
    /// A single byte of drift (wrong glyph, dropped row, mislabelled column,
    /// broken table framing) trips this assertion.
    #[test]
    fn move_fault_table_is_exact() {
        let ast = SomAst::Program(vec![Function {
            name: "main".to_string(),
            params: vec![],
            body: vec![
                Statement::VarDecl {
                    name: "x".to_string(),
                    ty: Type::Str,
                    init: Some(Expression::Literal(Literal::Str("s".to_string()))),
                },
                Statement::VarDecl {
                    name: "y".to_string(),
                    ty: Type::Str,
                    init: Some(Expression::Variable("x".to_string())),
                },
                Statement::VarDecl {
                    name: "z".to_string(),
                    ty: Type::Str,
                    init: Some(Expression::Variable("x".to_string())),
                },
            ],
        }]);
        let md = render_markdown(&OwnershipOracle::new().analyze(&ast));
        let expected = "\
| # | token | ownership | borrow | fault |
|---|-------|-----------|--------|-------|
| 0 | `fn main` | - | - |  |
| 1 | `let x` | Owned | None |  |
| 2 | `use x` | Moved | None |  |
| 3 | `let y` | Owned | None |  |
| 4 | `use x` | Moved | None | FAULT |
| 5 | `let z` | Owned | None |  |
| 6 | `drop z` | Dropped | None |  |
| 7 | `drop y` | Dropped | None |  |
| 8 | `drop x` | Moved | None |  |

Diagnostics:
- token 4 `x`: UseAfterMove
";
        assert_eq!(md, expected);
    }

    /// Whole-output oracle for a clean single-binding program: the table has
    /// exactly one data row and the diagnostics section reads "clean". A Copy
    /// (`Int`) binding is never moved, so its end-scope drop is `Dropped`.
    #[test]
    fn clean_program_table_is_exact() {
        let ast = SomAst::Program(vec![Function {
            name: "main".to_string(),
            params: vec![],
            body: vec![Statement::VarDecl {
                name: "x".to_string(),
                ty: Type::Int,
                init: Some(Expression::Literal(Literal::Int(1))),
            }],
        }]);
        let md = render_markdown(&OwnershipOracle::new().analyze(&ast));
        let expected = "\
| # | token | ownership | borrow | fault |
|---|-------|-----------|--------|-------|
| 0 | `fn main` | - | - |  |
| 1 | `let x` | Owned | None |  |
| 2 | `drop x` | Dropped | None |  |

No diagnostics — program is clean.
";
        assert_eq!(md, expected);
    }

    /// Structural well-formedness: the table is a GitHub-flavoured-markdown
    /// table — a header row, a delimiter row of dashes, then one row per
    /// token, and *every* row has exactly five columns (six `|` delimiters,
    /// leading and trailing). Derived independently of the cell contents so a
    /// stray or missing `|` (which would silently merge/split columns in a
    /// renderer) is caught.
    #[test]
    fn rendered_table_is_well_formed() {
        let ast = SomAst::Program(vec![Function {
            name: "f".to_string(),
            params: vec![],
            body: vec![
                Statement::VarDecl {
                    name: "a".to_string(),
                    ty: Type::Str,
                    init: Some(Expression::Literal(Literal::Str("s".to_string()))),
                },
                Statement::VarDecl {
                    name: "b".to_string(),
                    ty: Type::Str,
                    init: Some(Expression::Reference {
                        name: "a".to_string(),
                        mutable: false,
                    }),
                },
            ],
        }]);
        let analysis = OwnershipOracle::new().analyze(&ast);
        let md = render_markdown(&analysis);
        let rows = table_lines(&md);

        assert_eq!(rows[0], "| # | token | ownership | borrow | fault |");
        assert_eq!(rows[1], "|---|-------|-----------|--------|-------|");
        // header + delimiter + one row per token.
        assert_eq!(rows.len(), 2 + analysis.tokens.len());
        for (i, row) in rows.iter().enumerate()
        {
            assert!(
                row.starts_with('|'),
                "row {i} must start with a pipe: {row:?}"
            );
            assert!(row.ends_with('|'), "row {i} must end with a pipe: {row:?}");
            assert_eq!(
                row.matches('|').count(),
                6,
                "row {i} must have exactly 5 columns (6 pipes): {row:?}"
            );
        }
    }

    /// Per-kind glyph oracle. Each [`SomToken`] kind has a documented textual
    /// representation; render a hand-built one-token analysis and assert the
    /// rendered cell is exactly that glyph. This also covers the PCG token
    /// branches (`Node`/`Edge`/`Sep`), which the oracle path never emits.
    #[test]
    fn token_glyphs_match_documented_representation() {
        // (token, expected glyph rendered in the `token` cell)
        let cases: Vec<(SomToken, &str)> = vec![
            (SomToken::FnDecl("main".into()), "fn main"),
            (SomToken::Param("p".into()), "param p"),
            (SomToken::VarDecl("x".into()), "let x"),
            (SomToken::Assign("x".into()), "x = …"),
            (SomToken::Use("x".into()), "use x"),
            (SomToken::Ref("x".into()), "&x"),
            (SomToken::MutRef("x".into()), "&mut x"),
            (SomToken::Drop("x".into()), "drop x"),
            (SomToken::Return, "return"),
            (SomToken::ScopeStart, "{"),
            (SomToken::ScopeEnd, "}"),
            (SomToken::Sep, "—"),
            (
                SomToken::Node(PcgNode::Variable("a".into())),
                "Variable(\"a\")",
            ),
            (SomToken::Edge(PcgEdge::MutBorrows), "MutBorrows"),
        ];
        for (token, glyph) in cases
        {
            let analysis = Analysis {
                tokens: vec![token.clone()],
                labels: vec![TokenLabel {
                    ownership: OWNERSHIP_NA,
                    borrow: BORROW_NA,
                    invalid: false,
                }],
                diagnostics: Vec::new(),
            };
            let md = render_markdown(&analysis);
            let want = format!("| 0 | `{glyph}` |");
            assert!(
                md.contains(&want),
                "token {token:?} should render glyph {glyph:?}; row line was: {:?}",
                table_lines(&md).get(2)
            );
        }
    }

    /// The four label columns are populated from the symbolic crate's naming
    /// functions and the `invalid` flag — assert each independently on a
    /// hand-built analysis so a swapped column is caught. A `Ref` token with a
    /// live shared borrow renders ownership `Borrowed`, borrow `Shared`; an
    /// `NA` label renders `-`/`-`; the `invalid` flag renders `FAULT`.
    #[test]
    fn label_columns_render_independently() {
        let analysis = Analysis {
            tokens: vec![
                SomToken::Ref("x".into()),
                SomToken::MutRef("x".into()),
                SomToken::Drop("x".into()),
                SomToken::FnDecl("main".into()),
            ],
            labels: vec![
                TokenLabel {
                    ownership: OWNERSHIP_OWNED,
                    borrow: BORROW_SHARED,
                    invalid: false,
                },
                TokenLabel {
                    ownership: OWNERSHIP_MOVED,
                    borrow: BORROW_MUT,
                    invalid: true,
                },
                TokenLabel {
                    ownership: OWNERSHIP_DROPPED,
                    borrow: BORROW_NONE,
                    invalid: false,
                },
                TokenLabel {
                    ownership: OWNERSHIP_NA,
                    borrow: BORROW_NA,
                    invalid: false,
                },
            ],
            diagnostics: Vec::new(),
        };
        let md = render_markdown(&analysis);
        assert!(md.contains("| 0 | `&x` | Owned | Shared |  |"));
        assert!(md.contains("| 1 | `&mut x` | Moved | Mut | FAULT |"));
        assert!(md.contains("| 2 | `drop x` | Dropped | None |  |"));
        assert!(md.contains("| 3 | `fn main` | - | - |  |"));
    }

    /// The diagnostics section lists one bullet per diagnostic, in order,
    /// formatted `- token {index} \`{var}\`: {kind:?}`. Hand-built so the exact
    /// wording and the `{kind:?}` rendering are pinned.
    #[test]
    fn diagnostics_section_lists_each_fault() {
        let analysis = Analysis {
            tokens: vec![SomToken::Use("g".into()), SomToken::Assign("w".into())],
            labels: vec![
                TokenLabel {
                    ownership: OWNERSHIP_NA,
                    borrow: BORROW_NA,
                    invalid: true,
                },
                TokenLabel {
                    ownership: OWNERSHIP_OWNED,
                    borrow: BORROW_NONE,
                    invalid: true,
                },
            ],
            diagnostics: vec![
                Diagnostic {
                    token_index: 0,
                    var: "g".into(),
                    kind: FaultKind::UseOfUndeclared,
                },
                Diagnostic {
                    token_index: 1,
                    var: "w".into(),
                    kind: FaultKind::AssignToUndeclared,
                },
            ],
        };
        let md = render_markdown(&analysis);
        assert!(md.contains("\nDiagnostics:\n"));
        assert!(md.contains("- token 0 `g`: UseOfUndeclared\n"));
        assert!(md.contains("- token 1 `w`: AssignToUndeclared\n"));
        assert!(!md.contains("No diagnostics"));
    }
}
