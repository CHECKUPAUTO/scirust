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
    use scirust_som_symbolic::OwnershipOracle;

    #[test]
    fn renders_states_and_diagnostics() {
        // let x = 1; let y = x; let z = x;  → one use-after-move
        let ast = SomAst::Program(vec![Function {
            name: "main".to_string(),
            params: vec![],
            body: vec![
                Statement::VarDecl {
                    name: "x".to_string(),
                    ty: Type::Int,
                    init: Some(Expression::Literal(Literal::Int(1))),
                },
                Statement::VarDecl {
                    name: "y".to_string(),
                    ty: Type::Int,
                    init: Some(Expression::Variable("x".to_string())),
                },
                Statement::VarDecl {
                    name: "z".to_string(),
                    ty: Type::Int,
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
}
