//! **How wrong is the heuristic parser?** CCOS's causal graph is only as good as the
//! ingestion that builds it ("garbage in, garbage out"). The default parser is a
//! zero-dependency line-based heuristic; the `syn-parser` feature swaps in a real Rust
//! AST. This example parses CCOS's *own* `src/` tree and emits a canonical, sorted dump
//! of every symbol / `use` / module it finds — so running it both ways and diffing the
//! two dumps measures, exactly, where the heuristic disagrees with the AST (which, being
//! a real parser, is ground truth on valid Rust):
//!
//! ```bash
//! cargo run --release --example parse_accuracy                       > /tmp/heuristic.txt
//! cargo run --release --features syn-parser --example parse_accuracy > /tmp/ast.txt
//! diff /tmp/heuristic.txt /tmp/ast.txt          # every line = a heuristic error
//! ```
//!
//! Lines only in `ast.txt` are heuristic **false negatives** (real items it missed);
//! lines only in `heuristic.txt` are **false positives** (items it hallucinated). The
//! trailing `TOTALS` line quantifies the aggregate gap at a glance.

use ccos_core::parser::{ASTParser, ModuleDecl};
use std::fs;
use std::path::{Path, PathBuf};

/// All `*.rs` files under `dir`, recursively, in sorted (deterministic) order.
fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = fs::read_dir(&d) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Flatten a module tree to `M <name>` tokens (nested modules included).
fn module_tokens(mods: &[ModuleDecl], out: &mut Vec<String>) {
    for m in mods {
        out.push(format!("M {}", m.name));
        module_tokens(&m.children, out);
    }
}

fn main() {
    let backend = if cfg!(feature = "syn-parser") {
        "AST (syn)"
    } else {
        "heuristic (line-based)"
    };
    println!("# Parser accuracy dump — backend: {backend}\n");

    let parser = ASTParser::new();
    let (mut tot_sym, mut tot_use, mut tot_mod, mut n_files) = (0usize, 0usize, 0usize, 0usize);

    for path in rust_files(Path::new("src")) {
        let Ok(src) = fs::read_to_string(&path) else {
            continue;
        };
        let rel = path.to_string_lossy().replace('\\', "/");
        let r = parser.parse_source(&rel, &src);
        n_files += 1;
        tot_sym += r.symbols.len();
        tot_use += r.use_statements.len();

        // Canonical, sorted per-file token list so a diff is order-independent.
        let mut tokens = Vec::new();
        for s in &r.symbols {
            tokens.push(format!("S {:?} {}", s.kind, s.name));
        }
        for u in &r.use_statements {
            tokens.push(format!("U {}", u.full_path));
        }
        let mut mtok = Vec::new();
        module_tokens(&r.modules, &mut mtok);
        let n_mods = mtok.len();
        tot_mod += n_mods;
        tokens.extend(mtok);
        tokens.sort();

        println!(
            "## {rel}  ({} symbols, {} uses, {n_mods} mods)",
            r.symbols.len(),
            r.use_statements.len()
        );
        // File-scoped, TAB-separated so a global sort/diff treats the *same* token in
        // two different files as distinct (no cross-file collisions).
        for t in tokens {
            println!("{rel}\t{t}");
        }
    }

    println!(
        "\nTOTALS  files={n_files}  symbols={tot_sym}  uses={tot_use}  modules={tot_mod}  backend={backend}"
    );
}
