//! Python/NumPy subset front-end: source text -> [`ast::PyModule`].

pub mod ast;
pub mod lexer;
pub mod parser;

/// Parse a Python source string into the subset AST.
pub fn parse_python(src: &str) -> Result<ast::PyModule, String> {
    let toks = lexer::lex(src)?;
    parser::parse(&toks)
}
