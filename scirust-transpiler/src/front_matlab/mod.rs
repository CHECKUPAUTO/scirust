//! MATLAB/Octave subset front-end: source text -> [`ast::MModule`].

pub mod ast;
pub mod lexer;
pub mod parser;

/// Parse a MATLAB source string into the subset AST.
pub fn parse_matlab(src: &str) -> Result<ast::MModule, String> {
    let toks = lexer::lex(src)?;
    parser::parse(&toks)
}
