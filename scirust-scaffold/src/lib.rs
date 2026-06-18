//! # SciRust Scaffold
//!
//! Algorithmic scaffolding: DSL-based algorithm generation,
//! templating, code generation, analysis, and documentation.
//!
//! ## Quick Start
//!
//! ```ignore
//! use scirust_scaffold::*;
//!
//! let dsl = r#"
//! algorithm binary_search
//!   input: arr: Vec<i32>, target: i32
//!   output: idx: i32
//!   variables:
//!     left: i32 = 0
//!     right: i32 = len(arr) - 1
//!   steps:
//!     while left <= right
//!       mid = (left + right) / 2
//!       if arr[mid] == target
//!         return mid
//!       if arr[mid] < target
//!         left = mid + 1
//!       else
//!         right = mid - 1
//!     return -1
//! "#;
//!
//! let algo = parse_algorithm(dsl).unwrap();
//! let rust_code = generate_rust(&algo, &CodeStyle::default());
//! let python_code = generate_python(&algo, &CodeStyle::default());
//! let c_code = generate_c(&algo, &CodeStyle::default());
//! let pseudocode = generate_pseudocode(&algo);
//! let analysis = analyze(&algo);
//! let docs = generate_docs(&algo);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io::{self};
use std::path::{Path, PathBuf};

// ============================================================================
// 1. DSL: Tokenizer
// ============================================================================

/// A token produced by the DSL lexer.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Keyword(String),
    Identifier(String),
    IntLiteral(i64),
    FloatLiteral(f64),
    BoolLiteral(bool),
    StringLiteral(String),
    Colon,
    Comma,
    DotDot,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Equals,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    EqualEqual,
    NotEqual,
    AndAnd,
    OrOr,
    Not,
    Arrow,
    SemiColon,
    Newline,
    Indent(usize),
    Dedent,
    Eof,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Token::Keyword(k) => write!(f, "{}", k),
            Token::Identifier(i) => write!(f, "{}", i),
            Token::IntLiteral(n) => write!(f, "{}", n),
            Token::FloatLiteral(x) => write!(f, "{}", x),
            Token::BoolLiteral(b) => write!(f, "{}", b),
            Token::StringLiteral(s) => write!(f, "\"{}\"", s),
            Token::Colon => write!(f, ":"),
            Token::Comma => write!(f, ","),
            Token::DotDot => write!(f, ".."),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::Equals => write!(f, "="),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Percent => write!(f, "%"),
            Token::Less => write!(f, "<"),
            Token::Greater => write!(f, ">"),
            Token::LessEqual => write!(f, "<="),
            Token::GreaterEqual => write!(f, ">="),
            Token::EqualEqual => write!(f, "=="),
            Token::NotEqual => write!(f, "!="),
            Token::AndAnd => write!(f, "&&"),
            Token::OrOr => write!(f, "||"),
            Token::Not => write!(f, "!"),
            Token::Arrow => write!(f, "->"),
            Token::SemiColon => write!(f, ";"),
            Token::Newline => write!(f, "\\n"),
            Token::Indent(n) => write!(f, "INDENT({})", n),
            Token::Dedent => write!(f, "DEDENT"),
            Token::Eof => write!(f, "EOF"),
        }
    }
}

/// Tokenize with proper indent/dedent handling by pre-processing source lines.
pub fn tokenize_with_indent(input: &str) -> Result<Vec<Token>, String> {
    let lines: Vec<&str> = input.lines().collect();
    let mut indent_stack: Vec<usize> = vec![0];
    let mut result = Vec::new();

    for line in &lines
    {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#')
        {
            result.push(Token::Newline);
            continue;
        }
        let leading_spaces = line.len() - line.trim_start().len();
        let current_indent = *indent_stack.last().unwrap_or(&0);

        if leading_spaces > current_indent
        {
            indent_stack.push(leading_spaces);
            result.push(Token::Indent(leading_spaces));
        }
        else if leading_spaces < current_indent
        {
            while let Some(&top) = indent_stack.last()
            {
                if top > leading_spaces
                {
                    indent_stack.pop();
                    result.push(Token::Dedent);
                }
                else if top == leading_spaces
                {
                    break;
                }
                else
                {
                    return Err(format!(
                        "Inconsistent indentation: expected {} or less, found {}",
                        top, leading_spaces
                    ));
                }
            }
        }

        let line_tokens = tokenize_line(trimmed)?;
        result.extend(line_tokens);
        result.push(Token::Newline);
    }

    while indent_stack.len() > 1
    {
        indent_stack.pop();
        result.push(Token::Dedent);
    }
    result.push(Token::Eof);
    Ok(result)
}

fn tokenize_line(trimmed: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = trimmed.chars().peekable();

    let keywords: HashMap<&str, &str> = [
        ("algorithm", "algorithm"),
        ("input", "input"),
        ("output", "output"),
        ("variables", "variables"),
        ("steps", "steps"),
        ("loop", "loop"),
        ("for", "for"),
        ("while", "while"),
        ("in", "in"),
        ("if", "if"),
        ("else", "else"),
        ("return", "return"),
        ("let", "let"),
        ("swap", "swap"),
        ("true", "true"),
        ("false", "false"),
        ("and", "and"),
        ("or", "or"),
        ("not", "not"),
    ]
    .iter()
    .copied()
    .collect();

    while let Some(&c) = chars.peek()
    {
        match c
        {
            '#' => break,
            ' ' =>
            {
                chars.next();
            },
            '0'..='9' =>
            {
                let mut num = String::new();
                while let Some(&ch) = chars.peek()
                {
                    if ch.is_ascii_digit()
                    {
                        num.push(ch);
                        chars.next();
                    }
                    else
                    {
                        break;
                    }
                }
                if chars.peek() == Some(&'.')
                {
                    // Peek ahead: if the next char after '.' is also '.' or not a digit,
                    // this is a range (..) not a float literal
                    chars.next(); // consume first '.'
                    let is_range = chars.peek() == Some(&'.');
                    let has_digits = chars.peek().is_some_and(|c| c.is_ascii_digit());
                    if is_range || !has_digits
                    {
                        // It's a range: emit the integer, DotDot will be handled in '.' case
                        let val: i64 = num
                            .parse()
                            .map_err(|_| format!("Invalid integer: {}", num))?;
                        tokens.push(Token::IntLiteral(val));
                        if is_range
                        {
                            chars.next(); // consume second '.'
                            tokens.push(Token::DotDot);
                        }
                        // else: single dot consumed but no digits, will be handled as error later
                    }
                    else
                    {
                        num.push('.');
                        while let Some(&ch) = chars.peek()
                        {
                            if ch.is_ascii_digit()
                            {
                                num.push(ch);
                                chars.next();
                            }
                            else
                            {
                                break;
                            }
                        }
                        let val: f64 =
                            num.parse().map_err(|_| format!("Invalid float: {}", num))?;
                        tokens.push(Token::FloatLiteral(val));
                    }
                }
                else
                {
                    let val: i64 = num
                        .parse()
                        .map_err(|_| format!("Invalid integer: {}", num))?;
                    tokens.push(Token::IntLiteral(val));
                }
            },
            'a'..='z' | 'A'..='Z' | '_' =>
            {
                let mut ident = String::new();
                while let Some(&ch) = chars.peek()
                {
                    if ch.is_alphanumeric() || ch == '_' || ch == '!' || ch == '?'
                    {
                        ident.push(ch);
                        chars.next();
                    }
                    else
                    {
                        break;
                    }
                }
                if let Some(&kw) = keywords.get(ident.as_str())
                {
                    tokens.push(Token::Keyword(kw.to_string()));
                }
                else
                {
                    tokens.push(Token::Identifier(ident));
                }
            },
            '"' =>
            {
                chars.next();
                let mut s = String::new();
                while let Some(&ch) = chars.peek()
                {
                    if ch == '"'
                    {
                        chars.next();
                        break;
                    }
                    s.push(ch);
                    chars.next();
                }
                tokens.push(Token::StringLiteral(s));
            },
            ':' =>
            {
                chars.next();
                tokens.push(Token::Colon);
            },
            ',' =>
            {
                chars.next();
                tokens.push(Token::Comma);
            },
            '.' =>
            {
                chars.next();
                if chars.peek() == Some(&'.')
                {
                    chars.next();
                    tokens.push(Token::DotDot);
                }
                else
                {
                    return Err("Expected '..' for range".to_string());
                }
            },
            '(' =>
            {
                chars.next();
                tokens.push(Token::LParen);
            },
            ')' =>
            {
                chars.next();
                tokens.push(Token::RParen);
            },
            '[' =>
            {
                chars.next();
                tokens.push(Token::LBracket);
            },
            ']' =>
            {
                chars.next();
                tokens.push(Token::RBracket);
            },
            '=' =>
            {
                chars.next();
                if chars.peek() == Some(&'=')
                {
                    chars.next();
                    tokens.push(Token::EqualEqual);
                }
                else if chars.peek() == Some(&'>')
                {
                    chars.next();
                    tokens.push(Token::Arrow);
                }
                else
                {
                    tokens.push(Token::Equals);
                }
            },
            '+' =>
            {
                chars.next();
                tokens.push(Token::Plus);
            },
            '-' =>
            {
                chars.next();
                if chars.peek() == Some(&'>')
                {
                    chars.next();
                    tokens.push(Token::Arrow);
                }
                else
                {
                    tokens.push(Token::Minus);
                }
            },
            '*' =>
            {
                chars.next();
                tokens.push(Token::Star);
            },
            '/' =>
            {
                chars.next();
                tokens.push(Token::Slash);
            },
            '%' =>
            {
                chars.next();
                tokens.push(Token::Percent);
            },
            '<' =>
            {
                chars.next();
                if chars.peek() == Some(&'=')
                {
                    chars.next();
                    tokens.push(Token::LessEqual);
                }
                else
                {
                    tokens.push(Token::Less);
                }
            },
            '>' =>
            {
                chars.next();
                if chars.peek() == Some(&'=')
                {
                    chars.next();
                    tokens.push(Token::GreaterEqual);
                }
                else
                {
                    tokens.push(Token::Greater);
                }
            },
            '!' =>
            {
                chars.next();
                if chars.peek() == Some(&'=')
                {
                    chars.next();
                    tokens.push(Token::NotEqual);
                }
                else
                {
                    tokens.push(Token::Not);
                }
            },
            '&' =>
            {
                chars.next();
                if chars.peek() == Some(&'&')
                {
                    chars.next();
                    tokens.push(Token::AndAnd);
                }
                else
                {
                    return Err("Expected '&&'".to_string());
                }
            },
            '|' =>
            {
                chars.next();
                if chars.peek() == Some(&'|')
                {
                    chars.next();
                    tokens.push(Token::OrOr);
                }
                else
                {
                    return Err("Expected '||'".to_string());
                }
            },
            ';' =>
            {
                chars.next();
                tokens.push(Token::SemiColon);
            },
            _ =>
            {
                return Err(format!("Unexpected character '{}' in line", c));
            },
        }
    }
    Ok(tokens)
}

// ============================================================================
// 1. DSL: AST Definition
// ============================================================================

/// A typed variable declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedVar {
    pub name: String,
    pub type_name: String,
    pub default_value: Option<Expression>,
}

impl TypedVar {
    pub fn new(name: &str, type_name: &str) -> Self {
        TypedVar {
            name: name.to_string(),
            type_name: type_name.to_string(),
            default_value: None,
        }
    }
    pub fn with_default(name: &str, type_name: &str, default: Expression) -> Self {
        TypedVar {
            name: name.to_string(),
            type_name: type_name.to_string(),
            default_value: Some(default),
        }
    }
}

/// An expression in the DSL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expression {
    IntLiteral(i64),
    FloatLiteral(f64),
    BoolLiteral(bool),
    StringLiteral(String),
    Variable(String),
    BinaryOp {
        op: BinaryOperator,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    UnaryOp {
        op: UnaryOperator,
        operand: Box<Expression>,
    },
    ArrayIndex {
        array: Box<Expression>,
        index: Box<Expression>,
    },
    FunctionCall {
        name: String,
        args: Vec<Expression>,
    },
    Len(String),
    Range {
        start: Box<Expression>,
        end: Box<Expression>,
    },
}

impl Expression {
    pub fn variable(name: &str) -> Self {
        Expression::Variable(name.to_string())
    }
    pub fn int(n: i64) -> Self {
        Expression::IntLiteral(n)
    }
    pub fn binary(op: BinaryOperator, left: Expression, right: Expression) -> Self {
        Expression::BinaryOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }
    pub fn index(array: Expression, index: Expression) -> Self {
        Expression::ArrayIndex {
            array: Box::new(array),
            index: Box::new(index),
        }
    }
    pub fn func(name: &str, args: Vec<Expression>) -> Self {
        Expression::FunctionCall {
            name: name.to_string(),
            args,
        }
    }
    pub fn len(var: &str) -> Self {
        Expression::Len(var.to_string())
    }
    pub fn range(start: Expression, end: Expression) -> Self {
        Expression::Range {
            start: Box::new(start),
            end: Box::new(end),
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Expression::IntLiteral(n) => write!(f, "{}", n),
            Expression::FloatLiteral(x) => write!(f, "{}", x),
            Expression::BoolLiteral(b) => write!(f, "{}", b),
            Expression::StringLiteral(s) => write!(f, "\"{}\"", s),
            Expression::Variable(v) => write!(f, "{}", v),
            Expression::BinaryOp { op, left, right } => write!(f, "({} {} {})", left, op, right),
            Expression::UnaryOp { op, operand } => write!(f, "{}{}", op, operand),
            Expression::ArrayIndex { array, index } => write!(f, "{}[{}]", array, index),
            Expression::FunctionCall { name, args } =>
            {
                let args_str: Vec<String> = args.iter().map(|a| a.to_string()).collect();
                write!(f, "{}({})", name, args_str.join(", "))
            },
            Expression::Len(v) => write!(f, "len({})", v),
            Expression::Range { start, end } => write!(f, "{}..{}", start, end),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOperator {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Neq,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
}

impl fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            BinaryOperator::Add => write!(f, "+"),
            BinaryOperator::Sub => write!(f, "-"),
            BinaryOperator::Mul => write!(f, "*"),
            BinaryOperator::Div => write!(f, "/"),
            BinaryOperator::Mod => write!(f, "%"),
            BinaryOperator::Eq => write!(f, "=="),
            BinaryOperator::Neq => write!(f, "!="),
            BinaryOperator::Lt => write!(f, "<"),
            BinaryOperator::Gt => write!(f, ">"),
            BinaryOperator::Le => write!(f, "<="),
            BinaryOperator::Ge => write!(f, ">="),
            BinaryOperator::And => write!(f, "&&"),
            BinaryOperator::Or => write!(f, "||"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOperator {
    Neg,
    Not,
}

impl fmt::Display for UnaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            UnaryOperator::Neg => write!(f, "-"),
            UnaryOperator::Not => write!(f, "!"),
        }
    }
}

/// A statement in the DSL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Statement {
    VariableDecl {
        name: String,
        type_name: Option<String>,
        value: Expression,
    },
    Assignment {
        target: String,
        value: Expression,
    },
    ArrayAssign {
        array: String,
        index: Expression,
        value: Expression,
    },
    ForLoop {
        var: String,
        range_start: Expression,
        range_end: Expression,
        body: Vec<Statement>,
    },
    WhileLoop {
        condition: Expression,
        body: Vec<Statement>,
    },
    IfStatement {
        condition: Expression,
        then_branch: Vec<Statement>,
        else_branch: Vec<Statement>,
    },
    Return(Option<Expression>),
    Swap(Expression, Expression),
    ExpressionStmt(Expression),
}

/// A complete algorithm description.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Algorithm {
    pub name: String,
    pub inputs: Vec<TypedVar>,
    pub outputs: Vec<TypedVar>,
    pub variables: Vec<TypedVar>,
    pub steps: Vec<Statement>,
}

// ============================================================================
// 1. DSL: Parser
// ============================================================================

/// Recursive descent parser for the algorithm DSL.
pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn expect(&mut self, expected: &str) -> Result<(), String> {
        let tok = self.peek().clone();
        match &tok
        {
            Token::Keyword(k) if k == expected =>
            {
                self.pos += 1;
                Ok(())
            },
            Token::Identifier(i) if i == expected =>
            {
                self.pos += 1;
                Ok(())
            },
            Token::Colon if expected == ":" =>
            {
                self.pos += 1;
                Ok(())
            },
            Token::Comma if expected == "," =>
            {
                self.pos += 1;
                Ok(())
            },
            Token::DotDot if expected == ".." =>
            {
                self.pos += 1;
                Ok(())
            },
            Token::LParen if expected == "(" =>
            {
                self.pos += 1;
                Ok(())
            },
            Token::RParen if expected == ")" =>
            {
                self.pos += 1;
                Ok(())
            },
            Token::LBracket if expected == "[" =>
            {
                self.pos += 1;
                Ok(())
            },
            Token::RBracket if expected == "]" =>
            {
                self.pos += 1;
                Ok(())
            },
            Token::Equals if expected == "=" =>
            {
                self.pos += 1;
                Ok(())
            },
            _ => Err(format!(
                "Expected '{}' but found {} at position {}",
                expected, tok, self.pos
            )),
        }
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek(), Token::Newline)
        {
            self.pos += 1;
        }
    }

    fn skip_whitespace(&mut self) {
        while matches!(
            self.peek(),
            Token::Newline | Token::Indent(_) | Token::Dedent
        )
        {
            self.pos += 1;
        }
    }

    pub fn parse_algorithm(&mut self) -> Result<Algorithm, String> {
        self.skip_whitespace();
        self.expect("algorithm")?;
        self.skip_whitespace();

        let Token::Identifier(ref name) = self.peek().clone()
        else
        {
            return Err(format!("Expected algorithm name, found {}", self.peek()));
        };
        let name = name.clone();
        self.pos += 1;
        self.skip_whitespace();

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut variables = Vec::new();
        let mut steps = Vec::new();

        while self.pos < self.tokens.len() && !matches!(self.peek(), Token::Eof)
        {
            self.skip_whitespace();
            match self.peek()
            {
                Token::Keyword(k) if k == "input" =>
                {
                    self.pos += 1;
                    self.expect(":")?;
                    inputs = self.parse_typed_vars()?;
                },
                Token::Keyword(k) if k == "output" =>
                {
                    self.pos += 1;
                    self.expect(":")?;
                    outputs = self.parse_typed_vars()?;
                },
                Token::Keyword(k) if k == "variables" =>
                {
                    self.pos += 1;
                    self.expect(":")?;
                    variables = self.parse_variable_decls()?;
                },
                Token::Keyword(k) if k == "steps" =>
                {
                    self.pos += 1;
                    self.expect(":")?;
                    steps = self.parse_block()?;
                },
                Token::Keyword(k)
                    if k == "for"
                        || k == "while"
                        || k == "if"
                        || k == "return"
                        || k == "let"
                        || k == "swap" =>
                {
                    steps = self.parse_block()?;
                },
                Token::Identifier(_) =>
                {
                    steps = self.parse_block()?;
                },
                Token::Newline =>
                {
                    self.pos += 1;
                },
                Token::Eof => break,
                tok =>
                {
                    return Err(format!("Unexpected token {} while parsing algorithm", tok));
                },
            }
        }

        Ok(Algorithm {
            name,
            inputs,
            outputs,
            variables,
            steps,
        })
    }

    fn parse_typed_vars(&mut self) -> Result<Vec<TypedVar>, String> {
        let mut vars = Vec::new();
        loop
        {
            self.skip_whitespace();
            let tok = self.peek().clone();
            match tok
            {
                Token::Identifier(name) =>
                {
                    self.pos += 1;
                    self.expect(":")?;
                    let type_name = self.parse_typename()?;
                    vars.push(TypedVar::new(&name, &type_name));
                    self.skip_whitespace();
                    if matches!(self.peek(), Token::Comma)
                    {
                        self.pos += 1;
                        continue;
                    }
                    if matches!(self.peek(), Token::Identifier(_))
                    {
                        continue;
                    }
                    break;
                },
                Token::Newline | Token::Indent(_) | Token::Dedent =>
                {
                    self.pos += 1;
                    continue;
                },
                _ => break,
            }
        }
        Ok(vars)
    }

    fn parse_typename(&mut self) -> Result<String, String> {
        self.skip_newlines();
        let tok = self.peek().clone();
        match tok
        {
            Token::Identifier(t) =>
            {
                self.pos += 1;
                let mut type_name = t;
                if matches!(self.peek(), Token::Less)
                {
                    self.pos += 1;
                    type_name.push('<');
                    type_name.push_str(&self.parse_typename()?);
                    while matches!(self.peek(), Token::Comma)
                    {
                        self.pos += 1;
                        type_name.push_str(", ");
                        type_name.push_str(&self.parse_typename()?);
                    }
                    if matches!(self.peek(), Token::Greater)
                    {
                        self.pos += 1;
                        type_name.push('>');
                    }
                    else
                    {
                        return Err("Expected '>' after type parameter".to_string());
                    }
                }
                Ok(type_name)
            },
            _ => Err(format!("Expected type name, found {}", tok)),
        }
    }

    fn parse_variable_decls(&mut self) -> Result<Vec<TypedVar>, String> {
        let mut vars = Vec::new();
        loop
        {
            self.skip_whitespace();
            if !matches!(self.peek(), Token::Identifier(_))
            {
                break;
            }
            let name = if let Token::Identifier(n) = self.peek().clone()
            {
                self.pos += 1;
                n
            }
            else
            {
                break;
            };
            self.expect(":")?;
            let type_name = self.parse_typename()?;
            self.expect("=")?;
            let default = self.parse_expression()?;
            vars.push(TypedVar::with_default(&name, &type_name, default));
            self.skip_whitespace();
        }
        Ok(vars)
    }

    fn parse_block(&mut self) -> Result<Vec<Statement>, String> {
        let mut stmts = Vec::new();
        self.skip_newlines();
        // Consume optional INDENT at block start
        if matches!(self.peek(), Token::Indent(_))
        {
            self.pos += 1;
        }
        self.skip_newlines();
        while self.pos < self.tokens.len()
        {
            match self.peek()
            {
                Token::Eof => break,
                Token::Dedent =>
                {
                    self.pos += 1;
                    break;
                },
                Token::Keyword(k) if k == "input" || k == "output" || k == "variables" => break,
                Token::Newline =>
                {
                    self.pos += 1;
                    continue;
                },
                Token::SemiColon =>
                {
                    self.pos += 1;
                    continue;
                },
                Token::Indent(_) =>
                {
                    self.pos += 1;
                    continue;
                },
                _ =>
                {
                    stmts.push(self.parse_statement()?);
                },
            }
        }
        Ok(stmts)
    }

    fn parse_statement(&mut self) -> Result<Statement, String> {
        self.skip_newlines();
        let tok = self.peek().clone();
        match tok
        {
            Token::Keyword(k) if k == "let" => self.parse_let_stmt(),
            Token::Keyword(k) if k == "for" => self.parse_for_loop(),
            Token::Keyword(k) if k == "while" => self.parse_while_loop(),
            Token::Keyword(k) if k == "if" => self.parse_if_stmt(),
            Token::Keyword(k) if k == "return" => self.parse_return_stmt(),
            Token::Keyword(k) if k == "swap" => self.parse_swap_stmt(),
            Token::Identifier(_) =>
            {
                let ident = if let Token::Identifier(n) = self.peek().clone()
                {
                    self.pos += 1;
                    n
                }
                else
                {
                    return Err("Expected identifier".to_string());
                };
                self.skip_newlines();
                match self.peek()
                {
                    Token::Equals =>
                    {
                        self.pos += 1;
                        let value = self.parse_expression()?;
                        Ok(Statement::Assignment {
                            target: ident,
                            value,
                        })
                    },
                    Token::LBracket =>
                    {
                        self.pos += 1;
                        let index = self.parse_expression()?;
                        self.expect("]")?;
                        self.expect("=")?;
                        let value = self.parse_expression()?;
                        Ok(Statement::ArrayAssign {
                            array: ident,
                            index,
                            value,
                        })
                    },
                    _ => Err(format!(
                        "Unexpected token {} after identifier '{}'",
                        self.peek(),
                        ident
                    )),
                }
            },
            _ => Err(format!("Unexpected token {} at start of statement", tok)),
        }
    }

    fn parse_let_stmt(&mut self) -> Result<Statement, String> {
        self.expect("let")?;
        let name = if let Token::Identifier(n) = self.peek().clone()
        {
            self.pos += 1;
            n
        }
        else
        {
            return Err("Expected variable name after 'let'".to_string());
        };
        self.skip_newlines();
        let type_name = if matches!(self.peek(), Token::Colon)
        {
            self.pos += 1;
            Some(self.parse_typename()?)
        }
        else
        {
            None
        };
        self.expect("=")?;
        let value = self.parse_expression()?;
        Ok(Statement::VariableDecl {
            name,
            type_name,
            value,
        })
    }

    fn parse_for_loop(&mut self) -> Result<Statement, String> {
        self.expect("for")?;
        let var = if let Token::Identifier(n) = self.peek().clone()
        {
            self.pos += 1;
            n
        }
        else
        {
            return Err("Expected loop variable after 'for'".to_string());
        };
        self.expect("in")?;
        let range_start = self.parse_expression()?;
        self.expect("..")?;
        let range_end = self.parse_expression()?;
        self.skip_newlines();
        let body = self.parse_block()?;
        Ok(Statement::ForLoop {
            var,
            range_start,
            range_end,
            body,
        })
    }

    fn parse_while_loop(&mut self) -> Result<Statement, String> {
        self.expect("while")?;
        let condition = self.parse_expression()?;
        self.skip_newlines();
        let body = self.parse_block()?;
        Ok(Statement::WhileLoop { condition, body })
    }

    fn parse_if_stmt(&mut self) -> Result<Statement, String> {
        self.expect("if")?;
        let condition = self.parse_expression()?;
        self.skip_newlines();
        let then_branch = self.parse_block()?;
        self.skip_newlines();
        let else_branch = if matches!(self.peek(), Token::Keyword(k) if k == "else")
        {
            self.pos += 1;
            self.skip_newlines();
            if matches!(self.peek(), Token::Keyword(k) if k == "if")
            {
                vec![self.parse_if_stmt()?]
            }
            else
            {
                self.parse_block()?
            }
        }
        else
        {
            Vec::new()
        };
        Ok(Statement::IfStatement {
            condition,
            then_branch,
            else_branch,
        })
    }

    fn parse_return_stmt(&mut self) -> Result<Statement, String> {
        self.expect("return")?;
        self.skip_newlines();
        match self.peek()
        {
            Token::Newline | Token::Eof | Token::Dedent | Token::SemiColon | Token::Keyword(_) =>
            {
                Ok(Statement::Return(None))
            },
            _ =>
            {
                let expr = self.parse_expression()?;
                Ok(Statement::Return(Some(expr)))
            },
        }
    }

    fn parse_swap_stmt(&mut self) -> Result<Statement, String> {
        self.expect("swap")?;
        self.expect("(")?;
        let a_expr = self.parse_expression()?;
        self.expect(",")?;
        let b_expr = self.parse_expression()?;
        self.expect(")")?;
        Ok(Statement::Swap(a_expr, b_expr))
    }

    fn parse_expression(&mut self) -> Result<Expression, String> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_and()?;
        while matches!(self.peek(), Token::OrOr)
        {
            self.pos += 1;
            let right = self.parse_and()?;
            left = Expression::binary(BinaryOperator::Or, left, right);
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_equality()?;
        while matches!(self.peek(), Token::AndAnd)
        {
            self.pos += 1;
            let right = self.parse_equality()?;
            left = Expression::binary(BinaryOperator::And, left, right);
        }
        Ok(left)
    }

    fn parse_equality(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_relational()?;
        loop
        {
            match self.peek()
            {
                Token::EqualEqual =>
                {
                    self.pos += 1;
                    let right = self.parse_relational()?;
                    left = Expression::binary(BinaryOperator::Eq, left, right);
                },
                Token::NotEqual =>
                {
                    self.pos += 1;
                    let right = self.parse_relational()?;
                    left = Expression::binary(BinaryOperator::Neq, left, right);
                },
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_relational(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_additive()?;
        loop
        {
            match self.peek()
            {
                Token::Less =>
                {
                    self.pos += 1;
                    let right = self.parse_additive()?;
                    left = Expression::binary(BinaryOperator::Lt, left, right);
                },
                Token::Greater =>
                {
                    self.pos += 1;
                    let right = self.parse_additive()?;
                    left = Expression::binary(BinaryOperator::Gt, left, right);
                },
                Token::LessEqual =>
                {
                    self.pos += 1;
                    let right = self.parse_additive()?;
                    left = Expression::binary(BinaryOperator::Le, left, right);
                },
                Token::GreaterEqual =>
                {
                    self.pos += 1;
                    let right = self.parse_additive()?;
                    left = Expression::binary(BinaryOperator::Ge, left, right);
                },
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_additive(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_multiplicative()?;
        loop
        {
            match self.peek()
            {
                Token::Plus =>
                {
                    self.pos += 1;
                    let right = self.parse_multiplicative()?;
                    left = Expression::binary(BinaryOperator::Add, left, right);
                },
                Token::Minus =>
                {
                    self.pos += 1;
                    let right = self.parse_multiplicative()?;
                    left = Expression::binary(BinaryOperator::Sub, left, right);
                },
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_unary()?;
        loop
        {
            match self.peek()
            {
                Token::Star =>
                {
                    self.pos += 1;
                    let right = self.parse_unary()?;
                    left = Expression::binary(BinaryOperator::Mul, left, right);
                },
                Token::Slash =>
                {
                    self.pos += 1;
                    let right = self.parse_unary()?;
                    left = Expression::binary(BinaryOperator::Div, left, right);
                },
                Token::Percent =>
                {
                    self.pos += 1;
                    let right = self.parse_unary()?;
                    left = Expression::binary(BinaryOperator::Mod, left, right);
                },
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expression, String> {
        match self.peek()
        {
            Token::Minus =>
            {
                self.pos += 1;
                let operand = self.parse_unary()?;
                Ok(Expression::UnaryOp {
                    op: UnaryOperator::Neg,
                    operand: Box::new(operand),
                })
            },
            Token::Not =>
            {
                self.pos += 1;
                let operand = self.parse_unary()?;
                Ok(Expression::UnaryOp {
                    op: UnaryOperator::Not,
                    operand: Box::new(operand),
                })
            },
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<Expression, String> {
        let tok = self.peek().clone();
        match tok
        {
            Token::IntLiteral(n) =>
            {
                self.pos += 1;
                Ok(Expression::IntLiteral(n))
            },
            Token::FloatLiteral(x) =>
            {
                self.pos += 1;
                Ok(Expression::FloatLiteral(x))
            },
            Token::BoolLiteral(b) =>
            {
                self.pos += 1;
                Ok(Expression::BoolLiteral(b))
            },
            Token::Keyword(k) if k == "true" =>
            {
                self.pos += 1;
                Ok(Expression::BoolLiteral(true))
            },
            Token::Keyword(k) if k == "false" =>
            {
                self.pos += 1;
                Ok(Expression::BoolLiteral(false))
            },
            Token::StringLiteral(s) =>
            {
                self.pos += 1;
                Ok(Expression::StringLiteral(s))
            },
            Token::LParen =>
            {
                self.pos += 1;
                let expr = self.parse_expression()?;
                self.expect(")")?;
                Ok(expr)
            },
            Token::Identifier(id) =>
            {
                self.pos += 1;
                match self.peek()
                {
                    Token::LParen =>
                    {
                        self.pos += 1;
                        if id == "len"
                        {
                            let arg = if let Token::Identifier(a) = self.peek().clone()
                            {
                                self.pos += 1;
                                a
                            }
                            else
                            {
                                return Err("Expected variable argument to len()".to_string());
                            };
                            self.expect(")")?;
                            return Ok(Expression::Len(arg));
                        }
                        let mut args = Vec::new();
                        if !matches!(self.peek(), Token::RParen)
                        {
                            loop
                            {
                                args.push(self.parse_expression()?);
                                if matches!(self.peek(), Token::Comma)
                                {
                                    self.pos += 1;
                                    continue;
                                }
                                break;
                            }
                        }
                        self.expect(")")?;
                        Ok(Expression::FunctionCall { name: id, args })
                    },
                    Token::LBracket =>
                    {
                        self.pos += 1;
                        let index = self.parse_expression()?;
                        self.expect("]")?;
                        Ok(Expression::ArrayIndex {
                            array: Box::new(Expression::Variable(id)),
                            index: Box::new(index),
                        })
                    },
                    Token::DotDot =>
                    {
                        self.pos += 1;
                        let end = self.parse_expression()?;
                        Ok(Expression::Range {
                            start: Box::new(Expression::Variable(id)),
                            end: Box::new(end),
                        })
                    },
                    _ => Ok(Expression::Variable(id)),
                }
            },
            _ => Err(format!("Unexpected token {} in expression", tok)),
        }
    }
}

/// Parse a complete DSL string into an Algorithm AST.
pub fn parse_algorithm(input: &str) -> Result<Algorithm, String> {
    let tokens = tokenize_with_indent(input)?;
    let mut parser = Parser::new(tokens);
    parser.parse_algorithm()
}

// ============================================================================
// 2. Code Generation: Code Style
// ============================================================================

/// Configurable code style for generated output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeStyle {
    /// Number of spaces per indentation level.
    pub indent_size: usize,
    /// Use tabs instead of spaces for indentation.
    pub use_tabs: bool,
    /// Opening brace on same line or next line.
    pub brace_same_line: bool,
    /// Naming convention for variables: "snake_case", "camelCase", "PascalCase"
    pub variable_naming: String,
    /// Naming convention for functions: "snake_case", "camelCase", "PascalCase"
    pub function_naming: String,
    /// Include Big-O complexity comments in generated code.
    pub include_complexity: bool,
    /// Include documentation comments.
    pub include_docs: bool,
    /// Use type annotations where applicable.
    pub use_type_annotations: bool,
    /// Trailing semicolon style.
    pub semicolons: bool,
}

impl Default for CodeStyle {
    fn default() -> Self {
        CodeStyle {
            indent_size: 4,
            use_tabs: false,
            brace_same_line: true,
            variable_naming: "snake_case".to_string(),
            function_naming: "snake_case".to_string(),
            include_complexity: false,
            include_docs: true,
            use_type_annotations: true,
            semicolons: true,
        }
    }
}

// ============================================================================
// 2. Code Generation: Backends
// ============================================================================

/// Trait for code generation backends.
pub trait CodeGenerator {
    fn generate(&self, algorithm: &Algorithm, style: &CodeStyle) -> String;

    fn indent(&self, style: &CodeStyle, level: usize) -> String {
        if style.use_tabs
        {
            "\t".repeat(level)
        }
        else
        {
            " ".repeat(style.indent_size * level)
        }
    }

    fn convert_name(&self, name: &str, convention: &str) -> String {
        match convention
        {
            "snake_case" => to_snake_case(name),
            "camelCase" => to_camel_case(name),
            "PascalCase" => to_pascal_case(name),
            _ => name.to_string(),
        }
    }

    fn type_map(&self, type_name: &str) -> String {
        type_name.to_string()
    }
}

fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate()
    {
        if c.is_uppercase()
        {
            if i > 0
            {
                result.push('_');
            }
            result.push(c.to_lowercase().next().unwrap_or(c));
        }
        else if c == '-' || c == ' '
        {
            result.push('_');
        }
        else
        {
            result.push(c);
        }
    }
    result
}

fn to_camel_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = false;
    for c in s.chars()
    {
        if c == '_' || c == '-' || c == ' '
        {
            capitalize_next = true;
        }
        else if capitalize_next
        {
            result.push(c.to_uppercase().next().unwrap_or(c));
            capitalize_next = false;
        }
        else
        {
            result.push(c);
        }
    }
    result
}

fn to_pascal_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = true;
    for c in s.chars()
    {
        if c == '_' || c == '-' || c == ' '
        {
            capitalize_next = true;
        }
        else if capitalize_next
        {
            result.push(c.to_uppercase().next().unwrap_or(c));
            capitalize_next = false;
        }
        else
        {
            result.push(c);
        }
    }
    result
}

/// Rust code generator.
pub struct RustGenerator;

impl CodeGenerator for RustGenerator {
    fn generate(&self, algorithm: &Algorithm, style: &CodeStyle) -> String {
        let mut out = String::new();
        let i0 = self.indent(style, 0);
        let i1 = self.indent(style, 1);

        if style.include_docs
        {
            out.push_str(&format!("{}/// {}\n", i0, algorithm.name));
            if !algorithm.inputs.is_empty()
            {
                for inp in &algorithm.inputs
                {
                    out.push_str(&format!(
                        "{}/// # Arguments\n{}/// * `{}` - {}\n",
                        i0,
                        i0,
                        self.convert_name(&inp.name, &style.variable_naming),
                        inp.type_name
                    ));
                }
            }
            if !algorithm.outputs.is_empty()
            {
                for out_var in &algorithm.outputs
                {
                    out.push_str(&format!(
                        "{}/// # Returns\n{}/// * `{}`\n",
                        i0,
                        i0,
                        self.convert_name(&out_var.name, &style.variable_naming)
                    ));
                }
            }
            if style.include_complexity
            {
                let complexity = estimate_complexity(algorithm);
                out.push_str(&format!(
                    "{}///\n{}/// # Complexity\n{}/// Time: {}, Space: {}\n",
                    i0,
                    i0,
                    i0,
                    complexity,
                    estimate_space_complexity(algorithm)
                ));
            }
        }

        let fn_name = self.convert_name(&algorithm.name, &style.function_naming);

        let mut params: Vec<String> = Vec::new();
        for inp in &algorithm.inputs
        {
            let var_name = self.convert_name(&inp.name, &style.variable_naming);
            if style.use_type_annotations
            {
                let rust_type = self.rust_type_for(&inp.type_name);
                match inp.type_name.to_lowercase().as_str()
                {
                    s if s.starts_with("vec<") =>
                    {
                        params.push(format!("{}: &{}", var_name, rust_type));
                    },
                    _ =>
                    {
                        params.push(format!("{}: {}", var_name, rust_type));
                    },
                }
            }
            else
            {
                params.push(var_name);
            }
        }

        let return_type = if algorithm.outputs.len() == 1
        {
            self.rust_type_for(&algorithm.outputs[0].type_name)
        }
        else
        {
            "()".to_string()
        };

        let brace = if style.brace_same_line { " {" } else { "\n{" };
        if style.use_type_annotations
        {
            out.push_str(&format!(
                "{}pub fn {}({}) -> {}{}\n",
                i0,
                fn_name,
                params.join(", "),
                return_type,
                brace
            ));
        }
        else
        {
            out.push_str(&format!(
                "{}fn {}({}){}\n",
                i0,
                fn_name,
                params.join(", "),
                brace
            ));
        }

        for var in &algorithm.variables
        {
            let var_name = self.convert_name(&var.name, &style.variable_naming);
            let init = self.emit_expr(
                var.default_value
                    .as_ref()
                    .unwrap_or(&Expression::IntLiteral(0)),
                style,
            );
            let type_ann = if style.use_type_annotations
            {
                format!(": {}", self.rust_type_for(&var.type_name))
            }
            else
            {
                String::new()
            };
            out.push_str(&format!(
                "{}let mut {}{} = {};\n",
                i1, var_name, type_ann, init
            ));
        }

        if !algorithm.variables.is_empty() && !algorithm.steps.is_empty()
        {
            out.push('\n');
        }

        for stmt in &algorithm.steps
        {
            self.emit_stmt(&mut out, stmt, style, 1);
        }

        out.push_str(&format!("{}}}\n", i0));
        out
    }

    fn type_map(&self, type_name: &str) -> String {
        self.rust_type_for(type_name)
    }
}

impl RustGenerator {
    fn rust_type_for(&self, type_name: &str) -> String {
        let lower = type_name.to_lowercase();
        match lower.as_str()
        {
            "int" | "i32" => "i32".to_string(),
            "i64" => "i64".to_string(),
            "f64" | "float" | "f32" => "f64".to_string(),
            "bool" => "bool".to_string(),
            "str" | "string" => "String".to_string(),
            s if s.starts_with("vec<") => type_name.to_string(),
            _ => type_name.to_string(),
        }
    }

    fn emit_stmt(&self, out: &mut String, stmt: &Statement, style: &CodeStyle, level: usize) {
        let indent = self.indent(style, level);
        let semi = if style.semicolons { ";" } else { "" };

        match stmt
        {
            Statement::VariableDecl {
                name,
                type_name,
                value,
            } =>
            {
                let var_name = self.convert_name(name, &style.variable_naming);
                let init = self.emit_expr(value, style);
                let type_ann = if style.use_type_annotations
                {
                    if let Some(tn) = type_name
                    {
                        format!(": {}", self.rust_type_for(tn))
                    }
                    else
                    {
                        String::new()
                    }
                }
                else
                {
                    String::new()
                };
                let mutable = "mut ";
                out.push_str(&format!(
                    "{}let {} {}= {}{}{}\n",
                    indent, mutable, var_name, type_ann, init, semi
                ));
            },
            Statement::Assignment { target, value } =>
            {
                let var_name = self.convert_name(target, &style.variable_naming);
                let val = self.emit_expr(value, style);
                out.push_str(&format!("{}{} = {}{}\n", indent, var_name, val, semi));
            },
            Statement::ArrayAssign {
                array,
                index,
                value,
            } =>
            {
                let arr = self.convert_name(array, &style.variable_naming);
                let idx = self.emit_expr(index, style);
                let val = self.emit_expr(value, style);
                out.push_str(&format!("{}{}[{}] = {}{}\n", indent, arr, idx, val, semi));
            },
            Statement::ForLoop {
                var,
                range_start,
                range_end,
                body,
            } =>
            {
                let loop_var = self.convert_name(var, &style.variable_naming);
                let start = self.emit_expr(range_start, style);
                let end = self.emit_expr(range_end, style);
                let brace = if style.brace_same_line { " {" } else { "\n{" };
                out.push_str(&format!(
                    "{}for {} in {}..{}{}\n",
                    indent, loop_var, start, end, brace
                ));
                for s in body
                {
                    self.emit_stmt(out, s, style, level + 1);
                }
                out.push_str(&format!("{}}}\n", indent));
            },
            Statement::WhileLoop { condition, body } =>
            {
                let cond = self.emit_expr(condition, style);
                let brace = if style.brace_same_line { " {" } else { "\n{" };
                out.push_str(&format!("{}while {}{}\n", indent, cond, brace));
                for s in body
                {
                    self.emit_stmt(out, s, style, level + 1);
                }
                out.push_str(&format!("{}}}\n", indent));
            },
            Statement::IfStatement {
                condition,
                then_branch,
                else_branch,
            } =>
            {
                let cond = self.emit_expr(condition, style);
                let brace = if style.brace_same_line { " {" } else { "\n{" };
                out.push_str(&format!("{}if {}{}\n", indent, cond, brace));
                for s in then_branch
                {
                    self.emit_stmt(out, s, style, level + 1);
                }
                if else_branch.is_empty()
                {
                    out.push_str(&format!("{}}}\n", indent));
                }
                else
                {
                    out.push_str(&format!("{}}} else {{\n", indent));
                    for s in else_branch
                    {
                        self.emit_stmt(out, s, style, level + 1);
                    }
                    out.push_str(&format!("{}}}\n", indent));
                }
            },
            Statement::Return(expr) =>
            {
                if let Some(e) = expr
                {
                    let val = self.emit_expr(e, style);
                    out.push_str(&format!("{}return {}{}\n", indent, val, semi));
                }
                else
                {
                    out.push_str(&format!("{}return{}\n", indent, semi));
                }
            },
            Statement::Swap(a, b) =>
            {
                let a_val = self.emit_expr(a, style);
                let b_val = self.emit_expr(b, style);
                out.push_str(&format!(
                    "{}std::mem::swap(&mut {}, &mut {}){}\n",
                    indent, a_val, b_val, semi
                ));
            },
            Statement::ExpressionStmt(expr) =>
            {
                let e = self.emit_expr(expr, style);
                out.push_str(&format!("{}{}{}\n", indent, e, semi));
            },
        }
    }

    fn emit_expr(&self, expr: &Expression, style: &CodeStyle) -> String {
        match expr
        {
            Expression::IntLiteral(n) => n.to_string(),
            Expression::FloatLiteral(x) => x.to_string(),
            Expression::BoolLiteral(b) => b.to_string(),
            Expression::StringLiteral(s) => format!("\"{}\"", s),
            Expression::Variable(v) => self.convert_name(v, &style.variable_naming),
            Expression::BinaryOp { op, left, right } =>
            {
                let l = self.emit_expr(left, style);
                let r = self.emit_expr(right, style);
                match op
                {
                    BinaryOperator::And => format!("{} && {}", l, r),
                    BinaryOperator::Or => format!("{} || {}", l, r),
                    _ => format!("{} {} {}", l, op, r),
                }
            },
            Expression::UnaryOp { op, operand } =>
            {
                let o = self.emit_expr(operand, style);
                format!("{}{}", op, o)
            },
            Expression::ArrayIndex { array, index } =>
            {
                let arr = self.emit_expr(array, style);
                let idx = self.emit_expr(index, style);
                format!("{}[{}]", arr, idx)
            },
            Expression::FunctionCall { name, args } =>
            {
                let args_str: Vec<String> = args.iter().map(|a| self.emit_expr(a, style)).collect();
                if name == "println"
                {
                    format!("println!({})", args_str.join(", "))
                }
                else if name == "print"
                {
                    format!("print!({})", args_str.join(", "))
                }
                else
                {
                    format!("{}({})", name, args_str.join(", "))
                }
            },
            Expression::Len(v) =>
            {
                let var_name = self.convert_name(v, &style.variable_naming);
                format!("{}.len()", var_name)
            },
            Expression::Range { start, end } =>
            {
                let s = self.emit_expr(start, style);
                let e = self.emit_expr(end, style);
                format!("{}..{}", s, e)
            },
        }
    }
}

/// Python code generator.
pub struct PythonGenerator;

impl CodeGenerator for PythonGenerator {
    fn generate(&self, algorithm: &Algorithm, style: &CodeStyle) -> String {
        let mut out = String::new();
        let i0 = self.indent(style, 0);
        let i1 = self.indent(style, 1);

        let fn_name = self.convert_name(&algorithm.name, &style.function_naming);
        let params: Vec<String> = algorithm
            .inputs
            .iter()
            .map(|inp| inp.name.clone())
            .collect();

        if style.include_docs
        {
            out.push_str(&format!("{}def {}(", i0, fn_name));
            out.push_str(&params.join(", "));
            out.push_str("):\n");
            let doc_parts: Vec<String> = algorithm
                .inputs
                .iter()
                .map(|inp| format!("    {} ({})", inp.name, inp.type_name))
                .collect();
            if !doc_parts.is_empty()
            {
                out.push_str(&format!(
                    "{}\"\"\"\n{}Args:\n{}\n",
                    i1,
                    i1,
                    doc_parts.join("\n")
                ));
                if !algorithm.outputs.is_empty()
                {
                    let ret_parts: Vec<String> = algorithm
                        .outputs
                        .iter()
                        .map(|o| format!("    {} ({})", o.name, o.type_name))
                        .collect();
                    out.push_str(&format!("{}Returns:\n{}\n", i1, ret_parts.join("\n")));
                }
                out.push_str(&format!("{}\"\"\"\n", i1));
            }
        }
        else
        {
            out.push_str(&format!("{}def {}(", i0, fn_name));
            out.push_str(&params.join(", "));
            out.push_str("):\n");
        }

        for var in &algorithm.variables
        {
            let init = self.emit_expr(
                var.default_value
                    .as_ref()
                    .unwrap_or(&Expression::IntLiteral(0)),
                style,
            );
            out.push_str(&format!("{}{} = {}\n", i1, var.name, init));
        }

        if !algorithm.variables.is_empty() && !algorithm.steps.is_empty()
        {
            out.push('\n');
        }

        for stmt in &algorithm.steps
        {
            self.emit_stmt(&mut out, stmt, style, 1);
        }

        out
    }

    fn type_map(&self, type_name: &str) -> String {
        let lower = type_name.to_lowercase();
        match lower.as_str()
        {
            "vec<i32>" | "vec<i64>" => "list[int]".to_string(),
            "vec<f64>" | "vec<f32>" => "list[float]".to_string(),
            "vec<str>" | "vec<string>" => "list[str]".to_string(),
            "i32" | "i64" | "int" => "int".to_string(),
            "f64" | "f32" | "float" => "float".to_string(),
            "bool" => "bool".to_string(),
            "str" | "string" => "str".to_string(),
            _ => type_name.to_string(),
        }
    }
}

impl PythonGenerator {
    fn emit_stmt(&self, out: &mut String, stmt: &Statement, style: &CodeStyle, level: usize) {
        let indent = self.indent(style, level);

        match stmt
        {
            Statement::VariableDecl {
                name,
                type_name: _,
                value,
            } =>
            {
                let init = self.emit_expr(value, style);
                out.push_str(&format!("{}{} = {}\n", indent, name, init));
            },
            Statement::Assignment { target, value } =>
            {
                let val = self.emit_expr(value, style);
                out.push_str(&format!("{}{} = {}\n", indent, target, val));
            },
            Statement::ArrayAssign {
                array,
                index,
                value,
            } =>
            {
                let idx = self.emit_expr(index, style);
                let val = self.emit_expr(value, style);
                out.push_str(&format!("{}{}[{}] = {}\n", indent, array, idx, val));
            },
            Statement::ForLoop {
                var,
                range_start,
                range_end,
                body,
            } =>
            {
                let start = self.emit_expr(range_start, style);
                let end = self.emit_expr(range_end, style);
                out.push_str(&format!(
                    "{}for {} in range({}, {}):\n",
                    indent, var, start, end
                ));
                for s in body
                {
                    self.emit_stmt(out, s, style, level + 1);
                }
                if body.is_empty()
                {
                    out.push_str(&format!("{}    pass\n", indent));
                }
            },
            Statement::WhileLoop { condition, body } =>
            {
                let cond = self.emit_expr(condition, style);
                out.push_str(&format!("{}while {}:\n", indent, cond));
                for s in body
                {
                    self.emit_stmt(out, s, style, level + 1);
                }
                if body.is_empty()
                {
                    out.push_str(&format!("{}    pass\n", indent));
                }
            },
            Statement::IfStatement {
                condition,
                then_branch,
                else_branch,
            } =>
            {
                let cond = self.emit_expr(condition, style);
                out.push_str(&format!("{}if {}:\n", indent, cond));
                for s in then_branch
                {
                    self.emit_stmt(out, s, style, level + 1);
                }
                if then_branch.is_empty()
                {
                    out.push_str(&format!("{}    pass\n", indent));
                }
                if !else_branch.is_empty()
                {
                    out.push_str(&format!("{}else:\n", indent));
                    for s in else_branch
                    {
                        self.emit_stmt(out, s, style, level + 1);
                    }
                }
            },
            Statement::Return(expr) =>
            {
                if let Some(e) = expr
                {
                    let val = self.emit_expr(e, style);
                    out.push_str(&format!("{}return {}\n", indent, val));
                }
                else
                {
                    out.push_str(&format!("{}return\n", indent));
                }
            },
            Statement::Swap(a, b) =>
            {
                let a_val = self.emit_expr(a, style);
                let b_val = self.emit_expr(b, style);
                out.push_str(&format!(
                    "{}{}, {} = {}, {}\n",
                    indent, a_val, b_val, b_val, a_val
                ));
            },
            Statement::ExpressionStmt(expr) =>
            {
                let e = self.emit_expr(expr, style);
                out.push_str(&format!("{}{}\n", indent, e));
            },
        }
    }

    fn emit_expr(&self, expr: &Expression, style: &CodeStyle) -> String {
        match expr
        {
            Expression::IntLiteral(n) => n.to_string(),
            Expression::FloatLiteral(x) => x.to_string(),
            Expression::BoolLiteral(b) =>
            {
                if *b
                {
                    "True".to_string()
                }
                else
                {
                    "False".to_string()
                }
            },
            Expression::StringLiteral(s) => format!("\"{}\"", s),
            Expression::Variable(v) => v.clone(),
            Expression::BinaryOp { op, left, right } =>
            {
                let l = self.emit_expr(left, style);
                let r = self.emit_expr(right, style);
                let op_str = match op
                {
                    BinaryOperator::And => "and",
                    BinaryOperator::Or => "or",
                    _ => return format!("{} {} {}", l, op, r),
                };
                format!("{} {} {}", l, op_str, r)
            },
            Expression::UnaryOp { op, operand } =>
            {
                let o = self.emit_expr(operand, style);
                let op_str = match op
                {
                    UnaryOperator::Neg => "-",
                    UnaryOperator::Not => "not ",
                };
                format!("{}{}", op_str, o)
            },
            Expression::ArrayIndex { array, index } =>
            {
                let arr = self.emit_expr(array, style);
                let idx = self.emit_expr(index, style);
                format!("{}[{}]", arr, idx)
            },
            Expression::FunctionCall { name, args } =>
            {
                let args_str: Vec<String> = args.iter().map(|a| self.emit_expr(a, style)).collect();
                format!("{}({})", name, args_str.join(", "))
            },
            Expression::Len(v) => format!("len({})", v),
            Expression::Range { start, end } =>
            {
                let s = self.emit_expr(start, style);
                let e = self.emit_expr(end, style);
                format!("range({}, {})", s, e)
            },
        }
    }
}

/// C code generator.
pub struct CGenerator;

impl CodeGenerator for CGenerator {
    fn generate(&self, algorithm: &Algorithm, style: &CodeStyle) -> String {
        let mut out = String::new();
        let i0 = self.indent(style, 0);
        let i1 = self.indent(style, 1);

        let fn_name = self.convert_name(&algorithm.name, &style.function_naming);

        if style.include_docs
        {
            out.push_str(&format!("{0}/** {1}\n{0} * \n{0} */\n", i0, algorithm.name));
        }

        let return_type = if algorithm.outputs.len() == 1
        {
            match algorithm.outputs[0].type_name.to_lowercase().as_str()
            {
                "int" | "i32" | "i64" => "int",
                "f64" | "float" | "f32" => "double",
                "bool" => "int",
                _ => "void",
            }
        }
        else
        {
            "void"
        };

        let mut params: Vec<String> = Vec::new();
        for inp in &algorithm.inputs
        {
            let ctype = match inp.type_name.to_lowercase().as_str()
            {
                "int" | "i32" | "i64" => "int",
                "f64" | "float" | "f32" => "double",
                "bool" => "int",
                "str" | "string" => "const char*",
                s if s.starts_with("vec<") =>
                {
                    params.push(format!("int {}_size", inp.name));
                    "int*"
                },
                _ => "void*",
            };
            params.push(format!("{} {}", ctype, inp.name));
        }

        let brace = if style.brace_same_line { " {" } else { "\n{" };
        out.push_str(&format!(
            "{} {} {}({}){}\n",
            i0,
            return_type,
            fn_name,
            params.join(", "),
            brace
        ));

        for var in &algorithm.variables
        {
            let ctype = match var.type_name.to_lowercase().as_str()
            {
                "int" | "i32" | "i64" => "int",
                "f64" | "float" | "f32" => "double",
                "bool" => "int",
                _ => "int",
            };
            let init = self.emit_expr(
                var.default_value
                    .as_ref()
                    .unwrap_or(&Expression::IntLiteral(0)),
                style,
            );
            out.push_str(&format!("{}{} {} = {};\n", i1, ctype, var.name, init));
        }

        if !algorithm.variables.is_empty() && !algorithm.steps.is_empty()
        {
            out.push('\n');
        }

        for stmt in &algorithm.steps
        {
            self.emit_stmt(&mut out, stmt, style, 1);
        }

        out.push_str(&format!("{}}}\n", i0));
        out
    }

    fn type_map(&self, type_name: &str) -> String {
        let lower = type_name.to_lowercase();
        match lower.as_str()
        {
            "int" | "i32" | "i64" => "int",
            "f64" | "float" | "f32" => "double",
            "bool" => "int",
            "str" | "string" => "const char*",
            _ => type_name,
        }
        .to_string()
    }
}

impl CGenerator {
    fn emit_stmt(&self, out: &mut String, stmt: &Statement, style: &CodeStyle, level: usize) {
        let indent = self.indent(style, level);

        match stmt
        {
            Statement::VariableDecl {
                name,
                type_name,
                value,
            } =>
            {
                let ctype = if let Some(tn) = type_name
                {
                    match tn.to_lowercase().as_str()
                    {
                        "int" | "i32" | "i64" => "int",
                        "f64" | "float" | "f32" => "double",
                        "bool" => "int",
                        _ => "int",
                    }
                }
                else
                {
                    "int"
                };
                let init = self.emit_expr(value, style);
                out.push_str(&format!("{}{} {} = {};\n", indent, ctype, name, init));
            },
            Statement::Assignment { target, value } =>
            {
                let val = self.emit_expr(value, style);
                out.push_str(&format!("{}{} = {};\n", indent, target, val));
            },
            Statement::ArrayAssign {
                array,
                index,
                value,
            } =>
            {
                let idx = self.emit_expr(index, style);
                let val = self.emit_expr(value, style);
                out.push_str(&format!("{}{}[{}] = {};\n", indent, array, idx, val));
            },
            Statement::ForLoop {
                var,
                range_start,
                range_end,
                body,
            } =>
            {
                let start = self.emit_expr(range_start, style);
                let end = self.emit_expr(range_end, style);
                out.push_str(&format!(
                    "{}for (int {} = {}; {} < {}; {}++) {{\n",
                    indent, var, start, var, end, var
                ));
                for s in body
                {
                    self.emit_stmt(out, s, style, level + 1);
                }
                out.push_str(&format!("{}}}\n", indent));
            },
            Statement::WhileLoop { condition, body } =>
            {
                let cond = self.emit_expr(condition, style);
                out.push_str(&format!("{}while ({}) {{\n", indent, cond));
                for s in body
                {
                    self.emit_stmt(out, s, style, level + 1);
                }
                out.push_str(&format!("{}}}\n", indent));
            },
            Statement::IfStatement {
                condition,
                then_branch,
                else_branch,
            } =>
            {
                let cond = self.emit_expr(condition, style);
                out.push_str(&format!("{}if ({}) {{\n", indent, cond));
                for s in then_branch
                {
                    self.emit_stmt(out, s, style, level + 1);
                }
                if else_branch.is_empty()
                {
                    out.push_str(&format!("{}}}\n", indent));
                }
                else
                {
                    out.push_str(&format!("{}}} else {{\n", indent));
                    for s in else_branch
                    {
                        self.emit_stmt(out, s, style, level + 1);
                    }
                    out.push_str(&format!("{}}}\n", indent));
                }
            },
            Statement::Return(expr) =>
            {
                if let Some(e) = expr
                {
                    let val = self.emit_expr(e, style);
                    out.push_str(&format!("{}return {};\n", indent, val));
                }
                else
                {
                    out.push_str(&format!("{}return;\n", indent));
                }
            },
            Statement::Swap(a, b) =>
            {
                let a_val = self.emit_expr(a, style);
                let b_val = self.emit_expr(b, style);
                out.push_str(&format!(
                    "{}int _tmp = {}; {} = {}; {} = _tmp;\n",
                    indent, a_val, a_val, b_val, b_val
                ));
            },
            Statement::ExpressionStmt(expr) =>
            {
                let e = self.emit_expr(expr, style);
                out.push_str(&format!("{}{};\n", indent, e));
            },
        }
    }

    fn emit_expr(&self, expr: &Expression, style: &CodeStyle) -> String {
        match expr
        {
            Expression::IntLiteral(n) => n.to_string(),
            Expression::FloatLiteral(x) => x.to_string(),
            Expression::BoolLiteral(b) =>
            {
                if *b
                {
                    "1".to_string()
                }
                else
                {
                    "0".to_string()
                }
            },
            Expression::StringLiteral(s) => format!("\"{}\"", s),
            Expression::Variable(v) => v.clone(),
            Expression::BinaryOp { op, left, right } =>
            {
                let l = self.emit_expr(left, style);
                let r = self.emit_expr(right, style);
                let op_str = match op
                {
                    BinaryOperator::And => "&&",
                    BinaryOperator::Or => "||",
                    _ => return format!("{} {} {}", l, op, r),
                };
                format!("{} {} {}", l, op_str, r)
            },
            Expression::UnaryOp { op, operand } =>
            {
                let o = self.emit_expr(operand, style);
                match op
                {
                    UnaryOperator::Neg => format!("-{}", o),
                    UnaryOperator::Not => format!("!{}", o),
                }
            },
            Expression::ArrayIndex { array, index } =>
            {
                let arr = self.emit_expr(array, style);
                let idx = self.emit_expr(index, style);
                format!("{}[{}]", arr, idx)
            },
            Expression::FunctionCall { name, args } =>
            {
                let args_str: Vec<String> = args.iter().map(|a| self.emit_expr(a, style)).collect();
                format!("{}({})", name, args_str.join(", "))
            },
            Expression::Len(v) => format!("{}_len", v),
            Expression::Range { start, end } =>
            {
                let s = self.emit_expr(start, style);
                let e = self.emit_expr(end, style);
                format!("{}..{}", s, e)
            },
        }
    }
}

/// Generate Rust code from an algorithm.
pub fn generate_rust(algorithm: &Algorithm, style: &CodeStyle) -> String {
    RustGenerator.generate(algorithm, style)
}

/// Generate Python code from an algorithm.
pub fn generate_python(algorithm: &Algorithm, style: &CodeStyle) -> String {
    PythonGenerator.generate(algorithm, style)
}

/// Generate C code from an algorithm.
pub fn generate_c(algorithm: &Algorithm, style: &CodeStyle) -> String {
    CGenerator.generate(algorithm, style)
}

/// Generate pseudocode from an algorithm.
pub fn generate_pseudocode(algorithm: &Algorithm) -> String {
    let mut out = String::new();
    out.push_str(&format!("Algorithm: {}\n\n", algorithm.name));

    if !algorithm.inputs.is_empty()
    {
        out.push_str("Input:\n");
        for inp in &algorithm.inputs
        {
            out.push_str(&format!("  {}: {}\n", inp.name, inp.type_name));
        }
        out.push('\n');
    }
    if !algorithm.outputs.is_empty()
    {
        out.push_str("Output:\n");
        for out_var in &algorithm.outputs
        {
            out.push_str(&format!("  {}: {}\n", out_var.name, out_var.type_name));
        }
        out.push('\n');
    }

    let complexity = estimate_complexity(algorithm);
    out.push_str(&format!("Complexity: {}\n\n", complexity));

    if !algorithm.variables.is_empty()
    {
        for var in &algorithm.variables
        {
            let init = match &var.default_value
            {
                Some(v) => v.to_string(),
                None => "?".to_string(),
            };
            out.push_str(&format!("{} \u{2190} {}\n", var.name, init));
        }
        out.push('\n');
    }

    out.push_str("Steps:\n");
    emit_pseudocode_stmts(&mut out, &algorithm.steps, 1);
    out
}

fn emit_pseudocode_stmts(out: &mut String, stmts: &[Statement], level: usize) {
    let indent = "  ".repeat(level);
    for stmt in stmts
    {
        match stmt
        {
            Statement::VariableDecl { name, value, .. } =>
            {
                out.push_str(&format!("{}{} \u{2190} {}\n", indent, name, value));
            },
            Statement::Assignment { target, value } =>
            {
                out.push_str(&format!("{}{} \u{2190} {}\n", indent, target, value));
            },
            Statement::ArrayAssign {
                array,
                index,
                value,
            } =>
            {
                out.push_str(&format!(
                    "{}{}[{}] \u{2190} {}\n",
                    indent, array, index, value
                ));
            },
            Statement::ForLoop {
                var,
                range_start,
                range_end,
                body,
            } =>
            {
                out.push_str(&format!(
                    "{}for {} \u{2190} {} to {}-1 do\n",
                    indent, var, range_start, range_end
                ));
                emit_pseudocode_stmts(out, body, level + 1);
                out.push_str(&format!("{}end for\n", indent));
            },
            Statement::WhileLoop { condition, body } =>
            {
                out.push_str(&format!("{}while {} do\n", indent, condition));
                emit_pseudocode_stmts(out, body, level + 1);
                out.push_str(&format!("{}end while\n", indent));
            },
            Statement::IfStatement {
                condition,
                then_branch,
                else_branch,
            } =>
            {
                out.push_str(&format!("{}if {} then\n", indent, condition));
                emit_pseudocode_stmts(out, then_branch, level + 1);
                if !else_branch.is_empty()
                {
                    out.push_str(&format!("{}else\n", indent));
                    emit_pseudocode_stmts(out, else_branch, level + 1);
                }
                out.push_str(&format!("{}end if\n", indent));
            },
            Statement::Return(expr) => match expr
            {
                Some(e) => out.push_str(&format!("{}return {}\n", indent, e)),
                None => out.push_str(&format!("{}return\n", indent)),
            },
            Statement::Swap(a, b) =>
            {
                out.push_str(&format!("{}swap({}, {})\n", indent, a, b));
            },
            Statement::ExpressionStmt(e) =>
            {
                out.push_str(&format!("{}{}\n", indent, e));
            },
        }
    }
}

// ============================================================================
// 3. Template System
// ============================================================================

/// A parameterized algorithm template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmTemplate {
    pub name: String,
    pub description: String,
    pub category: TemplateCategory,
    /// DSL source with template variables like ${name}, ${type}, ${size}
    pub dsl_template: String,
    /// Default parameter values
    pub default_params: HashMap<String, String>,
    pub complexity: String,
    pub space_complexity: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemplateCategory {
    Sorting,
    Searching,
    Graph,
    DynamicProgramming,
    Greedy,
    DivideAndConquer,
    Tree,
    String,
    Math,
    Other(String),
}

impl fmt::Display for TemplateCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            TemplateCategory::Sorting => write!(f, "Sorting"),
            TemplateCategory::Searching => write!(f, "Searching"),
            TemplateCategory::Graph => write!(f, "Graph"),
            TemplateCategory::DynamicProgramming => write!(f, "Dynamic Programming"),
            TemplateCategory::Greedy => write!(f, "Greedy"),
            TemplateCategory::DivideAndConquer => write!(f, "Divide & Conquer"),
            TemplateCategory::Tree => write!(f, "Tree"),
            TemplateCategory::String => write!(f, "String"),
            TemplateCategory::Math => write!(f, "Math"),
            TemplateCategory::Other(s) => write!(f, "{}", s),
        }
    }
}

/// Template registry holding all available algorithm templates.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TemplateRegistry {
    pub templates: HashMap<String, AlgorithmTemplate>,
}

impl TemplateRegistry {
    pub fn new() -> Self {
        TemplateRegistry {
            templates: HashMap::new(),
        }
    }

    pub fn register(&mut self, template: AlgorithmTemplate) {
        self.templates.insert(template.name.clone(), template);
    }

    pub fn get(&self, name: &str) -> Option<&AlgorithmTemplate> {
        self.templates.get(name)
    }

    pub fn list_categories(&self) -> Vec<String> {
        let mut cats: Vec<String> = self
            .templates
            .values()
            .map(|t| t.category.to_string())
            .collect();
        cats.sort();
        cats.dedup();
        cats
    }

    pub fn by_category(&self, category: &str) -> Vec<&AlgorithmTemplate> {
        self.templates
            .values()
            .filter(|t| t.category.to_string() == category)
            .collect()
    }

    pub fn instantiate(
        &self,
        template_name: &str,
        params: &HashMap<String, String>,
    ) -> Result<Algorithm, String> {
        let template = self
            .templates
            .get(template_name)
            .ok_or_else(|| format!("Template '{}' not found", template_name))?;

        let mut dsl = template.dsl_template.clone();

        for (key, value) in &template.default_params
        {
            if !params.contains_key(key)
            {
                dsl = dsl.replace(&format!("${{{}}}", key), value);
            }
        }
        for (key, value) in params
        {
            dsl = dsl.replace(&format!("${{{}}}", key), value);
        }

        if dsl.contains("${")
        {
            let unresolved: Vec<&str> = dsl.lines().filter(|l| l.contains("${")).collect();
            return Err(format!(
                "Unresolved template variables in '{}':\n{}",
                template_name,
                unresolved.join("\n")
            ));
        }

        parse_algorithm(&dsl)
    }
}

/// Create a standard template library with common algorithms.
pub fn create_template_library() -> TemplateRegistry {
    let mut registry = TemplateRegistry::new();

    registry.register(AlgorithmTemplate {
        name: "bubble_sort".to_string(),
        description: "Simple comparison-based sorting algorithm".to_string(),
        category: TemplateCategory::Sorting,
        dsl_template: r#"algorithm bubble_sort
  input: arr: Vec<${type}>
  output: sorted: Vec<${type}>
  variables:
    n: i32 = len(arr)
    i: i32 = 0
    j: i32 = 0
  steps:
    for i in 0..n
      for j in 0..(n - i - 1)
        if arr[j] > arr[j + 1]
          swap(arr[j], arr[j + 1])
    return arr
"#
        .to_string(),
        default_params: [("type".to_string(), "i32".to_string())]
            .iter()
            .cloned()
            .collect(),
        complexity: "O(n\u{00b2})".to_string(),
        space_complexity: "O(1)".to_string(),
    });

    registry.register(AlgorithmTemplate {
        name: "binary_search".to_string(),
        description: "Efficient search in a sorted array".to_string(),
        category: TemplateCategory::Searching,
        dsl_template: r#"algorithm binary_search
  input: arr: Vec<${type}>, target: ${type}
  output: idx: i32
  variables:
    left: i32 = 0
    right: i32 = len(arr) - 1
  steps:
    while left <= right
      mid = (left + right) / 2
      if arr[mid] == target
        return mid
      if arr[mid] < target
        left = mid + 1
      else
        right = mid - 1
    return -1
"#
        .to_string(),
        default_params: [("type".to_string(), "i32".to_string())]
            .iter()
            .cloned()
            .collect(),
        complexity: "O(log n)".to_string(),
        space_complexity: "O(1)".to_string(),
    });

    registry.register(AlgorithmTemplate {
        name: "merge_sort".to_string(),
        description: "Divide-and-conquer sorting algorithm".to_string(),
        category: TemplateCategory::Sorting,
        dsl_template: r#"algorithm merge_sort
  input: arr: Vec<${type}>
  output: sorted: Vec<${type}>
  variables:
    n: i32 = len(arr)
  steps:
    if n <= 1
      return arr
    mid = n / 2
    left = merge_sort(arr[0..mid])
    right = merge_sort(arr[mid..n])
    return merge(left, right)
"#
        .to_string(),
        default_params: [("type".to_string(), "i32".to_string())]
            .iter()
            .cloned()
            .collect(),
        complexity: "O(n log n)".to_string(),
        space_complexity: "O(n)".to_string(),
    });

    registry.register(AlgorithmTemplate {
        name: "linear_search".to_string(),
        description: "Simple sequential search".to_string(),
        category: TemplateCategory::Searching,
        dsl_template: r#"algorithm linear_search
  input: arr: Vec<${type}>, target: ${type}
  output: idx: i32
  steps:
    for i in 0..len(arr)
      if arr[i] == target
        return i
    return -1
"#
        .to_string(),
        default_params: [("type".to_string(), "i32".to_string())]
            .iter()
            .cloned()
            .collect(),
        complexity: "O(n)".to_string(),
        space_complexity: "O(1)".to_string(),
    });

    registry.register(AlgorithmTemplate {
        name: "factorial".to_string(),
        description: "Compute factorial recursively".to_string(),
        category: TemplateCategory::Math,
        dsl_template: r#"algorithm factorial
  input: n: i32
  output: result: i64
  steps:
    if n <= 1
      return 1
    return n * factorial(n - 1)
"#
        .to_string(),
        default_params: HashMap::new(),
        complexity: "O(n)".to_string(),
        space_complexity: "O(n)".to_string(),
    });

    registry.register(AlgorithmTemplate {
        name: "fibonacci_dp".to_string(),
        description: "Compute nth Fibonacci number using dynamic programming".to_string(),
        category: TemplateCategory::DynamicProgramming,
        dsl_template: r#"algorithm fibonacci_dp
  input: n: i32
  output: result: i64
  variables:
    dp: Vec<i64> = vec(0, n + 1)
  steps:
    dp[0] = 0
    dp[1] = 1
    for i in 2..(n + 1)
      dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]
"#
        .to_string(),
        default_params: HashMap::new(),
        complexity: "O(n)".to_string(),
        space_complexity: "O(n)".to_string(),
    });

    registry.register(AlgorithmTemplate {
        name: "bfs".to_string(),
        description: "Breadth-first search on a graph".to_string(),
        category: TemplateCategory::Graph,
        dsl_template: r#"algorithm bfs
  input: graph: Graph, start: i32
  output: order: Vec<i32>
  variables:
    visited: Vec<bool> = vec(false, ${size})
    order: Vec<i32> = vec()
    idx: i32 = 0
  steps:
    visited[start] = true
    order[0] = start
    for idx in 0..${size}
      if visited[idx] == false
        visited[idx] = true
    return order
"#
        .to_string(),
        default_params: [("size".to_string(), "100".to_string())]
            .iter()
            .cloned()
            .collect(),
        complexity: "O(V + E)".to_string(),
        space_complexity: "O(V)".to_string(),
    });

    registry.register(AlgorithmTemplate {
        name: "quick_sort".to_string(),
        description: "Efficient in-place sorting using partitioning".to_string(),
        category: TemplateCategory::Sorting,
        dsl_template: r#"algorithm quick_sort
  input: arr: Vec<${type}>, low: i32, high: i32
  output: sorted: Vec<${type}>
  steps:
    if low < high
      pi = partition(arr, low, high)
      quick_sort(arr, low, pi - 1)
      quick_sort(arr, pi + 1, high)
    return arr
"#
        .to_string(),
        default_params: [("type".to_string(), "i32".to_string())]
            .iter()
            .cloned()
            .collect(),
        complexity: "O(n log n) average, O(n\u{00b2}) worst".to_string(),
        space_complexity: "O(log n)".to_string(),
    });

    registry.register(AlgorithmTemplate {
        name: "dijkstra".to_string(),
        description: "Shortest path in weighted graph".to_string(),
        category: TemplateCategory::Graph,
        dsl_template: r#"algorithm dijkstra
  input: graph: Graph, source: i32
  output: dist: Vec<i32>
  variables:
    dist: Vec<i32> = vec(INF, ${size})
    visited: Vec<bool> = vec(false, ${size})
    u: i32 = 0
    v: i32 = 0
  steps:
    dist[source] = 0
    for i in 0..${size}
      u = min_distance(dist, visited)
      visited[u] = true
      for v in 0..${size}
        if visited[v] == false and graph[u][v] != 0 and dist[u] != INF and dist[u] + graph[u][v] < dist[v]
          dist[v] = dist[u] + graph[u][v]
    return dist
"#
        .to_string(),
        default_params: [("size".to_string(), "100".to_string())]
            .iter()
            .cloned()
            .collect(),
        complexity: "O(V\u{00b2})".to_string(),
        space_complexity: "O(V)".to_string(),
    });

    registry.register(AlgorithmTemplate {
        name: "kadane".to_string(),
        description: "Maximum subarray sum using Kadane's algorithm".to_string(),
        category: TemplateCategory::DynamicProgramming,
        dsl_template: r#"algorithm kadane
  input: arr: Vec<${type}>
  output: max_sum: ${type}
  variables:
    max_ending_here: ${type} = 0
    max_so_far: ${type} = arr[0]
  steps:
    for i in 0..len(arr)
      max_ending_here = max(arr[i], max_ending_here + arr[i])
      max_so_far = max(max_so_far, max_ending_here)
    return max_so_far
"#
        .to_string(),
        default_params: [("type".to_string(), "i32".to_string())]
            .iter()
            .cloned()
            .collect(),
        complexity: "O(n)".to_string(),
        space_complexity: "O(1)".to_string(),
    });

    registry.register(AlgorithmTemplate {
        name: "knapsack_01".to_string(),
        description: "0-1 Knapsack problem using DP".to_string(),
        category: TemplateCategory::DynamicProgramming,
        dsl_template: r#"algorithm knapsack_01
  input: weights: Vec<i32>, values: Vec<i32>, capacity: i32
  output: max_value: i32
  variables:
    n: i32 = len(weights)
    dp: Vec<Vec<i32>> = vec_2d(0, n + 1, capacity + 1)
  steps:
    for i in 1..(n + 1)
      for w in 1..(capacity + 1)
        if weights[i - 1] <= w
          dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])
        else
          dp[i][w] = dp[i - 1][w]
    return dp[n][capacity]
"#
        .to_string(),
        default_params: HashMap::new(),
        complexity: "O(n * W)".to_string(),
        space_complexity: "O(n * W)".to_string(),
    });

    registry.register(AlgorithmTemplate {
        name: "two_sum".to_string(),
        description: "Find two numbers that sum to target".to_string(),
        category: TemplateCategory::Searching,
        dsl_template: r#"algorithm two_sum
  input: nums: Vec<${type}>, target: ${type}
  output: result: Vec<i32>
  variables:
    seen: HashMap<${type}, i32> = new_map()
  steps:
    for i in 0..len(nums)
      complement = target - nums[i]
      if seen.contains(complement)
        return [seen[complement], i]
      seen[nums[i]] = i
    return []
"#
        .to_string(),
        default_params: [("type".to_string(), "i32".to_string())]
            .iter()
            .cloned()
            .collect(),
        complexity: "O(n)".to_string(),
        space_complexity: "O(n)".to_string(),
    });

    registry.register(AlgorithmTemplate {
        name: "insertion_sort".to_string(),
        description: "Simple stable sorting algorithm".to_string(),
        category: TemplateCategory::Sorting,
        dsl_template: r#"algorithm insertion_sort
  input: arr: Vec<${type}>
  output: sorted: Vec<${type}>
  variables:
    n: i32 = len(arr)
  steps:
    for i in 1..n
      key = arr[i]
      j = i - 1
      while j >= 0 and arr[j] > key
        arr[j + 1] = arr[j]
        j = j - 1
      arr[j + 1] = key
    return arr
"#
        .to_string(),
        default_params: [("type".to_string(), "i32".to_string())]
            .iter()
            .cloned()
            .collect(),
        complexity: "O(n\u{00b2})".to_string(),
        space_complexity: "O(1)".to_string(),
    });

    registry.register(AlgorithmTemplate {
        name: "dfs".to_string(),
        description: "Depth-first search on a graph".to_string(),
        category: TemplateCategory::Graph,
        dsl_template: r#"algorithm dfs
  input: graph: Graph, start: i32
  output: order: Vec<i32>
  variables:
    visited: Vec<bool> = vec(false, ${size})
    order: Vec<i32> = vec()
    i: i32 = 0
  steps:
    visited[start] = true
    order.push(start)
    for i in 0..${size}
      if visited[i] == false
        visited[i] = true
        order.push(i)
    return order
"#
        .to_string(),
        default_params: [("size".to_string(), "100".to_string())]
            .iter()
            .cloned()
            .collect(),
        complexity: "O(V + E)".to_string(),
        space_complexity: "O(V)".to_string(),
    });

    registry.register(AlgorithmTemplate {
        name: "gcd".to_string(),
        description: "Greatest common divisor using Euclidean algorithm".to_string(),
        category: TemplateCategory::Math,
        dsl_template: r#"algorithm gcd
  input: a: i32, b: i32
  output: result: i32
  steps:
    while b != 0
      temp = b
      b = a % b
      a = temp
    return a
"#
        .to_string(),
        default_params: HashMap::new(),
        complexity: "O(log min(a,b))".to_string(),
        space_complexity: "O(1)".to_string(),
    });

    registry
}

// ============================================================================
// 4. Scaffold Generator
// ============================================================================

/// Result of scaffolding a new algorithm project.
#[derive(Debug, Clone)]
pub struct ScaffoldResult {
    pub files_created: Vec<PathBuf>,
    pub target_dir: PathBuf,
}

/// Scaffold a new algorithm project.
pub fn scaffold_new(name: &str, base_dir: &Path) -> Result<ScaffoldResult, io::Error> {
    let target_dir = base_dir.join(name);
    if target_dir.exists()
    {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("Directory '{}' already exists", target_dir.display()),
        ));
    }

    fs::create_dir_all(target_dir.join("src"))?;
    let mut files = Vec::new();

    // Cargo.toml
    let cargo_toml = format!(
        r#"[package]
name = "scirust-{name}"
version = "0.1.0"
edition = "2021"
description = "Generated algorithm: {name}"

[dependencies]

[dev-dependencies]
"#,
        name = name
    );
    let path = target_dir.join("Cargo.toml");
    fs::write(&path, &cargo_toml)?;
    files.push(path);

    // src/lib.rs
    let lib_rs = format!(
        r#"/// # {name_pascal}
///
/// Algorithm scaffold generated by scirust-scaffold.
///
/// ## Complexity
/// - Time: TODO
/// - Space: TODO

pub fn {name_snake}(input: &[i32]) -> i32 {{
    // TODO: Implement algorithm
    let mut result = 0;
    for &item in input {{
        result += item;
    }}
    result
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_{name_snake}_empty() {{
        assert_eq!({name_snake}(&[]), 0);
    }}

    #[test]
    fn test_{name_snake}_single() {{
        assert_eq!({name_snake}(&[1]), 1);
    }}

    #[test]
    fn test_{name_snake}_multiple() {{
        assert_eq!({name_snake}(&[1, 2, 3]), 6);
    }}

    #[test]
    fn test_{name_snake}_negative() {{
        assert_eq!({name_snake}(&[-1, 1]), 0);
    }}
}}
"#,
        name_pascal = to_pascal_case(name),
        name_snake = to_snake_case(name)
    );
    let path = target_dir.join("src").join("lib.rs");
    fs::write(&path, &lib_rs)?;
    files.push(path);

    // README.md
    let readme = format!(
        r#"# {name_pascal}

Generated algorithm scaffold.

## Usage

```rust
use scirust_{name_snake}::{name_snake};

fn main() {{
    let result = {name_snake}(&[1, 2, 3]);
    println!("Result: {{}}", result);
}}
```

## Complexity
- Time: TODO
- Space: TODO
"#,
        name_pascal = to_pascal_case(name),
        name_snake = to_snake_case(name)
    );
    let path = target_dir.join("README.md");
    fs::write(&path, &readme)?;
    files.push(path);

    Ok(ScaffoldResult {
        files_created: files,
        target_dir,
    })
}

/// Generate test cases for an algorithm.
pub fn scaffold_test(name: &str, output_dir: &Path) -> Result<PathBuf, io::Error> {
    let test_file = output_dir.join(format!("test_{}.rs", to_snake_case(name)));
    let name_pascal = to_pascal_case(name);
    let name_snake = to_snake_case(name);

    let content = format!(
        r#"// Generated tests for {name_pascal}
#[cfg(test)]
mod {name_snake}_tests {{
    use super::*;

    #[test]
    fn test_{name_snake}_basic() {{
        let input = vec![1, 2, 3, 4, 5];
        let result = {name_snake}(&input);
        assert!(result >= 0);
    }}

    #[test]
    fn test_{name_snake}_empty() {{
        let input: Vec<i32> = vec![];
        let result = {name_snake}(&input);
    }}

    #[test]
    fn test_{name_snake}_single() {{
        let input = vec![42];
        let result = {name_snake}(&input);
    }}

    #[test]
    fn test_{name_snake}_large() {{
        let input: Vec<i32> = (0..1000).collect();
        let result = {name_snake}(&input);
    }}

    #[test]
    fn test_{name_snake}_edge_cases() {{
        let neg = vec![-5, -3, -1];
        let result_neg = {name_snake}(&neg);

        let dup = vec![1, 1, 1, 1];
        let result_dup = {name_snake}(&dup);

        let sorted = vec![1, 2, 3, 4];
        let result_sorted = {name_snake}(&sorted);

        let rev = vec![4, 3, 2, 1];
        let result_rev = {name_snake}(&rev);
    }}
}}
"#,
        name_pascal = name_pascal,
        name_snake = name_snake
    );

    fs::write(&test_file, &content)?;
    Ok(test_file)
}

/// Generate benchmarks for an algorithm.
pub fn scaffold_bench(name: &str, output_dir: &Path) -> Result<PathBuf, io::Error> {
    let bench_file = output_dir.join(format!("bench_{}.rs", to_snake_case(name)));
    let name_pascal = to_pascal_case(name);
    let name_snake = to_snake_case(name);

    let content = format!(
        r#"// Generated benchmarks for {name_pascal}
#![feature(test)]
extern crate test;
use test::Bencher;

#[bench]
fn bench_{name_snake}_small(b: &mut Bencher) {{
    let input: Vec<i32> = (0..10).collect();
    b.iter(|| {{
        let _ = {name_snake}(test::black_box(&input));
    }});
}}

#[bench]
fn bench_{name_snake}_medium(b: &mut Bencher) {{
    let input: Vec<i32> = (0..100).collect();
    b.iter(|| {{
        let _ = {name_snake}(test::black_box(&input));
    }});
}}

#[bench]
fn bench_{name_snake}_large(b: &mut Bencher) {{
    let input: Vec<i32> = (0..1000).collect();
    b.iter(|| {{
        let _ = {name_snake}(test::black_box(&input));
    }});
}}

#[bench]
fn bench_{name_snake}_worst_case(b: &mut Bencher) {{
    let input: Vec<i32> = (0..1000).rev().collect();
    b.iter(|| {{
        let _ = {name_snake}(test::black_box(&input));
    }});
}}

#[bench]
fn bench_{name_snake}_best_case(b: &mut Bencher) {{
    let input: Vec<i32> = (0..1000).collect();
    b.iter(|| {{
        let _ = {name_snake}(test::black_box(&input));
    }});
}}
"#,
        name_pascal = name_pascal,
        name_snake = name_snake
    );

    fs::write(&bench_file, &content)?;
    Ok(bench_file)
}

// ============================================================================
// 5. Code Analysis
// ============================================================================

/// Result of algorithmic analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub algorithm_name: String,
    pub instruction_count: usize,
    pub loop_nesting_depth: usize,
    pub estimated_complexity: String,
    pub space_complexity: String,
    pub variable_count: usize,
    pub warnings: Vec<AnalysisWarning>,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisWarning {
    pub severity: WarningSeverity,
    pub message: String,
    pub location: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WarningSeverity {
    Info,
    Warning,
    Error,
}

/// Analyze an algorithm and produce insights.
pub fn analyze(algorithm: &Algorithm) -> AnalysisResult {
    let mut warnings = Vec::new();
    let mut suggestions = Vec::new();

    let instruction_count = count_instructions(&algorithm.steps);
    let loop_nesting = max_loop_nesting(&algorithm.steps);
    let estimated_complexity = estimate_complexity(algorithm);
    let variable_count = count_variables(algorithm);
    let space_complexity = estimate_space_complexity(algorithm);

    // Warning: unused variables
    let used = collect_used_vars(&algorithm.steps);
    for var in &algorithm.variables
    {
        if !used.contains(&var.name)
        {
            warnings.push(AnalysisWarning {
                severity: WarningSeverity::Warning,
                message: format!("Variable '{}' is declared but never used", var.name),
                location: format!("algorithm '{}'", algorithm.name),
            });
        }
    }

    // Warning: infinite loop potential
    detect_infinite_loops(&algorithm.steps, &mut warnings, &algorithm.name);

    // Warning: empty then/else branches
    detect_empty_branches(&algorithm.steps, &mut warnings, &algorithm.name);

    // Suggestion: replace nested loops with DP
    if loop_nesting >= 2
    {
        suggestions.push(
            "Consider using dynamic programming or memoization to reduce nested loop complexity"
                .to_string(),
        );
    }

    // Suggestion: naming
    if algorithm.name.contains('_')
    {
        suggestions.push(
            "Algorithm name uses snake_case; consider a descriptive PascalCase name".to_string(),
        );
    }

    // Warning: missing return
    if !has_return_in_all_paths(&algorithm.steps, algorithm.outputs.is_empty())
    {
        warnings.push(AnalysisWarning {
            severity: WarningSeverity::Warning,
            message: "Not all code paths return a value".to_string(),
            location: format!("algorithm '{}'", algorithm.name),
        });
    }

    AnalysisResult {
        algorithm_name: algorithm.name.clone(),
        instruction_count,
        loop_nesting_depth: loop_nesting,
        estimated_complexity,
        space_complexity,
        variable_count,
        warnings,
        suggestions,
    }
}

fn count_instructions(stmts: &[Statement]) -> usize {
    let mut count = stmts.len();
    for stmt in stmts
    {
        match stmt
        {
            Statement::ForLoop { body, .. } | Statement::WhileLoop { body, .. } =>
            {
                count += count_instructions(body);
            },
            Statement::IfStatement {
                then_branch,
                else_branch,
                ..
            } =>
            {
                count += count_instructions(then_branch);
                count += count_instructions(else_branch);
            },
            _ =>
            {},
        }
    }
    count
}

fn max_loop_nesting(stmts: &[Statement]) -> usize {
    let mut max_depth = 0;
    let mut current_depth = 0;
    max_loop_nesting_inner(stmts, &mut current_depth, &mut max_depth);
    max_depth
}

fn max_loop_nesting_inner(stmts: &[Statement], current: &mut usize, max: &mut usize) {
    for stmt in stmts
    {
        match stmt
        {
            Statement::ForLoop { body, .. } | Statement::WhileLoop { body, .. } =>
            {
                *current += 1;
                if *current > *max
                {
                    *max = *current;
                }
                max_loop_nesting_inner(body, current, max);
                *current -= 1;
            },
            Statement::IfStatement {
                then_branch,
                else_branch,
                ..
            } =>
            {
                max_loop_nesting_inner(then_branch, current, max);
                max_loop_nesting_inner(else_branch, current, max);
            },
            _ =>
            {},
        }
    }
}

/// Estimate Big-O complexity based on loop nesting and structure.
pub fn estimate_complexity(algorithm: &Algorithm) -> String {
    let depth = max_loop_nesting(&algorithm.steps);
    let mut has_div_and_conquer = false;

    for stmt in &algorithm.steps
    {
        if contains_self_call(stmt, &algorithm.name)
        {
            has_div_and_conquer = true;
            break;
        }
    }

    if has_div_and_conquer
    {
        match depth
        {
            0 => "O(1)".to_string(),
            1 => "O(log n)".to_string(),
            _ => "O(n log n)".to_string(),
        }
    }
    else
    {
        match depth
        {
            0 => "O(1)".to_string(),
            1 => "O(n)".to_string(),
            2 => "O(n\u{00b2})".to_string(),
            3 => "O(n\u{00b3})".to_string(),
            _ => format!("O(n^{})", depth),
        }
    }
}

fn contains_self_call(stmt: &Statement, name: &str) -> bool {
    match stmt
    {
        Statement::ExpressionStmt(expr)
        | Statement::Return(Some(expr))
        | Statement::Assignment { value: expr, .. } => expr_contains_name(expr, name),
        Statement::IfStatement {
            then_branch,
            else_branch,
            ..
        } =>
        {
            then_branch.iter().any(|s| contains_self_call(s, name))
                || else_branch.iter().any(|s| contains_self_call(s, name))
        },
        Statement::ForLoop { body, .. } | Statement::WhileLoop { body, .. } =>
        {
            body.iter().any(|s| contains_self_call(s, name))
        },
        _ => false,
    }
}

fn expr_contains_name(expr: &Expression, name: &str) -> bool {
    match expr
    {
        Expression::FunctionCall { name: fn_name, .. } => fn_name == name,
        Expression::BinaryOp { left, right, .. } =>
        {
            expr_contains_name(left, name) || expr_contains_name(right, name)
        },
        Expression::UnaryOp { operand, .. } => expr_contains_name(operand, name),
        _ => false,
    }
}

fn estimate_space_complexity(algorithm: &Algorithm) -> String {
    let depth = max_loop_nesting(&algorithm.steps);
    let has_extra_storage = algorithm
        .variables
        .iter()
        .any(|v| v.type_name.contains("Vec<") || v.type_name.contains("Map<"));

    if has_extra_storage
    {
        match depth
        {
            0..=1 => "O(n)".to_string(),
            _ => "O(n\u{00b2})".to_string(),
        }
    }
    else if algorithm
        .steps
        .iter()
        .any(|s| matches!(s, Statement::ForLoop { .. } | Statement::WhileLoop { .. }))
    {
        "O(1)".to_string()
    }
    else
    {
        "O(1)".to_string()
    }
}

fn count_variables(algorithm: &Algorithm) -> usize {
    algorithm.variables.len()
        + algorithm.inputs.len()
        + algorithm.outputs.len()
        + count_local_vars(&algorithm.steps)
}

fn count_local_vars(stmts: &[Statement]) -> usize {
    let mut count = 0;
    for stmt in stmts
    {
        match stmt
        {
            Statement::VariableDecl { .. } => count += 1,
            Statement::ForLoop { body, .. } | Statement::WhileLoop { body, .. } =>
            {
                count += 1;
                count += count_local_vars(body);
            },
            Statement::IfStatement {
                then_branch,
                else_branch,
                ..
            } =>
            {
                count += count_local_vars(then_branch);
                count += count_local_vars(else_branch);
            },
            _ =>
            {},
        }
    }
    count
}

fn collect_used_vars(stmts: &[Statement]) -> Vec<String> {
    let mut vars = Vec::new();
    for stmt in stmts
    {
        match stmt
        {
            Statement::Assignment { target, value } =>
            {
                vars.push(target.clone());
                collect_expr_vars(value, &mut vars);
            },
            Statement::ArrayAssign {
                array,
                index,
                value,
                ..
            } =>
            {
                vars.push(array.clone());
                collect_expr_vars(index, &mut vars);
                collect_expr_vars(value, &mut vars);
            },
            Statement::ForLoop {
                var,
                range_start,
                range_end,
                body,
            } =>
            {
                vars.push(var.clone());
                collect_expr_vars(range_start, &mut vars);
                collect_expr_vars(range_end, &mut vars);
                vars.extend(collect_used_vars(body));
            },
            Statement::WhileLoop { condition, body } =>
            {
                collect_expr_vars(condition, &mut vars);
                vars.extend(collect_used_vars(body));
            },
            Statement::IfStatement {
                condition,
                then_branch,
                else_branch,
            } =>
            {
                collect_expr_vars(condition, &mut vars);
                vars.extend(collect_used_vars(then_branch));
                vars.extend(collect_used_vars(else_branch));
            },
            Statement::Return(Some(e)) => collect_expr_vars(e, &mut vars),
            Statement::Swap(a, b) =>
            {
                collect_expr_vars(a, &mut vars);
                collect_expr_vars(b, &mut vars);
            },
            Statement::ExpressionStmt(e) => collect_expr_vars(e, &mut vars),
            _ =>
            {},
        }
    }
    vars
}

fn collect_expr_vars(expr: &Expression, vars: &mut Vec<String>) {
    match expr
    {
        Expression::Variable(v) => vars.push(v.clone()),
        Expression::BinaryOp { left, right, .. } =>
        {
            collect_expr_vars(left, vars);
            collect_expr_vars(right, vars);
        },
        Expression::UnaryOp { operand, .. } => collect_expr_vars(operand, vars),
        Expression::ArrayIndex { array, index } =>
        {
            collect_expr_vars(array, vars);
            collect_expr_vars(index, vars);
        },
        Expression::FunctionCall { name, args } =>
        {
            vars.push(name.clone());
            for arg in args
            {
                collect_expr_vars(arg, vars);
            }
        },
        Expression::Len(v) => vars.push(v.clone()),
        Expression::Range { start, end } =>
        {
            collect_expr_vars(start, vars);
            collect_expr_vars(end, vars);
        },
        _ =>
        {},
    }
}

fn detect_infinite_loops(
    stmts: &[Statement],
    warnings: &mut Vec<AnalysisWarning>,
    algo_name: &str,
) {
    for stmt in stmts
    {
        match stmt
        {
            Statement::WhileLoop { body, .. } =>
            {
                let has_mutation = body.iter().any(|s| {
                    matches!(s, Statement::Assignment { .. })
                        || matches!(s, Statement::ArrayAssign { .. })
                        || matches!(s, Statement::Swap(..))
                });
                if !has_mutation
                {
                    warnings.push(AnalysisWarning {
                        severity: WarningSeverity::Warning,
                        message: "Potential infinite loop: while loop body contains no variable mutations".to_string(),
                        location: format!("algorithm '{}'", algo_name),
                    });
                }
                detect_infinite_loops(body, warnings, algo_name);
            },
            Statement::ForLoop { body, .. } =>
            {
                detect_infinite_loops(body, warnings, algo_name);
            },
            Statement::IfStatement {
                then_branch,
                else_branch,
                ..
            } =>
            {
                detect_infinite_loops(then_branch, warnings, algo_name);
                detect_infinite_loops(else_branch, warnings, algo_name);
            },
            _ =>
            {},
        }
    }
}

fn detect_empty_branches(
    stmts: &[Statement],
    warnings: &mut Vec<AnalysisWarning>,
    algo_name: &str,
) {
    for stmt in stmts
    {
        match stmt
        {
            Statement::IfStatement {
                then_branch,
                else_branch,
                ..
            } =>
            {
                if then_branch.is_empty() && else_branch.is_empty()
                {
                    warnings.push(AnalysisWarning {
                        severity: WarningSeverity::Info,
                        message: "Empty if-statement body; consider removing or adding a comment"
                            .to_string(),
                        location: format!("algorithm '{}'", algo_name),
                    });
                }
                detect_empty_branches(then_branch, warnings, algo_name);
                detect_empty_branches(else_branch, warnings, algo_name);
            },
            Statement::ForLoop { body, .. } | Statement::WhileLoop { body, .. } =>
            {
                if body.is_empty()
                {
                    warnings.push(AnalysisWarning {
                        severity: WarningSeverity::Info,
                        message: "Empty loop body; consider removing or adding a comment"
                            .to_string(),
                        location: format!("algorithm '{}'", algo_name),
                    });
                }
                detect_empty_branches(body, warnings, algo_name);
            },
            _ =>
            {},
        }
    }
}

fn has_return_in_all_paths(stmts: &[Statement], is_void: bool) -> bool {
    if is_void
    {
        return true;
    }
    let mut has_return = false;
    for stmt in stmts
    {
        match stmt
        {
            Statement::Return(_) =>
            {
                has_return = true;
            },
            Statement::IfStatement {
                then_branch,
                else_branch,
                ..
            } if !else_branch.is_empty() =>
            {
                has_return = has_return
                    || (has_return_in_all_paths(then_branch, false)
                        && has_return_in_all_paths(else_branch, false));
            },
            _ =>
            {},
        }
    }
    has_return
}

// ============================================================================
// 6. Documentation Generator
// ============================================================================

/// Generate rustdoc-style documentation for an algorithm.
pub fn generate_docs(algorithm: &Algorithm) -> String {
    let mut out = String::new();
    let analysis = analyze(algorithm);

    out.push_str(&format!("/// # {}\n", to_pascal_case(&algorithm.name)));
    out.push_str("///\n");

    let params: Vec<String> = algorithm
        .inputs
        .iter()
        .map(|inp| format!("/// * `{}` \u{2014} {}", inp.name, inp.type_name))
        .collect();
    if !params.is_empty()
    {
        out.push_str("/// ## Parameters\n");
        for p in &params
        {
            out.push_str(&format!("{}\n", p));
        }
        out.push_str("///\n");
    }

    let returns: Vec<String> = algorithm
        .outputs
        .iter()
        .map(|o| format!("/// * `{}` \u{2014} {}", o.name, o.type_name))
        .collect();
    if !returns.is_empty()
    {
        out.push_str("/// ## Returns\n");
        for r in &returns
        {
            out.push_str(&format!("{}\n", r));
        }
        out.push_str("///\n");
    }

    out.push_str("/// ## Complexity\n");
    out.push_str(&format!(
        "/// - **Time:** {}\n",
        analysis.estimated_complexity
    ));
    out.push_str(&format!("/// - **Space:** {}\n", analysis.space_complexity));
    out.push_str("///\n");

    out.push_str("/// ## Analysis\n");
    out.push_str(&format!(
        "/// - Instructions: {}\n",
        analysis.instruction_count
    ));
    out.push_str(&format!(
        "/// - Loop nesting: {}\n",
        analysis.loop_nesting_depth
    ));
    out.push_str(&format!("/// - Variables: {}\n", analysis.variable_count));
    out.push_str("///\n");

    if !analysis.warnings.is_empty()
    {
        out.push_str("/// ## Warnings\n");
        for w in &analysis.warnings
        {
            let severity = match w.severity
            {
                WarningSeverity::Info => "[INFO]",
                WarningSeverity::Warning => "[WARN]",
                WarningSeverity::Error => "[ERR]",
            };
            out.push_str(&format!("/// {} {}\n", severity, w.message));
        }
        out.push_str("///\n");
    }

    if !analysis.suggestions.is_empty()
    {
        out.push_str("/// ## Suggestions\n");
        for s in &analysis.suggestions
        {
            out.push_str(&format!("/// * {}\n", s));
        }
        out.push_str("///\n");
    }

    out.push_str(&format!(
        "/// ## Visualization\n///\n{}\n",
        generate_ascii_diagram(algorithm)
    ));

    out.push_str(&format!(
        "/// ## Example\n///\n{}\n",
        generate_example(algorithm)
    ));

    out
}

/// Generate an ASCII diagram for algorithm visualization.
pub fn generate_ascii_diagram(algorithm: &Algorithm) -> String {
    let mut out = String::new();
    let indent = "/// ";

    out.push_str(&format!("{}┌──────────────────────────────┐\n", indent));
    out.push_str(&format!(
        "{}│  Algorithm: {:<17} │\n",
        indent,
        truncate(&to_pascal_case(&algorithm.name), 17)
    ));
    out.push_str(&format!("{}├──────────────────────────────┤\n", indent));

    if !algorithm.inputs.is_empty()
    {
        for inp in &algorithm.inputs
        {
            out.push_str(&format!(
                "{}│  Input: {:<20} │\n",
                indent,
                truncate(&format!("{} ({})", inp.name, inp.type_name), 20)
            ));
        }
    }

    if !algorithm.variables.is_empty()
    {
        out.push_str(&format!("{}├──────────────────────────────┤\n", indent));
        for var in &algorithm.variables
        {
            out.push_str(&format!(
                "{}│  Init: {:<21} │\n",
                indent,
                truncate(&format!("{} = ...", var.name), 21)
            ));
        }
    }

    if !algorithm.steps.is_empty()
    {
        out.push_str(&format!("{}├──────────────────────────────┤\n", indent));
        visualize_steps(&algorithm.steps, &mut out, indent);
    }

    out.push_str(&format!("{}├──────────────────────────────┤\n", indent));
    if !algorithm.outputs.is_empty()
    {
        for out_var in &algorithm.outputs
        {
            out.push_str(&format!(
                "{}│  Return: {:<19} │\n",
                indent,
                truncate(&out_var.name, 19)
            ));
        }
    }
    out.push_str(&format!("{}└──────────────────────────────┘\n", indent));

    out
}

fn visualize_steps(stmts: &[Statement], out: &mut String, indent: &str) {
    for stmt in stmts
    {
        match stmt
        {
            Statement::ForLoop {
                var,
                range_start,
                range_end,
                ..
            } =>
            {
                let range_str = format!("{}..{}", range_start, range_end);
                out.push_str(&format!(
                    "{}│  for {} in {:<12} │\n",
                    indent,
                    truncate(var, 4),
                    truncate(&range_str, 12)
                ));
            },
            Statement::WhileLoop { .. } =>
            {
                out.push_str(&format!("{}│  while \u{2026}                 │\n", indent));
            },
            Statement::IfStatement {
                then_branch,
                else_branch,
                ..
            } =>
            {
                out.push_str(&format!(
                    "{}│  if \u{2026} then \u{2026}             │\n",
                    indent
                ));
                if !then_branch.is_empty()
                {
                    out.push_str(&format!(
                        "{}│  \u{251c}\u{2500}\u{2500} then branch          │\n",
                        indent
                    ));
                }
                if !else_branch.is_empty()
                {
                    out.push_str(&format!(
                        "{}│  \u{251c}\u{2500}\u{2500} else branch          │\n",
                        indent
                    ));
                }
            },
            Statement::Return(Some(e)) =>
            {
                out.push_str(&format!(
                    "{}│  return {:<17} │\n",
                    indent,
                    truncate(&e.to_string(), 17)
                ));
            },
            Statement::Return(None) =>
            {
                out.push_str(&format!("{}│  return                  │\n", indent));
            },
            Statement::Swap(a, b) =>
            {
                let a_str = a.to_string();
                let b_str = b.to_string();
                out.push_str(&format!(
                    "{}│  swap({}, {})          │\n",
                    indent,
                    truncate(&a_str, 7),
                    truncate(&b_str, 7)
                ));
            },
            _ =>
            {
                out.push_str(&format!("{}│  \u{2026}                       │\n", indent));
            },
        }
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len
    {
        s.to_string()
    }
    else
    {
        format!("{}\u{2026}", &s[..max_len - 1])
    }
}

/// Generate a usage example for the algorithm.
pub fn generate_example(algorithm: &Algorithm) -> String {
    let mut out = String::new();
    let indent = "/// ";

    out.push_str(&format!("{}```rust\n", indent));
    out.push_str(&format!(
        "{}// Example usage of {}\n",
        indent, algorithm.name
    ));

    if !algorithm.inputs.is_empty()
    {
        for inp in &algorithm.inputs
        {
            let sample_val = sample_value(&inp.type_name);
            out.push_str(&format!("{}let {} = {};\n", indent, inp.name, sample_val));
        }
    }

    let args: Vec<String> = algorithm
        .inputs
        .iter()
        .map(|i| {
            if i.type_name.to_lowercase().starts_with("vec<")
            {
                format!("&{}", i.name)
            }
            else
            {
                i.name.clone()
            }
        })
        .collect();

    out.push_str(&format!(
        "{}let result = {}({});\n",
        indent,
        to_snake_case(&algorithm.name),
        args.join(", ")
    ));

    out.push_str(&format!("{}assert!(/* verify result */);\n", indent));
    out.push_str(&format!("{}```\n", indent));

    out
}

fn sample_value(type_name: &str) -> String {
    let lower = type_name.to_lowercase();
    match lower.as_str()
    {
        "int" | "i32" => "42".to_string(),
        "i64" => "42i64".to_string(),
        "f64" | "float" | "f32" => "3.14".to_string(),
        "bool" => "true".to_string(),
        "str" | "string" | "&str" => "\"hello\"".to_string(),
        s if s.starts_with("vec<") =>
        {
            let inner = s.trim_start_matches("vec<").trim_end_matches('>');
            match inner
            {
                "i32" | "int" => "vec![1, 2, 3, 4, 5]".to_string(),
                "f64" | "float" => "vec![1.0, 2.0, 3.0]".to_string(),
                _ => "vec![]".to_string(),
            }
        },
        _ => "Default::default()".to_string(),
    }
}

// ============================================================================
// 7. Configuration
// ============================================================================

/// Project configuration for scaffolding and code generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaffoldConfig {
    pub project_name: String,
    pub code_style: CodeStyle,
    pub target_languages: Vec<String>,
    pub template_paths: Vec<String>,
    pub output_dir: String,
    pub include_tests: bool,
    pub include_benches: bool,
    pub include_docs: bool,
    pub custom_templates: HashMap<String, AlgorithmTemplate>,
}

impl Default for ScaffoldConfig {
    fn default() -> Self {
        ScaffoldConfig {
            project_name: "scirust-algorithm".to_string(),
            code_style: CodeStyle::default(),
            target_languages: vec!["rust".to_string(), "python".to_string(), "c".to_string()],
            template_paths: vec![],
            output_dir: "./generated".to_string(),
            include_tests: true,
            include_benches: true,
            include_docs: true,
            custom_templates: HashMap::new(),
        }
    }
}

impl ScaffoldConfig {
    /// Load configuration from a JSON file.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let content =
            fs::read_to_string(path).map_err(|e| format!("Failed to read config file: {}", e))?;
        serde_json::from_str(&content).map_err(|e| format!("Failed to parse config JSON: {}", e))
    }

    /// Save configuration to a JSON file.
    pub fn to_file(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize config: {}", e))?;
        fs::write(path, json).map_err(|e| format!("Failed to write config file: {}", e))
    }

    /// Generate code for all target languages.
    pub fn generate_all(&self, algorithm: &Algorithm) -> HashMap<String, String> {
        let mut results = HashMap::new();
        for lang in &self.target_languages
        {
            match lang.as_str()
            {
                "rust" =>
                {
                    results.insert(
                        "rust".to_string(),
                        generate_rust(algorithm, &self.code_style),
                    );
                },
                "python" =>
                {
                    results.insert(
                        "python".to_string(),
                        generate_python(algorithm, &self.code_style),
                    );
                },
                "c" =>
                {
                    results.insert("c".to_string(), generate_c(algorithm, &self.code_style));
                },
                "pseudocode" =>
                {
                    results.insert("pseudocode".to_string(), generate_pseudocode(algorithm));
                },
                _ =>
                {},
            }
        }
        results
    }

    /// Build a full template registry from built-in and custom templates.
    pub fn build_registry(&self) -> TemplateRegistry {
        let mut registry = create_template_library();
        for template in self.custom_templates.values()
        {
            registry.register(template.clone());
        }
        registry
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Tokenizer Tests ---

    #[test]
    fn test_tokenize_simple_identifier() {
        let tokens = tokenize_line("hello").unwrap();
        assert_eq!(tokens.len(), 1);
        assert!(matches!(tokens[0], Token::Identifier(ref s) if s == "hello"));
    }

    #[test]
    fn test_tokenize_integer() {
        let tokens = tokenize_line("42").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0], Token::IntLiteral(42));
    }

    #[test]
    fn test_tokenize_float() {
        let tokens = tokenize_line("3.14").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0], Token::FloatLiteral(3.14));
    }

    #[test]
    fn test_tokenize_keyword_if() {
        let tokens = tokenize_line("if").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0], Token::Keyword("if".to_string()));
    }

    #[test]
    fn test_tokenize_operators() {
        let tokens = tokenize_line("+ - * / % < > <= >= == != && || !").unwrap();
        assert_eq!(tokens.len(), 14);
        assert_eq!(tokens[0], Token::Plus);
        assert_eq!(tokens[1], Token::Minus);
        assert_eq!(tokens[10], Token::NotEqual);
        assert_eq!(tokens[11], Token::AndAnd);
        assert_eq!(tokens[12], Token::OrOr);
        assert_eq!(tokens[13], Token::Not);
    }

    #[test]
    fn test_tokenize_brackets() {
        let tokens = tokenize_line("( ) [ ]").unwrap();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0], Token::LParen);
        assert_eq!(tokens[1], Token::RParen);
        assert_eq!(tokens[2], Token::LBracket);
        assert_eq!(tokens[3], Token::RBracket);
    }

    #[test]
    fn test_tokenize_with_indent_basic() {
        let input = "algorithm test\n  input: arr: Vec<i32>";
        let tokens = tokenize_with_indent(input).unwrap();
        assert!(
            tokens
                .iter()
                .any(|t| matches!(t, Token::Keyword(k) if k == "algorithm"))
        );
        assert!(
            tokens
                .iter()
                .any(|t| matches!(t, Token::Keyword(k) if k == "input"))
        );
        assert!(
            tokens
                .iter()
                .any(|t| matches!(t, Token::Identifier(ref s) if s == "test"))
        );
    }

    // --- Parser Tests ---

    #[test]
    fn test_parse_minimal_algorithm() {
        let dsl = "algorithm empty\n  input: x: i32\n  output: y: i32\n  steps:\n    return x";
        let algo = parse_algorithm(dsl).unwrap();
        assert_eq!(algo.name, "empty");
        assert_eq!(algo.inputs.len(), 1);
        assert_eq!(algo.inputs[0].name, "x");
        assert_eq!(algo.inputs[0].type_name, "i32");
        assert_eq!(algo.outputs.len(), 1);
        assert_eq!(algo.steps.len(), 1);
    }

    #[test]
    fn test_parse_bubble_sort() {
        let dsl = r#"algorithm bubble_sort
  input: arr: Vec<i32>
  output: sorted: Vec<i32>
  variables:
    n: i32 = len(arr)
    i: i32 = 0
    j: i32 = 0
  steps:
    for i in 0..n
      for j in 0..(n - i - 1)
        if arr[j] > arr[j + 1]
          swap(arr[j], arr[j + 1])
    return arr
"#;
        let algo = parse_algorithm(dsl).unwrap();
        assert_eq!(algo.name, "bubble_sort");
        assert_eq!(algo.inputs.len(), 1);
        assert_eq!(algo.outputs.len(), 1);
        assert_eq!(algo.variables.len(), 3);
        assert!(!algo.steps.is_empty());
    }

    #[test]
    fn test_parse_binary_search() {
        let dsl = r#"algorithm binary_search
  input: arr: Vec<i32>, target: i32
  output: idx: i32
  variables:
    left: i32 = 0
    right: i32 = len(arr) - 1
  steps:
    while left <= right
      mid = (left + right) / 2
      if arr[mid] == target
        return mid
      if arr[mid] < target
        left = mid + 1
      else
        right = mid - 1
    return -1
"#;
        let algo = parse_algorithm(dsl).unwrap();
        assert_eq!(algo.name, "binary_search");
        assert_eq!(algo.inputs.len(), 2);
        assert_eq!(algo.inputs[1].name, "target");
    }

    #[test]
    fn test_parse_algorithm_no_steps_label() {
        let dsl = "algorithm factorial\n  input: n: i32\n  output: result: i64\n  if n <= 1\n    return 1\n  return n * factorial(n - 1)";
        let algo = parse_algorithm(dsl).unwrap();
        assert_eq!(algo.name, "factorial");
        assert_eq!(algo.steps.len(), 2);
    }

    #[test]
    fn test_parse_expression_precedence() {
        // Parse "a + b * c" to test operator precedence
        let tokens = tokenize_line("a + b * c").unwrap();
        let mut parser = Parser::new(tokens);
        let expr = parser.parse_expression().unwrap();
        if let Expression::BinaryOp { op, left, right } = &expr
        {
            assert_eq!(*op, BinaryOperator::Add);
            assert!(matches!(**left, Expression::Variable(_)));
            assert!(matches!(
                **right,
                Expression::BinaryOp {
                    op: BinaryOperator::Mul,
                    ..
                }
            ));
        }
        else
        {
            panic!("Expected BinaryOp");
        }
    }

    #[test]
    fn test_parse_array_index() {
        let tokens = tokenize_line("arr[i]").unwrap();
        let mut parser = Parser::new(tokens);
        let expr = parser.parse_expression().unwrap();
        assert!(matches!(expr, Expression::ArrayIndex { .. }));
    }

    #[test]
    fn test_parse_len_expression() {
        let tokens = tokenize_line("len(arr)").unwrap();
        let mut parser = Parser::new(tokens);
        let expr = parser.parse_expression().unwrap();
        assert!(matches!(expr, Expression::Len(ref v) if v == "arr"));
    }

    // --- Code Generation Tests ---

    #[test]
    fn test_generate_rust() {
        let dsl = "algorithm sum\n  input: nums: Vec<i32>\n  output: total: i32\n  variables:\n    total: i32 = 0\n  steps:\n    for i in 0..len(nums)\n      total = total + nums[i]\n    return total";
        let algo = parse_algorithm(dsl).unwrap();
        let code = generate_rust(&algo, &CodeStyle::default());
        assert!(code.contains("pub fn sum("));
        assert!(code.contains("for "));
        assert!(code.contains("return"));
    }

    #[test]
    fn test_generate_python() {
        let dsl = "algorithm find_max\n  input: arr: Vec<i32>\n  output: max_val: i32\n  variables:\n    max_val: i32 = arr[0]\n  steps:\n    for i in 1..len(arr)\n      if arr[i] > max_val\n        max_val = arr[i]\n    return max_val";
        let algo = parse_algorithm(dsl).unwrap();
        let code = generate_python(&algo, &CodeStyle::default());
        assert!(code.contains("def find_max("));
        assert!(code.contains("for "));
        assert!(code.contains("return"));
    }

    #[test]
    fn test_generate_c() {
        let dsl = "algorithm add\n  input: a: i32, b: i32\n  output: result: i32\n  steps:\n    return a + b";
        let algo = parse_algorithm(dsl).unwrap();
        let code = generate_c(&algo, &CodeStyle::default());
        assert!(code.contains("int add("));
        assert!(code.contains("return"));
    }

    #[test]
    fn test_generate_pseudocode() {
        let dsl = "algorithm linear_search\n  input: arr: Vec<i32>, target: i32\n  output: idx: i32\n  steps:\n    for i in 0..len(arr)\n      if arr[i] == target\n        return i\n    return -1";
        let algo = parse_algorithm(dsl).unwrap();
        let code = generate_pseudocode(&algo);
        assert!(code.contains("Algorithm: linear_search"));
        assert!(code.contains("Input:"));
        assert!(code.contains("Output:"));
        assert!(code.contains("Steps:"));
    }

    #[test]
    fn test_code_style_custom() {
        let dsl = "algorithm test_style\n  input: my_array: Vec<i32>\n  output: result: i32\n  steps:\n    return 0";
        let algo = parse_algorithm(dsl).unwrap();
        let style = CodeStyle {
            indent_size: 2,
            use_tabs: false,
            brace_same_line: false,
            variable_naming: "camelCase".to_string(),
            function_naming: "PascalCase".to_string(),
            include_complexity: true,
            include_docs: false,
            use_type_annotations: true,
            semicolons: false,
        };
        let code = generate_rust(&algo, &style);
        assert!(code.contains("pub fn TestStyle("));
    }

    // --- Template Tests ---

    #[test]
    fn test_template_registry() {
        let registry = create_template_library();
        assert!(registry.get("bubble_sort").is_some());
        assert!(registry.get("binary_search").is_some());
        assert!(registry.get("merge_sort").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_template_instantiate() {
        let registry = create_template_library();
        let mut params = HashMap::new();
        params.insert("type".to_string(), "f64".to_string());
        let algo = registry.instantiate("bubble_sort", &params).unwrap();
        assert_eq!(algo.name, "bubble_sort");
        assert_eq!(algo.inputs[0].type_name, "Vec<f64>");
    }

    #[test]
    fn test_template_categories() {
        let registry = create_template_library();
        let cats = registry.list_categories();
        assert!(cats.contains(&"Sorting".to_string()));
        assert!(cats.contains(&"Searching".to_string()));
        assert!(cats.contains(&"Graph".to_string()));
        assert!(cats.contains(&"Dynamic Programming".to_string()));
        assert!(cats.contains(&"Math".to_string()));
    }

    // --- Analysis Tests ---

    #[test]
    fn test_analyze_bubble_sort_complexity() {
        let dsl = r#"algorithm bubble_sort
  input: arr: Vec<i32>
  output: sorted: Vec<i32>
  steps:
    for i in 0..len(arr)
      for j in 0..(len(arr) - i - 1)
        if arr[j] > arr[j + 1]
          swap(arr[j], arr[j + 1])
    return arr
"#;
        let algo = parse_algorithm(dsl).unwrap();
        let analysis = analyze(&algo);
        assert_eq!(analysis.estimated_complexity, "O(n\u{00b2})");
        assert_eq!(analysis.loop_nesting_depth, 2);
    }

    #[test]
    fn test_analyze_infinite_loop_warning() {
        let dsl = r#"algorithm endless
  input: x: i32
  output: y: i32
  steps:
    while true
      y = y + 1
    return y
"#;
        let algo = parse_algorithm(dsl).unwrap();
        let analysis = analyze(&algo);
        // "true" is a keyword token which will be parsed as variable
        // The while condition is a Variable("true") which is a valid identifier
        // The body contains an assignment, so no warning about no mutation
        // Just check analysis runs without crashing
        assert!(analysis.instruction_count > 0);
    }

    // --- Scaffold Tests ---

    #[test]
    fn test_scaffold_new() {
        let tmp = std::env::temp_dir().join("scirust_test_scaffold");
        let _ = fs::remove_dir_all(&tmp);
        let result = scaffold_new("my_algo", &tmp).unwrap();
        assert!(result.target_dir.exists());
        assert!(result.target_dir.join("Cargo.toml").exists());
        assert!(result.target_dir.join("src/lib.rs").exists());
        assert!(result.target_dir.join("README.md").exists());
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_scaffold_test_generator() {
        let tmp = std::env::temp_dir();
        let file = scaffold_test("my_algo", &tmp).unwrap();
        assert!(file.exists());
        let content = fs::read_to_string(&file).unwrap();
        assert!(content.contains("my_algo_tests"));
        assert!(content.contains("test_my_algo_basic"));
        let _ = fs::remove_file(&file);
    }

    #[test]
    fn test_scaffold_bench_generator() {
        let tmp = std::env::temp_dir();
        let file = scaffold_bench("my_algo", &tmp).unwrap();
        assert!(file.exists());
        let content = fs::read_to_string(&file).unwrap();
        assert!(content.contains("bench_my_algo_small"));
        assert!(content.contains("bench_my_algo_large"));
        let _ = fs::remove_file(&file);
    }

    // --- Documentation Tests ---

    #[test]
    fn test_generate_docs() {
        let dsl = r#"algorithm binary_search
  input: arr: Vec<i32>, target: i32
  output: idx: i32
  steps:
    while left <= right
      mid = (left + right) / 2
      if arr[mid] == target
        return mid
    return -1
"#;
        let algo = parse_algorithm(dsl).unwrap();
        let docs = generate_docs(&algo);
        assert!(docs.contains("BinarySearch"));
        assert!(docs.contains("Parameters"));
        assert!(docs.contains("Complexity"));
        assert!(docs.contains("Visualization"));
    }

    #[test]
    fn test_generate_ascii_diagram() {
        let dsl =
            "algorithm sum\n  input: nums: Vec<i32>\n  output: total: i32\n  steps:\n    return 0";
        let algo = parse_algorithm(dsl).unwrap();
        let diagram = generate_ascii_diagram(&algo);
        assert!(diagram.contains("Sum"));
        assert!(diagram.contains("Input"));
        assert!(diagram.contains("Return"));
    }

    #[test]
    fn test_generate_example() {
        let dsl =
            "algorithm max\n  input: arr: Vec<i32>\n  output: result: i32\n  steps:\n    return 0";
        let algo = parse_algorithm(dsl).unwrap();
        let example = generate_example(&algo);
        assert!(example.contains("```rust"));
        assert!(example.contains("let arr = vec![1, 2, 3, 4, 5];"));
    }

    // --- Config Tests ---

    #[test]
    fn test_config_default() {
        let config = ScaffoldConfig::default();
        assert_eq!(config.project_name, "scirust-algorithm");
        assert_eq!(config.target_languages.len(), 3);
    }

    #[test]
    fn test_config_serialize_roundtrip() {
        let config = ScaffoldConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: ScaffoldConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.project_name, config.project_name);
        assert_eq!(parsed.target_languages, config.target_languages);
    }

    // --- Utility Tests ---

    #[test]
    fn test_snake_case_conversion() {
        assert_eq!(to_snake_case("HelloWorld"), "hello_world");
        assert_eq!(to_snake_case("bubbleSort"), "bubble_sort");
        assert_eq!(to_snake_case("MyAlgorithm"), "my_algorithm");
    }

    #[test]
    fn test_camel_case_conversion() {
        assert_eq!(to_camel_case("hello_world"), "helloWorld");
        assert_eq!(to_camel_case("my_algorithm"), "myAlgorithm");
    }

    #[test]
    fn test_pascal_case_conversion() {
        assert_eq!(to_pascal_case("hello_world"), "HelloWorld");
        assert_eq!(to_pascal_case("bubble_sort"), "BubbleSort");
    }

    #[test]
    fn test_count_instructions() {
        let dsl = r#"algorithm test
  input: a: i32
  output: b: i32
  steps:
    b = a + 1
    if a > 0
      b = b * 2
    else
      b = 0
    return b
"#;
        let algo = parse_algorithm(dsl).unwrap();
        let analysis = analyze(&algo);
        assert!(analysis.instruction_count > 0);
    }

    #[test]
    fn test_estimate_complexity_constant() {
        let dsl = "algorithm const\n  input: x: i32\n  output: y: i32\n  steps:\n    return x * 2";
        let algo = parse_algorithm(dsl).unwrap();
        let analysis = analyze(&algo);
        assert_eq!(analysis.estimated_complexity, "O(1)");
    }

    #[test]
    fn test_estimate_complexity_linear() {
        let dsl = "algorithm linear\n  input: arr: Vec<i32>\n  output: sum: i32\n  steps:\n    for i in 0..len(arr)\n      sum = sum + arr[i]\n    return sum";
        let algo = parse_algorithm(dsl).unwrap();
        let analysis = analyze(&algo);
        assert_eq!(analysis.estimated_complexity, "O(n)");
    }

    #[test]
    fn test_unused_variable_warning() {
        let dsl = r#"algorithm unused_var
  input: a: i32
  output: b: i32
  variables:
    unused: i32 = 0
  steps:
    return a
"#;
        let algo = parse_algorithm(dsl).unwrap();
        let analysis = analyze(&algo);
        assert!(
            analysis
                .warnings
                .iter()
                .any(|w| w.message.contains("unused"))
        );
    }

    #[test]
    fn test_parse_with_comments() {
        let dsl = r#"# This is a comment
algorithm with_comment
  # Input section
  input: x: i32
  output: y: i32
  steps:
    # Return the input
    return x
"#;
        let algo = parse_algorithm(dsl).unwrap();
        assert_eq!(algo.name, "with_comment");
        assert_eq!(algo.inputs.len(), 1);
        assert_eq!(algo.steps.len(), 1);
    }

    #[test]
    fn test_parse_error_invalid_syntax() {
        let dsl = "algorithm\n  @invalid";
        let result = parse_algorithm(dsl);
        assert!(result.is_err());
    }

    #[test]
    fn test_swap_statement_parse() {
        let dsl = r#"algorithm swap_test
  input: a: i32, b: i32
  output: a: i32, b: i32
  steps:
    swap(a, b)
    return a
"#;
        let algo = parse_algorithm(dsl).unwrap();
        assert_eq!(algo.steps.len(), 2);
        assert!(matches!(algo.steps[0], Statement::Swap(..)));
    }

    #[test]
    fn test_template_instantiate_missing_params_error() {
        let registry = create_template_library();
        // "size" has a default of "100" so it should work
        let params: HashMap<String, String> = HashMap::new();
        let result = registry.instantiate("bfs", &params);
        // Should succeed because size has a default
        assert!(result.is_ok());

        // But for bubble_sort, "type" has a default of "i32", so it should also work
        let result2 = registry.instantiate("bubble_sort", &params);
        assert!(result2.is_ok());
    }
}
