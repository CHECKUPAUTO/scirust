//! # scirust-codetrans
//!
//! A code-to-code transformation library in pure Rust.
//!
//! ## Features
//!
//! - **Code AST**: Expression nodes with pretty-printing and S-expression parsing
//! - **Pattern Matching**: Match code patterns with variables, priorities, and multi-pattern support
//! - **Optimization Rules**: Constant folding, DCE, CSE, LICM, strength reduction, inlining, TCO
//! - **Refactoring**: Extract function, rename, inline, loop-to-iterator, match-to-if-let, boolean simplification
//! - **Transpilation**: Rust to Python and Rust to C subsets
//! - **Pattern Database**: JSON persistence, conflict detection, composition
//! - **Transformation Engine**: Fixed-point application, selection strategies, history logging

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

// ============================================================================
// 1. CODE AST
// ============================================================================

/// Binary operator kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Range,
    RangeInclusive,
    Index,
}

impl fmt::Display for BinOpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self
        {
            BinOpKind::Add => "+",
            BinOpKind::Sub => "-",
            BinOpKind::Mul => "*",
            BinOpKind::Div => "/",
            BinOpKind::Rem => "%",
            BinOpKind::And => "&&",
            BinOpKind::Or => "||",
            BinOpKind::BitAnd => "&",
            BinOpKind::BitOr => "|",
            BinOpKind::BitXor => "^",
            BinOpKind::Shl => "<<",
            BinOpKind::Shr => ">>",
            BinOpKind::Eq => "==",
            BinOpKind::Ne => "!=",
            BinOpKind::Lt => "<",
            BinOpKind::Le => "<=",
            BinOpKind::Gt => ">",
            BinOpKind::Ge => ">=",
            BinOpKind::Range => "..",
            BinOpKind::RangeInclusive => "..=",
            BinOpKind::Index => "[]",
        };
        write!(f, "{}", s)
    }
}

/// Unary operator kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnaryOpKind {
    Neg,
    Not,
    Deref,
    Ref,
    RefMut,
}

impl fmt::Display for UnaryOpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self
        {
            UnaryOpKind::Neg => "-",
            UnaryOpKind::Not => "!",
            UnaryOpKind::Deref => "*",
            UnaryOpKind::Ref => "&",
            UnaryOpKind::RefMut => "&mut ",
        };
        write!(f, "{}", s)
    }
}

/// Literal values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Literal {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Unit,
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Literal::Int(v) => write!(f, "{}", v),
            Literal::Float(v) =>
            {
                if v.fract() == 0.0
                {
                    write!(f, "{:.1}", v)
                }
                else
                {
                    write!(f, "{}", v)
                }
            },
            Literal::String(v) => write!(f, "\"{}\"", v),
            Literal::Bool(v) => write!(f, "{}", v),
            Literal::Unit => write!(f, "()"),
        }
    }
}

impl Literal {
    pub fn as_i64(&self) -> Option<i64> {
        match self
        {
            Literal::Int(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self
        {
            Literal::Float(v) => Some(*v),
            Literal::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self
        {
            Literal::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

/// The main expression AST node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    Lit(Literal),
    Var(String),
    BinOp {
        op: BinOpKind,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    UnaryOp {
        op: UnaryOpKind,
        expr: Box<Expr>,
    },
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
    },
    If {
        cond: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Option<Box<Expr>>,
    },
    Let {
        name: String,
        value: Box<Expr>,
        body: Box<Expr>,
    },
    LetMut {
        name: String,
        value: Box<Expr>,
        body: Box<Expr>,
    },
    While {
        cond: Box<Expr>,
        body: Box<Expr>,
    },
    For {
        var: String,
        iter: Box<Expr>,
        body: Box<Expr>,
    },
    Assign {
        name: String,
        value: Box<Expr>,
    },
    Block(Vec<Expr>),
    Return(Option<Box<Expr>>),
    Function {
        name: String,
        params: Vec<String>,
        return_type: Option<String>,
        body: Box<Expr>,
    },
    Struct {
        name: String,
        fields: Vec<(String, String)>,
    },
    Enum {
        name: String,
        variants: Vec<(String, Vec<(String, String)>)>,
    },
    Match {
        expr: Box<Expr>,
        arms: Vec<(Expr, Expr)>,
    },
    Break,
    Continue,
    TypeAnnotation {
        expr: Box<Expr>,
        ty: String,
    },
    FieldAccess {
        expr: Box<Expr>,
        field: String,
    },
    Index {
        expr: Box<Expr>,
        index: Box<Expr>,
    },
}

// ============================================================================
// PRETTY-PRINTING
// ============================================================================

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.pretty_print(f, 0)
    }
}

impl Expr {
    fn pretty_print(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let pad = "    ".repeat(indent);
        match self
        {
            Expr::Lit(lit) => write!(f, "{}", lit),
            Expr::Var(name) => write!(f, "{}", name),
            Expr::BinOp { op, left, right } =>
            {
                let needs_parens_left = matches!(left.as_ref(), Expr::BinOp { .. });
                let needs_parens_right = matches!(right.as_ref(), Expr::BinOp { .. });
                if needs_parens_left
                {
                    write!(f, "({})", left)?;
                }
                else
                {
                    write!(f, "{}", left)?;
                }
                write!(f, " {} ", op)?;
                if needs_parens_right
                {
                    write!(f, "({})", right)
                }
                else
                {
                    write!(f, "{}", right)
                }
            },
            Expr::UnaryOp { op, expr } =>
            {
                write!(f, "{}{}", op, expr)
            },
            Expr::Call { func, args } =>
            {
                write!(f, "{}(", func)?;
                for (i, arg) in args.iter().enumerate()
                {
                    if i > 0
                    {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            },
            Expr::If {
                cond,
                then_branch,
                else_branch,
            } =>
            {
                write!(f, "if {} ", cond)?;
                write!(f, "{}", then_branch)?;
                if let Some(else_br) = else_branch
                {
                    write!(f, " else {}", else_br)?;
                }
                Ok(())
            },
            Expr::Let { name, value, body } =>
            {
                write!(f, "{}let {} = {};\n{}", pad, name, value, body)
            },
            Expr::LetMut { name, value, body } =>
            {
                write!(f, "{}let mut {} = {};\n{}", pad, name, value, body)
            },
            Expr::While { cond, body } =>
            {
                write!(f, "while {} {}", cond, body)
            },
            Expr::For { var, iter, body } =>
            {
                write!(f, "for {} in {} {}", var, iter, body)
            },
            Expr::Assign { name, value } =>
            {
                write!(f, "{}{} = {};", pad, name, value)
            },
            Expr::Block(stmts) =>
            {
                writeln!(f, "{{")?;
                for stmt in stmts
                {
                    stmt.pretty_print(f, indent + 1)?;
                }
                write!(f, "{}}}", pad)
            },
            Expr::Return(Some(expr)) =>
            {
                write!(f, "{}return {};", pad, expr)
            },
            Expr::Return(None) =>
            {
                write!(f, "{}return;", pad)
            },
            Expr::Function {
                name,
                params,
                return_type,
                body,
            } =>
            {
                write!(f, "fn {}(", name)?;
                for (i, param) in params.iter().enumerate()
                {
                    if i > 0
                    {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", param)?;
                }
                write!(f, ")")?;
                if let Some(rt) = return_type
                {
                    write!(f, " -> {}", rt)?;
                }
                write!(f, " {}", body)
            },
            Expr::Struct { name, fields } =>
            {
                writeln!(f, "struct {} {{", name)?;
                for (field_name, field_type) in fields
                {
                    writeln!(f, "{}    {}: {},", pad, field_name, field_type)?;
                }
                write!(f, "{}}}", pad)
            },
            Expr::Enum { name, variants } =>
            {
                writeln!(f, "enum {} {{", name)?;
                for (var_name, var_fields) in variants
                {
                    if var_fields.is_empty()
                    {
                        writeln!(f, "{}    {},", pad, var_name)?;
                    }
                    else
                    {
                        write!(f, "{}    {}(", pad, var_name)?;
                        for (j, (fname, ftype)) in var_fields.iter().enumerate()
                        {
                            if j > 0
                            {
                                write!(f, ", ")?;
                            }
                            write!(f, "{}: {}", fname, ftype)?;
                        }
                        writeln!(f, "),")?;
                    }
                }
                write!(f, "{}}}", pad)
            },
            Expr::Match { expr, arms } =>
            {
                writeln!(f, "match {} {{", expr)?;
                for (pat, body) in arms
                {
                    writeln!(f, "{}    {} => {},", pad, pat, body)?;
                }
                write!(f, "{}}}", pad)
            },
            Expr::Break => write!(f, "{}break;", pad),
            Expr::Continue => write!(f, "{}continue;", pad),
            Expr::TypeAnnotation { expr, ty } =>
            {
                write!(f, "{}: {}", expr, ty)
            },
            Expr::FieldAccess { expr, field } =>
            {
                write!(f, "{}.{}", expr, field)
            },
            Expr::Index { expr, index } =>
            {
                write!(f, "{}[{}]", expr, index)
            },
        }
    }
}

// ============================================================================
// S-EXPRESSION PARSING
// ============================================================================

#[derive(Debug, Clone)]
enum SExpr {
    Atom(String),
    List(Vec<SExpr>),
}

struct SExprParser {
    tokens: Vec<String>,
    pos: usize,
}

impl SExprParser {
    fn new(input: &str) -> Self {
        let tokens = Self::tokenize(input);
        SExprParser { tokens, pos: 0 }
    }

    fn tokenize(input: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        let mut chars = input.chars().peekable();

        while let Some(&ch) = chars.peek()
        {
            match ch
            {
                '(' | ')' =>
                {
                    if !current.is_empty()
                    {
                        tokens.push(current.clone());
                        current.clear();
                    }
                    tokens.push(ch.to_string());
                    chars.next();
                },
                ' ' | '\t' | '\n' | '\r' =>
                {
                    if !current.is_empty()
                    {
                        tokens.push(current.clone());
                        current.clear();
                    }
                    chars.next();
                },
                '"' =>
                {
                    if !current.is_empty()
                    {
                        tokens.push(current.clone());
                        current.clear();
                    }
                    chars.next();
                    let mut s = String::new();
                    while let Some(&c) = chars.peek()
                    {
                        chars.next();
                        if c == '"'
                        {
                            break;
                        }
                        if c == '\\'
                        {
                            if let Some(&nc) = chars.peek()
                            {
                                chars.next();
                                match nc
                                {
                                    'n' => s.push('\n'),
                                    't' => s.push('\t'),
                                    '"' => s.push('"'),
                                    '\\' => s.push('\\'),
                                    _ =>
                                    {
                                        s.push('\\');
                                        s.push(nc);
                                    },
                                }
                            }
                        }
                        else
                        {
                            s.push(c);
                        }
                    }
                    tokens.push(format!("\"{}\"", s));
                },
                _ =>
                {
                    current.push(ch);
                    chars.next();
                },
            }
        }
        if !current.is_empty()
        {
            tokens.push(current);
        }
        tokens
    }

    fn parse(&mut self) -> Result<SExpr, String> {
        if self.pos >= self.tokens.len()
        {
            return Err("unexpected end of input".to_string());
        }
        let token = self.tokens[self.pos].clone();
        self.pos += 1;
        if token == "("
        {
            let mut items = Vec::new();
            while self.pos < self.tokens.len() && self.tokens[self.pos] != ")"
            {
                items.push(self.parse()?);
            }
            if self.pos >= self.tokens.len()
            {
                return Err("unclosed parenthesis".to_string());
            }
            self.pos += 1;
            Ok(SExpr::List(items))
        }
        else if token == ")"
        {
            Err("unexpected closing parenthesis".to_string())
        }
        else
        {
            Ok(SExpr::Atom(token))
        }
    }
}

/// Parse an expression from an S-expression string.
///
/// # Examples
///
/// ```
/// use scirust_codetrans::parse_expr;
/// let expr = parse_expr("(+ 2 3)").unwrap();
/// assert_eq!(expr.to_string(), "2 + 3");
/// ```
pub fn parse_expr(input: &str) -> Result<Expr, String> {
    let mut parser = SExprParser::new(input);
    let sexpr = parser.parse()?;
    sexpr_to_expr(&sexpr)
}

fn sexpr_to_expr(sexpr: &SExpr) -> Result<Expr, String> {
    match sexpr
    {
        SExpr::Atom(s) => parse_atom(s),
        SExpr::List(items) =>
        {
            if items.is_empty()
            {
                return Err("empty list".to_string());
            }
            let head = &items[0];
            let op_name = match head
            {
                SExpr::Atom(s) => s.as_str(),
                _ => return Err("list head must be an atom".to_string()),
            };
            let tail = &items[1..];
            match op_name
            {
                "+" | "-" | "*" | "/" | "%" | "&&" | "||" | "&" | "|" | "^" | "<<" | ">>"
                | "==" | "!=" | "<" | "<=" | ">" | ">=" =>
                {
                    if tail.len() != 2
                    {
                        return Err(format!("binary op {} requires 2 args", op_name));
                    }
                    Ok(Expr::BinOp {
                        op: parse_binop(op_name)?,
                        left: Box::new(sexpr_to_expr(&tail[0])?),
                        right: Box::new(sexpr_to_expr(&tail[1])?),
                    })
                },
                "neg" | "!" | "not" =>
                {
                    if tail.len() != 1
                    {
                        return Err(format!("unary op {} requires 1 arg", op_name));
                    }
                    Ok(Expr::UnaryOp {
                        op: if op_name == "neg"
                        {
                            UnaryOpKind::Neg
                        }
                        else
                        {
                            UnaryOpKind::Not
                        },
                        expr: Box::new(sexpr_to_expr(&tail[0])?),
                    })
                },
                "call" =>
                {
                    let func = sexpr_to_expr(&tail[0])?;
                    let args: Result<Vec<Expr>, String> =
                        tail[1..].iter().map(sexpr_to_expr).collect();
                    Ok(Expr::Call {
                        func: Box::new(func),
                        args: args?,
                    })
                },
                "if" =>
                {
                    if tail.len() < 2 || tail.len() > 3
                    {
                        return Err("if requires 2 or 3 args".to_string());
                    }
                    Ok(Expr::If {
                        cond: Box::new(sexpr_to_expr(&tail[0])?),
                        then_branch: Box::new(sexpr_to_expr(&tail[1])?),
                        else_branch: if tail.len() == 3
                        {
                            Some(Box::new(sexpr_to_expr(&tail[2])?))
                        }
                        else
                        {
                            None
                        },
                    })
                },
                "let" =>
                {
                    if tail.len() != 3
                    {
                        return Err("let requires 3 args".to_string());
                    }
                    let name = match &tail[0]
                    {
                        SExpr::Atom(s) => s.clone(),
                        _ => return Err("let name must be atom".to_string()),
                    };
                    Ok(Expr::Let {
                        name,
                        value: Box::new(sexpr_to_expr(&tail[1])?),
                        body: Box::new(sexpr_to_expr(&tail[2])?),
                    })
                },
                "let-mut" =>
                {
                    if tail.len() != 3
                    {
                        return Err("let-mut requires 3 args".to_string());
                    }
                    let name = match &tail[0]
                    {
                        SExpr::Atom(s) => s.clone(),
                        _ => return Err("let-mut name must be atom".to_string()),
                    };
                    Ok(Expr::LetMut {
                        name,
                        value: Box::new(sexpr_to_expr(&tail[1])?),
                        body: Box::new(sexpr_to_expr(&tail[2])?),
                    })
                },
                "while" =>
                {
                    if tail.len() != 2
                    {
                        return Err("while requires 2 args".to_string());
                    }
                    Ok(Expr::While {
                        cond: Box::new(sexpr_to_expr(&tail[0])?),
                        body: Box::new(sexpr_to_expr(&tail[1])?),
                    })
                },
                "for" =>
                {
                    if tail.len() != 3
                    {
                        return Err("for requires 3 args".to_string());
                    }
                    let var = match &tail[0]
                    {
                        SExpr::Atom(s) => s.clone(),
                        _ => return Err("for var must be atom".to_string()),
                    };
                    Ok(Expr::For {
                        var,
                        iter: Box::new(sexpr_to_expr(&tail[1])?),
                        body: Box::new(sexpr_to_expr(&tail[2])?),
                    })
                },
                "assign" | "set!" =>
                {
                    if tail.len() != 2
                    {
                        return Err("assign requires 2 args".to_string());
                    }
                    let name = match &tail[0]
                    {
                        SExpr::Atom(s) => s.clone(),
                        _ => return Err("assign name must be atom".to_string()),
                    };
                    Ok(Expr::Assign {
                        name,
                        value: Box::new(sexpr_to_expr(&tail[1])?),
                    })
                },
                "block" | "do" | "begin" =>
                {
                    let stmts: Result<Vec<Expr>, String> = tail.iter().map(sexpr_to_expr).collect();
                    Ok(Expr::Block(stmts?))
                },
                "return" =>
                {
                    if tail.is_empty()
                    {
                        Ok(Expr::Return(None))
                    }
                    else if tail.len() == 1
                    {
                        Ok(Expr::Return(Some(Box::new(sexpr_to_expr(&tail[0])?))))
                    }
                    else
                    {
                        Err("return takes 0 or 1 args".to_string())
                    }
                },
                "fn" | "function" =>
                {
                    if tail.len() < 3
                    {
                        return Err("fn requires at least 3 args".to_string());
                    }
                    let name = match &tail[0]
                    {
                        SExpr::Atom(s) => s.clone(),
                        _ => return Err("fn name must be atom".to_string()),
                    };
                    let params: Result<Vec<String>, String> = match &tail[1]
                    {
                        SExpr::List(params_list) => params_list
                            .iter()
                            .map(|p| match p
                            {
                                SExpr::Atom(s) => Ok(s.clone()),
                                _ => Err("param must be atom".to_string()),
                            })
                            .collect(),
                        SExpr::Atom(a) => Ok(vec![a.clone()]),
                    };
                    let params = params?;
                    Ok(Expr::Function {
                        name,
                        params,
                        return_type: None,
                        body: Box::new(sexpr_to_expr(&tail[2])?),
                    })
                },
                "struct" =>
                {
                    if tail.len() < 2
                    {
                        return Err("struct requires at least 2 args".to_string());
                    }
                    let name = match &tail[0]
                    {
                        SExpr::Atom(s) => s.clone(),
                        _ => return Err("struct name must be atom".to_string()),
                    };
                    let mut fields = Vec::new();
                    for field_spec in &tail[1..]
                    {
                        match field_spec
                        {
                            SExpr::List(fs) if fs.len() == 2 =>
                            {
                                let fname = match &fs[0]
                                {
                                    SExpr::Atom(s) => s.clone(),
                                    _ => return Err("field name must be atom".to_string()),
                                };
                                let ftype = match &fs[1]
                                {
                                    SExpr::Atom(s) => s.clone(),
                                    _ => return Err("field type must be atom".to_string()),
                                };
                                fields.push((fname, ftype));
                            },
                            SExpr::Atom(a) =>
                            {
                                fields.push((a.clone(), "T".to_string()));
                            },
                            _ => return Err("invalid field spec".to_string()),
                        }
                    }
                    Ok(Expr::Struct { name, fields })
                },
                "enum" =>
                {
                    if tail.len() < 2
                    {
                        return Err("enum requires at least 2 args".to_string());
                    }
                    let name = match &tail[0]
                    {
                        SExpr::Atom(s) => s.clone(),
                        _ => return Err("enum name must be atom".to_string()),
                    };
                    let mut variants = Vec::new();
                    for var_spec in &tail[1..]
                    {
                        match var_spec
                        {
                            SExpr::Atom(a) =>
                            {
                                variants.push((a.clone(), Vec::new()));
                            },
                            SExpr::List(vs) if !vs.is_empty() =>
                            {
                                let vname = match &vs[0]
                                {
                                    SExpr::Atom(s) => s.clone(),
                                    _ => return Err("variant name must be atom".to_string()),
                                };
                                let mut vfields = Vec::new();
                                for vf in &vs[1..]
                                {
                                    match vf
                                    {
                                        SExpr::List(fs) if fs.len() == 2 =>
                                        {
                                            let fname = match &fs[0]
                                            {
                                                SExpr::Atom(s) => s.clone(),
                                                _ =>
                                                {
                                                    return Err(
                                                        "field name must be atom".to_string()
                                                    );
                                                },
                                            };
                                            let ftype = match &fs[1]
                                            {
                                                SExpr::Atom(s) => s.clone(),
                                                _ =>
                                                {
                                                    return Err(
                                                        "field type must be atom".to_string()
                                                    );
                                                },
                                            };
                                            vfields.push((fname, ftype));
                                        },
                                        _ => return Err("invalid variant field spec".to_string()),
                                    }
                                }
                                variants.push((vname, vfields));
                            },
                            _ => return Err("invalid variant spec".to_string()),
                        }
                    }
                    Ok(Expr::Enum { name, variants })
                },
                "match" =>
                {
                    if tail.len() < 3
                    {
                        return Err("match requires at least 3 args".to_string());
                    }
                    let expr = sexpr_to_expr(&tail[0])?;
                    let mut arms = Vec::new();
                    let mut i = 1;
                    while i < tail.len()
                    {
                        if i + 1 >= tail.len()
                        {
                            return Err("match arms must come in pattern,body pairs".to_string());
                        }
                        arms.push((sexpr_to_expr(&tail[i])?, sexpr_to_expr(&tail[i + 1])?));
                        i += 2;
                    }
                    Ok(Expr::Match {
                        expr: Box::new(expr),
                        arms,
                    })
                },
                "break" => Ok(Expr::Break),
                "continue" => Ok(Expr::Continue),
                ".." if tail.len() == 2 => Ok(Expr::BinOp {
                    op: BinOpKind::Range,
                    left: Box::new(sexpr_to_expr(&tail[0])?),
                    right: Box::new(sexpr_to_expr(&tail[1])?),
                }),
                "..=" if tail.len() == 2 => Ok(Expr::BinOp {
                    op: BinOpKind::RangeInclusive,
                    left: Box::new(sexpr_to_expr(&tail[0])?),
                    right: Box::new(sexpr_to_expr(&tail[1])?),
                }),
                "index" if tail.len() == 2 => Ok(Expr::Index {
                    expr: Box::new(sexpr_to_expr(&tail[0])?),
                    index: Box::new(sexpr_to_expr(&tail[1])?),
                }),
                "field" if tail.len() == 2 =>
                {
                    let field = match &tail[1]
                    {
                        SExpr::Atom(s) => s.clone(),
                        _ => return Err("field name must be atom".to_string()),
                    };
                    Ok(Expr::FieldAccess {
                        expr: Box::new(sexpr_to_expr(&tail[0])?),
                        field,
                    })
                },
                _ => Err(format!("unknown operator: {}", op_name)),
            }
        },
    }
}

fn parse_atom(s: &str) -> Result<Expr, String> {
    if s == "true"
    {
        return Ok(Expr::Lit(Literal::Bool(true)));
    }
    if s == "false"
    {
        return Ok(Expr::Lit(Literal::Bool(false)));
    }
    if s == "()"
    {
        return Ok(Expr::Lit(Literal::Unit));
    }

    if s.starts_with('"') && s.ends_with('"')
    {
        let inner = &s[1..s.len() - 1];
        return Ok(Expr::Lit(Literal::String(inner.to_string())));
    }

    if s.starts_with("0x") || s.starts_with("0X")
    {
        if let Ok(v) = i64::from_str_radix(&s[2..], 16)
        {
            return Ok(Expr::Lit(Literal::Int(v)));
        }
    }
    if s.starts_with("0o") || s.starts_with("0O")
    {
        if let Ok(v) = i64::from_str_radix(&s[2..], 8)
        {
            return Ok(Expr::Lit(Literal::Int(v)));
        }
    }
    if s.starts_with("0b") || s.starts_with("0B")
    {
        if let Ok(v) = i64::from_str_radix(&s[2..], 2)
        {
            return Ok(Expr::Lit(Literal::Int(v)));
        }
    }

    if let Ok(v) = s.parse::<i64>()
    {
        return Ok(Expr::Lit(Literal::Int(v)));
    }
    if let Ok(v) = s.parse::<f64>()
    {
        return Ok(Expr::Lit(Literal::Float(v)));
    }

    Ok(Expr::Var(s.to_string()))
}

fn parse_binop(s: &str) -> Result<BinOpKind, String> {
    match s
    {
        "+" => Ok(BinOpKind::Add),
        "-" => Ok(BinOpKind::Sub),
        "*" => Ok(BinOpKind::Mul),
        "/" => Ok(BinOpKind::Div),
        "%" => Ok(BinOpKind::Rem),
        "&&" => Ok(BinOpKind::And),
        "||" => Ok(BinOpKind::Or),
        "&" => Ok(BinOpKind::BitAnd),
        "|" => Ok(BinOpKind::BitOr),
        "^" => Ok(BinOpKind::BitXor),
        "<<" => Ok(BinOpKind::Shl),
        ">>" => Ok(BinOpKind::Shr),
        "==" => Ok(BinOpKind::Eq),
        "!=" => Ok(BinOpKind::Ne),
        "<" => Ok(BinOpKind::Lt),
        "<=" => Ok(BinOpKind::Le),
        ">" => Ok(BinOpKind::Gt),
        ">=" => Ok(BinOpKind::Ge),
        _ => Err(format!("unknown binary operator: {}", s)),
    }
}

// ============================================================================
// AST VISITORS
// ============================================================================

/// A generic AST visitor trait that walks the AST.
pub trait AstVisitor {
    fn visit_expr(&mut self, expr: &Expr) {
        walk_expr(self, expr);
    }
}

pub fn walk_expr<V: AstVisitor + ?Sized>(visitor: &mut V, expr: &Expr) {
    match expr
    {
        Expr::Lit(_) | Expr::Var(_) | Expr::Break | Expr::Continue =>
        {},
        Expr::BinOp { left, right, .. } =>
        {
            visitor.visit_expr(left);
            visitor.visit_expr(right);
        },
        Expr::UnaryOp { expr: e, .. } =>
        {
            visitor.visit_expr(e);
        },
        Expr::Call { func, args } =>
        {
            visitor.visit_expr(func);
            for arg in args
            {
                visitor.visit_expr(arg);
            }
        },
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } =>
        {
            visitor.visit_expr(cond);
            visitor.visit_expr(then_branch);
            if let Some(e) = else_branch
            {
                visitor.visit_expr(e);
            }
        },
        Expr::Let { value, body, .. } | Expr::LetMut { value, body, .. } =>
        {
            visitor.visit_expr(value);
            visitor.visit_expr(body);
        },
        Expr::While { cond, body } =>
        {
            visitor.visit_expr(cond);
            visitor.visit_expr(body);
        },
        Expr::For { iter, body, .. } =>
        {
            visitor.visit_expr(iter);
            visitor.visit_expr(body);
        },
        Expr::Assign { value, .. } =>
        {
            visitor.visit_expr(value);
        },
        Expr::Block(stmts) =>
        {
            for s in stmts
            {
                visitor.visit_expr(s);
            }
        },
        Expr::Return(Some(e)) =>
        {
            visitor.visit_expr(e);
        },
        Expr::Return(None) =>
        {},
        Expr::Function { body, .. } =>
        {
            visitor.visit_expr(body);
        },
        Expr::Struct { .. } | Expr::Enum { .. } =>
        {},
        Expr::Match { expr: e, arms } =>
        {
            visitor.visit_expr(e);
            for (pat, body) in arms
            {
                visitor.visit_expr(pat);
                visitor.visit_expr(body);
            }
        },
        Expr::TypeAnnotation { expr: e, .. } =>
        {
            visitor.visit_expr(e);
        },
        Expr::FieldAccess { expr: e, .. } =>
        {
            visitor.visit_expr(e);
        },
        Expr::Index { expr: e, index } =>
        {
            visitor.visit_expr(e);
            visitor.visit_expr(index);
        },
    }
}

/// A fold (transform) visitor that produces new AST.
pub trait AstFold {
    fn fold_expr(&mut self, expr: &Expr) -> Expr {
        fold_expr(self, expr)
    }
}

pub fn fold_expr<F: AstFold + ?Sized>(folder: &mut F, expr: &Expr) -> Expr {
    match expr
    {
        e @ Expr::Lit(_) | e @ Expr::Var(_) | e @ Expr::Break | e @ Expr::Continue => e.clone(),
        Expr::BinOp { op, left, right } => Expr::BinOp {
            op: *op,
            left: Box::new(folder.fold_expr(left)),
            right: Box::new(folder.fold_expr(right)),
        },
        Expr::UnaryOp { op, expr: e } => Expr::UnaryOp {
            op: *op,
            expr: Box::new(folder.fold_expr(e)),
        },
        Expr::Call { func, args } => Expr::Call {
            func: Box::new(folder.fold_expr(func)),
            args: args.iter().map(|a| folder.fold_expr(a)).collect(),
        },
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => Expr::If {
            cond: Box::new(folder.fold_expr(cond)),
            then_branch: Box::new(folder.fold_expr(then_branch)),
            else_branch: else_branch.as_ref().map(|e| Box::new(folder.fold_expr(e))),
        },
        Expr::Let { name, value, body } => Expr::Let {
            name: name.clone(),
            value: Box::new(folder.fold_expr(value)),
            body: Box::new(folder.fold_expr(body)),
        },
        Expr::LetMut { name, value, body } => Expr::LetMut {
            name: name.clone(),
            value: Box::new(folder.fold_expr(value)),
            body: Box::new(folder.fold_expr(body)),
        },
        Expr::While { cond, body } => Expr::While {
            cond: Box::new(folder.fold_expr(cond)),
            body: Box::new(folder.fold_expr(body)),
        },
        Expr::For { var, iter, body } => Expr::For {
            var: var.clone(),
            iter: Box::new(folder.fold_expr(iter)),
            body: Box::new(folder.fold_expr(body)),
        },
        Expr::Assign { name, value } => Expr::Assign {
            name: name.clone(),
            value: Box::new(folder.fold_expr(value)),
        },
        Expr::Block(stmts) => Expr::Block(stmts.iter().map(|s| folder.fold_expr(s)).collect()),
        Expr::Return(Some(e)) => Expr::Return(Some(Box::new(folder.fold_expr(e)))),
        Expr::Return(None) => Expr::Return(None),
        Expr::Function {
            name,
            params,
            return_type,
            body,
        } => Expr::Function {
            name: name.clone(),
            params: params.clone(),
            return_type: return_type.clone(),
            body: Box::new(folder.fold_expr(body)),
        },
        Expr::Struct { name, fields } => Expr::Struct {
            name: name.clone(),
            fields: fields.clone(),
        },
        Expr::Enum { name, variants } => Expr::Enum {
            name: name.clone(),
            variants: variants.clone(),
        },
        Expr::Match { expr: e, arms } => Expr::Match {
            expr: Box::new(folder.fold_expr(e)),
            arms: arms
                .iter()
                .map(|(p, b)| (folder.fold_expr(p), folder.fold_expr(b)))
                .collect(),
        },
        Expr::TypeAnnotation { expr: e, ty } => Expr::TypeAnnotation {
            expr: Box::new(folder.fold_expr(e)),
            ty: ty.clone(),
        },
        Expr::FieldAccess { expr: e, field } => Expr::FieldAccess {
            expr: Box::new(folder.fold_expr(e)),
            field: field.clone(),
        },
        Expr::Index { expr: e, index } => Expr::Index {
            expr: Box::new(folder.fold_expr(e)),
            index: Box::new(folder.fold_expr(index)),
        },
    }
}

// ============================================================================
// 2. PATTERN MATCHING
// ============================================================================

/// A pattern variable (starts with `$` in S-expression form).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PatternVar(pub String);

impl fmt::Display for PatternVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "${}", self.0)
    }
}

/// A pattern node in the matching tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Pattern {
    Any(PatternVar),
    Lit(Literal),
    Var(String),
    BinOp {
        op: Option<BinOpKind>,
        left: Box<Pattern>,
        right: Box<Pattern>,
    },
    UnaryOp {
        op: Option<UnaryOpKind>,
        expr: Box<Pattern>,
    },
    Call {
        func: Box<Pattern>,
        args: Vec<Pattern>,
    },
    If {
        cond: Box<Pattern>,
        then_branch: Box<Pattern>,
        else_branch: Option<Box<Pattern>>,
    },
    Let {
        name: Option<String>,
        value: Box<Pattern>,
        body: Box<Pattern>,
    },
    While {
        cond: Box<Pattern>,
        body: Box<Pattern>,
    },
    For {
        var: Option<String>,
        iter: Box<Pattern>,
        body: Box<Pattern>,
    },
    Block(Vec<Pattern>),
    Function {
        name: Option<String>,
        params: Option<Vec<String>>,
        body: Box<Pattern>,
    },
    Return(Option<Box<Pattern>>),
    FieldAccess {
        expr: Box<Pattern>,
        field: Option<String>,
    },
    Wildcard,
    Repeat(Box<Pattern>),
    Rest(PatternVar),
}

impl fmt::Display for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Pattern::Any(pv) => write!(f, "{}", pv),
            Pattern::Lit(l) => write!(f, "{}", l),
            Pattern::Var(s) => write!(f, "{}", s),
            Pattern::BinOp { op, left, right } =>
            {
                let op_s = op.map(|o| o.to_string()).unwrap_or_else(|| "_".to_string());
                write!(f, "({} {} {})", op_s, left, right)
            },
            Pattern::UnaryOp { op, expr } =>
            {
                let op_s = op.map(|o| o.to_string()).unwrap_or_else(|| "_".to_string());
                write!(f, "({} {})", op_s, expr)
            },
            Pattern::Call { func, args } =>
            {
                write!(f, "(call {}", func)?;
                for a in args
                {
                    write!(f, " {}", a)?;
                }
                write!(f, ")")
            },
            Pattern::If {
                cond,
                then_branch,
                else_branch,
            } =>
            {
                write!(f, "(if {} {}", cond, then_branch)?;
                if let Some(e) = else_branch
                {
                    write!(f, " {}", e)?;
                }
                write!(f, ")")
            },
            Pattern::FieldAccess { expr, field } =>
            {
                write!(f, "(field {} {})", expr, field.as_deref().unwrap_or("_"))
            },
            Pattern::Wildcard => write!(f, "_"),
            Pattern::Repeat(p) => write!(f, "(... {})", p),
            Pattern::Rest(pv) => write!(f, "{}...", pv),
            _ => write!(f, "<pattern>"),
        }
    }
}

/// Parse a pattern from an S-expression string.
pub fn parse_pattern(input: &str) -> Result<Pattern, String> {
    let mut parser = SExprParser::new(input);
    let sexpr = parser.parse()?;
    sexpr_to_pattern(&sexpr)
}

fn sexpr_to_pattern(sexpr: &SExpr) -> Result<Pattern, String> {
    match sexpr
    {
        SExpr::Atom(s) =>
        {
            if s == "_"
            {
                Ok(Pattern::Wildcard)
            }
            else if s.starts_with('$')
            {
                Ok(Pattern::Any(PatternVar(s[1..].to_string())))
            }
            else if let Ok(v) = s.parse::<i64>()
            {
                Ok(Pattern::Lit(Literal::Int(v)))
            }
            else if let Ok(v) = s.parse::<f64>()
            {
                Ok(Pattern::Lit(Literal::Float(v)))
            }
            else if s == "true"
            {
                Ok(Pattern::Lit(Literal::Bool(true)))
            }
            else if s == "false"
            {
                Ok(Pattern::Lit(Literal::Bool(false)))
            }
            else
            {
                Ok(Pattern::Var(s.clone()))
            }
        },
        SExpr::List(items) =>
        {
            if items.is_empty()
            {
                return Err("empty pattern list".to_string());
            }
            let head = &items[0];
            let op_name = match head
            {
                SExpr::Atom(s) => s.as_str(),
                _ => return Err("pattern head must be atom".to_string()),
            };
            let tail = &items[1..];
            match op_name
            {
                "+" | "-" | "*" | "/" | "%" | "&&" | "||" | "&" | "|" | "^" | "<<" | ">>"
                | "==" | "!=" | "<" | "<=" | ">" | ">=" =>
                {
                    let op_parsed = parse_binop(op_name);
                    Ok(Pattern::BinOp {
                        op: op_parsed.ok(),
                        left: Box::new(sexpr_to_pattern(
                            &tail
                                .first()
                                .cloned()
                                .unwrap_or(SExpr::Atom("_".to_string())),
                        )?),
                        right: Box::new(sexpr_to_pattern(
                            &tail.get(1).cloned().unwrap_or(SExpr::Atom("_".to_string())),
                        )?),
                    })
                },
                "!" | "not" | "neg" =>
                {
                    if tail.is_empty()
                    {
                        return Err("unary op requires 1 arg".to_string());
                    }
                    Ok(Pattern::UnaryOp {
                        op: if op_name == "neg"
                        {
                            Some(UnaryOpKind::Neg)
                        }
                        else if op_name == "not" || op_name == "!"
                        {
                            Some(UnaryOpKind::Not)
                        }
                        else
                        {
                            None
                        },
                        expr: Box::new(sexpr_to_pattern(&tail[0])?),
                    })
                },
                "call" =>
                {
                    let func = sexpr_to_pattern(&tail[0])?;
                    let args: Result<Vec<Pattern>, String> =
                        tail[1..].iter().map(sexpr_to_pattern).collect();
                    Ok(Pattern::Call {
                        func: Box::new(func),
                        args: args?,
                    })
                },
                "if" => Ok(Pattern::If {
                    cond: Box::new(sexpr_to_pattern(&tail[0])?),
                    then_branch: Box::new(sexpr_to_pattern(&tail[1])?),
                    else_branch: if tail.len() > 2
                    {
                        Some(Box::new(sexpr_to_pattern(&tail[2])?))
                    }
                    else
                    {
                        None
                    },
                }),
                "for" =>
                {
                    let var = match &tail[0]
                    {
                        SExpr::Atom(a) if a.starts_with('$') => None,
                        SExpr::Atom(a) => Some(a.clone()),
                        SExpr::List(_) => None,
                    };
                    let var_idx = if var.is_some() { 1 } else { 0 };
                    if tail.len() <= var_idx + 1
                    {
                        return Err("for pattern requires iter and body".to_string());
                    }
                    Ok(Pattern::For {
                        var,
                        iter: Box::new(sexpr_to_pattern(&tail[var_idx])?),
                        body: Box::new(sexpr_to_pattern(&tail[var_idx + 1])?),
                    })
                },
                "field" => Ok(Pattern::FieldAccess {
                    expr: Box::new(sexpr_to_pattern(&tail[0])?),
                    field: match &tail.get(1)
                    {
                        Some(SExpr::Atom(a)) if a != "_" => Some(a.clone()),
                        _ => None,
                    },
                }),
                "..." => Ok(Pattern::Repeat(Box::new(sexpr_to_pattern(
                    &tail
                        .first()
                        .cloned()
                        .unwrap_or(SExpr::Atom("_".to_string())),
                )?))),
                _ => Err(format!("unknown pattern operator: {}", op_name)),
            }
        },
    }
}

/// A match result: which pattern variables were bound to which expressions.
pub type Bindings = HashMap<PatternVar, Expr>;

/// Try to match a pattern against an expression. Returns bindings if successful.
pub fn match_pattern(pattern: &Pattern, expr: &Expr) -> Option<Bindings> {
    let mut bindings = Bindings::new();
    if match_pattern_impl(pattern, expr, &mut bindings)
    {
        Some(bindings)
    }
    else
    {
        None
    }
}

fn match_pattern_impl(pattern: &Pattern, expr: &Expr, bindings: &mut Bindings) -> bool {
    match pattern
    {
        Pattern::Wildcard => true,
        Pattern::Any(pv) =>
        {
            if let Some(existing) = bindings.get(pv)
            {
                *existing == *expr
            }
            else
            {
                bindings.insert(pv.clone(), expr.clone());
                true
            }
        },
        Pattern::Lit(pl) => match expr
        {
            Expr::Lit(el) => pl == el,
            _ => false,
        },
        Pattern::Var(pv_name) => match expr
        {
            Expr::Var(ev_name) => pv_name == ev_name || pv_name == "_",
            _ => false,
        },
        Pattern::BinOp {
            op: po,
            left: pl,
            right: pr,
        } => match expr
        {
            Expr::BinOp {
                op: eo,
                left: el,
                right: er,
            } =>
            {
                if let Some(po_val) = po
                {
                    if *po_val != *eo
                    {
                        return false;
                    }
                }
                match_pattern_impl(pl, el, bindings) && match_pattern_impl(pr, er, bindings)
            },
            _ => false,
        },
        Pattern::UnaryOp { op: po, expr: pe } => match expr
        {
            Expr::UnaryOp { op: eo, expr: ee } =>
            {
                if let Some(po_val) = po
                {
                    if *po_val != *eo
                    {
                        return false;
                    }
                }
                match_pattern_impl(pe, ee, bindings)
            },
            _ => false,
        },
        Pattern::Call {
            func: pf,
            args: pargs,
        } => match expr
        {
            Expr::Call {
                func: ef,
                args: eargs,
            } =>
            {
                if !match_pattern_impl(pf, ef, bindings)
                {
                    return false;
                }
                if let Some(repeat_pos) = pargs.iter().position(|p| matches!(p, Pattern::Repeat(_)))
                {
                    let before = &pargs[..repeat_pos];
                    let after = &pargs[repeat_pos + 1..];
                    if eargs.len() < before.len() + after.len()
                    {
                        return false;
                    }
                    for (p, e) in before.iter().zip(eargs.iter())
                    {
                        if !match_pattern_impl(p, e, bindings)
                        {
                            return false;
                        }
                    }
                    let after_start = eargs.len() - after.len();
                    for (p, e) in after.iter().zip(eargs[after_start..].iter())
                    {
                        if !match_pattern_impl(p, e, bindings)
                        {
                            return false;
                        }
                    }
                    match &pargs[repeat_pos]
                    {
                        Pattern::Repeat(rp) =>
                        {
                            for e in &eargs[before.len()..after_start]
                            {
                                if !match_pattern_impl(rp, e, bindings)
                                {
                                    return false;
                                }
                            }
                        },
                        _ => unreachable!(),
                    }
                }
                else
                {
                    if pargs.len() != eargs.len()
                    {
                        return false;
                    }
                    for (p, e) in pargs.iter().zip(eargs.iter())
                    {
                        if !match_pattern_impl(p, e, bindings)
                        {
                            return false;
                        }
                    }
                }
                true
            },
            _ => false,
        },
        Pattern::If {
            cond: pc,
            then_branch: pt,
            else_branch: pe,
        } => match expr
        {
            Expr::If {
                cond: ec,
                then_branch: et,
                else_branch: ee,
            } =>
            {
                match_pattern_impl(pc, ec, bindings)
                    && match_pattern_impl(pt, et, bindings)
                    && match (pe, ee)
                    {
                        (Some(pp), Some(ep)) => match_pattern_impl(pp, ep, bindings),
                        (None, None) => true,
                        _ => false,
                    }
            },
            _ => false,
        },
        Pattern::For {
            var: pv,
            iter: pi,
            body: pb,
        } => match expr
        {
            Expr::For {
                var: ev,
                iter: ei,
                body: eb,
            } =>
            {
                if let Some(pv_name) = pv
                {
                    if pv_name != ev
                    {
                        return false;
                    }
                }
                match_pattern_impl(pi, ei, bindings) && match_pattern_impl(pb, eb, bindings)
            },
            _ => false,
        },
        Pattern::FieldAccess {
            expr: pe,
            field: pf,
        } => match expr
        {
            Expr::FieldAccess {
                expr: ee,
                field: ef,
            } =>
            {
                if let Some(pf_name) = pf
                {
                    if pf_name != ef
                    {
                        return false;
                    }
                }
                match_pattern_impl(pe, ee, bindings)
            },
            _ => false,
        },
        Pattern::Block(ppats) => match expr
        {
            Expr::Block(estmts) =>
            {
                if ppats.len() != estmts.len()
                {
                    return false;
                }
                ppats
                    .iter()
                    .zip(estmts.iter())
                    .all(|(p, e)| match_pattern_impl(p, e, bindings))
            },
            _ => false,
        },
        Pattern::Let {
            name: pn,
            value: pv,
            body: pb,
        } => match expr
        {
            Expr::Let {
                name: en,
                value: ev,
                body: eb,
            } =>
            {
                if let Some(pn_val) = pn
                {
                    if pn_val != en
                    {
                        return false;
                    }
                }
                match_pattern_impl(pv, ev, bindings) && match_pattern_impl(pb, eb, bindings)
            },
            _ => false,
        },
        Pattern::While { cond: pc, body: pb } => match expr
        {
            Expr::While { cond: ec, body: eb } =>
            {
                match_pattern_impl(pc, ec, bindings) && match_pattern_impl(pb, eb, bindings)
            },
            _ => false,
        },
        Pattern::Function {
            name: pn,
            params: pps,
            body: pb,
        } => match expr
        {
            Expr::Function {
                name: en,
                params: eps,
                body: eb,
                ..
            } =>
            {
                if let Some(pn_val) = pn
                {
                    if pn_val != en
                    {
                        return false;
                    }
                }
                if let Some(pp_vals) = pps
                {
                    if pp_vals.len() != eps.len()
                    {
                        return false;
                    }
                    for (p_param, e_param) in pp_vals.iter().zip(eps.iter())
                    {
                        if p_param != e_param && p_param != "_"
                        {
                            return false;
                        }
                    }
                }
                match_pattern_impl(pb, eb, bindings)
            },
            _ => false,
        },
        Pattern::Return(Some(rp)) => match expr
        {
            Expr::Return(Some(re)) => match_pattern_impl(rp, re, bindings),
            _ => false,
        },
        Pattern::Return(None) => matches!(expr, Expr::Return(_)),
        Pattern::Repeat(_) => true,
        Pattern::Rest(_) => true,
    }
}

/// A rewrite rule: pattern -> replacement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    pub name: String,
    pub pattern: Pattern,
    pub replacement: Pattern,
    pub priority: i32,
    pub description: String,
}

impl Rule {
    pub fn new(
        name: &str,
        pattern: Pattern,
        replacement: Pattern,
        priority: i32,
        description: &str,
    ) -> Self {
        Rule {
            name: name.to_string(),
            pattern,
            replacement,
            priority,
            description: description.to_string(),
        }
    }

    /// Try to apply this rule to an expression.
    pub fn apply(&self, expr: &Expr) -> Option<Expr> {
        let bindings = match_pattern(&self.pattern, expr)?;
        Some(substitute_pattern(&self.replacement, &bindings))
    }
}

/// Substitute pattern variables in a replacement pattern with their bound expressions.
fn substitute_pattern(pattern: &Pattern, bindings: &Bindings) -> Expr {
    match pattern
    {
        Pattern::Any(pv) =>
        {
            if let Some(expr) = bindings.get(pv)
            {
                expr.clone()
            }
            else
            {
                Expr::Var(format!("<unbound {}>", pv.0))
            }
        },
        Pattern::Lit(l) => Expr::Lit(l.clone()),
        Pattern::Var(s) => Expr::Var(s.clone()),
        Pattern::BinOp { op, left, right } => Expr::BinOp {
            op: op.unwrap_or(BinOpKind::Add),
            left: Box::new(substitute_pattern(left, bindings)),
            right: Box::new(substitute_pattern(right, bindings)),
        },
        Pattern::UnaryOp { op, expr } => Expr::UnaryOp {
            op: op.unwrap_or(UnaryOpKind::Neg),
            expr: Box::new(substitute_pattern(expr, bindings)),
        },
        Pattern::Call { func, args } => Expr::Call {
            func: Box::new(substitute_pattern(func, bindings)),
            args: args
                .iter()
                .map(|a| substitute_pattern(a, bindings))
                .collect(),
        },
        Pattern::If {
            cond,
            then_branch,
            else_branch,
        } => Expr::If {
            cond: Box::new(substitute_pattern(cond, bindings)),
            then_branch: Box::new(substitute_pattern(then_branch, bindings)),
            else_branch: else_branch
                .as_ref()
                .map(|e| Box::new(substitute_pattern(e, bindings))),
        },
        Pattern::For { var, iter, body } => Expr::For {
            var: var.clone().unwrap_or_else(|| "i".to_string()),
            iter: Box::new(substitute_pattern(iter, bindings)),
            body: Box::new(substitute_pattern(body, bindings)),
        },
        Pattern::FieldAccess { expr, field } => Expr::FieldAccess {
            expr: Box::new(substitute_pattern(expr, bindings)),
            field: field.clone().unwrap_or_else(|| "?".to_string()),
        },
        Pattern::Wildcard => Expr::Lit(Literal::Unit),
        Pattern::Let { name, value, body } => Expr::Let {
            name: name.clone().unwrap_or_else(|| "x".to_string()),
            value: Box::new(substitute_pattern(value, bindings)),
            body: Box::new(substitute_pattern(body, bindings)),
        },
        Pattern::While { cond, body } => Expr::While {
            cond: Box::new(substitute_pattern(cond, bindings)),
            body: Box::new(substitute_pattern(body, bindings)),
        },
        Pattern::Function { name, params, body } =>
        {
            let params = params.clone().unwrap_or_default();
            Expr::Function {
                name: name.clone().unwrap_or_else(|| "f".to_string()),
                params,
                return_type: None,
                body: Box::new(substitute_pattern(body, bindings)),
            }
        },
        Pattern::Return(Some(rp)) => Expr::Return(Some(Box::new(substitute_pattern(rp, bindings)))),
        Pattern::Return(None) => Expr::Return(None),
        Pattern::Block(ppats) => Expr::Block(
            ppats
                .iter()
                .map(|p| substitute_pattern(p, bindings))
                .collect(),
        ),
        Pattern::Repeat(inner) => substitute_pattern(inner, bindings),
        Pattern::Rest(_) => Expr::Lit(Literal::Unit),
    }
}

// ============================================================================
// 3. CODE OPTIMIZATION RULES
// ============================================================================

/// Create a standard set of optimization rules.
pub fn optimization_rules() -> Vec<Rule> {
    vec![
        Rule::new(
            "const-fold-add",
            parse_pattern("(+ $a $b)").unwrap(),
            Pattern::Any(PatternVar("result".to_string())),
            100,
            "Constant fold addition",
        ),
        Rule::new(
            "const-fold-sub",
            parse_pattern("(- $a $b)").unwrap(),
            Pattern::Any(PatternVar("result".to_string())),
            100,
            "Constant fold subtraction",
        ),
        Rule::new(
            "const-fold-mul",
            parse_pattern("(* $a $b)").unwrap(),
            Pattern::Any(PatternVar("result".to_string())),
            100,
            "Constant fold multiplication",
        ),
        Rule::new(
            "const-fold-div",
            parse_pattern("(/ $a $b)").unwrap(),
            Pattern::Any(PatternVar("result".to_string())),
            100,
            "Constant fold division",
        ),
        Rule::new(
            "add-zero",
            parse_pattern("(+ $x 0)").unwrap(),
            parse_pattern("$x").unwrap(),
            90,
            "x + 0 -> x",
        ),
        Rule::new(
            "add-zero-left",
            parse_pattern("(+ 0 $x)").unwrap(),
            parse_pattern("$x").unwrap(),
            90,
            "0 + x -> x",
        ),
        Rule::new(
            "sub-zero",
            parse_pattern("(- $x 0)").unwrap(),
            parse_pattern("$x").unwrap(),
            90,
            "x - 0 -> x",
        ),
        Rule::new(
            "mul-one",
            parse_pattern("(* $x 1)").unwrap(),
            parse_pattern("$x").unwrap(),
            90,
            "x * 1 -> x",
        ),
        Rule::new(
            "mul-one-left",
            parse_pattern("(* 1 $x)").unwrap(),
            parse_pattern("$x").unwrap(),
            90,
            "1 * x -> x",
        ),
        Rule::new(
            "mul-zero",
            parse_pattern("(* $x 0)").unwrap(),
            parse_pattern("0").unwrap(),
            90,
            "x * 0 -> 0",
        ),
        Rule::new(
            "mul-zero-left",
            parse_pattern("(* 0 $x)").unwrap(),
            parse_pattern("0").unwrap(),
            90,
            "0 * x -> 0",
        ),
        Rule::new(
            "mul-to-shl",
            parse_pattern("(* $x 2)").unwrap(),
            parse_pattern("(<< $x 1)").unwrap(),
            80,
            "x * 2 -> x << 1",
        ),
        Rule::new(
            "div-to-shr",
            parse_pattern("(/ $x 2)").unwrap(),
            parse_pattern("(>> $x 1)").unwrap(),
            80,
            "x / 2 -> x >> 1",
        ),
        Rule::new(
            "double-neg",
            parse_pattern("(! (! $x))").unwrap(),
            parse_pattern("$x").unwrap(),
            90,
            "!!x -> x",
        ),
        Rule::new(
            "and-true",
            parse_pattern("(&& $x true)").unwrap(),
            parse_pattern("$x").unwrap(),
            90,
            "x && true -> x",
        ),
        Rule::new(
            "and-true-left",
            parse_pattern("(&& true $x)").unwrap(),
            parse_pattern("$x").unwrap(),
            90,
            "true && x -> x",
        ),
        Rule::new(
            "or-false",
            parse_pattern("(|| $x false)").unwrap(),
            parse_pattern("$x").unwrap(),
            90,
            "x || false -> x",
        ),
        Rule::new(
            "or-false-left",
            parse_pattern("(|| false $x)").unwrap(),
            parse_pattern("$x").unwrap(),
            90,
            "false || x -> x",
        ),
        Rule::new(
            "and-false",
            parse_pattern("(&& $x false)").unwrap(),
            parse_pattern("false").unwrap(),
            90,
            "x && false -> false",
        ),
        Rule::new(
            "or-true",
            parse_pattern("(|| $x true)").unwrap(),
            parse_pattern("true").unwrap(),
            90,
            "x || true -> true",
        ),
    ]
}

// ============================================================================
// CONSTANT FOLDING ENGINE
// ============================================================================

fn try_const_fold(expr: &Expr) -> Option<Literal> {
    match expr
    {
        Expr::BinOp { op, left, right } =>
        {
            let l = try_const_fold(left)?;
            let r = try_const_fold(right)?;
            match op
            {
                BinOpKind::Add => match (&l, &r)
                {
                    (Literal::Int(a), Literal::Int(b)) => Some(Literal::Int(a + b)),
                    (a, b) => Some(Literal::Float(a.as_f64()? + b.as_f64()?)),
                },
                BinOpKind::Sub => match (&l, &r)
                {
                    (Literal::Int(a), Literal::Int(b)) => Some(Literal::Int(a - b)),
                    (a, b) => Some(Literal::Float(a.as_f64()? - b.as_f64()?)),
                },
                BinOpKind::Mul => match (&l, &r)
                {
                    (Literal::Int(a), Literal::Int(b)) => Some(Literal::Int(a * b)),
                    (a, b) => Some(Literal::Float(a.as_f64()? * b.as_f64()?)),
                },
                BinOpKind::Div =>
                {
                    let rv = r.as_f64()?;
                    if rv == 0.0
                    {
                        return None;
                    }
                    match (&l, &r)
                    {
                        (Literal::Int(a), Literal::Int(b)) if *b != 0 => Some(Literal::Int(a / b)),
                        (a, _b) => Some(Literal::Float(a.as_f64()? / rv)),
                    }
                },
                BinOpKind::Rem => match (&l, &r)
                {
                    (Literal::Int(a), Literal::Int(b)) if *b != 0 => Some(Literal::Int(a % b)),
                    _ => None,
                },
                BinOpKind::Eq => Some(Literal::Bool(
                    std::mem::discriminant(&l) == std::mem::discriminant(&r) && l == r,
                )),
                BinOpKind::Ne => Some(Literal::Bool(l != r)),
                BinOpKind::Lt => match (&l, &r)
                {
                    (Literal::Int(a), Literal::Int(b)) => Some(Literal::Bool(a < b)),
                    (a, b) => Some(Literal::Bool(a.as_f64()? < b.as_f64()?)),
                },
                BinOpKind::Le => match (&l, &r)
                {
                    (Literal::Int(a), Literal::Int(b)) => Some(Literal::Bool(a <= b)),
                    (a, b) => Some(Literal::Bool(a.as_f64()? <= b.as_f64()?)),
                },
                BinOpKind::Gt => match (&l, &r)
                {
                    (Literal::Int(a), Literal::Int(b)) => Some(Literal::Bool(a > b)),
                    (a, b) => Some(Literal::Bool(a.as_f64()? > b.as_f64()?)),
                },
                BinOpKind::Ge => match (&l, &r)
                {
                    (Literal::Int(a), Literal::Int(b)) => Some(Literal::Bool(a >= b)),
                    (a, b) => Some(Literal::Bool(a.as_f64()? >= b.as_f64()?)),
                },
                BinOpKind::And => match (&l, &r)
                {
                    (Literal::Bool(a), Literal::Bool(b)) => Some(Literal::Bool(*a && *b)),
                    _ => None,
                },
                BinOpKind::Or => match (&l, &r)
                {
                    (Literal::Bool(a), Literal::Bool(b)) => Some(Literal::Bool(*a || *b)),
                    _ => None,
                },
                BinOpKind::BitAnd => match (&l, &r)
                {
                    (Literal::Int(a), Literal::Int(b)) => Some(Literal::Int(a & b)),
                    _ => None,
                },
                BinOpKind::BitOr => match (&l, &r)
                {
                    (Literal::Int(a), Literal::Int(b)) => Some(Literal::Int(a | b)),
                    _ => None,
                },
                BinOpKind::BitXor => match (&l, &r)
                {
                    (Literal::Int(a), Literal::Int(b)) => Some(Literal::Int(a ^ b)),
                    _ => None,
                },
                BinOpKind::Shl => match (&l, &r)
                {
                    (Literal::Int(a), Literal::Int(b)) =>
                    {
                        Some(Literal::Int(a.wrapping_shl(*b as u32)))
                    },
                    _ => None,
                },
                BinOpKind::Shr => match (&l, &r)
                {
                    (Literal::Int(a), Literal::Int(b)) =>
                    {
                        Some(Literal::Int(a.wrapping_shr(*b as u32)))
                    },
                    _ => None,
                },
                _ => None,
            }
        },
        Expr::UnaryOp { op, expr } =>
        {
            let v = try_const_fold(expr)?;
            match op
            {
                UnaryOpKind::Neg => match v
                {
                    Literal::Int(i) => Some(Literal::Int(-i)),
                    Literal::Float(f) => Some(Literal::Float(-f)),
                    _ => None,
                },
                UnaryOpKind::Not => match v
                {
                    Literal::Bool(b) => Some(Literal::Bool(!b)),
                    _ => None,
                },
                _ => None,
            }
        },
        Expr::Lit(lit) => Some(lit.clone()),
        _ => None,
    }
}

fn apply_const_fold_rule(rule: &Rule, expr: &Expr) -> Option<Expr> {
    if ![
        "const-fold-add",
        "const-fold-sub",
        "const-fold-mul",
        "const-fold-div",
    ]
    .contains(&rule.name.as_str())
    {
        return None;
    }
    try_const_fold(expr).map(Expr::Lit)
}

// ============================================================================
// DEAD CODE ELIMINATION
// ============================================================================

/// Eliminate code after `return`, `break`, `continue` within a block.
pub fn dead_code_elimination(expr: &Expr) -> Expr {
    match expr
    {
        Expr::Block(stmts) =>
        {
            let mut alive = Vec::new();
            for stmt in stmts
            {
                let should_terminate = is_terminator(stmt);
                alive.push(dead_code_elimination(stmt));
                if should_terminate
                {
                    break;
                }
            }
            Expr::Block(alive)
        },
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => Expr::If {
            cond: cond.clone(),
            then_branch: Box::new(dead_code_elimination(then_branch)),
            else_branch: else_branch
                .as_ref()
                .map(|e| Box::new(dead_code_elimination(e))),
        },
        Expr::While { cond, body } => Expr::While {
            cond: cond.clone(),
            body: Box::new(dead_code_elimination(body)),
        },
        Expr::For { var, iter, body } => Expr::For {
            var: var.clone(),
            iter: iter.clone(),
            body: Box::new(dead_code_elimination(body)),
        },
        Expr::Function {
            name,
            params,
            return_type,
            body,
        } => Expr::Function {
            name: name.clone(),
            params: params.clone(),
            return_type: return_type.clone(),
            body: Box::new(dead_code_elimination(body)),
        },
        Expr::Match { expr: e, arms } => Expr::Match {
            expr: e.clone(),
            arms: arms
                .iter()
                .map(|(p, b)| (p.clone(), dead_code_elimination(b)))
                .collect(),
        },
        Expr::Let { name, value, body } => Expr::Let {
            name: name.clone(),
            value: value.clone(),
            body: Box::new(dead_code_elimination(body)),
        },
        Expr::LetMut { name, value, body } => Expr::LetMut {
            name: name.clone(),
            value: value.clone(),
            body: Box::new(dead_code_elimination(body)),
        },
        other => other.clone(),
    }
}

fn is_terminator(expr: &Expr) -> bool {
    matches!(expr, Expr::Return(_) | Expr::Break | Expr::Continue)
}

// ============================================================================
// COMMON SUBEXPRESSION ELIMINATION (simple)
// ============================================================================

/// Simple common subexpression elimination within a block.
pub fn common_subexpression_elimination(expr: &Expr) -> Expr {
    match expr
    {
        Expr::Block(stmts) =>
        {
            let mut seen: HashMap<String, String> = HashMap::new();
            let mut new_stmts = Vec::new();
            let mut counter = 0u64;
            for stmt in stmts
            {
                let processed = common_subexpression_elimination(stmt);
                if let Expr::Assign { name, value } = &processed
                {
                    let val_str = value.to_string();
                    if !matches!(value.as_ref(), Expr::Lit(_) | Expr::Var(_))
                    {
                        if let Some(tmp) = seen.get(&val_str)
                        {
                            new_stmts.push(Expr::Assign {
                                name: name.clone(),
                                value: Box::new(Expr::Var(tmp.clone())),
                            });
                            continue;
                        }
                        else if let std::collections::hash_map::Entry::Vacant(e) = seen.entry(val_str) {
                            counter += 1;
                            let tmp_name = format!("__cse_tmp_{}", counter);
                            new_stmts.push(Expr::Let {
                                name: tmp_name.clone(),
                                value: value.clone(),
                                body: Box::new(Expr::Block(vec![])),
                            });
                            e.insert(tmp_name.clone());
                            new_stmts.push(Expr::Assign {
                                name: name.clone(),
                                value: Box::new(Expr::Var(tmp_name)),
                            });
                            continue;
                        }
                    }
                }
                new_stmts.push(processed);
            }
            Expr::Block(new_stmts)
        },
        other => other.clone(),
    }
}

// ============================================================================
// LOOP-INVARIANT CODE MOTION
// ============================================================================

/// Move loop-invariant computations out of a For/While loop.
pub fn loop_invariant_code_motion(expr: &Expr) -> Expr {
    match expr
    {
        Expr::For { var, iter, body } =>
        {
            let loop_var = var.clone();
            let invariant = collect_invariant_exprs(body, &loop_var);
            let body_processed = loop_invariant_code_motion(body);
            if invariant.is_empty()
            {
                Expr::For {
                    var: loop_var,
                    iter: iter.clone(),
                    body: Box::new(body_processed),
                }
            }
            else
            {
                let mut stmts = invariant
                    .into_iter()
                    .enumerate()
                    .map(|(i, e)| Expr::Let {
                        name: format!("__inv_{}", i),
                        value: Box::new(e),
                        body: Box::new(Expr::Block(vec![])),
                    })
                    .collect::<Vec<_>>();
                stmts.push(Expr::For {
                    var: loop_var,
                    iter: iter.clone(),
                    body: Box::new(body_processed),
                });
                Expr::Block(stmts)
            }
        },
        Expr::While { cond, body } => Expr::While {
            cond: cond.clone(),
            body: Box::new(loop_invariant_code_motion(body)),
        },
        Expr::Block(stmts) => Expr::Block(stmts.iter().map(loop_invariant_code_motion).collect()),
        other => other.clone(),
    }
}

fn collect_invariant_exprs(expr: &Expr, loop_var: &str) -> Vec<Expr> {
    let mut result = Vec::new();
    collect_invariant_impl(expr, loop_var, &mut HashSet::new(), &mut result);
    result
}

fn collect_invariant_impl(
    expr: &Expr,
    loop_var: &str,
    seen: &mut HashSet<String>,
    result: &mut Vec<Expr>,
) {
    match expr
    {
        Expr::BinOp { left, right, .. } =>
        {
            if !references_var(left, loop_var) && !references_var(right, loop_var)
            {
                let s = expr.to_string();
                if seen.insert(s) && !matches!(expr, Expr::Lit(_) | Expr::Var(_))
                {
                    result.push(expr.clone());
                }
            }
            collect_invariant_impl(left, loop_var, seen, result);
            collect_invariant_impl(right, loop_var, seen, result);
        },
        Expr::Block(stmts) =>
        {
            for s in stmts
            {
                collect_invariant_impl(s, loop_var, seen, result);
            }
        },
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } =>
        {
            collect_invariant_impl(cond, loop_var, seen, result);
            collect_invariant_impl(then_branch, loop_var, seen, result);
            if let Some(e) = else_branch
            {
                collect_invariant_impl(e, loop_var, seen, result);
            }
        },
        Expr::Assign { value, .. } =>
        {
            collect_invariant_impl(value, loop_var, seen, result);
        },
        Expr::Call { args, .. } =>
        {
            for a in args
            {
                collect_invariant_impl(a, loop_var, seen, result);
            }
        },
        _ =>
        {},
    }
}

fn references_var(expr: &Expr, var: &str) -> bool {
    match expr
    {
        Expr::Var(v) => v == var,
        Expr::BinOp { left, right, .. } => references_var(left, var) || references_var(right, var),
        Expr::UnaryOp { expr: e, .. } => references_var(e, var),
        Expr::Call { func, args } =>
        {
            references_var(func, var) || args.iter().any(|a| references_var(a, var))
        },
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } =>
        {
            references_var(cond, var)
                || references_var(then_branch, var)
                || else_branch
                    .as_ref()
                    .is_some_and(|e| references_var(e, var))
        },
        Expr::Block(stmts) => stmts.iter().any(|s| references_var(s, var)),
        Expr::Assign { value, .. } => references_var(value, var),
        Expr::Index { expr: e, index } => references_var(e, var) || references_var(index, var),
        Expr::FieldAccess { expr: e, .. } => references_var(e, var),
        Expr::For { iter, body, .. } => references_var(iter, var) || references_var(body, var),
        Expr::While { cond: c, body: b } => references_var(c, var) || references_var(b, var),
        Expr::Return(Some(e)) => references_var(e, var),
        Expr::Match { expr: e, arms } =>
        {
            references_var(e, var) || arms.iter().any(|(_, b)| references_var(b, var))
        },
        _ => false,
    }
}

// ============================================================================
// 4. CODE REFACTORING
// ============================================================================

/// Rename a variable throughout an expression.
pub fn rename_variable(expr: &Expr, old_name: &str, new_name: &str) -> Expr {
    let mut renamer = VariableRenamer {
        old: old_name.to_string(),
        new: new_name.to_string(),
    };
    renamer.fold_expr(expr)
}

struct VariableRenamer {
    old: String,
    new: String,
}

impl AstFold for VariableRenamer {
    fn fold_expr(&mut self, expr: &Expr) -> Expr {
        match expr
        {
            Expr::Var(name) if *name == self.old => Expr::Var(self.new.clone()),
            Expr::Let { name, value, body } if *name == self.old => Expr::Let {
                name: self.new.clone(),
                value: Box::new(self.fold_expr(value)),
                body: Box::new(self.fold_expr(body)),
            },
            Expr::LetMut { name, value, body } if *name == self.old => Expr::LetMut {
                name: self.new.clone(),
                value: Box::new(self.fold_expr(value)),
                body: Box::new(self.fold_expr(body)),
            },
            Expr::For { var, iter, body } if *var == self.old => Expr::For {
                var: self.new.clone(),
                iter: Box::new(self.fold_expr(iter)),
                body: Box::new(self.fold_expr(body)),
            },
            Expr::Assign { name, value } if *name == self.old => Expr::Assign {
                name: self.new.clone(),
                value: Box::new(self.fold_expr(value)),
            },
            Expr::Function {
                name,
                params,
                return_type,
                body,
            } =>
            {
                let new_params: Vec<String> = params
                    .iter()
                    .map(|p| {
                        if p == &self.old
                        {
                            self.new.clone()
                        }
                        else
                        {
                            p.clone()
                        }
                    })
                    .collect();
                Expr::Function {
                    name: name.clone(),
                    params: new_params,
                    return_type: return_type.clone(),
                    body: Box::new(self.fold_expr(body)),
                }
            },
            _ => fold_expr(self, expr),
        }
    }
}

/// Inline a variable: replace uses with its value.
pub fn inline_variable(expr: &Expr, var_name: &str) -> Expr {
    match expr
    {
        Expr::Let { name, value, body } if name == var_name =>
        {
            let mut inliner = ExpressionInliner {
                var: var_name.to_string(),
                value: value.as_ref().clone(),
            };
            inliner.fold_expr(body)
        },
        Expr::LetMut { name, value, body } if name == var_name =>
        {
            let mut inliner = ExpressionInliner {
                var: var_name.to_string(),
                value: value.as_ref().clone(),
            };
            inliner.fold_expr(body)
        },
        _ => expr.clone(),
    }
}

struct ExpressionInliner {
    var: String,
    value: Expr,
}

impl AstFold for ExpressionInliner {
    fn fold_expr(&mut self, expr: &Expr) -> Expr {
        match expr
        {
            Expr::Var(name) if *name == self.var => self.value.clone(),
            _ => fold_expr(self, expr),
        }
    }
}

/// Convert a for loop that pushes to a vec into an iterator collect pattern.
pub fn convert_loop_to_iterator(expr: &Expr) -> Option<Expr> {
    match expr
    {
        Expr::For { var, iter, body } =>
        {
            let (target, val) = match body.as_ref()
            {
                Expr::Block(stmts) if stmts.len() == 1 => match &stmts[0]
                {
                    Expr::Call { func, args } if is_push_call(func, args) =>
                    {
                        (args[0].clone(), args[1].clone())
                    },
                    _ => return None,
                },
                Expr::Call { func, args } if is_push_call(func, args) =>
                {
                    (args[0].clone(), args[1].clone())
                },
                _ => return None,
            };
            Some(Expr::Let {
                name: target.to_string(),
                value: Box::new(Expr::Call {
                    func: Box::new(Expr::FieldAccess {
                        expr: Box::new(Expr::Call {
                            func: Box::new(Expr::FieldAccess {
                                expr: Box::new(iter.as_ref().clone()),
                                field: "map".to_string(),
                            }),
                            args: vec![Expr::Function {
                                name: "".to_string(),
                                params: vec![var.clone()],
                                return_type: None,
                                body: Box::new(val),
                            }],
                        }),
                        field: "collect".to_string(),
                    }),
                    args: vec![],
                }),
                body: Box::new(Expr::Block(vec![])),
            })
        },
        _ => None,
    }
}

fn is_push_call(func: &Expr, args: &[Expr]) -> bool {
    matches!(func, Expr::Var(s) if s == "push") && args.len() == 2
}

/// Convert match with a single arm into if-let.
pub fn convert_match_to_if_let(expr: &Expr) -> Option<Expr> {
    match expr
    {
        Expr::Match { expr: e, arms } if arms.len() == 1 =>
        {
            let (pat, body) = &arms[0];
            if let Expr::Var(pname) = pat
            {
                Some(Expr::Let {
                    name: pname.clone(),
                    value: e.clone(),
                    body: Box::new(body.clone()),
                })
            }
            else
            {
                Some(Expr::If {
                    cond: e.clone(),
                    then_branch: Box::new(body.clone()),
                    else_branch: None,
                })
            }
        },
        _ => None,
    }
}

/// Simplify boolean expression.
pub fn simplify_boolean(expr: &Expr) -> Expr {
    match expr
    {
        Expr::BinOp { op, left, right } =>
        {
            let l = simplify_boolean(left);
            let r = simplify_boolean(right);
            match op
            {
                BinOpKind::And =>
                {
                    if matches!(&l, Expr::Lit(Literal::Bool(true)))
                    {
                        return r;
                    }
                    if matches!(&r, Expr::Lit(Literal::Bool(true)))
                    {
                        return l;
                    }
                    if matches!(&l, Expr::Lit(Literal::Bool(false)))
                        || matches!(&r, Expr::Lit(Literal::Bool(false)))
                    {
                        return Expr::Lit(Literal::Bool(false));
                    }
                    Expr::BinOp {
                        op: *op,
                        left: Box::new(l),
                        right: Box::new(r),
                    }
                },
                BinOpKind::Or =>
                {
                    if matches!(&l, Expr::Lit(Literal::Bool(false)))
                    {
                        return r;
                    }
                    if matches!(&r, Expr::Lit(Literal::Bool(false)))
                    {
                        return l;
                    }
                    if matches!(&l, Expr::Lit(Literal::Bool(true)))
                        || matches!(&r, Expr::Lit(Literal::Bool(true)))
                    {
                        return Expr::Lit(Literal::Bool(true));
                    }
                    Expr::BinOp {
                        op: *op,
                        left: Box::new(l),
                        right: Box::new(r),
                    }
                },
                BinOpKind::Eq =>
                {
                    if l == r
                    {
                        return Expr::Lit(Literal::Bool(true));
                    }
                    Expr::BinOp {
                        op: *op,
                        left: Box::new(l),
                        right: Box::new(r),
                    }
                },
                _ => Expr::BinOp {
                    op: *op,
                    left: Box::new(l),
                    right: Box::new(r),
                },
            }
        },
        Expr::UnaryOp {
            op: UnaryOpKind::Not,
            expr,
        } =>
        {
            let e = simplify_boolean(expr);
            match &e
            {
                Expr::Lit(Literal::Bool(true)) => Expr::Lit(Literal::Bool(false)),
                Expr::Lit(Literal::Bool(false)) => Expr::Lit(Literal::Bool(true)),
                Expr::UnaryOp {
                    op: UnaryOpKind::Not,
                    expr: inner,
                } => *inner.clone(),
                Expr::BinOp {
                    op: BinOpKind::Eq,
                    left,
                    right,
                } => Expr::BinOp {
                    op: BinOpKind::Ne,
                    left: left.clone(),
                    right: right.clone(),
                },
                Expr::BinOp {
                    op: BinOpKind::Ne,
                    left,
                    right,
                } => Expr::BinOp {
                    op: BinOpKind::Eq,
                    left: left.clone(),
                    right: right.clone(),
                },
                Expr::BinOp {
                    op: BinOpKind::Lt,
                    left,
                    right,
                } => Expr::BinOp {
                    op: BinOpKind::Ge,
                    left: left.clone(),
                    right: right.clone(),
                },
                Expr::BinOp {
                    op: BinOpKind::Le,
                    left,
                    right,
                } => Expr::BinOp {
                    op: BinOpKind::Gt,
                    left: left.clone(),
                    right: right.clone(),
                },
                Expr::BinOp {
                    op: BinOpKind::Gt,
                    left,
                    right,
                } => Expr::BinOp {
                    op: BinOpKind::Le,
                    left: left.clone(),
                    right: right.clone(),
                },
                Expr::BinOp {
                    op: BinOpKind::Ge,
                    left,
                    right,
                } => Expr::BinOp {
                    op: BinOpKind::Lt,
                    left: left.clone(),
                    right: right.clone(),
                },
                _ => Expr::UnaryOp {
                    op: UnaryOpKind::Not,
                    expr: Box::new(e),
                },
            }
        },
        _ => expr.clone(),
    }
}

/// Extract repeated code patterns into a function (simplified heuristic).
pub fn extract_function(expr: &Expr, threshold: usize) -> Expr {
    match expr
    {
        Expr::Block(stmts) =>
        {
            let mut pattern_counts: HashMap<String, Vec<usize>> = HashMap::new();
            for (i, stmt) in stmts.iter().enumerate()
            {
                let s = stmt.to_string();
                if s.len() > 10
                {
                    pattern_counts.entry(s).or_default().push(i);
                }
            }
            let mut extracted: HashSet<usize> = HashSet::new();
            let mut new_body = Vec::new();
            for (repeated, indices) in &pattern_counts
            {
                if indices.len() >= threshold
                {
                    for &idx in indices
                    {
                        extracted.insert(idx);
                    }
                    let fn_name = format!("__extracted_{}", repeated.len().min(20));
                    new_body.push(Expr::Function {
                        name: fn_name.clone(),
                        params: vec![],
                        return_type: None,
                        body: Box::new(stmts[indices[0]].clone()),
                    });
                    for _idx in indices
                    {
                        new_body.push(Expr::Call {
                            func: Box::new(Expr::Var(fn_name.clone())),
                            args: vec![],
                        });
                    }
                }
            }
            for (i, stmt) in stmts.iter().enumerate()
            {
                if !extracted.contains(&i)
                {
                    new_body.push(extract_function(stmt, threshold));
                }
            }
            if new_body.is_empty()
            {
                expr.clone()
            }
            else
            {
                Expr::Block(new_body)
            }
        },
        _ => expr.clone(),
    }
}

// ============================================================================
// 5. CODE TRANSPILATION
// ============================================================================

/// Transpile a Rust-like expression to Python-like code.
pub fn transpile_rust_to_python(expr: &Expr) -> String {
    transpile_python(expr, 0)
}

fn transpile_python(expr: &Expr, indent: usize) -> String {
    let pad = "    ".repeat(indent);
    match expr
    {
        Expr::Lit(Literal::Bool(true)) => "True".to_string(),
        Expr::Lit(Literal::Bool(false)) => "False".to_string(),
        Expr::Lit(Literal::Unit) => "None".to_string(),
        Expr::Lit(lit) => format!("{}", lit),
        Expr::Var(name) => name.clone(),
        Expr::BinOp { op, left, right } =>
        {
            let op_str = match op
            {
                BinOpKind::And => "and",
                BinOpKind::Or => "or",
                _ =>
                {
                    return format!(
                        "({} {} {})",
                        transpile_python(left, 0),
                        op,
                        transpile_python(right, 0)
                    );
                },
            };
            format!(
                "{} {} {}",
                transpile_python(left, 0),
                op_str,
                transpile_python(right, 0)
            )
        },
        Expr::UnaryOp {
            op: UnaryOpKind::Not,
            expr,
        } =>
        {
            format!("not {}", transpile_python(expr, 0))
        },
        Expr::UnaryOp { op, expr } =>
        {
            format!("{}({})", op, transpile_python(expr, 0))
        },
        Expr::Call { func, args } =>
        {
            let args_str: Vec<String> = args.iter().map(|a| transpile_python(a, 0)).collect();
            format!("{}({})", transpile_python(func, 0), args_str.join(", "))
        },
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } =>
        {
            let cond_str = transpile_python(cond, 0);
            let then_str = transpile_python(then_branch, indent);
            if let Some(else_br) = else_branch
            {
                format!(
                    "if {}:\n{}\n{}else:\n{}",
                    cond_str,
                    then_str,
                    pad,
                    transpile_python(else_br, indent)
                )
            }
            else
            {
                format!("if {}:\n{}", cond_str, then_str)
            }
        },
        Expr::Let { name, value, body } =>
        {
            let val_str = transpile_python(value, 0);
            let body_str = transpile_python(body, indent);
            format!("{}{} = {}\n{}", pad, name, val_str, body_str)
        },
        Expr::LetMut { name, value, body } =>
        {
            let val_str = transpile_python(value, 0);
            let body_str = transpile_python(body, indent);
            format!("{}{} = {}\n{}", pad, name, val_str, body_str)
        },
        Expr::While { cond, body } =>
        {
            let cond_str = transpile_python(cond, 0);
            let body_str = transpile_python(body, indent + 1);
            format!("{}while {}:\n{}", pad, cond_str, body_str)
        },
        Expr::For { var, iter, body } =>
        {
            let iter_str = transpile_python(iter, 0);
            let body_str = transpile_python(body, indent + 1);
            format!("{}for {} in {}:\n{}", pad, var, iter_str, body_str)
        },
        Expr::Assign { name, value } =>
        {
            format!("{}{} = {}", pad, name, transpile_python(value, 0))
        },
        Expr::Block(stmts) =>
        {
            let mut result = String::new();
            for s in stmts
            {
                if !result.is_empty()
                {
                    result.push('\n');
                }
                result.push_str(&transpile_python(s, indent));
            }
            result
        },
        Expr::Return(Some(e)) => format!("{}return {}", pad, transpile_python(e, 0)),
        Expr::Return(None) => format!("{}return", pad),
        Expr::Function {
            name, params, body, ..
        } =>
        {
            let params_str = params.join(", ");
            let body_str = transpile_python(body, indent + 1);
            format!("{}def {}({}):\n{}", pad, name, params_str, body_str)
        },
        Expr::Struct { name, fields } =>
        {
            let fields_str: Vec<String> = fields.iter().map(|(n, _)| n.clone()).collect();
            format!(
                "{}class {}:\n{}    def __init__(self, {}):\n{}        pass",
                pad,
                name,
                pad,
                fields_str.join(", "),
                "        ".repeat(indent + 1),
            )
        },
        Expr::Match { expr: e, arms } =>
        {
            let mut result = format!("{}match {}:", pad, transpile_python(e, 0));
            for (pat, body) in arms
            {
                result.push_str(&format!(
                    "\n{}    case {}:\n{}",
                    pad,
                    transpile_python(pat, 0),
                    transpile_python(body, indent + 1),
                ));
            }
            result
        },
        Expr::Break => format!("{}break", pad),
        Expr::Continue => format!("{}continue", pad),
        Expr::FieldAccess { expr: e, field } => format!("{}.{}", transpile_python(e, 0), field),
        Expr::Index { expr: e, index } =>
        {
            format!("{}[{}]", transpile_python(e, 0), transpile_python(index, 0))
        },
        _ => "<unsupported>".to_string(),
    }
}

/// Transpile a Rust-like expression to C-like code.
pub fn transpile_rust_to_c(expr: &Expr) -> String {
    transpile_c(expr, 0)
}

fn transpile_c(expr: &Expr, indent: usize) -> String {
    let pad = "    ".repeat(indent);
    match expr
    {
        Expr::Lit(Literal::Bool(true)) => "1".to_string(),
        Expr::Lit(Literal::Bool(false)) => "0".to_string(),
        Expr::Lit(Literal::Unit) => "((void)0)".to_string(),
        Expr::Lit(lit) => format!("{}", lit),
        Expr::Var(name) => name.clone(),
        Expr::BinOp { op, left, right } =>
        {
            let op_str = match op
            {
                BinOpKind::And => "&&",
                BinOpKind::Or => "||",
                _ =>
                {
                    return format!(
                        "({} {} {})",
                        transpile_c(left, 0),
                        op,
                        transpile_c(right, 0)
                    );
                },
            };
            format!(
                "({} {} {})",
                transpile_c(left, 0),
                op_str,
                transpile_c(right, 0)
            )
        },
        Expr::UnaryOp {
            op: UnaryOpKind::Not,
            expr,
        } =>
        {
            format!("!({})", transpile_c(expr, 0))
        },
        Expr::UnaryOp { op, expr } => format!("{}({})", op, transpile_c(expr, 0)),
        Expr::Call { func, args } =>
        {
            let args_str: Vec<String> = args.iter().map(|a| transpile_c(a, 0)).collect();
            format!("{}({})", transpile_c(func, 0), args_str.join(", "))
        },
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } =>
        {
            let then_str = transpile_c(then_branch, indent);
            if let Some(else_br) = else_branch
            {
                format!(
                    "{}if ({}) {}\n{}else {}",
                    pad,
                    transpile_c(cond, 0),
                    then_str.trim(),
                    pad,
                    transpile_c(else_br, indent).trim()
                )
            }
            else
            {
                format!("{}if ({}) {}", pad, transpile_c(cond, 0), then_str.trim())
            }
        },
        Expr::Let { name, value, body } =>
        {
            format!(
                "{}int {} = {};\n{}",
                pad,
                name,
                transpile_c(value, 0),
                transpile_c(body, indent)
            )
        },
        Expr::LetMut { name, value, body } =>
        {
            format!(
                "{}int {} = {};\n{}",
                pad,
                name,
                transpile_c(value, 0),
                transpile_c(body, indent)
            )
        },
        Expr::While { cond, body } =>
        {
            format!(
                "{}while ({}) {}",
                pad,
                transpile_c(cond, 0),
                transpile_c(body, indent)
            )
        },
        Expr::For { var, iter, body } =>
        {
            let iter_str = transpile_c(iter, 0);
            let body_str = transpile_c(body, indent);
            format!(
                "{}for (int {} = 0; {} < {}; {}++) {}",
                pad,
                var,
                var,
                iter_str,
                var,
                body_str.trim()
            )
        },
        Expr::Assign { name, value } =>
        {
            format!("{}{} = {};", pad, name, transpile_c(value, 0))
        },
        Expr::Block(stmts) =>
        {
            let inner: Vec<String> = stmts.iter().map(|s| transpile_c(s, indent)).collect();
            if stmts.len() == 1 && indent == 0
            {
                inner[0].clone()
            }
            else
            {
                format!("{{\n{}\n{}}}", inner.join("\n"), pad)
            }
        },
        Expr::Return(Some(e)) => format!("{}return {};", pad, transpile_c(e, 0)),
        Expr::Return(None) => format!("{}return;", pad),
        Expr::Function {
            name,
            params,
            return_type,
            body,
        } =>
        {
            let ret = return_type.as_deref().unwrap_or("void");
            let params_str = params
                .iter()
                .map(|p| format!("int {}", p))
                .collect::<Vec<_>>()
                .join(", ");
            let body_str = transpile_c(body, indent + 1);
            format!("{} {}({}) {}", ret, name, params_str, body_str)
        },
        Expr::Struct { name, fields } =>
        {
            let fields_str: Vec<String> = fields
                .iter()
                .map(|(n, t)| format!("    {} {};", t, n))
                .collect();
            format!("typedef struct {{\n{}\n}} {};", fields_str.join("\n"), name)
        },
        Expr::Break => format!("{}break;", pad),
        Expr::Continue => format!("{}continue;", pad),
        Expr::FieldAccess { expr: e, field } => format!("{}.{}", transpile_c(e, 0), field),
        Expr::Index { expr: e, index } =>
        {
            format!("{}[{}]", transpile_c(e, 0), transpile_c(index, 0))
        },
        _ => "/* unsupported */".to_string(),
    }
}

// ============================================================================
// 6. PATTERN DATABASE
// ============================================================================

/// A database of transformation rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternDatabase {
    pub name: String,
    pub version: String,
    pub rules: Vec<Rule>,
}

impl PatternDatabase {
    pub fn new(name: &str) -> Self {
        PatternDatabase {
            name: name.to_string(),
            version: "0.1.0".to_string(),
            rules: Vec::new(),
        }
    }

    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    pub fn add_rules(&mut self, rules: Vec<Rule>) {
        self.rules.extend(rules);
    }

    pub fn sort_by_priority(&mut self) {
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("Failed to parse JSON: {}", e))
    }

    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| format!("Failed to serialize: {}", e))
    }

    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let content =
            std::fs::read_to_string(path).map_err(|e| format!("Failed to read file: {}", e))?;
        Self::from_json(&content)
    }

    pub fn save_to_file(&self, path: &str) -> Result<(), String> {
        let json = self.to_json()?;
        std::fs::write(path, json).map_err(|e| format!("Failed to write file: {}", e))
    }

    pub fn find_conflicts(&self) -> Vec<(usize, usize, String)> {
        let mut conflicts = Vec::new();
        for i in 0..self.rules.len()
        {
            for j in (i + 1)..self.rules.len()
            {
                if patterns_overlap(&self.rules[i].pattern, &self.rules[j].pattern)
                {
                    conflicts.push((
                        i,
                        j,
                        format!(
                            "Patterns '{}' and '{}' may conflict",
                            self.rules[i].name, self.rules[j].name
                        ),
                    ));
                }
            }
        }
        conflicts
    }

    pub fn compose_rules(&self, a_idx: usize, b_idx: usize) -> Option<Rule> {
        if a_idx >= self.rules.len() || b_idx >= self.rules.len()
        {
            return None;
        }
        let a = &self.rules[a_idx];
        let b = &self.rules[b_idx];
        Some(Rule::new(
            &format!("{}->{}", a.name, b.name),
            a.pattern.clone(),
            b.replacement.clone(),
            (a.priority + b.priority) / 2,
            &format!("Composed: {} then {}", a.description, b.description),
        ))
    }
}

fn patterns_overlap(a: &Pattern, b: &Pattern) -> bool {
    use Pattern::*;
    match (a, b)
    {
        (Wildcard, _) | (_, Wildcard) => true,
        (Any(_), _) | (_, Any(_)) => true,
        (Lit(la), Lit(lb)) => la == lb,
        (Var(va), Var(vb)) => va == vb,
        (
            BinOp {
                op: oa,
                left: la,
                right: ra,
            },
            BinOp {
                op: ob,
                left: lb,
                right: rb,
            },
        ) => oa == ob && patterns_overlap(la, lb) && patterns_overlap(ra, rb),
        (UnaryOp { op: oa, expr: ea }, UnaryOp { op: ob, expr: eb }) =>
        {
            oa == ob && patterns_overlap(ea, eb)
        },
        (Call { func: fa, args: aa }, Call { func: fb, args: ab }) =>
        {
            aa.len() == ab.len()
                && patterns_overlap(fa, fb)
                && aa
                    .iter()
                    .zip(ab.iter())
                    .all(|(a, b)| patterns_overlap(a, b))
        },
        (
            If {
                cond: ca,
                then_branch: ta,
                else_branch: ea,
            },
            If {
                cond: cb,
                then_branch: tb,
                else_branch: eb,
            },
        ) =>
        {
            patterns_overlap(ca, cb)
                && patterns_overlap(ta, tb)
                && match (ea, eb)
                {
                    (Some(ea), Some(eb)) => patterns_overlap(ea, eb),
                    (None, None) => true,
                    _ => true,
                }
        },
        (
            For {
                var: va,
                iter: ia,
                body: ba,
            },
            For {
                var: vb,
                iter: ib,
                body: bb,
            },
        ) =>
        {
            (va.is_none() || vb.is_none() || va == vb)
                && patterns_overlap(ia, ib)
                && patterns_overlap(ba, bb)
        },
        _ => std::mem::discriminant(a) == std::mem::discriminant(b),
    }
}

// ============================================================================
// 7. TRANSFORMATION ENGINE
// ============================================================================

/// Strategies for applying rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionStrategy {
    First,
    Best,
    All,
    Exhaustive,
    CostBased,
}

/// A log entry recording a transformation step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformLogEntry {
    pub rule_name: String,
    pub before: String,
    pub after: String,
    pub depth: usize,
}

/// A logger that tracks transformation history.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransformLogger {
    pub entries: Vec<TransformLogEntry>,
}

impl TransformLogger {
    pub fn new() -> Self {
        TransformLogger {
            entries: Vec::new(),
        }
    }

    pub fn log(&mut self, rule_name: &str, before: &Expr, after: &Expr, depth: usize) {
        self.entries.push(TransformLogEntry {
            rule_name: rule_name.to_string(),
            before: before.to_string(),
            after: after.to_string(),
            depth,
        });
    }

    pub fn summary(&self) -> String {
        let mut s = format!("Transformation log ({} steps):\n", self.entries.len());
        for (i, entry) in self.entries.iter().enumerate()
        {
            s.push_str(&format!(
                "  {}. [{}] {} -> {}\n",
                i + 1,
                entry.rule_name,
                entry.before,
                entry.after,
            ));
        }
        s
    }
}

/// The transformation engine.
#[derive(Debug, Clone)]
pub struct TransformEngine {
    pub rules: Vec<Rule>,
    pub logger: TransformLogger,
    pub max_depth: usize,
    pub max_iterations: usize,
}

impl TransformEngine {
    pub fn new() -> Self {
        TransformEngine {
            rules: Vec::new(),
            logger: TransformLogger::new(),
            max_depth: 50,
            max_iterations: 100,
        }
    }

    pub fn with_rules(rules: Vec<Rule>) -> Self {
        TransformEngine {
            rules,
            logger: TransformLogger::new(),
            max_depth: 50,
            max_iterations: 100,
        }
    }

    pub fn add_rules(&mut self, rules: Vec<Rule>) {
        self.rules.extend(rules);
    }

    /// Apply a single named rule.
    pub fn apply_rule(&mut self, rule_name: &str, expr: &Expr) -> Option<Expr> {
        let rule = self.rules.iter().find(|r| r.name == rule_name)?.clone();
        self.apply_single_rule(&rule, expr)
    }

    fn apply_single_rule(&mut self, rule: &Rule, expr: &Expr) -> Option<Expr> {
        let mut transformed = None;

        if rule.name.starts_with("const-fold-")
        {
            transformed = apply_const_fold_rule(rule, expr);
        }

        if transformed.is_none()
        {
            transformed = rule.apply(expr);
        }

        if let Some(ref result) = transformed
        {
            self.logger.log(&rule.name, expr, result, 0);
            return Some(result.clone());
        }

        match expr
        {
            Expr::BinOp { op, left, right } =>
            {
                if let Some(new_left) = self.apply_single_rule(rule, left)
                {
                    return Some(Expr::BinOp {
                        op: *op,
                        left: Box::new(new_left),
                        right: right.clone(),
                    });
                }
                if let Some(new_right) = self.apply_single_rule(rule, right)
                {
                    return Some(Expr::BinOp {
                        op: *op,
                        left: left.clone(),
                        right: Box::new(new_right),
                    });
                }
                None
            },
            Expr::UnaryOp { op, expr: e } =>
            {
                self.apply_single_rule(rule, e).map(|new_e| Expr::UnaryOp {
                    op: *op,
                    expr: Box::new(new_e),
                })
            },
            Expr::Call { func, args } =>
            {
                if let Some(new_func) = self.apply_single_rule(rule, func)
                {
                    return Some(Expr::Call {
                        func: Box::new(new_func),
                        args: args.clone(),
                    });
                }
                for (i, arg) in args.iter().enumerate()
                {
                    if let Some(new_arg) = self.apply_single_rule(rule, arg)
                    {
                        let mut new_args = args.clone();
                        new_args[i] = new_arg;
                        return Some(Expr::Call {
                            func: func.clone(),
                            args: new_args,
                        });
                    }
                }
                None
            },
            Expr::Block(stmts) =>
            {
                for (i, stmt) in stmts.iter().enumerate()
                {
                    if let Some(new_stmt) = self.apply_single_rule(rule, stmt)
                    {
                        let mut new_stmts = stmts.clone();
                        new_stmts[i] = new_stmt;
                        return Some(Expr::Block(new_stmts));
                    }
                }
                None
            },
            Expr::If {
                cond,
                then_branch,
                else_branch,
            } =>
            {
                if let Some(new_cond) = self.apply_single_rule(rule, cond)
                {
                    return Some(Expr::If {
                        cond: Box::new(new_cond),
                        then_branch: then_branch.clone(),
                        else_branch: else_branch.clone(),
                    });
                }
                if let Some(new_then) = self.apply_single_rule(rule, then_branch)
                {
                    return Some(Expr::If {
                        cond: cond.clone(),
                        then_branch: Box::new(new_then),
                        else_branch: else_branch.clone(),
                    });
                }
                if let Some(else_br) = else_branch
                {
                    self.apply_single_rule(rule, else_br)
                        .map(|new_else| Expr::If {
                            cond: cond.clone(),
                            then_branch: then_branch.clone(),
                            else_branch: Some(Box::new(new_else)),
                        })
                }
                else
                {
                    None
                }
            },
            Expr::Let { name, value, body } =>
            {
                if let Some(new_value) = self.apply_single_rule(rule, value)
                {
                    return Some(Expr::Let {
                        name: name.clone(),
                        value: Box::new(new_value),
                        body: body.clone(),
                    });
                }
                self.apply_single_rule(rule, body)
                    .map(|new_body| Expr::Let {
                        name: name.clone(),
                        value: value.clone(),
                        body: Box::new(new_body),
                    })
            },
            Expr::For { var, iter, body } =>
            {
                if let Some(new_iter) = self.apply_single_rule(rule, iter)
                {
                    return Some(Expr::For {
                        var: var.clone(),
                        iter: Box::new(new_iter),
                        body: body.clone(),
                    });
                }
                self.apply_single_rule(rule, body)
                    .map(|new_body| Expr::For {
                        var: var.clone(),
                        iter: iter.clone(),
                        body: Box::new(new_body),
                    })
            },
            Expr::While { cond, body } =>
            {
                if let Some(new_cond) = self.apply_single_rule(rule, cond)
                {
                    return Some(Expr::While {
                        cond: Box::new(new_cond),
                        body: body.clone(),
                    });
                }
                self.apply_single_rule(rule, body)
                    .map(|new_body| Expr::While {
                        cond: cond.clone(),
                        body: Box::new(new_body),
                    })
            },
            Expr::Assign { name, value } =>
            {
                self.apply_single_rule(rule, value)
                    .map(|new_value| Expr::Assign {
                        name: name.clone(),
                        value: Box::new(new_value),
                    })
            },
            Expr::Return(Some(e)) => self
                .apply_single_rule(rule, e)
                .map(|new_e| Expr::Return(Some(Box::new(new_e)))),
            Expr::Function {
                name,
                params,
                return_type,
                body,
            } => self
                .apply_single_rule(rule, body)
                .map(|new_body| Expr::Function {
                    name: name.clone(),
                    params: params.clone(),
                    return_type: return_type.clone(),
                    body: Box::new(new_body),
                }),
            Expr::Match { expr: e, arms } =>
            {
                if let Some(new_e) = self.apply_single_rule(rule, e)
                {
                    return Some(Expr::Match {
                        expr: Box::new(new_e),
                        arms: arms.clone(),
                    });
                }
                for (i, (pat, body)) in arms.iter().enumerate()
                {
                    if let Some(new_body) = self.apply_single_rule(rule, body)
                    {
                        let mut new_arms = arms.clone();
                        new_arms[i] = (pat.clone(), new_body);
                        return Some(Expr::Match {
                            expr: e.clone(),
                            arms: new_arms,
                        });
                    }
                }
                None
            },
            _ => None,
        }
    }

    /// Apply rules using the specified strategy.
    pub fn transform(&mut self, expr: &Expr, strategy: SelectionStrategy) -> Vec<Expr> {
        match strategy
        {
            SelectionStrategy::First =>
            {
                for rule in &self.rules.clone()
                {
                    if let Some(result) = self.apply_single_rule(rule, expr)
                    {
                        return vec![result];
                    }
                }
                vec![]
            },
            SelectionStrategy::Best =>
            {
                let mut best: Option<(Expr, i32)> = None;
                for rule in &self.rules.clone()
                {
                    if let Some(result) = self.apply_single_rule(rule, expr)
                    {
                        if best.is_none() || rule.priority > best.as_ref().unwrap().1
                        {
                            best = Some((result, rule.priority));
                        }
                    }
                }
                best.map(|(e, _)| vec![e]).unwrap_or_default()
            },
            SelectionStrategy::All =>
            {
                let mut results = Vec::new();
                for rule in &self.rules.clone()
                {
                    if let Some(result) = self.apply_single_rule(rule, expr)
                    {
                        results.push(result);
                    }
                }
                results
            },
            SelectionStrategy::Exhaustive => self.transform_fixed_point(expr),
            SelectionStrategy::CostBased =>
            {
                let mut results = Vec::new();
                for rule in &self.rules.clone()
                {
                    if let Some(result) = self.apply_single_rule(rule, expr)
                    {
                        let cost = cost_estimate(&result);
                        results.push((result, cost));
                    }
                }
                results.sort_by_key(|a| a.1);
                results.into_iter().map(|(e, _)| e).collect()
            },
        }
    }

    /// Apply rules repeatedly until no more changes (fixed point).
    pub fn transform_fixed_point(&mut self, expr: &Expr) -> Vec<Expr> {
        let mut current = expr.clone();
        let mut results = vec![expr.clone()];
        for iteration in 0..self.max_iterations
        {
            let mut changed = false;
            let rules = self.rules.clone();
            for rule in &rules
            {
                if let Some(new_expr) = self.apply_rule_recursive(rule, &current, 0)
                {
                    let new_str = new_expr.to_string();
                    let cur_str = current.to_string();
                    if new_str != cur_str
                    {
                        self.logger.log(&rule.name, &current, &new_expr, iteration);
                        current = new_expr;
                        results.push(current.clone());
                        changed = true;
                    }
                }
            }
            if !changed
            {
                break;
            }
        }
        results
    }

    fn apply_rule_recursive(&mut self, rule: &Rule, expr: &Expr, depth: usize) -> Option<Expr> {
        if depth > self.max_depth
        {
            return None;
        }

        if let Some(result) = self.apply_single_rule_nolog(rule, expr)
        {
            return Some(result);
        }

        match expr
        {
            Expr::BinOp { op, left, right } =>
            {
                if let Some(new_left) = self.apply_rule_recursive(rule, left, depth + 1)
                {
                    Some(Expr::BinOp {
                        op: *op,
                        left: Box::new(new_left),
                        right: right.clone(),
                    })
                }
                else { self.apply_rule_recursive(rule, right, depth + 1).map(|new_right| Expr::BinOp {
                        op: *op,
                        left: left.clone(),
                        right: Box::new(new_right),
                    }) }
            },
            Expr::UnaryOp { op, expr: e } =>
            {
                self.apply_rule_recursive(rule, e, depth + 1)
                    .map(|new_e| Expr::UnaryOp {
                        op: *op,
                        expr: Box::new(new_e),
                    })
            },
            Expr::Block(stmts) =>
            {
                for (i, stmt) in stmts.iter().enumerate()
                {
                    if let Some(new_stmt) = self.apply_rule_recursive(rule, stmt, depth + 1)
                    {
                        let mut new_stmts = stmts.clone();
                        new_stmts[i] = new_stmt;
                        return Some(Expr::Block(new_stmts));
                    }
                }
                None
            },
            Expr::Call { func, args } =>
            {
                if let Some(new_func) = self.apply_rule_recursive(rule, func, depth + 1)
                {
                    return Some(Expr::Call {
                        func: Box::new(new_func),
                        args: args.clone(),
                    });
                }
                for (i, arg) in args.iter().enumerate()
                {
                    if let Some(new_arg) = self.apply_rule_recursive(rule, arg, depth + 1)
                    {
                        let mut new_args = args.clone();
                        new_args[i] = new_arg;
                        return Some(Expr::Call {
                            func: func.clone(),
                            args: new_args,
                        });
                    }
                }
                None
            },
            Expr::Let { name, value, body } =>
            {
                if let Some(new_val) = self.apply_rule_recursive(rule, value, depth + 1)
                {
                    Some(Expr::Let {
                        name: name.clone(),
                        value: Box::new(new_val),
                        body: body.clone(),
                    })
                }
                else
                {
                    self.apply_rule_recursive(rule, body, depth + 1)
                        .map(|new_body| Expr::Let {
                            name: name.clone(),
                            value: value.clone(),
                            body: Box::new(new_body),
                        })
                }
            },
            Expr::For { var, iter, body } =>
            {
                if let Some(new_iter) = self.apply_rule_recursive(rule, iter, depth + 1)
                {
                    Some(Expr::For {
                        var: var.clone(),
                        iter: Box::new(new_iter),
                        body: body.clone(),
                    })
                }
                else
                {
                    self.apply_rule_recursive(rule, body, depth + 1)
                        .map(|new_body| Expr::For {
                            var: var.clone(),
                            iter: iter.clone(),
                            body: Box::new(new_body),
                        })
                }
            },
            Expr::While { cond, body } =>
            {
                if let Some(new_cond) = self.apply_rule_recursive(rule, cond, depth + 1)
                {
                    Some(Expr::While {
                        cond: Box::new(new_cond),
                        body: body.clone(),
                    })
                }
                else
                {
                    self.apply_rule_recursive(rule, body, depth + 1)
                        .map(|new_body| Expr::While {
                            cond: cond.clone(),
                            body: Box::new(new_body),
                        })
                }
            },
            Expr::If {
                cond,
                then_branch,
                else_branch,
            } =>
            {
                if let Some(new_cond) = self.apply_rule_recursive(rule, cond, depth + 1)
                {
                    Some(Expr::If {
                        cond: Box::new(new_cond),
                        then_branch: then_branch.clone(),
                        else_branch: else_branch.clone(),
                    })
                }
                else if let Some(new_then) =
                    self.apply_rule_recursive(rule, then_branch, depth + 1)
                {
                    Some(Expr::If {
                        cond: cond.clone(),
                        then_branch: Box::new(new_then),
                        else_branch: else_branch.clone(),
                    })
                }
                else if let Some(else_br) = else_branch
                {
                    self.apply_rule_recursive(rule, else_br, depth + 1)
                        .map(|new_else| Expr::If {
                            cond: cond.clone(),
                            then_branch: then_branch.clone(),
                            else_branch: Some(Box::new(new_else)),
                        })
                }
                else
                {
                    None
                }
            },
            Expr::Assign { name, value } =>
            {
                self.apply_rule_recursive(rule, value, depth + 1)
                    .map(|new_val| Expr::Assign {
                        name: name.clone(),
                        value: Box::new(new_val),
                    })
            },
            Expr::Function {
                name,
                params,
                return_type,
                body,
            } => self
                .apply_rule_recursive(rule, body, depth + 1)
                .map(|new_body| Expr::Function {
                    name: name.clone(),
                    params: params.clone(),
                    return_type: return_type.clone(),
                    body: Box::new(new_body),
                }),
            _ => None,
        }
    }

    fn apply_single_rule_nolog(&self, rule: &Rule, expr: &Expr) -> Option<Expr> {
        if rule.name.starts_with("const-fold-")
        {
            apply_const_fold_rule(rule, expr)
        }
        else
        {
            rule.apply(expr)
        }
    }
}

impl Default for TransformEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Estimate the cost (complexity) of an expression.
pub fn cost_estimate(expr: &Expr) -> usize {
    match expr
    {
        Expr::Lit(_) | Expr::Var(_) | Expr::Break | Expr::Continue => 1,
        Expr::BinOp { left, right, .. } => 1 + cost_estimate(left) + cost_estimate(right),
        Expr::UnaryOp { expr: e, .. } => 1 + cost_estimate(e),
        Expr::Call { func, args } =>
        {
            1 + cost_estimate(func) + args.iter().map(cost_estimate).sum::<usize>()
        },
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } =>
        {
            1 + cost_estimate(cond)
                + cost_estimate(then_branch)
                + else_branch.as_ref().map_or(0, |e| cost_estimate(e))
        },
        Expr::Let { value, body, .. } | Expr::LetMut { value, body, .. } =>
        {
            1 + cost_estimate(value) + cost_estimate(body)
        },
        Expr::While { cond, body } => 1 + cost_estimate(cond) + cost_estimate(body),
        Expr::For { iter, body, .. } => 1 + cost_estimate(iter) + cost_estimate(body),
        Expr::Assign { value, .. } => 1 + cost_estimate(value),
        Expr::Block(stmts) => 1 + stmts.iter().map(cost_estimate).sum::<usize>(),
        Expr::Return(Some(e)) => 1 + cost_estimate(e),
        Expr::Return(None) => 1,
        Expr::Function { body, .. } => 1 + cost_estimate(body),
        Expr::Struct { .. } | Expr::Enum { .. } => 1,
        Expr::Match { expr: e, arms } =>
        {
            1 + cost_estimate(e) + arms.iter().map(|(_, b)| cost_estimate(b)).sum::<usize>()
        },
        Expr::TypeAnnotation { expr: e, .. } => 1 + cost_estimate(e),
        Expr::FieldAccess { expr: e, .. } => 1 + cost_estimate(e),
        Expr::Index { expr: e, index } => 1 + cost_estimate(e) + cost_estimate(index),
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::*;

    // --- Parse tests ---

    #[test]
    fn test_parse_literal_int() {
        let expr = parse_expr("42").unwrap();
        assert_eq!(expr, Expr::Lit(Literal::Int(42)));
    }

    #[test]
    fn test_parse_literal_float() {
        let expr = parse_expr("3.14").unwrap();
        assert_eq!(expr, Expr::Lit(Literal::Float(3.14)));
    }

    #[test]
    fn test_parse_literal_bool() {
        assert_eq!(parse_expr("true").unwrap(), Expr::Lit(Literal::Bool(true)));
        assert_eq!(
            parse_expr("false").unwrap(),
            Expr::Lit(Literal::Bool(false))
        );
    }

    #[test]
    fn test_parse_var() {
        let expr = parse_expr("x").unwrap();
        assert_eq!(expr, Expr::Var("x".to_string()));
    }

    #[test]
    fn test_parse_binop() {
        let expr = parse_expr("(+ 2 3)").unwrap();
        assert_eq!(
            expr,
            Expr::BinOp {
                op: BinOpKind::Add,
                left: Box::new(Expr::Lit(Literal::Int(2))),
                right: Box::new(Expr::Lit(Literal::Int(3))),
            }
        );
    }

    #[test]
    fn test_parse_nested_binop() {
        let expr = parse_expr("(* (+ 2 3) 4)").unwrap();
        assert!(matches!(
            expr,
            Expr::BinOp {
                op: BinOpKind::Mul,
                ..
            }
        ));
    }

    #[test]
    fn test_parse_let() {
        let expr = parse_expr("(let x 5 x)").unwrap();
        assert!(matches!(expr, Expr::Let { .. }));
    }

    #[test]
    fn test_parse_if() {
        let expr = parse_expr("(if true 1 2)").unwrap();
        assert!(matches!(expr, Expr::If { .. }));
    }

    #[test]
    fn test_parse_fn() {
        let expr = parse_expr("(fn add (a b) (+ a b))").unwrap();
        assert!(matches!(expr, Expr::Function { .. }));
    }

    #[test]
    fn test_parse_while() {
        let expr = parse_expr("(while (< x 10) (set! x (+ x 1)))").unwrap();
        assert!(matches!(expr, Expr::While { .. }));
    }

    #[test]
    fn test_parse_for() {
        let expr = parse_expr("(for i (.. 0 10) (call push v i))").unwrap();
        assert!(matches!(expr, Expr::For { .. }));
    }

    #[test]
    fn test_pretty_print_binop() {
        let expr = parse_expr("(+ 1 2)").unwrap();
        assert_eq!(expr.to_string(), "1 + 2");
    }

    #[test]
    fn test_pretty_print_nested() {
        let expr = parse_expr("(* (+ 1 2) 3)").unwrap();
        assert_eq!(expr.to_string(), "(1 + 2) * 3");
    }

    #[test]
    fn test_pretty_print_function() {
        let expr = parse_expr("(fn double (x) (* x 2))").unwrap();
        assert_eq!(expr.to_string(), "fn double(x) x * 2");
    }

    // --- Pattern matching tests ---

    #[test]
    fn test_pattern_match_add_zero() {
        let pat = parse_pattern("(+ $x 0)").unwrap();
        let expr = parse_expr("(+ y 0)").unwrap();
        let bindings = match_pattern(&pat, &expr).unwrap();
        assert_eq!(
            bindings[&PatternVar("x".to_string())],
            Expr::Var("y".to_string())
        );
    }

    #[test]
    fn test_pattern_match_no_match() {
        let pat = parse_pattern("(+ $x 0)").unwrap();
        let expr = parse_expr("(+ y 1)").unwrap();
        assert!(match_pattern(&pat, &expr).is_none());
    }

    #[test]
    fn test_rule_apply_add_zero() {
        let rule = Rule::new(
            "add-zero",
            parse_pattern("(+ $x 0)").unwrap(),
            parse_pattern("$x").unwrap(),
            90,
            "x + 0 -> x",
        );
        let expr = parse_expr("(+ a 0)").unwrap();
        let result = rule.apply(&expr).unwrap();
        assert_eq!(result, Expr::Var("a".to_string()));
    }

    #[test]
    fn test_rule_apply_mul_one() {
        let rule = Rule::new(
            "mul-one",
            parse_pattern("(* $x 1)").unwrap(),
            parse_pattern("$x").unwrap(),
            90,
            "x * 1 -> x",
        );
        let expr = parse_expr("(* y 1)").unwrap();
        let result = rule.apply(&expr).unwrap();
        assert_eq!(result, Expr::Var("y".to_string()));
    }

    // --- Constant folding tests ---

    #[test]
    fn test_const_fold_add() {
        let lit = try_const_fold(&parse_expr("(+ 2 3)").unwrap()).unwrap();
        assert_eq!(lit, Literal::Int(5));
    }

    #[test]
    fn test_const_fold_mul() {
        let lit = try_const_fold(&parse_expr("(* 4 5)").unwrap()).unwrap();
        assert_eq!(lit, Literal::Int(20));
    }

    #[test]
    fn test_const_fold_nested() {
        let lit = try_const_fold(&parse_expr("(* (+ 2 3) 4)").unwrap()).unwrap();
        assert_eq!(lit, Literal::Int(20));
    }

    // --- Dead code elimination tests ---

    #[test]
    fn test_dce_after_return() {
        let expr = parse_expr("(block (return 1) (set! x 2))").unwrap();
        let result = dead_code_elimination(&expr);
        match result
        {
            Expr::Block(stmts) => assert_eq!(stmts.len(), 1),
            _ => panic!("Expected block"),
        }
    }

    #[test]
    fn test_dce_no_terminator() {
        let expr = parse_expr("(block (set! x 1) (set! y 2))").unwrap();
        let result = dead_code_elimination(&expr);
        match result
        {
            Expr::Block(stmts) => assert_eq!(stmts.len(), 2),
            _ => panic!("Expected block"),
        }
    }

    // --- Rename variable tests ---

    #[test]
    fn test_rename_variable() {
        let expr = parse_expr("(+ x 1)").unwrap();
        let result = rename_variable(&expr, "x", "new_x");
        assert_eq!(result.to_string(), "new_x + 1");
    }

    #[test]
    fn test_rename_in_let_binding() {
        let expr = parse_expr("(let x 5 (+ x 1))").unwrap();
        let result = rename_variable(&expr, "x", "y");
        assert_eq!(result, parse_expr("(let y 5 (+ y 1))").unwrap());
    }

    // --- Boolean simplification tests ---

    #[test]
    fn test_simplify_and_true() {
        let expr = parse_expr("(&& x true)").unwrap();
        let result = simplify_boolean(&expr);
        assert_eq!(result, Expr::Var("x".to_string()));
    }

    #[test]
    fn test_simplify_or_false() {
        let expr = parse_expr("(|| x false)").unwrap();
        let result = simplify_boolean(&expr);
        assert_eq!(result, Expr::Var("x".to_string()));
    }

    #[test]
    fn test_simplify_double_negation() {
        let expr = parse_expr("(! (! x))").unwrap();
        let result = simplify_boolean(&expr);
        assert_eq!(result, Expr::Var("x".to_string()));
    }

    // --- Transpilation tests ---

    #[test]
    fn test_transpile_python_add() {
        let expr = parse_expr("(+ a b)").unwrap();
        let py = transpile_rust_to_python(&expr);
        assert_eq!(py, "(a + b)");
    }

    #[test]
    fn test_transpile_python_fn() {
        let expr = parse_expr("(fn add (a b) (+ a b))").unwrap();
        let py = transpile_rust_to_python(&expr);
        assert!(py.contains("def add(a, b):"));
    }

    #[test]
    fn test_transpile_c_add() {
        let expr = parse_expr("(+ a b)").unwrap();
        let c = transpile_rust_to_c(&expr);
        assert_eq!(c, "(a + b)");
    }

    #[test]
    fn test_transpile_c_fn() {
        let expr = parse_expr("(fn add (a b) (+ a b))").unwrap();
        let c = transpile_rust_to_c(&expr);
        assert!(c.contains("void add(int a, int b)"));
    }

    // --- Transformation engine tests ---

    #[test]
    fn test_engine_apply_rule() {
        let mut engine = TransformEngine::with_rules(optimization_rules());
        let expr = parse_expr("(+ x 0)").unwrap();
        let result = engine.apply_rule("add-zero", &expr).unwrap();
        assert_eq!(result, Expr::Var("x".to_string()));
    }

    #[test]
    fn test_engine_const_fold() {
        let mut engine = TransformEngine::with_rules(optimization_rules());
        let expr = parse_expr("(+ 2 3)").unwrap();
        let result = engine.apply_rule("const-fold-add", &expr).unwrap();
        assert_eq!(result, Expr::Lit(Literal::Int(5)));
    }

    #[test]
    fn test_pattern_database_json() {
        let mut db = PatternDatabase::new("test");
        db.add_rules(optimization_rules());
        let json = db.to_json().unwrap();
        let db2 = PatternDatabase::from_json(&json).unwrap();
        assert_eq!(db2.rules.len(), db.rules.len());
    }

    #[test]
    fn test_engine_exhaustive_strategy() {
        let mut engine = TransformEngine::with_rules(optimization_rules());
        let expr = parse_expr("(+ (+ 1 2) 3)").unwrap();
        let results = engine.transform(&expr, SelectionStrategy::Exhaustive);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_engine_logger() {
        let mut engine = TransformEngine::with_rules(optimization_rules());
        let expr = parse_expr("(+ x 0)").unwrap();
        engine.apply_rule("add-zero", &expr);
        assert!(!engine.logger.entries.is_empty());
    }

    #[test]
    fn test_inline_variable() {
        let expr = parse_expr("(let x 5 (+ x 1))").unwrap();
        let result = inline_variable(&expr, "x");
        assert_eq!(result.to_string(), "5 + 1");
    }

    #[test]
    fn test_loop_to_iterator() {
        let expr = parse_expr("(for i (.. 0 n) (call push v (* i 2)))").unwrap();
        let result = convert_loop_to_iterator(&expr).unwrap();
        let s = result.to_string();
        assert!(s.contains("map") || s.contains("collect"));
    }

    #[test]
    fn test_cost_estimate() {
        let expr = parse_expr("(+ (+ 1 2) (* 3 4))").unwrap();
        let cost = cost_estimate(&expr);
        assert!(cost > 3);
    }
}
