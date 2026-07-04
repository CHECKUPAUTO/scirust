//! Recursive-descent parser for the supported Python subset.
//!
//! Grammar (informal):
//!   module   := (def)*
//!   def      := 'def' NAME '(' params ')' ['->' hint] ':' NEWLINE block
//!   block    := INDENT stmt+ DEDENT
//!   stmt     := assign | assign_index | for | return
//!   for      := 'for' NAME 'in' 'range' '(' args ')' ':' NEWLINE block
//!   expr     := add ; add := mul (('+'|'-') mul)* ; mul := unary (('*'|'/') unary)*
//!   unary    := '-' unary | pow ; pow := postfix ('**' unary)?
//!   postfix  := atom ('[' expr ']')*
//!   atom     := INT | FLOAT | '(' expr ')' | dotted ['(' args ')']

use super::ast::*;
use super::lexer::Tok;

pub fn parse(toks: &[Tok]) -> Result<PyModule, String> {
    let mut p = Parser { toks, pos: 0 };
    let mut funcs = Vec::new();
    while !p.at_eof()
    {
        if p.is_name("def")
        {
            funcs.push(p.parse_func()?);
        }
        else
        {
            return Err(format!(
                "only top-level `def`s are supported; got {:?}",
                p.peek()
            ));
        }
    }
    Ok(PyModule { funcs })
}

struct Parser<'a> {
    toks: &'a [Tok],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> &Tok {
        self.toks.get(self.pos).unwrap_or(&Tok::Eof)
    }
    fn peek2(&self) -> &Tok {
        self.toks.get(self.pos + 1).unwrap_or(&Tok::Eof)
    }
    fn bump(&mut self) -> Tok {
        let t = self.peek().clone();
        self.pos += 1;
        t
    }
    fn at_eof(&self) -> bool {
        matches!(self.peek(), Tok::Eof)
    }
    fn is_name(&self, s: &str) -> bool {
        matches!(self.peek(), Tok::Name(n) if n == s)
    }
    fn is_sym(&self, s: &str) -> bool {
        matches!(self.peek(), Tok::Sym(x) if x == s)
    }
    fn eat_sym(&mut self, s: &str) -> Result<(), String> {
        if self.is_sym(s)
        {
            self.pos += 1;
            Ok(())
        }
        else
        {
            Err(format!("expected `{}`, got {:?}", s, self.peek()))
        }
    }
    fn eat_name(&mut self, s: &str) -> Result<(), String> {
        if self.is_name(s)
        {
            self.pos += 1;
            Ok(())
        }
        else
        {
            Err(format!("expected `{}`, got {:?}", s, self.peek()))
        }
    }
    fn eat_newlines(&mut self) {
        while matches!(self.peek(), Tok::Newline)
        {
            self.pos += 1;
        }
    }
    fn take_name(&mut self) -> Result<String, String> {
        match self.bump()
        {
            Tok::Name(n) => Ok(n),
            other => Err(format!("expected identifier, got {:?}", other)),
        }
    }

    fn parse_func(&mut self) -> Result<PyFunc, String> {
        self.eat_name("def")?;
        let name = self.take_name()?;
        self.eat_sym("(")?;
        let mut params = Vec::new();
        while !self.is_sym(")")
        {
            let pname = self.take_name()?;
            let hint = if self.is_sym(":")
            {
                self.eat_sym(":")?;
                self.parse_hint()?
            }
            else
            {
                None
            };
            params.push(PyParam { name: pname, hint });
            if self.is_sym(",")
            {
                self.eat_sym(",")?;
            }
            else
            {
                break;
            }
        }
        self.eat_sym(")")?;
        let ret_hint = if self.is_sym("->")
            || (self.is_sym("-") && matches!(self.peek2(), Tok::Sym(s) if s == ">"))
        {
            // '->' may arrive as one Sym or as '-' '>' depending on lexing.
            if self.is_sym("->")
            {
                self.bump();
            }
            else
            {
                self.bump();
                self.bump();
            }
            self.parse_hint()?
        }
        else
        {
            None
        };
        self.eat_sym(":")?;
        let body = self.parse_block()?;
        Ok(PyFunc {
            name,
            params,
            ret_hint,
            body,
        })
    }

    /// Parse a type annotation and classify it. Unknown hints -> None (inferred).
    fn parse_hint(&mut self) -> Result<Option<TypeHint>, String> {
        // Read a dotted name or a string literal token.
        let mut s = match self.bump()
        {
            Tok::Name(n) => n,
            other => return Err(format!("expected type hint, got {:?}", other)),
        };
        while self.is_sym(".")
        {
            self.eat_sym(".")?;
            s.push('.');
            s.push_str(&self.take_name()?);
        }
        let low = s.trim_start_matches('\u{1}').to_ascii_lowercase();
        let low = low.strip_prefix("str:").unwrap_or(&low).to_string();
        let hint = if low.contains("ndarray") || low.contains("array") || low.contains("list")
        {
            Some(TypeHint::Array)
        }
        else if low.contains("float")
        {
            Some(TypeHint::Float)
        }
        else if low.contains("int")
        {
            Some(TypeHint::Int)
        }
        else
        {
            None
        };
        Ok(hint)
    }

    fn parse_block(&mut self) -> Result<Vec<PyStmt>, String> {
        self.eat_newlines();
        if !matches!(self.peek(), Tok::Indent)
        {
            return Err(format!("expected an indented block, got {:?}", self.peek()));
        }
        self.bump(); // INDENT
        let mut stmts = Vec::new();
        loop
        {
            self.eat_newlines();
            if matches!(self.peek(), Tok::Dedent | Tok::Eof)
            {
                break;
            }
            stmts.push(self.parse_stmt()?);
        }
        if matches!(self.peek(), Tok::Dedent)
        {
            self.bump();
        }
        if stmts.is_empty()
        {
            return Err("empty block".into());
        }
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<PyStmt, String> {
        if self.is_name("return")
        {
            self.bump();
            if matches!(self.peek(), Tok::Newline | Tok::Dedent | Tok::Eof)
            {
                return Ok(PyStmt::Return(None));
            }
            let e = self.parse_expr()?;
            return Ok(PyStmt::Return(Some(e)));
        }
        if self.is_name("for")
        {
            return self.parse_for();
        }
        if self.is_name("if")
        {
            return self.parse_if();
        }
        if self.is_name("while")
        {
            return self.parse_while();
        }
        // assignment: NAME ['[' idx ']'] '=' expr
        let target = self.take_name()?;
        if self.is_sym("[")
        {
            self.eat_sym("[")?;
            let index = self.parse_expr()?;
            self.eat_sym("]")?;
            self.eat_sym("=")?;
            let value = self.parse_expr()?;
            return Ok(PyStmt::AssignIndex {
                target,
                index,
                value,
            });
        }
        self.eat_sym("=")?;
        let value = self.parse_expr()?;
        Ok(PyStmt::Assign { target, value })
    }

    fn parse_for(&mut self) -> Result<PyStmt, String> {
        self.eat_name("for")?;
        let var = self.take_name()?;
        self.eat_name("in")?;
        self.eat_name("range")?;
        self.eat_sym("(")?;
        let mut args = Vec::new();
        while !self.is_sym(")")
        {
            args.push(self.parse_expr()?);
            if self.is_sym(",")
            {
                self.eat_sym(",")?;
            }
            else
            {
                break;
            }
        }
        self.eat_sym(")")?;
        self.eat_sym(":")?;
        let (start, end) = match args.len()
        {
            1 => (PyExpr::Int(0), args.into_iter().next().unwrap()),
            2 =>
            {
                let mut it = args.into_iter();
                (it.next().unwrap(), it.next().unwrap())
            },
            _ => return Err("range() takes 1 or 2 arguments in this subset".into()),
        };
        let body = self.parse_block()?;
        Ok(PyStmt::For {
            var,
            start,
            end,
            body,
        })
    }

    /// `if cond: block (elif cond: block)* (else: block)?`
    fn parse_if(&mut self) -> Result<PyStmt, String> {
        self.eat_name("if")?;
        self.parse_if_tail()
    }

    /// `while cond: block`
    fn parse_while(&mut self) -> Result<PyStmt, String> {
        self.eat_name("while")?;
        let cond = self.parse_condition()?;
        self.eat_sym(":")?;
        let body = self.parse_block()?;
        Ok(PyStmt::While { cond, body })
    }

    /// Parse the part after `if`/`elif`: `cond ':' block` then optional
    /// `elif`/`else`. `elif` desugars into a nested `If` in the else branch.
    fn parse_if_tail(&mut self) -> Result<PyStmt, String> {
        let cond = self.parse_condition()?;
        self.eat_sym(":")?;
        let then = self.parse_block()?;
        self.eat_newlines();
        let els = if self.is_name("elif")
        {
            self.eat_name("elif")?;
            vec![self.parse_if_tail()?]
        }
        else if self.is_name("else")
        {
            self.eat_name("else")?;
            self.eat_sym(":")?;
            self.parse_block()?
        }
        else
        {
            Vec::new()
        };
        Ok(PyStmt::If { cond, then, els })
    }

    /// A boolean condition: a single comparison `add <op> add`.
    fn parse_condition(&mut self) -> Result<PyExpr, String> {
        let l = self.parse_add()?;
        let op = if self.is_sym("<")
        {
            CmpOp::Lt
        }
        else if self.is_sym("<=")
        {
            CmpOp::Le
        }
        else if self.is_sym(">")
        {
            CmpOp::Gt
        }
        else if self.is_sym(">=")
        {
            CmpOp::Ge
        }
        else if self.is_sym("==")
        {
            CmpOp::Eq
        }
        else if self.is_sym("!=")
        {
            CmpOp::Ne
        }
        else
        {
            return Err(format!(
                "condition must be a comparison (`<`,`<=`,`>`,`>=`,`==`,`!=`), got {:?}",
                self.peek()
            ));
        };
        self.bump();
        let r = self.parse_add()?;
        Ok(PyExpr::Cmp {
            op,
            l: Box::new(l),
            r: Box::new(r),
        })
    }

    // ---- expressions ------------------------------------------------------

    fn parse_expr(&mut self) -> Result<PyExpr, String> {
        self.parse_add()
    }

    fn parse_add(&mut self) -> Result<PyExpr, String> {
        let mut left = self.parse_mul()?;
        loop
        {
            let op = if self.is_sym("+")
            {
                BinOp::Add
            }
            else if self.is_sym("-")
            {
                BinOp::Sub
            }
            else
            {
                break;
            };
            self.bump();
            let right = self.parse_mul()?;
            left = PyExpr::Bin {
                op,
                l: Box::new(left),
                r: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_mul(&mut self) -> Result<PyExpr, String> {
        let mut left = self.parse_unary()?;
        loop
        {
            let op = if self.is_sym("*")
            {
                BinOp::Mul
            }
            else if self.is_sym("/")
            {
                BinOp::Div
            }
            else if self.is_sym("@")
            {
                BinOp::MatMul
            }
            else
            {
                break;
            };
            self.bump();
            let right = self.parse_unary()?;
            left = PyExpr::Bin {
                op,
                l: Box::new(left),
                r: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<PyExpr, String> {
        if self.is_sym("-")
        {
            self.bump();
            let e = self.parse_unary()?;
            return Ok(PyExpr::Neg(Box::new(e)));
        }
        if self.is_sym("+")
        {
            self.bump();
            return self.parse_unary();
        }
        self.parse_pow()
    }

    fn parse_pow(&mut self) -> Result<PyExpr, String> {
        let base = self.parse_postfix()?;
        if self.is_sym("**")
        {
            self.bump();
            let exp = self.parse_unary()?; // right-assoc
            return Ok(PyExpr::Bin {
                op: BinOp::Pow,
                l: Box::new(base),
                r: Box::new(exp),
            });
        }
        Ok(base)
    }

    fn parse_postfix(&mut self) -> Result<PyExpr, String> {
        let mut e = self.parse_atom()?;
        while self.is_sym("[")
        {
            self.eat_sym("[")?;
            let idx = self.parse_expr()?;
            self.eat_sym("]")?;
            // `x.shape[0]` -> len(x)  (1-D subset)
            if let PyExpr::Name(n) = &e
            {
                if let Some(base) = n.strip_suffix(".shape")
                {
                    e = PyExpr::Call {
                        func: "len".into(),
                        args: vec![PyExpr::Name(base.to_string())],
                    };
                    continue;
                }
            }
            e = PyExpr::Index {
                base: Box::new(e),
                index: Box::new(idx),
            };
        }
        Ok(e)
    }

    fn parse_atom(&mut self) -> Result<PyExpr, String> {
        match self.peek().clone()
        {
            Tok::Int(v) =>
            {
                self.bump();
                Ok(PyExpr::Int(v))
            },
            Tok::Float(v) =>
            {
                self.bump();
                Ok(PyExpr::Float(v))
            },
            Tok::Sym(s) if s == "(" =>
            {
                self.bump();
                let e = self.parse_expr()?;
                self.eat_sym(")")?;
                Ok(e)
            },
            Tok::Name(_) =>
            {
                // dotted name
                let mut name = self.take_name()?;
                while self.is_sym(".") && matches!(self.peek2(), Tok::Name(_))
                {
                    self.eat_sym(".")?;
                    name.push('.');
                    name.push_str(&self.take_name()?);
                }
                if self.is_sym("(")
                {
                    self.eat_sym("(")?;
                    let mut args = Vec::new();
                    while !self.is_sym(")")
                    {
                        args.push(self.parse_expr()?);
                        if self.is_sym(",")
                        {
                            self.eat_sym(",")?;
                        }
                        else
                        {
                            break;
                        }
                    }
                    self.eat_sym(")")?;
                    Ok(PyExpr::Call { func: name, args })
                }
                else
                {
                    Ok(PyExpr::Name(name))
                }
            },
            other => Err(format!("unexpected token in expression: {:?}", other)),
        }
    }
}
