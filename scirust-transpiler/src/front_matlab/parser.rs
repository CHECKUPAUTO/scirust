//! Recursive-descent parser for the supported MATLAB/Octave subset.

use super::ast::*;
use super::lexer::MTok;

pub fn parse(toks: &[MTok]) -> Result<MModule, String> {
    let mut p = Parser { toks, pos: 0 };
    let mut funcs = Vec::new();
    p.skip_terms();
    while !p.at_eof()
    {
        if p.is_ident("function")
        {
            funcs.push(p.parse_func()?);
        }
        else
        {
            return Err(format!("expected `function`, got {:?}", p.peek()));
        }
        p.skip_terms();
    }
    Ok(MModule { funcs })
}

struct Parser<'a> {
    toks: &'a [MTok],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> &MTok {
        self.toks.get(self.pos).unwrap_or(&MTok::Eof)
    }
    fn bump(&mut self) -> MTok {
        let t = self.peek().clone();
        self.pos += 1;
        t
    }
    fn at_eof(&self) -> bool {
        matches!(self.peek(), MTok::Eof)
    }
    fn is_ident(&self, s: &str) -> bool {
        matches!(self.peek(), MTok::Ident(n) if n == s)
    }
    fn is_sym(&self, s: &str) -> bool {
        matches!(self.peek(), MTok::Sym(x) if x == s)
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
    fn eat_ident(&mut self, s: &str) -> Result<(), String> {
        if self.is_ident(s)
        {
            self.pos += 1;
            Ok(())
        }
        else
        {
            Err(format!("expected `{}`, got {:?}", s, self.peek()))
        }
    }
    fn take_ident(&mut self) -> Result<String, String> {
        match self.bump()
        {
            MTok::Ident(n) => Ok(n),
            other => Err(format!("expected identifier, got {:?}", other)),
        }
    }
    fn skip_terms(&mut self) {
        while matches!(self.peek(), MTok::Term)
        {
            self.pos += 1;
        }
    }
    fn at_block_end(&self) -> bool {
        self.is_ident("end")
            || self.is_ident("endfunction")
            || self.is_ident("else")
            || self.is_ident("elseif")
            || self.at_eof()
    }

    fn parse_func(&mut self) -> Result<MFunc, String> {
        self.eat_ident("function")?;
        // Output list: a single `out` or a bracketed `[o1, o2, …]`.
        let outs = if self.is_sym("[")
        {
            self.eat_sym("[")?;
            let mut outs = Vec::new();
            while !self.is_sym("]")
            {
                outs.push(self.take_ident()?);
                if self.is_sym(",")
                {
                    self.eat_sym(",")?;
                }
                else
                {
                    break;
                }
            }
            self.eat_sym("]")?;
            if outs.is_empty()
            {
                return Err("a function must declare at least one output".into());
            }
            outs
        }
        else
        {
            vec![self.take_ident()?]
        };
        self.eat_sym("=")?;
        let name = self.take_ident()?;
        self.eat_sym("(")?;
        let mut params = Vec::new();
        while !self.is_sym(")")
        {
            params.push(self.take_ident()?);
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
        let body = self.parse_block()?;
        // close with `end` or `endfunction`
        if self.is_ident("end") || self.is_ident("endfunction")
        {
            self.bump();
        }
        else
        {
            return Err(format!(
                "expected `end` to close function, got {:?}",
                self.peek()
            ));
        }
        Ok(MFunc {
            name,
            outs,
            params,
            body,
        })
    }

    /// Parse statements until a block-closing keyword.
    fn parse_block(&mut self) -> Result<Vec<MStmt>, String> {
        let mut stmts = Vec::new();
        self.skip_terms();
        while !self.at_block_end()
        {
            stmts.push(self.parse_stmt()?);
            self.skip_terms();
        }
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<MStmt, String> {
        if self.is_ident("for")
        {
            return self.parse_for();
        }
        if self.is_ident("if")
        {
            return self.parse_if();
        }
        if self.is_ident("while")
        {
            return self.parse_while();
        }
        // assignment: NAME ['(' idx ')'] '=' expr
        let target = self.take_ident()?;
        if self.is_sym("(")
        {
            self.eat_sym("(")?;
            let index = self.parse_expr()?;
            self.eat_sym(")")?;
            self.eat_sym("=")?;
            let value = self.parse_expr()?;
            return Ok(MStmt::AssignIndex {
                target,
                index,
                value,
            });
        }
        self.eat_sym("=")?;
        let value = self.parse_expr()?;
        Ok(MStmt::Assign { target, value })
    }

    fn parse_for(&mut self) -> Result<MStmt, String> {
        self.eat_ident("for")?;
        let var = self.take_ident()?;
        self.eat_sym("=")?;
        let lo = self.parse_add()?;
        self.eat_sym(":")?;
        let hi = self.parse_add()?;
        let body = self.parse_block()?;
        self.eat_ident("end")?;
        Ok(MStmt::For { var, lo, hi, body })
    }

    fn parse_while(&mut self) -> Result<MStmt, String> {
        self.eat_ident("while")?;
        let cond = self.parse_condition()?;
        let body = self.parse_block()?;
        self.eat_ident("end")?;
        Ok(MStmt::While { cond, body })
    }

    fn parse_if(&mut self) -> Result<MStmt, String> {
        self.eat_ident("if")?;
        self.parse_if_tail()
    }

    fn parse_if_tail(&mut self) -> Result<MStmt, String> {
        let cond = self.parse_condition()?;
        let then = self.parse_block()?;
        let els = if self.is_ident("elseif")
        {
            self.eat_ident("elseif")?;
            vec![self.parse_if_tail()?]
        }
        else if self.is_ident("else")
        {
            self.eat_ident("else")?;
            let e = self.parse_block()?;
            self.eat_ident("end")?;
            return Ok(MStmt::If { cond, then, els: e });
        }
        else
        {
            self.eat_ident("end")?;
            Vec::new()
        };
        // `elseif` path already consumed its own `end` via the nested tail.
        Ok(MStmt::If { cond, then, els })
    }

    fn parse_condition(&mut self) -> Result<MExpr, String> {
        let l = self.parse_add()?;
        let op = if self.is_sym("<")
        {
            MCmpOp::Lt
        }
        else if self.is_sym("<=")
        {
            MCmpOp::Le
        }
        else if self.is_sym(">")
        {
            MCmpOp::Gt
        }
        else if self.is_sym(">=")
        {
            MCmpOp::Ge
        }
        else if self.is_sym("==")
        {
            MCmpOp::Eq
        }
        else if self.is_sym("~=")
        {
            MCmpOp::Ne
        }
        else
        {
            return Err(format!(
                "expected a comparison operator, got {:?}",
                self.peek()
            ));
        };
        self.bump();
        let r = self.parse_add()?;
        Ok(MExpr::Cmp {
            op,
            l: Box::new(l),
            r: Box::new(r),
        })
    }

    fn parse_expr(&mut self) -> Result<MExpr, String> {
        self.parse_add()
    }

    fn parse_add(&mut self) -> Result<MExpr, String> {
        let mut left = self.parse_mul()?;
        loop
        {
            let op = if self.is_sym("+")
            {
                MBinOp::Add
            }
            else if self.is_sym("-")
            {
                MBinOp::Sub
            }
            else
            {
                break;
            };
            self.bump();
            let right = self.parse_mul()?;
            left = MExpr::Bin {
                op,
                l: Box::new(left),
                r: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_mul(&mut self) -> Result<MExpr, String> {
        let mut left = self.parse_unary()?;
        loop
        {
            let op = if self.is_sym("*")
            {
                MBinOp::Mul
            }
            else if self.is_sym("/")
            {
                MBinOp::Div
            }
            else if self.is_sym(".*")
            {
                MBinOp::EMul
            }
            else if self.is_sym("./")
            {
                MBinOp::EDiv
            }
            else if self.is_sym("\\")
            {
                MBinOp::LDiv
            }
            else
            {
                break;
            };
            self.bump();
            let right = self.parse_unary()?;
            left = MExpr::Bin {
                op,
                l: Box::new(left),
                r: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<MExpr, String> {
        if self.is_sym("-")
        {
            self.bump();
            return Ok(MExpr::Neg(Box::new(self.parse_unary()?)));
        }
        if self.is_sym("+")
        {
            self.bump();
            return self.parse_unary();
        }
        self.parse_pow()
    }

    fn parse_pow(&mut self) -> Result<MExpr, String> {
        let base = self.parse_postfix()?;
        let op = if self.is_sym("^")
        {
            Some(MBinOp::Pow)
        }
        else if self.is_sym(".^")
        {
            Some(MBinOp::EPow)
        }
        else
        {
            None
        };
        if let Some(op) = op
        {
            self.bump();
            let exp = self.parse_unary()?;
            return Ok(MExpr::Bin {
                op,
                l: Box::new(base),
                r: Box::new(exp),
            });
        }
        Ok(base)
    }

    /// An atom followed by any number of postfix transpose operators (`A'`,
    /// `A.'`). Transpose binds tighter than `^`, so it is applied here first.
    fn parse_postfix(&mut self) -> Result<MExpr, String> {
        let mut e = self.parse_atom()?;
        while self.is_sym("'") || self.is_sym(".'")
        {
            self.bump();
            e = MExpr::Transpose(Box::new(e));
        }
        Ok(e)
    }

    fn parse_atom(&mut self) -> Result<MExpr, String> {
        match self.peek().clone()
        {
            MTok::Num(v) =>
            {
                self.bump();
                Ok(MExpr::Num(v))
            },
            MTok::Sym(s) if s == "(" =>
            {
                self.bump();
                let e = self.parse_expr()?;
                self.eat_sym(")")?;
                Ok(e)
            },
            MTok::Ident(name) =>
            {
                self.bump();
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
                    // Resolved to intrinsic vs 1-based indexing during lowering.
                    Ok(MExpr::Call { func: name, args })
                }
                else
                {
                    Ok(MExpr::Ident(name))
                }
            },
            other => Err(format!("unexpected token in expression: {:?}", other)),
        }
    }
}
