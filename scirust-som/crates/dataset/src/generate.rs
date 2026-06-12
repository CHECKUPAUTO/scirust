//! Seeded program generator + oracle-labelled training samples.
//!
//! Programs are generated from a [`PcgEngine`] stream (the framework's own
//! deterministic RNG), then labelled by [`OwnershipOracle`]. The generator
//! never invents labels: the oracle is the single source of truth, so a
//! "fault biased" pick that happens to be legal is simply labelled legal.

use scirust_core::nn::rng::PcgEngine;
use scirust_som_pcg::ast::{Expression, Function, Literal, SomAst, Statement, Type};
use scirust_som_symbolic::OwnershipOracle;
use scirust_som_tokenizer::{MAX_VARS, SomVocab, StructuredTokenizer};

#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Statements per function body (before nesting).
    pub min_statements: usize,
    pub max_statements: usize,
    /// Maximum nesting depth of explicit scopes.
    pub max_scope_depth: usize,
    /// Probability (percent) of picking a possibly-moved variable on
    /// purpose when generating a use — drives fault coverage.
    pub fault_bias_pct: u32,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            min_statements: 3,
            max_statements: 8,
            max_scope_depth: 2,
            fault_bias_pct: 35,
        }
    }
}

/// Deterministic toy-program generator.
pub struct ProgramGenerator {
    rng: PcgEngine,
    cfg: GeneratorConfig,
}

impl ProgramGenerator {
    pub fn new(seed: u64) -> Self {
        Self::with_config(seed, GeneratorConfig::default())
    }

    pub fn with_config(seed: u64, cfg: GeneratorConfig) -> Self {
        Self {
            rng: PcgEngine::new(seed),
            cfg,
        }
    }

    fn pick(&mut self, n: usize) -> usize {
        (self.rng.next_u32() as usize) % n.max(1)
    }

    fn chance(&mut self, pct: u32) -> bool {
        self.rng.next_u32() % 100 < pct
    }

    /// Generate one program (a single `main` function).
    pub fn generate(&mut self) -> SomAst {
        let mut names: Vec<String> = Vec::new();
        let n_stmts = self.cfg.min_statements
            + self.pick(self.cfg.max_statements - self.cfg.min_statements + 1);
        let mut body = Vec::with_capacity(n_stmts);
        for _ in 0..n_stmts
        {
            body.push(self.gen_statement(&mut names, 0));
        }
        if self.chance(25) && !names.is_empty()
        {
            let target = names[self.pick(names.len())].clone();
            let expr = if self.chance(30)
            {
                // Escaping borrow fault candidate.
                Expression::Reference {
                    name: target,
                    mutable: false,
                }
            }
            else
            {
                Expression::Variable(target)
            };
            body.push(Statement::Return(Some(expr)));
        }
        SomAst::Program(vec![Function {
            name: "main".to_string(),
            params: vec![],
            body,
        }])
    }

    fn fresh_or_reuse(&mut self, names: &mut Vec<String>) -> String {
        if names.len() < MAX_VARS
        {
            let name = format!("v{}", names.len());
            names.push(name.clone());
            name
        }
        else
        {
            // Shadow an existing name to stay within the vocab slots.
            names[self.pick(names.len())].clone()
        }
    }

    fn any_name(&mut self, names: &[String]) -> Option<String> {
        if names.is_empty()
        {
            None
        }
        else
        {
            Some(names[self.pick(names.len())].clone())
        }
    }

    fn gen_statement(&mut self, names: &mut Vec<String>, depth: usize) -> Statement {
        let roll = self.rng.next_u32() % 100;
        match roll
        {
            0..=24 =>
            {
                let name = self.fresh_or_reuse(names);
                Statement::VarDecl {
                    name,
                    ty: Type::Int,
                    init: Some(Expression::Literal(Literal::Int(
                        (self.rng.next_u32() % 100) as i64,
                    ))),
                }
            },
            25..=44 => match self.any_name(names)
            {
                Some(from) =>
                {
                    let name = self.fresh_or_reuse(names);
                    Statement::VarDecl {
                        name,
                        ty: Type::Int,
                        init: Some(Expression::Variable(from)),
                    }
                },
                None => self.gen_literal_decl(names),
            },
            45..=59 => match self.any_name(names)
            {
                Some(of) =>
                {
                    let mutable = self.chance(35);
                    let name = self.fresh_or_reuse(names);
                    Statement::VarDecl {
                        name,
                        ty: Type::Ref(Box::new(Type::Int), mutable),
                        init: Some(Expression::Reference { name: of, mutable }),
                    }
                },
                None => self.gen_literal_decl(names),
            },
            60..=69 => match self.any_name(names)
            {
                Some(lhs) => Statement::Assignment {
                    lhs,
                    rhs: Expression::Literal(Literal::Int((self.rng.next_u32() % 100) as i64)),
                },
                None => self.gen_literal_decl(names),
            },
            70..=84 =>
            {
                // A bare use; with fault_bias the picked variable may well
                // already be moved — the oracle decides.
                let biased = self.chance(self.cfg.fault_bias_pct);
                match self.any_name(names)
                {
                    Some(mut target) =>
                    {
                        if biased && names.len() > 1
                        {
                            target = names[self.pick(names.len())].clone();
                        }
                        Statement::Expression(Expression::Variable(target))
                    },
                    None => self.gen_literal_decl(names),
                }
            },
            85..=92 if depth < self.cfg.max_scope_depth =>
            {
                let n_inner = 1 + self.pick(3);
                let mut inner = Vec::with_capacity(n_inner);
                for _ in 0..n_inner
                {
                    inner.push(self.gen_statement(names, depth + 1));
                }
                Statement::Scope(inner)
            },
            _ => match self.any_name(names)
            {
                Some(a) =>
                {
                    let b = self.any_name(names).expect("non-empty");
                    let name = self.fresh_or_reuse(names);
                    Statement::VarDecl {
                        name,
                        ty: Type::Int,
                        init: Some(Expression::BinaryOp {
                            left: Box::new(Expression::Variable(a)),
                            op: scirust_som_pcg::ast::BinaryOp::Add,
                            right: Box::new(Expression::Variable(b)),
                        }),
                    }
                },
                None => self.gen_literal_decl(names),
            },
        }
    }

    fn gen_literal_decl(&mut self, names: &mut Vec<String>) -> Statement {
        let name = self.fresh_or_reuse(names);
        Statement::VarDecl {
            name,
            ty: Type::Int,
            init: Some(Expression::Literal(Literal::Int(0))),
        }
    }
}

/// One per-token training sample (already integer-encoded).
#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub token_ids: Vec<usize>,
    /// Ownership class id per token (see `scirust-som-symbolic`).
    pub ownership: Vec<usize>,
    /// Borrow class id per token.
    pub borrow: Vec<usize>,
    /// 1.0 where the token is a fault, else 0.0.
    pub invalid: Vec<f32>,
}

impl TrainingSample {
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }
}

/// Generate `n` oracle-labelled samples with sequence length in
/// `[4, max_len]`. Fully deterministic in `seed`.
pub fn build_training_set(seed: u64, n: usize, max_len: usize) -> Vec<TrainingSample> {
    let mut generator = ProgramGenerator::new(seed);
    let oracle = OwnershipOracle::new();
    let tokenizer = StructuredTokenizer::new();
    let mut samples = Vec::with_capacity(n);
    let mut attempts = 0usize;
    let max_attempts = n.saturating_mul(20).max(64);
    while samples.len() < n && attempts < max_attempts
    {
        attempts += 1;
        let ast = generator.generate();
        let analysis = oracle.analyze(&ast);
        debug_assert_eq!(
            analysis.tokens,
            tokenizer.tokenize_ast_with_drops(&ast),
            "oracle stream must match tokenizer stream"
        );
        if analysis.tokens.len() < 4 || analysis.tokens.len() > max_len
        {
            continue;
        }
        samples.push(TrainingSample {
            token_ids: SomVocab::encode(&analysis.tokens),
            ownership: analysis.ownership_ids(),
            borrow: analysis.borrow_ids(),
            invalid: analysis.invalid_flags(),
        });
    }
    samples
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_som_symbolic::{OWNERSHIP_DROPPED, OWNERSHIP_MOVED, OWNERSHIP_OWNED};

    #[test]
    fn generation_is_deterministic() {
        let a = build_training_set(7, 10, 64);
        let b = build_training_set(7, 10, 64);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b)
        {
            assert_eq!(x.token_ids, y.token_ids);
            assert_eq!(x.ownership, y.ownership);
            assert_eq!(x.borrow, y.borrow);
            assert_eq!(x.invalid, y.invalid);
        }
    }

    #[test]
    fn dataset_has_class_coverage() {
        let samples = build_training_set(42, 60, 64);
        assert!(samples.len() >= 50, "generator should reach the target");
        let mut seen = [false; 5];
        let mut any_invalid = false;
        for s in &samples
        {
            assert_eq!(s.token_ids.len(), s.ownership.len());
            assert_eq!(s.token_ids.len(), s.borrow.len());
            assert_eq!(s.token_ids.len(), s.invalid.len());
            for &o in &s.ownership
            {
                seen[o] = true;
            }
            if s.invalid.iter().any(|&f| f > 0.5)
            {
                any_invalid = true;
            }
        }
        assert!(seen[OWNERSHIP_OWNED], "Owned must appear");
        assert!(seen[OWNERSHIP_MOVED], "Moved must appear");
        assert!(seen[OWNERSHIP_DROPPED], "Dropped must appear");
        assert!(any_invalid, "fault tokens must appear in the dataset");
    }

    #[test]
    fn token_ids_within_vocab() {
        let samples = build_training_set(3, 20, 64);
        let vs = SomVocab::vocab_size();
        for s in &samples
        {
            assert!(s.token_ids.iter().all(|&id| id < vs));
        }
    }
}
